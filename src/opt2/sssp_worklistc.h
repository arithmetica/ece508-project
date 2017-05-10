#define SSSP_VARIANT "worklistc"

#define MAXDIST		100

#define AVGDEGREE	2.5
#define WORKPERTHREAD	1
//#define THRESHOLD 10000
//#define DELTA 1000

unsigned int NVERTICES;

#include <cub/cub.cuh>
#include "worklistc.h"
#include "gbar.cuh"
#include "cutil_subset.h"

const int BLKSIZE = 512;

struct workprogress 
{
  Worklist2 *wl[4];
  int in_wl;
  int iteration;
};

texture <int, 1, cudaReadModeElementType> columns;
texture <int, 1, cudaReadModeElementType> row_offsets;

__global__
void initialize(foru *dist, unsigned int nv) {
	unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("ii=%d, nv=%d.\n", ii, *nv);
	if (ii < nv) {
		dist[ii] = MYINFINITY;
	}
}

__device__
foru processedge2(foru *dist, Graph &graph, unsigned iteration, unsigned src, unsigned edge, unsigned &dst, Worklist2 &inwl, foru &altdist) {
  
  dst = tex1Dfetch(columns, edge);
  if (dst >= graph.nnodes) return 0;

  foru wt = graph.edgessrcwt[edge];
  if (wt >= MYINFINITY) return 0;

  foru dstwt = cub::ThreadLoad<cub::LOAD_CG>(dist + dst);
  altdist = cub::ThreadLoad<cub::LOAD_CG>(dist + src) + wt;  

  //printf("%d %d %d %d %d\n", src, dst, wt, dstwt, altdist);

  if(altdist < dstwt)
    {
      atomicAdd(inwl.dedges_touched, (int) 1);
      atomicMin(&dist[dst], altdist);
      return 1;

      /* foru olddist = atomicMin(&dist[dst], altdist); */
      /* if (altdist < olddist) { */
      /* 	return olddist; */
      /* }  */
    }
  
  return 0;
}

/*
__device__ void expandByCTA(foru *dist, Graph &graph, Worklist2 &inwl, Worklist2 &outwl, unsigned iteration)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  int nn;

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  __shared__ int owner;
  __shared__ int shnn;

  int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);

  owner = -1;

  while(total_inputs-- > 0)
    {      
      int neighborsize = 0;
      int neighboroffset = 0;
      int nnsize = 0;

      if(inwl.pop_id(id, nn))
	{	  
	  if(nn != -1)
	    {
	      neighborsize = nnsize = graph.getOutDegree(nn);
	      neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[nn]);
	    }
	}

      while(true)
	{
	  if(nnsize > BLKSIZE)
	    owner = threadIdx.x;

	  __syncthreads();
	  
	  if(owner == -1)
	    break;

	  if(owner == threadIdx.x)
	    {
	      shnn = nn;
	      cub::ThreadStore<cub::STORE_CG>(inwl.dwl + id, -1);
	      owner = -1;
	      nnsize = 0;
	    }

	  __syncthreads();

	  neighborsize = graph.getOutDegree(shnn);
	  neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[shnn]);
	  int xy = ((neighborsize + blockDim.x - 1) / blockDim.x) * blockDim.x;
	  
	  for(int i = threadIdx.x; i < xy; i+= blockDim.x)
	    {
	      int ncnt = 0;
	      unsigned to_push = 0;
	      if(i < neighborsize)
		if(processedge2(dist, graph, iteration, shnn, neighboroffset + i, to_push))
		  {
		    ncnt = 1;
		  }
	    
	      outwl.push_1item<BlockScan>(ncnt, (int) to_push, BLKSIZE);
	    }
	}

      id += gridDim.x * blockDim.x;
    }
}*/

__device__
unsigned processnode2(foru *dist, Graph &graph, Worklist2 &inwl, Worklist2 &outwl1, Worklist2 &outwl2, unsigned iteration, long unsigned base, long unsigned delta) 
{
  //expandByCTA(dist, graph, inwl, outwl, iteration);

  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  int nn;

  typedef cub::BlockScan<int, BLKSIZE> BlockScan;

  const int SCRATCHSIZE = BLKSIZE;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ int gather_offsets[SCRATCHSIZE];
  __shared__ int src[SCRATCHSIZE];

  gather_offsets[threadIdx.x] = 0;

  int total_inputs = (*inwl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);
  
  while(total_inputs-- > 0)
    {      
      int neighborsize = 0;
      int neighboroffset = 0;
      int scratch_offset = 0;
      int total_edges = 0;

      if(inwl.pop_id(id, nn))
	{	  
	  if(nn != -1)
	    {
	      
	      neighborsize = graph.getOutDegree(nn);
	      neighboroffset = tex1Dfetch(row_offsets, graph.srcsrc[nn]);
	    }
	}

      BlockScan(temp_storage).ExclusiveSum(neighborsize, scratch_offset, total_edges);
  
      int done = 0;
      int neighborsdone = 0;

      /* if(total_edges) */
      /* 	if(threadIdx.x == 0) */
      /* 	  printf("total edges: %d\n", total_edges); */

      while(total_edges > 0)
	{
	  __syncthreads();

	  int i;
	  for(i = 0; neighborsdone + i < neighborsize && (scratch_offset + i - done) < SCRATCHSIZE; i++)
	    {
	      gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
	      src[scratch_offset + i - done] = nn;
	    }

	  neighborsdone += i;
	  scratch_offset += i;

	  __syncthreads();

	  int ncnt = 0, ncnt1 = 0, ncnt2 = 0;
	  unsigned to_push = 0, to_push1=0, to_push2=0;
	  foru altdist;

	  if(threadIdx.x < total_edges)
	  	{
	      if(processedge2(dist, graph, iteration, src[threadIdx.x], gather_offsets[threadIdx.x], to_push, inwl, altdist))
		{
		  ncnt = 1;
		}
	    }

	  //outwl.push_1item<BlockScan>(ncnt, (int) to_push, BLKSIZE);
	  /*heidars2*/
	    //printf("to_push:%d altdist:%d\n",to_push, altdist);
	    int threshold = base + iteration*delta;
	    if (altdist < threshold)
	    {
	    	to_push1 = to_push, to_push2 = -1;
	    }
	    else
	    {
	    	to_push1 = -1, to_push2 = to_push;
	    }

	    ncnt1 = to_push1 == -1 ? 0 : ncnt;
	    ncnt2 = to_push2 == -1 ? 0 : ncnt;

	  	outwl1.push_1item<BlockScan>(ncnt1, (int) to_push1, BLKSIZE);
	  	//printf("[INFO] Adding %d to outlist 1, ncnt is %d, nitems is %d\n", to_push, ncnt, *outwl1.dindex);
	  	outwl2.push_1item<BlockScan>(ncnt2, (int) to_push2, BLKSIZE);
	  	//printf("[INFO] Adding %d to outlist 2, ncnt is %d, nitems is %d\n", to_push, ncnt, *outwl2.dindex);
	  

	  	/*
	  	__shared__ BlockScan::TempStorage temp_storage_1;
    		__shared__ int queue_index_1;
    		int total_items_1 = 0;
   			int thread_data_1 = ncnt;
   			__shared__ int flag_1;
   			int old_flag_1;

   		__shared__ BlockScan::TempStorage temp_storage_2;
    		__shared__ int queue_index_2;
    		int total_items_2 = 0;
   			int thread_data_2 = ncnt;
   			printf("thread_data_2 is %d\n", thread_data_2);
   			__shared__ int flag_2;
   			int old_flag_2;



	  	if (altdist < threshold)
	  	{
	  		
   			BlockScan(temp_storage).ExclusiveSum(thread_data_1, thread_data_1, total_items_1);
	  		old_flag_1 = atomicExch(&flag_1, 1); 
	  		 if(old_flag_1 != 1)
      		 {	
	     		queue_index_1 = atomicAdd((int *) outwl1.dindex, total_items_1);
       			printf("atomically added to outwl1.dindex is now %d, thread_data_1 was %d\n", *outwl1.dindex, ncnt);

	     //printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
      		}
	     		
	  	} else
	  	{
	  		
   			BlockScan(temp_storage).ExclusiveSum(thread_data_2, thread_data_2, total_items_2);
	  		old_flag_2 = atomicExch(&flag_2, 1); 
	  		 if(old_flag_2 != 1)
      		 {	
	     		queue_index_2 = atomicAdd((int *) outwl2.dindex, total_items_2);
	     		printf("atomically added to outwl2.dindex is now %d, thread_data_2 was %d\n", *outwl2.dindex, thread_data_2);
	     //printf("queueindex: %d %d %d %d %d\n", blockIdx.x, threadIdx.x, queue_index, thread_data + n_items, total_items);
      		}

	  	}
    	__syncthreads();

    
    	if(ncnt == 1 && altdist < threshold)
      	{
	      	cub::ThreadStore<cub::STORE_CG>(outwl1.dwl + queue_index_1 + thread_data_1, (int) to_push);
	     	
      	}
      	if(ncnt == 1 && altdist >= threshold){
      		cub::ThreadStore<cub::STORE_CG>(outwl2.dwl + queue_index_2 + thread_data_2, (int) to_push);
      	}
      	*/

      
	  total_edges -= BLKSIZE;
	  done += BLKSIZE;
	}

      id += blockDim.x * gridDim.x;
    }

  return 0;
}


__device__
void drelax(foru *dist, Graph& graph, unsigned *gerrno, Worklist2 &inwl1, Worklist2 &inwl2, Worklist2 &outwl1, Worklist2 &outwl2, int iteration, long unsigned base, long unsigned delta) {
	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	if(iteration == 0)
	  {
	    if(id == 0)
	      {
	      	printf("[INFO] Initialization Succesfull\n");
			int item = 0;
			inwl1.push(item);
	      }
	    return;	    
	  }
	else
	  {
	    //if
	    (processnode2(dist, graph, inwl1, outwl1, outwl2, iteration, base, delta));
	    //  *gerrno = 1;
	  	//if
	  	(processnode2(dist, graph, inwl2, outwl1, outwl2, iteration, base, delta));
	  	//  *gerrno = 1;
	  }
}

__global__ void drelax3(foru *dist, Graph graph, unsigned *gerrno, Worklist2 inwl1, Worklist2 inwl2, Worklist2 outwl1, Worklist2 outwl2, int iteration, GlobalBarrier gb, long unsigned base, long unsigned delta)
{
  drelax(dist, graph, gerrno, inwl1, inwl2, outwl1, outwl2, iteration);
}


__global__ void print_array(int *a, int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    printf("%d %d\n", id, a[id]);
}

__global__ void print_texture(int n)
{
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < n)
    printf("%d %d\n", id, tex1Dfetch(columns, id));
}

__global__ void remove_dups(Worklist2 wl, int *node_owner, GlobalBarrier gb)
{
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  int nn;
  
  int total_inputs = (*wl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);
  
  while(total_inputs-- > 0)
    {      
      if(wl.pop_id(id, nn))
	{
	  node_owner[nn] = id;
	}

      id += gridDim.x * blockDim.x;
    }

  id = blockIdx.x * blockDim.x + threadIdx.x;
  total_inputs = (*wl.dindex + gridDim.x * blockDim.x - 1)/(gridDim.x * blockDim.x);

  gb.Sync();
  
  while(total_inputs-- > 0)
    { 
      if(wl.pop_id(id, nn))
	{
	  if(node_owner[nn] != id)
	  {
	  	wl.dwl[id] = -1;
	  	//printf("[WARNING: node %d had a duplicate\n", nn);
	  }
	    
	}

      id += gridDim.x * blockDim.x;    
    }
}

void sssp(foru *hdist, foru *dist, Graph &graph, unsigned long totalcomm)
{
	foru foruzero = 0.0;
	unsigned int NBLOCKS, FACTOR = 128;
	bool *changed;
	int iteration = 0;
	unsigned *nerr;

	double starttime, endtime;
	double runtime;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	NBLOCKS = deviceProp.multiProcessorCount;

	NVERTICES = graph.nnodes;

	FACTOR = (NVERTICES + MAXBLOCKSIZE * NBLOCKS - 1) / (MAXBLOCKSIZE * NBLOCKS);

	//printf("initializing (nblocks=%d, blocksize=%d).\n", NBLOCKS*FACTOR, MAXBLOCKSIZE);
	initialize <<<NBLOCKS*FACTOR, MAXBLOCKSIZE>>> (dist, graph.nnodes);
	CudaTest("initializing failed");
	cudaMemcpy(&dist[0], &foruzero, sizeof(foruzero), cudaMemcpyHostToDevice);

	if (cudaMalloc((void **)&changed, sizeof(bool)) != cudaSuccess) CudaTest("allocating changed failed");

	if (cudaMalloc((void **)&nerr, sizeof(unsigned)) != cudaSuccess) CudaTest("allocating nerr failed");

	Worklist2 wl1(graph.nedges * 2), wl2(graph.nedges * 2), wl3(graph.nedges * 2), wl4(graph.nedges * 2);
	Worklist2 *inwl1 = &wl1, *outwl1 = &wl2, *inwl2 = &wl3, *outwl2 = &wl4;
	int nitems = 1;

	struct workprogress hwp, *dwp;

	hwp.wl[0] = NULL;
	hwp.wl[1] = NULL;
	hwp.wl[2] = NULL;
	hwp.wl[3] = NULL;
 	hwp.in_wl = 0;
	hwp.iteration = 1;

	int *node_owners;
	CUDA_SAFE_CALL(cudaMalloc(&node_owners, graph.nnodes * sizeof(int)));

	CUDA_SAFE_CALL(cudaMalloc(&dwp, sizeof(hwp)));
	CUDA_SAFE_CALL(cudaMemcpy(dwp, &hwp, sizeof(*dwp), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemset(inwl1->dedges_touched, 0, 1 * sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset(inwl2->dedges_touched, 0, 1 * sizeof(unsigned int)));

	cudaBindTexture(0, columns, graph.edgessrcdst, (graph.nedges + 1) * sizeof(int));
	cudaBindTexture(0, row_offsets, graph.psrc, (graph.nnodes + 1) * sizeof(int));

	//print_array<<<1, graph.nedges + 1>>>((int *) graph.edgessrcdst, graph.nedges + 1);
	//print_texture<<<1, graph.nedges + 1>>>(graph.nedges + 1);
	//return;


	/* currently not used due to ensuing launch timeouts*/
	GlobalBarrierLifetime gb;
	gb.Setup(28);

	printf("solving.\n");
	printf("starting...\n");
	//printf("worklist size: %d\n", inwl->nitems());
	//printf("WL: 0 0, \n");

	starttime = rtclock();
	drelax3<<<1, BLKSIZE>>>(dist, graph, nerr, *inwl1, *inwl2, *outwl1, *outwl2, 0, gb, base, delta);
	unsigned int curr_edges_touched1 = 0;
	unsigned int curr_edges_touched2 = 0;
	unsigned long total_edges_touched = 0;
	do {
	        ++iteration;
		unsigned nblocks = (nitems + BLKSIZE - 1) / BLKSIZE; 
		//printf("%d %d %d %d\n", nblocks, BLKSIZE, iteration, nitems);
		//printf("ITERATION: %d\n", iteration);
		//inwl->display_items();
		//drelax2 <<<14, BLKSIZE>>> (dist, graph, nerr, *inwl, *outwl, dwp, gb);
		
		drelax3 <<<nblocks, BLKSIZE>>> (dist, graph, nerr, *inwl1, *inwl2, *outwl1, *outwl2, iteration, gb, base, delta);
		nitems = outwl1->nitems() + outwl2->nitems();

		remove_dups<<<14, 1024>>>(*outwl1, node_owners, gb);
		remove_dups<<<14, 1024>>>(*outwl2, node_owners, gb);
		
		//printf("%d\n", iteration);
		//outwl->display_items();

		//printf("worklist size: %d\n", nitems);
		
		/*Worklist2 *tmp = inwl;
		inwl = outwl;
		outwl = tmp;*/

		CUDA_SAFE_CALL(cudaMemcpy(&curr_edges_touched1, inwl1->dedges_touched, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&curr_edges_touched2, inwl2->dedges_touched, sizeof(unsigned int), cudaMemcpyDeviceToHost));

		total_edges_touched += curr_edges_touched1;
		total_edges_touched += curr_edges_touched2;


		CUDA_SAFE_CALL(cudaMemset(inwl1->dedges_touched, 0, 1 * sizeof(unsigned int)));
		CUDA_SAFE_CALL(cudaMemset(inwl2->dedges_touched, 0, 1 * sizeof(unsigned int)));


		Worklist2 *tmp1 = inwl1;
		inwl1 = outwl1;
		outwl1 = tmp1;

		Worklist2 *tmp2 = inwl2;
		inwl2 = outwl2;
		outwl2 = tmp2;


		outwl1->reset();
		outwl2->reset();
		//printf("nitems:%d\n",nitems);
	} while (nitems > 0);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	endtime = rtclock();

	CUDA_SAFE_CALL(cudaMemcpy(&hwp, dwp, sizeof(hwp), cudaMemcpyDeviceToHost));

	printf("millions of edges touched: %.2f\n", (float) total_edges_touched / 1000000);

	printf("\titerations = %d %d.\n", iteration, hwp.iteration);
	runtime = (1000.0f * (endtime - starttime));
	printf("\truntime [%s] = %f ms.\n", SSSP_VARIANT, runtime);

	return;
}
