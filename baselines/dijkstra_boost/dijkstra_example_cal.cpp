//=======================================================================
// Copyright 2001 Jeremy G. Siek, Andrew Lumsdaine, Lie-Quan Lee, 
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//=======================================================================
#include <boost/config.hpp>
#include <iostream>
#include <fstream>

#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>

using namespace boost;

int
main(int argc, char ** argv)
{
  char * input = argv[1];
  std::ifstream infile(input);

  int num_nodes, num_arcs;
  int node1,node2, dist;

  struct timeval tstart_init, tend_init, tstart_sssp, tend_sssp;
  unsigned long long time[2];

  typedef adjacency_list < listS, vecS, directedS,
    no_property, property < edge_weight_t, int > > graph_t;
  typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
  typedef graph_traits < graph_t >::edge_descriptor edge_descriptor;
  typedef std::pair<int, int> Edge;

  infile >> num_nodes >> num_arcs;
  std::printf("num nodes: %d, num arcs: %d\n",num_nodes, num_arcs);

  gettimeofday(&tstart_init,NULL);

  Edge *edge_array = new Edge [num_arcs];
  int *weights = new int [num_arcs];
  int *name = new int[num_nodes];
  graph_t g(num_nodes);

  property_map<graph_t, edge_weight_t>::type weightmap = get(edge_weight, g);

  for (std::size_t j = 0; j < num_arcs; ++j) {
    edge_descriptor e; bool inserted;
    infile >> node1 >> node2 >> dist;
    weights[j] = dist;
    tie(e, inserted) = add_edge(node1, node2, g);
    weightmap[e] = weights[j];
    name[node1] = node1; name[node2] = node2; 
  }
  std::vector<vertex_descriptor> p(num_vertices(g));
  std::vector<int> d(num_vertices(g));
  vertex_descriptor s = vertex('1', g);

  gettimeofday(&tend_init,NULL);
  gettimeofday(&tstart_sssp,NULL);

  property_map<graph_t, vertex_index_t>::type indexmap = get(vertex_index, g);
  dijkstra_shortest_paths(g, s, &p[0], &d[0], weightmap, indexmap, 
                          std::less<int>(), closed_plus<int>(), 
                          (std::numeric_limits<int>::max)(), 0,
                          default_dijkstra_visitor());

  gettimeofday(&tend_sssp,NULL);

  time[0] =  (tend_init.tv_sec - tstart_init.tv_sec)*1000000 + (tend_init.tv_usec - tstart_init.tv_usec);
  time[1] =  (tend_sssp.tv_sec - tstart_sssp.tv_sec)*1000000 + (tend_sssp.tv_usec - tstart_sssp.tv_usec);

  std::printf("Initialization takes: %llu usecs\nComputation takes: %llu usecs\n", time[0], time[1]);
/*
  std::cout << "distances and parents:" << std::endl;
  graph_traits < graph_t >::vertex_iterator vi, vend;
  for (tie(vi, vend) = vertices(g); vi != vend; ++vi) {
    std::cout << "distance(" << name[*vi] << ") = " << d[*vi] << ", ";
    std::cout << "parent(" << name[*vi] << ") = " << name[p[*vi]] << std::
      endl;
  }
  std::cout << std::endl;
*/
/*
  std::ofstream dot_file("figs/dijkstra-eg.dot");

  dot_file << "digraph D {\n"
    << "  rankdir=LR\n"
    << "  size=\"4,3\"\n"
    << "  ratio=\"fill\"\n"
    << "  edge[style=\"bold\"]\n" << "  node[shape=\"circle\"]\n";

  graph_traits < graph_t >::edge_iterator ei, ei_end;
  for (tie(ei, ei_end) = edges(g); ei != ei_end; ++ei) {
    graph_traits < graph_t >::edge_descriptor e = *ei;
    graph_traits < graph_t >::vertex_descriptor
      u = source(e, g), v = target(e, g);
    dot_file << name[u] << " -> " << name[v]
      << "[label=\"" << get(weightmap, e) << "\"";
    if (p[v] == u)
      dot_file << ", color=\"black\"";
    else
      dot_file << ", color=\"grey\"";
    dot_file << "]";
  }
  dot_file << "}";

*/
  return EXIT_SUCCESS;
}
