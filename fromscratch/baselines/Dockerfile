FROM rai/cuda_ppc64le:devel-8.0
MAINTAINER Azin Heidarshenas <heidars2@illinois.edu>

# Install dependencies
RUN \ 
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libboost-all-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


#RUN cd /home && wget http://iss.ices.utexas.edu/projects/galois/downloads/lonestargpu-2.0.tar.bz2 && tar -vxjf lonestargpu-2.0.tar.bz2 
#RUN cd /home && git clone https://github.com/NVlabs/cub.git && mv cub cub-1.1.1
#ENV LSGDIR /home/lonestargpu-2.0
#ENV CUBDIR /home/cub-1.1.1
#RUN cd /home/lonestargpu-2.0 && ln -s $CUBDIR && make inputs  && make
