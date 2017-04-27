FROM rai/cuda_ppc64le:devel-8.0
MAINTAINER Hadi Asghari-Moghaddam <asghari2@illinois.edu>

# Install dependencies
RUN \ 
    apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        build-essential \
        cmake \
        libboost-all-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

