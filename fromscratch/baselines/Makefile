TARGET := SSSP
NVCC := nvcc
GCC := g++
CC := $(GCC)

COMPUTECAPABILITY := sm_20
FLAGS := -O3 -arch=$(COMPUTECAPABILITY) -g -Xptxas -v 

SRC := main.cu sssp.cu
$(TARGET) : $(SRC) 
	    $(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
