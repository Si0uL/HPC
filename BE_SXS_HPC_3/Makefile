CC=nvcc
CFLAGS= -I./inc --ptxas-options=-v -arch=sm_30
LDFLAGS= 

% : %.cu
	$(CC)  $(CFLAGS) -o $@.out $< $(LDFLAGS)


