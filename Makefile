#
# Bridges - PSC
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#
CC = gcc
MPCC = mpicc
OPENMP = -fopenmp
CFLAGS = -O3 
CFLAGS = -Wall -std=c++11 $(OPT)
LDFLAGS = -Wall
# librt is needed for clock_gettime
#LDLIBS = -lrt 
#LDLIBS = -lrt -lblas -llapack
LDLIBS = -lrt  -L/opt/aci/sw/blas/3.6.0_gcc-5.3.1/usr/lib64/ -L/opt/aci/sw/lapack/3.6.0_gcc-5.3.1/usr/lib64/ -llapack -lblas -lgfortran


TARGETS = openmp 
#pthreads openmp mpi autograder

all:	$(TARGETS) -I /storage/home/yug125/work/WTMF-Parallel/armadillo-8.400.0/include -o -llapack -lblas

openmp: openmp.o 
	$(CC) -o $@ $(LIBS) $(OPENMP) openmp.o 

openmp.o: openmp.cpp 
	$(CC) -c $(OPENMP) $(CFLAGS) openmp.cpp

clean:
	rm -f *.o $(TARGETS) *.stdout *.txt
