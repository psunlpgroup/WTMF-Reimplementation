#
# Bridges - PSC
#
# Intel Compilers are loaded by default; for other compilers please check the module list
#

CC = g++
OPENMP = -fopenmp
CFLAGS = -O3 -Wall -std=c++11 $(OPT)

LDLIBS = -I /storage/home/yug125/work/WTMF-Parallel/armadillo-8.400.0/include -L/opt/aci/sw/blas/3.6.0_gcc-5.3.1/usr/lib64/ -L/opt/aci/sw/lapack/3.6.0_gcc-5.3.1/usr/lib64/ -lblas -llapack
LOC = -larmadillo

TARGET = ormf
SOURCE = ormf.cpp

all:	
	$(CC) $(SOURCE) $(CFLAGS) -o $(TARGET) $(ACI) $(LDLIBS)

omp:
	$(CC) $(SOURCE) $(CFLAGS) $(OPENMP) -o $(TARGET) $(ACI) $(LDLIBS)

local:	
	$(CC) $(SOURCE) $(CFLAGS) -o $(TARGET) $(LOC)

lomp:
	$(CC) $(SOURCE) $(CFLAGS) $(OPENMP) -o $(TARGET) $(LOC)

clean:
	rm -f *.o $(TARGET) *.stdout *.txt
