# WTMF-Reimplementation


This is a working git repo for WTMF re-implementation for ormf algorithm. 

## Input and Output 
Input will be "train.ind" file from preprocessing; output should be a .txt file with all the word vectors. 

## Requirement 
Armadillo, OpenMP, HPC environment with blas, lapack, fortran (e.g.,Penn State ACI-ICS)

## Steps (Running on ACI-ICS)
Go to Yanjun's directory: 
```
cd /storage/work/yug125/WTMF-Parallel
```

Submit the job under interactive node:

```
qsub -I <resource requirement>
```
Load gcc module for cpp compiler: 
```
module load gcc
```
Compile ormf file, with link to Armadillo, blas, lapack: 
```
g++ -std=c++11 ormf.cpp -O2 -I /storage/work/yug125/WTMF-Parallel/armadillo-8.400.0/include -o ormf1  -lrt -L/opt/aci/sw/blas/3.6.0_gcc-5.3.1/usr/lib64/ -L/opt/aci/sw/lapack/3.6.0_gcc-5.3.1/usr/lib64/ -llapack -lblas -lgfortran
```
Run the program:
```
./ormf1
```
