# WTMF-Reimplementation
This is a working git repo for WTMF re-implementation for ormf algorithm. 

## Input and Output 
Input will be "train.ind" file from preprocessing; output should be a .txt file with all the word vectors. 

## Requirements
Armadillo, OpenMP, HPC environment with blas, lapack

## Rationale 
Please refer to train_ormf.m and ormf.m for the algorithm. 

## Running

### ACI-ICS

(Requires access to Yanjun's ACI files for Armadillo.)

Load GCC (7.3.1):
```
module load gcc
```

Use make:
```
make
```

Submit the job:
```
qsub scripts/wtmf-job.pbs
```

### Locally

(Requires libraries installed properly.)

Use make:
```
make local
```

Run the executable:
```
./ormf
```

## File Structure

### matlab/
Original Matlab implementation of ORMF.

### data/
Tiny input, Matlab output on tiny input.

### script/
Python helper scripts, PBS job script.
