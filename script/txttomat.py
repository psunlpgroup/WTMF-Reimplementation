from scipy.io import savemat
import numpy as n
import sys

infile = sys.argv[1]

p_in = []

with open(infile, 'r') as f:
    f.readline()
    for l in f:
        p_in.append(map(float, l.split()))

matout = {
        'P': n.array(p_in).T,
        'dim': 0,
        'lambda': 0,
        'w_m': 0,
        'alpha': 0,
        'n_iters': 0
        }

savemat(infile, matout)
