import os, sys, shutil, pickle
import matplotlib.pyplot as plt
import numpy as np
import csv


# REMEMBER TO USE PYTHON 3.7 DUE TO INCOMPATBILITY WITH MATPLOTLIB AXES UNPACKING

suffix = '_epsilon_0_7088000000000001.pkl'
qoi = ['P', 'N', 'O', 'NO', 'N2', 'O2', 'Ttr', 'Tve', 'M']

folder = 'Plots/'

for i in range(1, len(qoi)+1):

    plot_filename = folder + str(i) + '_' + qoi[i-1] + suffix

    with open(plot_filename, 'rb') as f:
        _, x, ub,lb, m = pickle.load(f)

    fieldnames = ['x','UB','LB','M']

    with open(folder+qoi[i-1]+'.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for j in range(len(x)):
            writer.writerow({'x': x[j], 'UB': ub[j], 'LB': lb[j], 'M': m[j]})



