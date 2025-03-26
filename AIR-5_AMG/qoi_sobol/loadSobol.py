import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_csv(file_path):
    """
    Load data from a CSV file.
    Assumes the first column contains x-values and the others are y-values for different curves.
    
    :param file_path: Path to the CSV file
    :return: x_values (list), curves (list of lists)
    """
    x_values = []
    curves = []

    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header if present
        
        for row in csvreader:
            x_values.append(float(row[0]))
            for i, value in enumerate(row[1:]):
                if len(curves) <= i:
                    curves.append([])  # Initialize new column lists
                curves[i].append(float(value))
    
    return x_values, curves


plt.rcParams.update({
    "text.usetex": True,
    "font.size": 20 })

selected_colors = ['#1f77b4',   # blue
                   '#ff7f0e',   # orange
                   '#2ca02c',   # green
                   '#d62728']   # red

# smoothing factor
SF = 0.0075

# Plot 1: N
file_path = 'N.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='upper right', framealpha=1.0, fontsize=35) #bbox_to_anchor=(1.285, 1)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0.193, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_N_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 2: O
file_path = 'O.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='lower right', framealpha=1.0, fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0.193, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_O_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 3: NO
file_path = 'NO.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='upper right', framealpha=1.0, fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0.193, 0.48)  
plt.ylim(-0.02, 1.02)

stackplot_filename = 'dw5_NO_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 4: N2
file_path = 'N2.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='upper left', framealpha=1.0, fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_N2_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 5: O2
file_path = 'O2.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='upper left', framealpha=1.0, fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_O2_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 6: P
file_path = 'P.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.8), framealpha=1.0, fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_P_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 7: Ttr
file_path = 'Ttr.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='upper right', framealpha=1.0, fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_Ttr_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 8: Tve
file_path = 'Tve.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.8), framealpha=1.0, fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_Tve_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 9: M
file_path = 'M.csv'
x, curves = load_csv(file_path)

# Smoothing
ws = max(int(len(x) * SF), 1)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
plt.ylabel(r'$S_1$', fontsize=35)
plt.legend(loc='lower left', framealpha=1.0, fontsize=35)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

stackplot_filename = 'dw5_M_sobol.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()
