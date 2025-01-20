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


font = {'family': 'Arial',
        'weight': 'normal',
        'size': 25}
plt.rc('font', **font)

selected_colors = ['#1f77b4',   # blue
                   '#ff7f0e',   # orange
                   '#2ca02c',   # green
                   '#d62728']   # red

# Plot 1: N
file_path = 'N.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1.285, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0.193, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_1_Nfirst_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 2: O
file_path = 'O.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0.193, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_2_Ofirst_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 3: NO
file_path = 'NO.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1.285, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0.193, 0.48)  
plt.ylim(-0.02, 1.02)

stackplot_filename = 'dw5_3_NOfirst_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 4: N2
file_path = 'N2.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_4_N2first_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 5: O2
file_path = 'O2.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1.285, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_5_O2first_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 6: P
file_path = 'P.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1.285, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_6_Pfirst_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 7: Ttr
file_path = 'Ttr.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1.285, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_7_Ttrfirst_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 8: Tve
file_path = 'Tve.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1.285, 1), fontsize=35)

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

stackplot_filename = 'dw5_8_Tvefirst_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()

# Plot 9: M
file_path = 'M.csv'
x, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
plt.stackplot(x, curves[0], curves[1], curves[2], curves[3], labels=[r'$M$', r'$P$', r'$T$', r'$Y_{N_2}$'], colors=selected_colors)
plt.xlabel('X coordinate [m]', fontsize=35)
plt.ylabel('First order Sobol index [-]', fontsize=35)
plt.legend(loc='upper right', bbox_to_anchor=(1.285, 1), fontsize=35)

plt.xlim(0, 0.48)  
plt.ylim(-0.02, 1.02)  

plt.grid(True, which='both')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=30)

stackplot_filename = 'dw5_9_Mfirst_order_sobol_indices.svg'
plt.savefig(stackplot_filename, format='svg', bbox_inches='tight')
plt.close()
