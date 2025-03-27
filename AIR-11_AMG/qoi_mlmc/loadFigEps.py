import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np

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

# Set custom font parameters
# Utilities
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 20 })
shaded_color = [0.6, 0.8, 1]

eps_folder = 'eps04/'

#####################################################
# Plot 1: normalized pressure
file_path = eps_folder + 'P.csv'
loaded_grid, curves = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis
ax.fill_between(loaded_grid,  curves[0], curves[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'$\mu \pm \sigma$')# \left[ \frac{P}{P_\infty} \right]$')
ax.plot(loaded_grid, curves[2], linewidth=3, color='red', label=r'$\mu$') #\left[ \frac{P}{P_\infty} \right]$')

ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{P}{P_\infty}$', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel


ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0,3, 1, 2]
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.legend(loc='best', fontsize=35, framealpha=1.0)
ax.grid(True)
ax.set_xlim(-0.06, 0.5)
ax.set_ylim(-25, 200) 

save_name = eps_folder + 'dw11_P.svg'
plt.savefig(save_name, bbox_inches='tight')

#####################################################
# Plot 2: N2, O2
file_path = eps_folder + 'N2.csv'
loaded_gridN2, curvesN2 = load_csv(file_path)

file_path = eps_folder + 'O2.csv'
loaded_gridO2, curvesO2 = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()
ax.fill_between(loaded_gridN2, curvesN2[0], curvesN2[1], color='green', alpha=0.2, edgecolor='none', label=r'$N_2: \mu \pm \sigma$')
ax.plot(loaded_gridN2, curvesN2[2], linewidth=3, color='green', label=r'$N_2: \mu $')

ax.fill_between(loaded_gridO2, curvesO2[0], curvesO2[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'$O_2: \mu \pm \sigma$')
ax.plot(loaded_gridO2, curvesO2[2], linewidth=3, color='blue', label=r'$O_2: \mu$')

ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
ax.set_ylabel(r'$Y$', fontsize=35)
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0,1, 3, 4, 2]
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', bbox_to_anchor=(0.7, 0.85), fontsize=35)
ax.legend(loc='best', fontsize=35, framealpha=1.0)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
# Set y limits as the union of all individual y-limits
ax.set_ylim(-0.025, 0.82)

save_name = eps_folder +'dw11_N2_O2.svg'
plt.savefig(save_name, bbox_inches='tight')

#####################################################
# Plot 3: NO, O, N
file_path = eps_folder + 'N.csv'
loaded_gridN, curvesN = load_csv(file_path)

file_path = eps_folder + 'O.csv'
loaded_gridO, curvesO = load_csv(file_path)

file_path = eps_folder + 'NO.csv'
loaded_gridNO, curvesNO = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridN, curvesN[0], curvesN[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'$N: \mu \pm \sigma$')
ax.plot(loaded_gridN, curvesN[2], linewidth=3, color='blue', label=r'$N: \mu$')

ax.fill_between(loaded_gridO, curvesO[0], curvesO[1], color='green', alpha=0.2, edgecolor='none', label=r'$O: \mu \pm \sigma$')
ax.plot(loaded_gridO, curvesO[2], linewidth=3, color='green', label=r'$O: \mu $')

ax.fill_between(loaded_gridNO, curvesNO[0], curvesNO[1], color='orange', alpha=0.2, edgecolor='none', label=r'$NO: \mu \pm \sigma$')
ax.plot(loaded_gridNO, curvesNO[2], linewidth=3, color='orange', label=r'$NO: \mu$')

ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
ax.set_ylabel(r'$Y$', fontsize=35)
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 1, 2, 4, 5, 6, 3]
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=28)
ax.legend(loc='best', fontsize=35, framealpha=1.0)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
# Set y limits as the union of all individual y-limits
ax.set_ylim(-0.025, 0.2)

# Save the N, O, and NO figure
save_name = eps_folder + 'dw11_N_O_NO.svg'
plt.savefig(save_name, bbox_inches='tight')

#####################################################
# Plot 4: Number Densities
file_path = eps_folder + 'nd_N+.csv'
loaded_gridNPlus, curvesNPlus = load_csv(file_path)

file_path = eps_folder + 'nd_O+.csv'
loaded_gridOPlus, curvesOPlus = load_csv(file_path)

file_path = eps_folder + 'nd_NO+.csv'
loaded_gridNOPlus, curvesNOPlus = load_csv(file_path)

file_path = eps_folder + 'nd_N2+.csv'
loaded_gridN2Plus, curvesN2Plus = load_csv(file_path)

file_path = eps_folder + 'nd_O2+.csv'
loaded_gridO2Plus, curvesO2Plus = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridNPlus, curvesNPlus[0], curvesNPlus[1], color='blue', alpha=0.4, edgecolor='none', label=r'$N^+: \mu \pm \sigma$')
ax.plot(loaded_gridNPlus, curvesNPlus[2], linewidth=3, color='blue', label=r'$N^+: \mu$')

ax.fill_between(loaded_gridOPlus, curvesOPlus[0], curvesOPlus[1], color='green', alpha=0.4, edgecolor='none', label=r'$O^+: \mu \pm \sigma$')
ax.plot(loaded_gridOPlus, curvesOPlus[2], linewidth=3, color='green', label=r'$O^+: \mu$')

ax.fill_between(loaded_gridNOPlus, curvesNOPlus[0], curvesNOPlus[1], color='red', alpha=0.4, edgecolor='none', label=r'$NO^+: \mu \pm \sigma$')
ax.plot(loaded_gridNOPlus, curvesNOPlus[2], linewidth=3, color='red', label=r'$NO^+: \mu$')

ax.fill_between(loaded_gridN2Plus, curvesN2Plus[0], curvesN2Plus[1], color='orange', alpha=0.4, edgecolor='none', label=r'$N_2^+: \mu \pm \sigma$')
ax.plot(loaded_gridN2Plus, curvesN2Plus[2], linewidth=3, color='orange', label=r'$N_2^+: \mu$')

ax.fill_between(loaded_gridO2Plus, curvesO2Plus[0], curvesO2Plus[1], color='purple', alpha=0.4, edgecolor='none', label=r'$O_2^+: \mu \pm \sigma$')
ax.plot(loaded_gridO2Plus, curvesO2Plus[2], linewidth=3, color='purple', label=r'$O_2^+: \mu$')

ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)
ax.set_ylabel(r'$n_D, \; \mathrm{m}^{-3}$', fontsize=35)
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 1, 2, 4, 5, 6, 3]
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=28)
ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=35, framealpha=1.0)
ax.grid(True)
ax.set_xlim(0.18, 0.48)
# Set y limits as the union of all individual y-limits
ax.set_yscale('log')
ax.set_ylim(1e8, 1e20)

# Save the N, O, and NO figure
save_name = eps_folder + 'dw11_nds.svg'
plt.savefig(save_name, bbox_inches='tight')

#####################################################
# Plot 7: Ttr
file_path = eps_folder + 'Ttr.csv'
loaded_gridTtr, curvesTtr = load_csv(file_path)

# file_path = 'diffTtr.csv'
# x_newTtr, diffcurvesTtr = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridTtr, curvesTtr[0], curvesTtr[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'$\mu \pm \sigma$')
ax.plot(loaded_gridTtr, curvesTtr[2], linewidth=3, color='red', label=r'$\mu$')

ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{T_{tr}}{T_{tr,\infty}}$', fontsize=35) # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 2, 1]
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.legend(loc='best', fontsize=35, framealpha=1.0)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
ax.set_ylim(-0.5, 8.5)

save_name = eps_folder + 'dw11_Ttr.svg'
plt.savefig(save_name, bbox_inches='tight')

#####################################################
# Plot 8: Tve
file_path = eps_folder + 'Tve.csv'
loaded_gridTve, curvesTve= load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridTve, curvesTve[0], curvesTve[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'$\mu \pm \sigma$')
ax.plot(loaded_gridTve, curvesTve[2], linewidth=3, color='red', label=r'$\mu$')

ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{T_{ve}}{T_{ve,\infty}}$', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 2, 1]
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.legend(loc='best', fontsize=35, framealpha=1.0)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
ax.set_ylim(-1.725, 6.2)

save_name = eps_folder + 'dw11_Tve.svg'
plt.savefig(save_name, bbox_inches='tight')

#####################################################
# Plot 9: M
file_path = eps_folder + 'M.csv'
loaded_gridM,curvesM= load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridM, curvesM[0], curvesM[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'$\mu \pm \sigma$')
ax.plot(loaded_gridM, curvesM[2], linewidth=3, color='red', label=r'$\mu$')

ax.set_xlabel(r'$x, \; \mathrm{m} $', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{M}{M_\infty}$', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
# order = [0, 2, 1]
# ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.legend(loc='best', fontsize=35, framealpha=1.0)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
ax.set_ylim(-0.125, 1.05)

save_name = eps_folder + 'dw11_M.svg'
plt.savefig(save_name, bbox_inches='tight')
