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
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 35}

plt.rc('font', **font)  # Apply custom font parameters
shaded_color = [0.6, 0.8, 1]

epsstring = '3_371'

data = pd.read_table('normalized_pressure.dat', header=None, delimiter=',')
data.columns = ['Var1', 'Var2']
var1 = data['Var1'].to_numpy()
var2 = data['Var2'].to_numpy()

# Plot 1: normalized pressure
file_path = 'P.csv'
loaded_grid, curves = load_csv(file_path)

file_path = 'diffP.csv'
x_newP, diffcurves = load_csv(file_path)


plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis
ax.fill_between(loaded_grid,  curves[0], curves[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'1$\sigma[\frac{P}{P_\infty}]$')
ax.plot(loaded_grid, curves[2], linewidth=3, color='red', label=r'$\mu[\frac{P}{P_\infty}]$')
ax.plot(var1, var2, 'd', label='CFD [12]',markersize=10, linestyle='None', markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.5)

ax.set_xlabel('X coordinate [m]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{P}{P_\infty}$ [-]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel

ax.plot(x_newP, diffcurves[2], linewidth=3, label='Error to [21]', color='gray')
ax.fill_between(x_newP, diffcurves[0], diffcurves[1], color='gray', alpha=0.2)

ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,3, 1, 2]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.grid(True)
ax.set_xlim(-0.06, 0.5)
ax.set_ylim(-25, 200) 

save_name = 'dw5_fig1_p.svg'
plt.savefig(save_name, bbox_inches='tight')

# Plot 2: N2, O2
file_path = 'N2.csv'
loaded_gridN2, curvesN2 = load_csv(file_path)

file_path = 'O2.csv'
loaded_gridO2, curvesO2 = load_csv(file_path)

file_path = 'diffO2.csv'
x_newO2, diffcurves = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()
ax.fill_between(loaded_gridN2, curvesN2[0], curvesN2[1], color='green', alpha=0.2, edgecolor='none', label=f'1$\sigma$[$YN2$]')
ax.plot(loaded_gridN2, curvesN2[2], linewidth=3, color='green', label=f'$\mu$[$YN2$]')

ax.fill_between(loaded_gridO2, curvesO2[0], curvesO2[1], color=shaded_color, alpha=0.7, edgecolor='none', label=f'1$\sigma$[$YO2$]')
ax.plot(loaded_gridO2, curvesO2[2], linewidth=3, color='blue', label=f'$\mu$[$YO2$]')

ax.plot(x_newO2, diffcurves[2], linewidth=3, label=r'Error to [21] on $YO2$', color='gray')
ax.fill_between(x_newO2, diffcurves[0], diffcurves[1], color='gray', alpha=0.2)

ax.set_xlabel('X coordinate [m]', fontsize=35)
ax.set_ylabel('Mass fraction Y [-]', fontsize=35)
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1, 3, 4, 2]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', bbox_to_anchor=(0.7, 0.85), fontsize=35)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
# Set y limits as the union of all individual y-limits
ax.set_ylim(-0.025, 0.82)

save_name = 'dw5_fig2_N2_O2.svg'
plt.savefig(save_name, bbox_inches='tight')

# Plot 3: NO, O, N
file_path = 'N.csv'
loaded_gridN, curvesN = load_csv(file_path)

file_path = 'O.csv'
loaded_gridO, curvesO = load_csv(file_path)

file_path = 'NO.csv'
loaded_gridNO, curvesNO = load_csv(file_path)

file_path = 'diffNO.csv'
x_newNO, diffcurvesNO = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridN, curvesN[0], curvesN[1], color=shaded_color, alpha=0.7, edgecolor='none', label=f'1$\sigma$[$YN$]')
ax.plot(loaded_gridN, curvesN[2], linewidth=3, color='blue', label=f'$\mu$[$YN$]')

ax.fill_between(loaded_gridO, curvesO[0], curvesO[1], color='green', alpha=0.2, edgecolor='none', label=f'1$\sigma$[$YO$]')
ax.plot(loaded_gridO, curvesO[2], linewidth=3, color='green', label=f'$\mu$[$YO$]')


ax.fill_between(loaded_gridNO, curvesNO[0], curvesNO[1], color='orange', alpha=0.2, edgecolor='none', label=f'1$\sigma$[$YNO$]')
ax.plot(loaded_gridNO, curvesNO[2], linewidth=3, color='orange', label=f'$\mu$[$YNO$]')


ax.plot(x_newNO, diffcurvesNO[2], linewidth=3, label=r'Error to [21] on $YNO$', color='gray')
ax.fill_between(x_newNO, diffcurvesNO[0], diffcurvesNO[1], color='gray', alpha=0.2)


ax.set_xlabel('X coordinate [m]', fontsize=35)
ax.set_ylabel('Mass fraction Y [-]', fontsize=35)
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 1, 2, 4, 5, 6, 3]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=28)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
# Set y limits as the union of all individual y-limits
ax.set_ylim(-0.025, 0.2)

# Save the N, O, and NO figure
save_name = 'dw5_fig3_N_O_NO.svg'
plt.savefig(save_name, bbox_inches='tight')

# Plot 7: Ttr
file_path = 'Ttr.csv'
loaded_gridTtr, curvesTtr = load_csv(file_path)

file_path = 'diffTtr.csv'
x_newTtr, diffcurvesTtr = load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridTtr, curvesTtr[0], curvesTtr[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'1$\sigma[\frac{T_{tr}}{T_{tr_\infty}}]$')
ax.plot(loaded_gridTtr, curvesTtr[2], linewidth=3, color='red', label=r'$\mu[\frac{T_{tr}}{T_{tr_\infty}}]$')

ax.plot(x_newTtr, diffcurvesTtr[2], linewidth=3, label=r'Error to [21]', color='gray')
ax.fill_between(x_newTtr, diffcurvesTtr[0], diffcurvesTtr[1], color='gray', alpha=0.2)


ax.set_xlabel('X coordinate [m]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{T_{tr}}{T_{tr_\infty}}$ [-]', fontsize=35) # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 2, 1]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
ax.set_ylim(-0.5, 8.5)

save_name = 'dw5_fig7_Ttr.svg'
plt.savefig(save_name, bbox_inches='tight')

# Plot 8: Tve
file_path = 'Tve.csv'
loaded_gridTve, curvesTve= load_csv(file_path)

file_path = 'diffTve.csv'
x_newTve, diffcurvesTve= load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridTve, curvesTve[0], curvesTve[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'1$\sigma[\frac{T_{ve}}{T_{ve_\infty}}]$')

ax.plot(loaded_gridTve, curvesTve[2], linewidth=3, color='red', label=r'$\mu[\frac{T_{ve}}{T_{ve_\infty}}]$')


ax.plot(x_newTve, diffcurvesTve[2], linewidth=3, label=r'Error to [21]', color='gray')
ax.fill_between(x_newTve, diffcurvesTve[0], diffcurvesTve[1], color='gray', alpha=0.2)


ax.set_xlabel('X coordinate [m]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{T_{ve}}{T_{{ve}_\infty}}$ [-]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 2, 1]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
ax.set_ylim(-1.725, 6.2)

save_name = 'dw5_fig8_Tve.svg'
plt.savefig(save_name, bbox_inches='tight')

# Plot 9: M
file_path = 'M.csv'
loaded_gridM,curvesM= load_csv(file_path)

file_path = 'diffM.csv'
x_newM, diffcurvesM= load_csv(file_path)

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis

ax.fill_between(loaded_gridM, curvesM[0], curvesM[1], color=shaded_color, alpha=0.7, edgecolor='none', label=r'1$\sigma[\frac{M}{M_\infty}]$')
ax.plot(loaded_gridM, curvesM[2], linewidth=3, color='red', label=r'$\mu[\frac{M}{M_\infty}]$')

ax.plot(x_newM, diffcurvesM[2], linewidth=3, label=r'Error to [21]', color='gray')
ax.fill_between(x_newM, diffcurvesM[0], diffcurvesM[1], color='gray', alpha=0.2)


ax.set_xlabel('X coordinate [m]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{M}{M_\infty}$ [-]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 2, 1]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.grid(True)
ax.set_xlim(-0.06, 0.48)
ax.set_ylim(-0.125, 1.05)

save_name = 'dw5_fig9_M.svg'
plt.savefig(save_name, bbox_inches='tight')
