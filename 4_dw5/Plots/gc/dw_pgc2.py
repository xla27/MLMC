import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage.restoration import denoise_tv_chambolle
import numpy as np

def total_variation_denoising(data, weight):
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, pd.Series):
        data = data.to_numpy()
    return denoise_tv_chambolle(data, weight=weight)

def moving_average(data, window_size):
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1
    padded_data = np.pad(data, (pad_left, pad_right), mode='edge')
    smoothed_data = np.convolve(padded_data, np.ones(window_size), 'valid') / window_size
    return smoothed_data

def process_csv(csv_filename):

    csv_path_surf = os.path.join(os.getcwd(), csv_filename)
    # Read CSV into a DataFrame
    data_surf = pd.read_csv(csv_path_surf, header=None)
    
    # Convert all columns to numeric, coerce errors to NaN, and drop rows with NaN values
    data_surf = data_surf.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Rename columns to 'Var1', 'Var2', ..., 'VarN'
    data_surf.columns = [f'Var{j}' for j in range(1, len(data_surf.columns) + 1)]
    
    # Sort DataFrame by 'Var2'
    data_surf = data_surf.sort_values(by='Var2')
    
    # Calculate p_i and get xnodesf
    p_i = (data_surf['Var18'] / 390).tolist()
    xnodesf = data_surf['Var2'].tolist()
    
    # Post-processing
    SF = 0.0075
    ws = max(int(len(p_i) * SF), 1);
    p_i = moving_average(p_i, ws);
    p_i = total_variation_denoising(p_i, weight=0.6);
    
    return xnodesf, p_i

f0 = 'surface_flow0.csv'
f1 = 'surface_flow1.csv'
f2 = 'surface_flow2.csv'
f3 = 'surface_flow3.csv'
f4 = 'surface_flow4.csv'

x0, p0 = process_csv(f0)
x1, p1 = process_csv(f1)
x2, p2 = process_csv(f2)
x3, p3 = process_csv(f3)
x4, p4 = process_csv(f4)

# Plotting
# Set custom font parameters
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 35}

plt.rc('font', **font)  # Apply custom font parameters

plt.figure(figsize=(13, 11))
ax = plt.gca()  # Get the current axis
ax.plot(x0, p0, linewidth=3, color='red', linestyle='solid', label='Level 0')
ax.plot(x1, p1, linewidth=3, color='blue', linestyle='dashdot', label='Level 1')
ax.plot(x2, p2, linewidth=3, color='green', linestyle='dashed', label='Level 2')
ax.plot(x3, p3, linewidth=3, color='black', linestyle='dotted', label='Level 3')
ax.plot(x4, p4, linewidth=3, color='orange', linestyle=':', label='Level 4')



ax.set_xlabel('X coordinate [m]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{P}{P_\infty}$ [-]', fontsize=35)  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2,3,4]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', fontsize=35)
ax.grid(True)
ax.set_xlim(0.17, 0.38)
ax.set_ylim(-5, 165) 

save_name = 'dwp_gc2.svg'
plt.savefig(save_name, bbox_inches='tight')