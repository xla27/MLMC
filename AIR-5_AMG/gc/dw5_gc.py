import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt
import os
from skimage.restoration import denoise_tv_chambolle
import numpy as np
from scipy.integrate import trapz

###########################################################################################
#  Utility functions

# Utilities
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 20 })

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
    SF = 0.015
    ws = max(int(len(p_i) * SF), 1)
    p_i = moving_average(p_i, ws)
    p_i = total_variation_denoising(p_i, weight=0.6)
    
    return xnodesf, p_i

def Lpnorm(x, f, p):
    return trapz(np.abs(f)**p, x)**(1/p)

###############################################################################################
# Wall pressure plot - Nominal conditions

f0 = 'surface_000.csv'
f1 = 'surface_001.csv'
f2 = 'surface_002.csv'
f3 = 'surface_003.csv'
f4 = 'surface_004.csv'

x0, p0 = process_csv(f0)
x1, p1 = process_csv(f1)
x2, p2 = process_csv(f2)
x3, p3 = process_csv(f3)
x4, p4 = process_csv(f4)



plt.figure()
ax = plt.gca()  # Get the current axis
ax.plot(x0, p0, linewidth=3, color='red',    linestyle='-', label='$L0$')
ax.plot(x1, p1, linewidth=3, color='blue',   linestyle='-', label='$L1$')
ax.plot(x2, p2, linewidth=3, color='green',  linestyle='-', label='$L2$')
ax.plot(x3, p3, linewidth=3, color='black',  linestyle='-', label='$L3$')
ax.plot(x4, p4, linewidth=3, color='orange', linestyle='-', label='$L4$')



ax.set_xlabel(r'$x, \; \mathrm{m}$')  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
ax.set_ylabel(r'$\frac{P}{P_\infty}$')  # Use set_xlabel and set_ylabel instead of plt.xlabel and plt.ylabel
#ax.tick_params(labelsize=30)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1,2,3,4]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='best', framealpha=1.0)
ax.grid(True)
ax.set_xlim(0.17, 0.38)
ax.set_ylim(-5, 200) 

save_name = 'dw5_nominal_pwall.svg'
plt.savefig(save_name, bbox_inches='tight')

###############################################################################################
# Convergence on ||pw/pinf||L1 on grid refinement

qoi = [Lpnorm(x0, p0, 1), Lpnorm(x1, p1, 1), Lpnorm(x2, p2, 1), Lpnorm(x3, p3, 1), Lpnorm(x4, p4, 1)]


# Data vectors
x = []
y = []
for i in range(4):
    x.append(i)
    y.append(np.abs(qoi[i] - qoi[-1])/qoi[-1])


# Set up the plot with custom figure size
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y, 'ro-', linewidth=3.5,  markersize=11)

# Add labels and title with custom font sizes
ax.set_xlabel('$\mathrm{Refinement} \, \mathrm{level}$')
ax.set_ylabel('$\mathrm{Error}, \; \%$')

# Customize the y-ticks to display in scientific notation
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#ax.get_xaxis().get_offset_text().set_fontsize(30)

# # Customize tick parameters
# y_ticks = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
# ax.set_yticks(y_ticks)
# ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks])  # Format labels to two decimal places
# ax.tick_params(labelsize=30)

# Set the axes limits
ax.set_xlim(-0.5, 3.5)  # Adjusted to match the data range
#ax.set_ylim(0.08, 0.3)  # Adjusted to match the data range

# Add grid
ax.grid(True)



# Set the aspect ratio to be equal
ax.set_aspect('auto')  # Changed from 'equal' to 'auto' for a better fit with the data range

# # Add the legend with custom font size
# ax.legend(loc='best')

# Save the plot
save_name = 'dw_GC.svg'
plt.savefig(save_name, bbox_inches='tight')

# Show the plot
#plt.show()
