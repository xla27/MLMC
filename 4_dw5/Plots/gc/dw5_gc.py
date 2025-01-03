import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Data vectors
x = [70132, 140324, 280424, 560132]
y = [0.2872433707, 0.2076475752, 0.1435286126, 0.09796356193]

# Set custom font parameters
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 35}

plt.rc('font', **font)  # Apply custom font parameters

# Set up the plot with custom figure size
fig, ax = plt.subplots(figsize=(13, 11))

# Plot the data
ax.plot(x, y, 'ro-', linewidth=3.5,  markersize=11, label=r'Relative error on $C_D$')

# Add labels and title with custom font sizes
ax.set_xlabel('Number of elements [-]', fontsize=35)
ax.set_ylabel('Relative error [%]', fontsize=30)

# Customize the x-ticks to display in scientific notation
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.get_xaxis().get_offset_text().set_fontsize(30)

# Customize tick parameters
y_ticks = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{tick:.2f}' for tick in y_ticks])  # Format labels to two decimal places
ax.tick_params(labelsize=30)

# Set the axes limits
ax.set_xlim(60000, 600000)  # Adjusted to match the data range
ax.set_ylim(0.08, 0.3)  # Adjusted to match the data range

# Add grid
ax.grid(True)



# Set the aspect ratio to be equal
ax.set_aspect('auto')  # Changed from 'equal' to 'auto' for a better fit with the data range

# Add the legend with custom font size
ax.legend(loc='best', fontsize=35)

# Save the plot
save_name = 'dw_GC.svg'
plt.savefig(save_name, bbox_inches='tight')

# Show the plot
plt.show()
