import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import csv
from cycler import cycler
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

class HandlerLine2D(HandlerTuple):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        l1, l2 = orig_handle
        y = height / 2 - ydescent

        # Adjust marker size here
        l1_marker_size = 10  # adjust this value as needed
        l2_marker_size = 10  # adjust this value as needed

        l1 = Line2D([xdescent, xdescent + width / 2], [y, y], color=l1.get_color(), 
                    linestyle=l1.get_linestyle(), marker=l1.get_marker(), lw=l1.get_linewidth(), markersize=l1_marker_size)
        l2 = Line2D([xdescent + width / 2, xdescent + width], [y, y], color=l2.get_color(), 
                    linestyle=l2.get_linestyle(), marker=l2.get_marker(), lw=l2.get_linewidth(), markersize=l2_marker_size)

        l1.set_transform(trans)
        l2.set_transform(trans)

        return [l1, l2]

l = [0, 1, 2, 3, 4]
eps = [0.04, 0.05]  # Define epss values
mlmc_cost = [8.771e+06, 4.117e+06]  # Define mlmc_cost values
std_cost = [2.806e+07, 1.696e+07]  # Define std_cost values
Ns = [[208, 24, 9, 4, 2], [102, 11, 4, 2, 1]]  # Number of samples values
ls = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]  # List of lists for ls

del1 = [154.9673151, 34.901789, 9.9885191, 4.8179677, 4.360772]
del2 = [154.9673151, 189.8691041, 179.880585, 184.6985527, 189.059324]
var1 = [1487.2, 145.2, 29.37, 12.82, 5.58]
var2 = [1487.2, 1889.5, 1805.1, 1846.3, 1671.8]

nproc = 4
cost0 = nproc * (0.14 * 45000) # nproc * tempo singola iter * numero iter
cost1 = nproc * (0.195 * 60000) 
cost2 = nproc * (0.275 * 115000)
cost3 = nproc * (0.33 * 199000)
cost4 = nproc * (0.375 * 320000)
cost = [cost0, cost1, cost2, cost3, cost4]


epss_array = np.array(eps) # Convert 'epss' to a numpy array
del22 = np.array(del2)
denominator = abs(del22[-1])
epsss = np.array(eps)
mlmc_costs = np.array(mlmc_cost)

# Set custom font parameters
font = {'family': 'Arial',
        'weight': 'normal',
        'size': 35}

plt.rc('font', **font)  # Apply custom font parameters

plt.rc("legend", framealpha=None)
plt.rc("legend", edgecolor='black')
plt.rc("font", family="serif")

# Define figure size
figsize = (13, 11)

# Alto: log2 media e varianza
fig, ax1 = plt.subplots(figsize=figsize)

line1, = ax1.plot(l, np.log2(del2), label='mean $Q_l$', clip_on=False, markersize=10, marker='o', linewidth=3.5, linestyle='-', color='blue')
line2, = ax1.plot(l[1:], np.log2(del1[1:]), label='mean $Q_l - Q_{l-1}$', clip_on=False, markersize=10, marker='x', linewidth=3.5, linestyle='--', color='blue')
ax1.set_xlabel('Level $\ell$', fontsize=35)
ax1.set_ylabel(r'$\log_2||\mathrm{mean}||_\infty$', fontsize=35, color='blue')
ax1.tick_params(axis='y', labelcolor='blue', labelsize=30)
ax1.tick_params(axis='x', labelsize=30)
ax1.grid(True)
ax1.set_xticks(range(max(l) + 1))
ax1.set_xlim([0, max(l)])

ax2 = ax1.twinx()
line3, = ax2.plot(l, np.log2(var2), label='variance $Q_l$', clip_on=False, markersize=10, marker='o', linewidth=3.5, linestyle='-', color='red')
line4, = ax2.plot(l[1:], np.log2(var1[1:]), label='variance $Q_l - Q_{l-1}$', clip_on=False, markersize=10, marker='x', linewidth=3.5, linestyle='--', color='red')
ax2.set_ylabel(r'$\log_2 ||\mathrm{variance}||_\infty$', fontsize=35, color='red')
ax2.tick_params(axis='y', labelcolor='red', labelsize=30)
ax2.set_xlim([0, max(l)])

legend_elements = [
    (Line2D([0], [0], color='blue', lw=3.5, linestyle='-', marker='o'), 
     Line2D([0], [0], color='red', lw=3.5, linestyle='-', marker='o')),
    (Line2D([0], [0], color='blue', lw=3.5, linestyle='--', marker='x'), 
     Line2D([0], [0], color='red', lw=3.5, linestyle='--', marker='x'))
]


ax1.legend(legend_elements, [r'$\vec{Q}_\ell$', r'$\vec{Q}_\ell - \vec{Q}_{\ell-1}$'],
           handler_map={tuple: HandlerLine2D()}, loc='best', fontsize=35)

plt.savefig('dw11_figsix6_12_combined_amg.svg')
plt.close()


# Centro sx: Plot cost per level
plt.figure(figsize=figsize)
plt.plot(l, np.log2(cost), 'o-', markersize=10, clip_on=False, linewidth=3.5, color='blue')
plt.xlabel('Level $\ell$', fontsize=35)
plt.ylabel(r'$\log_2$ cost per sample', fontsize=35)
plt.grid(True)
plt.axis([0,max(l), min(np.log2(cost)), max(np.log2(cost))])
plt.xticks(range(max(l) + 1), fontsize=30)
plt.yticks(np.arange(15, 19, 1), fontsize=30)
plt.tick_params(labelsize=30)
plt.savefig('dw11_figsix6_3_amg.svg')
plt.close()


# basso sx: Plot number of samples per level
plt.figure(figsize=figsize)
colors = ['blue', 'red', 'green','purple']
custom_cycler = (cycler('color', ['blue', 'red', 'green','purple']) *      # Colors for each plot 
                 cycler('marker', ['o', 'x', 'd','*']) *              # Markers for each plot
                 cycler('markersize', [10, 10, 10, 10]) *             # Marker sizes for each plot
                 cycler('linewidth', [3.5, 3.5, 3.5, 3.5]))            # Linewidths for each plot

# Set the rcParams with the custom cycler
plt.rcParams['axes.prop_cycle'] = custom_cycler
for eps_c, ll, n, color in zip(epsss, ls, Ns, colors):
    plt.semilogy(ll, n, label='$\\varepsilon_r$ = {:.2f}'.format(eps_c), markerfacecolor='none', clip_on=False, linewidth=3.5, markersize=10, color=color)

plt.xlabel('Level $\ell$', fontsize=35)
plt.ylabel('$N_l$', fontsize=35)
plt.legend(loc='best', fontsize=35)
plt.grid(True, which="both")
axis = plt.axis()
plt.axis([0, max([max(x) for x in ls]), 1, axis[3]])
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
plt.xticks(range(max(l) + 1), fontsize=30)
plt.tick_params(labelsize=30)
plt.savefig('dw11_figsix6_5_amg.svg')
plt.close()

# basso dx: Plot normalized cost for given accuracy
plt.figure(figsize=figsize)
I = np.argsort(eps)
std_cost = np.array(std_cost)
mlmc_cost = np.array(mlmc_cost)
plt.loglog(epsss[I], std_cost[I] / 3600,  'o-',  label='MC', markersize=10, clip_on=False, linewidth=3.5, color='blue')
plt.loglog(epsss[I], mlmc_cost[I] / 3600, 'o--', label='MLMC', markersize=10,   clip_on=False, linewidth=3.5, color='red')
plt.xlabel(r'Relative accuracy $\varepsilon_r$', fontsize=35)
plt.ylabel(r'Cost [$\mathrm{N_{CPU} \cdot hours}$]', fontsize=35)
plt.legend(loc='best', fontsize=35)
plt.grid(True, which="both")
plt.tick_params(labelsize=30)

yticks = [1e3, 1e4]
ytick_labels = [r'$10^3$', r'$10^4$']
plt.yticks(yticks, ytick_labels)

axis = plt.axis()
plt.axis([min(epsss), max(epsss), axis[2], axis[3]])

plt.axis([min(epsss), max(epsss), min(yticks) * 0.9, max(yticks)])

plt.savefig('dw11_figsix6_6_amg.svg')
plt.close()