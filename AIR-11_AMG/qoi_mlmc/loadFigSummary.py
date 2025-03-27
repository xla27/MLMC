import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import csv
import pickle
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


# log2del1 = [7.27582015162542, 5.125229083049479, 3.320270799593645, 2.268424721678081, 2.12458356206175]
# log2del2 = [7.27582015162542, 7.568861355988556, 7.490895670773469, 7.529028751164304, 7.562695191476765]
# log2var1 = [10.53838296003212, 7.181897643108388, 4.876271360550127, 3.6803243568440163, 2.4802651220544627]
# log2var2 = [10.53838296003212, 10.88378880325996, 10.817863047263648, 10.850421276078848, 10.707186550581676]
# log2cost = [14.621136113274641, 15.51422090935813, 16.948777859356397, 18.002946834789466, 18.872674880270605]
del1 = [1.5269e+01, 8.3743e-01, 3.2850e-01, 1.8196e-01, 9.0537e-02]
del2 = [1.5269e+01, 1.4041e+01, 1.3803e+01, 1.4709e+01, 1.5485e+01]
var1 = [4.190e+01, 6.201e-01, 2.103e-01, 8.214e-02, 3.468e-02]
var2 = [4.190e+01, 6.310e+01, 3.788e+01, 5.411e+01, 2.504e+01]
cost = [2**14.621136113274641, 
        2**15.51422090935813, 
        2**16.948777859356397,
        2**18.002946834789466, 
        2**18.872674880270605]
l = [0, 1, 2, 3, 4]; l = [int(x) for x in l]
# epss = plot_data['epss']
# mlmc_cost = plot_data['mlmc_cost']
# std_cost = plot_data['std_cost']
# Ns = plot_data['Ns']
# ls = plot_data['ls']
L1 = 1
L2 = 5
# pa = np.polyfit(range(L1, L2), log2del1[L1:L2], 1);  alpha = -pa[0]
# pb = np.polyfit(range(L1, L2), log2var1[L1:L2], 1);  beta  = -pb[0]
# pg = np.polyfit(range(L1, L2), log2cost[L1:L2], 1);  gamma =  pg[0]
pa = np.polyfit(range(L1, L2), np.log2(del1[L1:L2]), 1);  alpha = -pa[0]
pb = np.polyfit(range(L1, L2), np.log2(var1[L1:L2]), 1);  beta  = -pb[0]
pg = np.polyfit(range(L1, L2), np.log2(cost[L1:L2]), 1);  gamma =  pg[0]


eps = [0.04, 0.05]  # Define epss values
mlmc_cost = [2436.389*3600, 1143.611*3600]  # Define mlmc_cost values
std_cost = [7794.444*3600, 4711.111*3600]  # Define std_cost values
Ns = [[208, 24, 9, 4, 2],[102, 11, 4, 2, 1]]  # Number of samples values
ls = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]  # List of lists for ls

epss_array = np.array([0.04, 0.05]) # Convert 'epss' to a numpy array
# log2del22 = np.array(log2del2)
del22 = np.array(del2)


# Utilities
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 20 })

# Define figure size
figsize = (13, 11)

# Alto: log2 media e varianza
fig, ax1 = plt.subplots(figsize=figsize)

ax1.plot(l, np.log2(del2), label=r'$\mathrm{mean} \, Q_l$', clip_on=False, markersize=15, marker='o', linewidth=3.5, linestyle='-', color='blue')
ax1.plot(l[1:], np.log2(del1[1:]), label=r'$\mathrm{mean} \, $Q_l - Q_{l-1}$', clip_on=False, markersize=15, marker='v', linewidth=3.5, linestyle='--', color='blue')
ax1.set_xlabel(r'$\mathrm{Level} \, \ell$', fontsize=35)
ax1.set_ylabel(r'$\left \| \mathrm{E} \, [\cdot] \right \|_{L^1} , \; \log_2$', fontsize=35, color='blue')
ax1.tick_params(axis='y', labelcolor='blue', labelsize=30)
ax1.tick_params(axis='x', labelsize=30)
ax1.grid(True)
ax1.set_xticks(range(max(l) + 1))
ax1.set_xlim([0, max(l)])

ax2 = ax1.twinx()
ax2.plot(l, np.log2(var2), label=r'$\mathrm{variance} \, Q_l$', clip_on=False, markersize=15, marker='o', linewidth=3.5, linestyle='-', color='red')
ax2.plot(l[1:], np.log2(var1[1:]), label=r'\mathrm{variance} \, Q_l - Q_{l-1}$', clip_on=False, markersize=15, marker='v', linewidth=3.5, linestyle='--', color='red')
#ax2.set_ylabel(r'$\log_2 ||\mathrm{variance}||_\infty$', fontsize=35, color='red')
ax2.set_ylabel(r'$\left \| \mathrm{V} \, [\cdot] \right \|_{L^1} , \; \log_2$', fontsize=35, color='red')
ax2.tick_params(axis='y', labelcolor='red', labelsize=30)
ax2.set_xlim([0, max(l)])

legend_elements = [
    (Line2D([0], [0], color='blue', lw=3.5, linestyle='-', marker='o'), 
     Line2D([0], [0], color='red', lw=3.5, linestyle='-', marker='o')),
    (Line2D([0], [0], color='blue', lw=3.5, linestyle='--', marker='v'), 
     Line2D([0], [0], color='red', lw=3.5, linestyle='--', marker='v'))
]


ax1.legend(legend_elements, [r'$Q_\ell$', r'$Q_\ell - Q_{\ell-1}$'],
           handler_map={tuple: HandlerLine2D()}, loc='best', fontsize=35)

plt.savefig('dw5_summary_12_combined.svg')
plt.close()


# Centro sx: Plot cost per level
plt.figure(figsize=figsize)
plt.plot(l, np.log2(cost), 'o-', markersize=10, clip_on=False, linewidth=3.5, color='blue')
plt.xlabel(r'$\mathrm{Level} \, \ell$', fontsize=35)
#plt.ylabel(r'$\log_2$ cost per sample', fontsize=35)
plt.ylabel(r'$\mathrm{Cost} \, \mathrm{per} \, \mathrm{sample}, \: \log_2$', fontsize=35)
plt.grid(True)
plt.axis([0,max(l), min(np.log2(cost)), max(np.log2(cost))])
plt.xticks(range(max(l) + 1), fontsize=30)
plt.tick_params(labelsize=30)
plt.savefig('dw5_summary_3.svg')
plt.close()


# centro dx: Plot theoretical cost
plt.figure(figsize=figsize)
plt.rc('axes', prop_cycle=(cycler('color', ['k']) * cycler('linestyle', ['--', ':']) * cycler('marker', ['*'])))

epsss = np.array(eps)
mlmc_costs = np.array(mlmc_cost)

plt.loglog(epsss, mlmc_costs / max(mlmc_costs), 'o--', markersize=10, label=r'$\mathrm{Simulation}$', linewidth=3.5, color='blue')
plt.loglog(epsss, epsss**(-2) / max(epsss**(-2)), 'x--', markersize=10, label=r'$\beta > \gamma$', linewidth=3.5, color='red')
plt.loglog(epsss, epsss**(-2 - (gamma - beta) / alpha) / max(epsss**(-2 - (gamma - beta) / alpha)), 'd--', markersize=10, label=r'$\beta < \gamma$', linewidth=3.5, color='green')
plt.xlabel(r'$\varepsilon_r$', fontsize=35)
plt.ylabel(r'$\mathrm{Normalized} \, \mathrm{cost}$', fontsize=35)
plt.grid(True, which="both")
plt.tick_params(labelsize=30)
axis = plt.axis()
plt.axis([min(epsss), max(epsss), axis[2], axis[3]])
plt.legend(loc='best', fontsize=35, framealpha=1.0)
plt.savefig('dw5_summary_4.svg')
plt.close()

# basso sx: Plot number of samples per level
plt.figure(figsize=figsize)
colors = ['blue', 'red', 'green','purple']
custom_cycler = (cycler('color', ['blue', 'red', 'green','purple']) *      # Colors for each plot 
                 cycler('marker', ['o', 'x', 'd','*']) *              # Markers for each plot
                 cycler('markersize', [10, 10, 10, 10]) *             # Marker sizes for each plot
                 cycler('linewidth', [3, 3, 3, 3]))            # Linewidths for each plot

# Set the rcParams with the custom cycler
plt.rcParams['axes.prop_cycle'] = custom_cycler
for eps_c, ll, n, color in zip(epsss, ls, Ns, colors):
    plt.semilogy(ll, n, label='$\\varepsilon_r = {:.2f}$'.format(eps_c), markerfacecolor=color, clip_on=False, linewidth=3, markersize=10, color=color)

plt.xlabel(r'$\mathrm{Level} \, \ell$', fontsize=35)
plt.ylabel(r'$N_l$', fontsize=35)
plt.legend(loc='best', fontsize=35, framealpha=1.0)
plt.grid(True, which="both")
axis = plt.axis()
plt.axis([0, max([max(x) for x in ls]), axis[2], axis[3]])
plt.xticks(range(max(l) + 1), fontsize=30)
plt.tick_params(labelsize=30)
plt.savefig('dw5_summary_5.svg')
plt.close()

# basso dx: Plot normalized cost for given accuracy
plt.figure(figsize=figsize)
I = np.argsort(eps)
std_cost = np.array(std_cost)
mlmc_cost = np.array(mlmc_cost)
plt.loglog(epsss[I], std_cost[I] / 3600,  'o-',  label=r'$\mathrm{MC}$', markersize=10, clip_on=False, linewidth=3.5, color='blue')
plt.loglog(epsss[I], mlmc_cost[I] / 3600, 'o--', label=r'$\mathrm{MLMC}$', markersize=10,   clip_on=False, linewidth=3.5, color='red')
plt.xlabel(r'$\varepsilon_r$', fontsize=35)
plt.ylabel(r'$\mathrm{Cost}, \; [N_{\mathrm{CPU}} \cdot \mathrm{hours}]$', fontsize=35)
plt.legend(loc='best', fontsize=35, framealpha=1.0)
plt.grid(True, which="both")
plt.tick_params(labelsize=30)
axis = plt.axis()
plt.axis([min(epsss), max(epsss), axis[2], axis[3]])
plt.savefig('dw5_summary_6.svg')
plt.close()
