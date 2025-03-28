import matplotlib
import matplotlib.pyplot as plt
import numpy
import pickle
import re
import os
from cycler import cycler

plt.close("all")
plt.cla()
plt.clf()
# Set matplotlib defaults to agree with MATLAB code
plt.rc("legend", framealpha=None)
plt.rc("legend", edgecolor='black')
plt.rc("font", family="serif")
# Following option for TeX text seems to not work with EPS figures?
#plt.rc("text", usetex=True)

# NOTE: set clip_on=False in plotting to get consistent style with MATLAB

def mlmc_plot(filename, nvert):
    """
    Utility to generate MLMC diagnostic plots based on
    input text file generated by MLMC driver code mlmc_test.

    mlmc_plot(filename, nvert, error_bars=False)

    Inputs:
      filename: string, (base of) filename with output from mlmc_test routine
      nvert   : int, number of vertical plots <= 3
                    nvert == 1   generates fig1: (1),(2) fig2: (5),(6)
                    nvert == 2   generates fig1: (1),(2),(5),(6)
                    nvert == 3   generates fig1: (1)-(6)
      error_bars: bool, flag to add error bars in plots of level differences

    Outputs:
      Matplotlib figure(s) for
        Convergence tests
        (1) Var[P_l - P_{l-1}] per level
        (2) E[|P_l - P_{l-1}|] per level
        (3) cost per level
        (5) number of samples per level
        (6) normalised cost per accuracy target
    """

    #
    # read in data
    #

    file = open(filename, "r")
    file.seek(0)     

    # Declare lists for data
    del1 = []
    del2 = []
    var1 = []
    var2 = []
    cost = []
    l    = []

    epss = []
    mlmc_cost = []
    std_cost = []
    Ns = []
    ls = []

    # Default values for number of samples and file_version
    N = 0
 
    complexity_flag = False # first read convergence tests rather than complexity tests

    for line in file: 
        # Recognise number of samples line from the fact that it starts with '*** using'
        if line[0:9] == '*** using':
            N = int(line[14:20])

        # Recognise whether we should switch to reading complexity tests
        if line[0:19] == '*** MLMC complexity':
            complexity_flag = True # now start to read complexity tests

        # Recognise MLMC complexity test lines from the fact that line[0] is an integer
        # Also need complexity_flag == True because line[0] is an integer also identifies
        # the convergence test lines
        if '0' <= line[0] <= '9' and complexity_flag:
            splitline = [float(x) for x in line.split()]
            epss.append(splitline[0])
            mlmc_cost.append(splitline[2])
            std_cost.append(splitline[3])
            Ns.append(splitline[5:])
            ls.append(list(range(0,len(splitline[5:]))))
        
        # Recognise convergence test lines from the fact that line[1] is an integer
        # and possibly also line[0] (or line[0] is whitespace)
        if (line[0] == ' ' or '0' <= line[0] <= '9') and '0' <= line[1] <= '9':
            splitline = [float(x) for x in line.split()]
            l.append(splitline[0])
            del1.append(splitline[1])
            del2.append(splitline[2])
            var1.append(splitline[3])
            var2.append(splitline[4])
            cost.append(splitline[5])
            continue

    #
    # plot figures
    #
    file.close()

    # Fudge to get comparable size to default MATLAB fig size
    # width_MATLAB = 0.9*8; height_MATLAB = 0.9*6.5;
    # plt.figure(figsize=([width_MATLAB, height_MATLAB*0.75*nvert]))
    plt.tight_layout()

    plt.rc('axes', prop_cycle=(cycler('color', ['k']) *
                               cycler('linestyle', ['--', ':']) *
                               cycler('marker', ['*'])))

    # Alto sx: Var[P_l - P_{l-1}] per level
    plt.subplot(3, 2, 1)
    plt.plot(l,     numpy.log2(var2),     label=r'$P_\ell$',              clip_on=False)
    plt.plot(l[1:], numpy.log2(var1[1:]), label=r'$P_\ell - P_{\ell-1}$', clip_on=False)
    plt.xlabel('level $\ell$')
    plt.ylabel(r'$\mathrm{log}_2(\mathrm{variance})$')
    plt.legend(loc='lower left', fontsize='medium')
    axis = plt.axis()
    plt.axis([0, max(l), axis[2], axis[3]])


    # Alto dx: E[|P_l - P_{l-1}|] per level
    plt.subplot(3, 2, 2)
    plt.plot(l,     numpy.log2(numpy.abs(del2)),     label=r'$P_\ell$',              clip_on=False)
    plt.plot(l[1:], numpy.log2(numpy.abs(del1[1:])), label=r'$P_\ell - P_{\ell-1}$', clip_on=False)
    plt.xlabel('level $\ell$')
    plt.ylabel(r'$\mathrm{log}_2(|\mathrm{mean}|)$')
    plt.legend(loc='lower left', fontsize='medium')
    axis = plt.axis()
    plt.axis([0, max(l), axis[2], axis[3]])

    # Centro sx: cost per level
    plt.subplot(3, 2, 3)
    plt.plot(l, numpy.log2(cost), '*--', clip_on=False)
    plt.xlabel('level $\ell$')
    plt.ylabel(r'$\log_2$ cost per sample')
    axis = plt.axis()
    plt.axis([0, max(l), axis[2], axis[3]])  



    marker_styles = ['o', 'x', 'd', '*', 's']
    plt.rc('axes', prop_cycle=(cycler('color', ['k']) *
                               cycler('linestyle', ['--']) *
                               cycler('marker', marker_styles)))
    # centro dx: costo teorico
    
    with open(filename, 'r') as fid:
        file_content = fid.read()

    # Regular expressions patterns
    alpha_pattern = r'alpha\s*=\s*([\d.]+)'
    beta_pattern  = r'beta\s*=\s*([\d.]+)'
    gamma_pattern = r'gamma\s*=\s*([\d.]+)'

    # Extract matches using regular expressions
    alpha_match = re.search(alpha_pattern, file_content)
    beta_match  = re.search(beta_pattern, file_content)
    gamma_match = re.search(gamma_pattern, file_content)

    # Extract values from matches
    alpha = float(alpha_match.group(1))
    beta  = float(beta_match.group(1))
    gamma = float(gamma_match.group(1))

    # Plotting 
    mlmc_cost_array = numpy.array(mlmc_cost)
    epss_array = numpy.array(epss)
    plt.subplot(3, 2, 4)
    plt.loglog(epss_array, mlmc_cost_array / max(mlmc_cost_array), color='blue', label='Simulation')
    plt.grid(True)
    plt.loglog(epss_array, epss_array**(-2) / max(epss_array**(-2)), color='red', label=r'$\beta$ > $\gamma$')
    plt.loglog(epss_array, epss_array**(-2 - (gamma - beta) / alpha) / max(epss_array**(-2 - (gamma - beta) / alpha)), color='green', label=r'$\beta$ < $\gamma$')
    plt.xlabel('Relative accuracy $\epsilon_r$', fontsize=14, fontweight='bold')
    plt.ylabel('Cost normalized trend', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    # basso sx: number of samples per level
    plt.subplot(3, 2, 5)
    for (eps, ll, n) in zip(epss, ls, Ns):
        # Check if any element in n is zero
        if numpy.any(n == 0):
            # Replace zero values with 1
            n[n == 0] = 1
        plt.semilogy(ll, n, label=eps, markerfacecolor='none', clip_on=False)
    plt.xlabel('level $\ell$')
    plt.ylabel('$N_\ell$')
    plt.legend(loc='upper right', frameon=True, fontsize='medium')
    axis = plt.axis()
    plt.axis([0, max([max(x) for x in ls]), axis[2], axis[3]])


    plt.rc('axes', prop_cycle=(cycler('color', ['k']) *
                               cycler('linestyle', ['--', ':']) *
                               cycler('marker', ['*'])))

    # basso dx: normalised cost for given accuracy
    eps = numpy.array(epss)
    std_cost = numpy.array(std_cost)
    mlmc_cost = numpy.array(mlmc_cost)
    I = numpy.argsort(eps)
    plt.subplot(3, 2, 6)
    plt.loglog(eps[I], eps[I]**2 * std_cost[I],  '*-',  label='MC',     clip_on=False)
    plt.loglog(eps[I], eps[I]**2 * mlmc_cost[I], '*--', label='MLMC',   clip_on=False)
    plt.xlabel(r'accuracy $\varepsilon$')
    plt.ylabel(r'$\varepsilon^2$ cost')
    plt.legend(fontsize='medium')
    axis = plt.axis()
    plt.axis([min(eps), max(eps), axis[2], axis[3]])

    

    save=True
    summary_plot = 'plot_summary.pkl'
    if save:
        # Save data with pickle
        plot_data = {
            'del1': del1,
            'del2': del2,
            'var1': var1,
            'var2': var2,
            'cost': cost,
            'l': l,
            'epss': epss,
            'mlmc_cost': mlmc_cost,
            'std_cost': std_cost,
            'Ns': Ns,
            'ls': ls,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
        }
        with open(summary_plot, 'wb') as f:
            pickle.dump(plot_data, f)    
