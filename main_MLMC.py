# Authors: Nicolò Sarman, Alberto Perlini
# Description: Multi level Monte Carlo, hypersonic flow of air-5 over a 15-45° double wedge in thermal and chemical non equilibrium

import os, sys
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['OPENBLAS_NUM_THREADS'] = "1"
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

def main(nproc, testcase, workingFolder):
    """
    Main script for MLMC UQ on hypersonic double wedge.
    """

    # folder for with MLMC code
    baseFolder = '/home/xla27/Softwares/MLMC'

    if testcase == 'AIR5':

        baseFolder = baseFolder + '/AIR5'

        from AIR5 import mlmc, screening, mlmc_plot, dw_l

    if testcase == 'AIR5_AMG':

        basefolder = basefolder + '/AIR5_AMG'

        from AIR5_AMG import mlmc, screening, mlmc_plot, dw_l

    if testcase == 'AIR11':

        basefolder = basefolder + '/AIR11'

        from AIR11 import mlmc, screening, mlmc_plot, dw_l

    if testcase == 'AIR11_AMG':

        basefolder = basefolder + '/AIR11_AMG'

        from AIR11_AMG import mlmc, screening, mlmc_plot, dw_l


    varargin = (nproc, baseFolder, workingFolder) 

    N = 10    # samples for convergence tests
    L = 4     # levels for convergence tests

    N0   = 2   # initial samples on coarsest levels
    Lmin = 2   # minimum refinement level
    Lmax = 4   # maximum refinement level

    Eps = [0.03, 0.04, 0.07]     # required relative tolerance
    
    # log file
    filename = "dw.txt"
    logfile = open(filename, "w")

    # Starting recording time
    t_Start = time.time()

    # 1: Testing phase
    now = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "*** MLMC file version 1.0 produced by                  ***\n")
    write(logfile, "*** Python mlmc_test on %s        ***\n" % now )
    write(logfile, "**********************************************************\n")
    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "*** Convergence tests                                  ***\n")
    write(logfile, "*** using N =%7d samples                           ***\n" % N)
    write(logfile, "**********************************************************\n")
    write(logfile, "\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)  var(Pf)")
    write(logfile, "  cost\n-------------------------")
    write(logfile, "------------------------------------------------------\n")

    del1, del2, var1, var2, cost, alpha, beta, gamma = screening(dw_l, L, N, logfile)

    write(logfile, "\n******************************************************\n")
    write(logfile, "*** Linear regression estimates of MLMC parameters ***\n")
    write(logfile, "******************************************************\n")
    write(logfile, "\n alpha = %f  (exponent for MLMC weak convergence)\n" % alpha)
    write(logfile, " beta  = %f  (exponent for MLMC variance) \n" % beta)
    write(logfile, " gamma = %f  (exponent for MLMC cost) \n" % gamma)

    # 2: MLMC call
    write(logfile, "\n")
    write(logfile, "***************************** \n")
    write(logfile, "*** MLMC complexity tests *** \n")
    write(logfile, "***************************** \n\n")
    write(logfile, "   eps       value     mlmc_cost   std_cost  savings     N_l \n")
    write(logfile, "------------------------------------------------------------ \n")

    alpha = max(alpha, 0.5)
    beta = max(beta, 0.5)
    theta = 0.5

    Eps = np.array(Eps) * np.abs(del2[-1])  # absolute tolerance (rel * average at the finest level)

    for eps in Eps:

        epsName = str(eps)[:5]
        filenameEps = f"Nl{str(epsName).replace('.', '_')}.txt"
        Nlfile = open(filenameEps, "w")

        P, Nl, Cl = mlmc(dw_l, N0, eps, Lmin, Lmax, alpha, beta, gamma, Nlfile, *varargin)

        Nlfile.close()
        l = len(Nl) - 1
        mlmc_cost = np.dot(Nl,Cl)
        std_cost = var2[min(len(var2)-1,l)] * Cl[-1] / ((1.0 - theta) * eps ** 2)     # cost of a classic MC at highest level

        write(logfile, "%.3e %11.4e  %.3e  %.3e  %7.2f " % (eps, P, mlmc_cost, std_cost, std_cost/mlmc_cost))
        write(logfile, " ".join(["%9d" % n for n in Nl]))
        write(logfile, "\n")

    write(logfile, "\n")

    # Ending recording time
    time_elapsed = time.time() - t_Start
    
    logfile.write('\nTime elapsed: {} seconds\n'.format(time_elapsed))
    logfile.close() 
    del logfile
    
    plt.close()
    plt.cla()

    # MLMC plots
    mlmc_plot(filename, nvert=3)
    plt.savefig(filename.replace('.txt', '.svg'))

    return

def write(logfile, msg):
    """
    Write to a logfile.
    """
    logfile.write(msg)
    logfile.flush()

if __name__ == "__main__":

    nproc = int(sys.argv[1])    # number of processors for CFD
    testcase = sys.argv[2]
    workingFolder = os.getcwd() # working directory for data saving

    main(nproc, testcase, workingFolder)
