from mlmc import mlmc
from datetime import datetime
import numpy as np

def mlmc_test(mlmc_l, N, L, N0, Eps, Lmin, Lmax, logfile, *varargin):
    """
        mlmc_l: function to call CFD
        N:    number of samples for convergence tests
        L:    number of levels for convergence tests

        N0:   initial number of samples for MLMC calculations
        Eps:  desired accuracy (rms error) array for MLMC calculations
        Lmin: minimum number of levels for MLMC calculations
        Lmax: maximum number of levels for MLMC calculations

        logfile: file handle for printing to file

        *args, **kwargs: optional additional variables to be passed to mlmc_l
    """
    
    # 1: Testing phase
    now = datetime.now().strftime("%d-%B-%Y %H:%M:%S")
    write(logfile, "\n");
    write(logfile, "**********************************************************\n");
    write(logfile, "*** MLMC file version 1.0 produced by              ***\n");
    write(logfile, "*** Python mlmc_test on %s           ***\n" % now );
    write(logfile, "**********************************************************\n");
    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "*** Convergence tests ***\n")
    write(logfile, "*** using N =%7d samples                           ***\n" % N)
    write(logfile, "**********************************************************\n")
    write(logfile, "\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)  var(Pf)")
    write(logfile, "  cost\n-------------------------")
    write(logfile, "------------------------------------------------------\n")

    del1 = [158.86, 36.158, 18.461, 10.97, 7.2238]
    del2 = [158.86, 152.81, 178.25, 161.21, 168.55]
    var1 = [921.9, 112.9, 29.11, 13.47, 4.299]
    var2 = [921.9, 1927, 1848, 1837, 1609]
    cost = [2190, 9280, 27300, 74100, 210000]

    for l in range(L + 1):

        write(logfile, "%2d  %11.4e %11.4e  %.3e  %.3e %.2e \n" % \
                      (l, del1[l], del2[l], var1[l], var2[l], cost[l]))

    alpha = 0.772133;
    beta  = 1.525488;
    gamma =  1.494503;

    write(logfile, "\n******************************************************\n");
    write(logfile, "*** Linear regression estimates of MLMC parameters ***\n");
    write(logfile, "******************************************************\n");
    write(logfile, "\n alpha = %f  (exponent for MLMC weak convergence)\n" % alpha);
    write(logfile, " beta  = %f  (exponent for MLMC variance) \n" % beta);
    write(logfile, " gamma = %f  (exponent for MLMC cost) \n" % gamma);

    # 2: MLMC call
    write(logfile, "\n");
    write(logfile, "***************************** \n");
    write(logfile, "*** MLMC complexity tests *** \n");
    write(logfile, "***************************** \n\n");
    write(logfile, "   eps       value     mlmc_cost   std_cost  savings     N_l \n");
    write(logfile, "------------------------------------------------------------ \n");

    alpha = max(alpha, 0.5); beta = max(beta, 0.5); theta = 0.5;
    Eps = np.array(Eps) * np.abs(del2[-1])

    for eps in Eps:
        epsName = str(eps)[:5]
        filename = f"Nl{str(epsName).replace('.', '_')}.txt"
        Nlfile = open(filename, "w")
        P, Nl, Cl = mlmc(mlmc_l, N0, eps, Lmin, Lmax, alpha, beta, gamma, Nlfile, *varargin)
        Nlfile.close()
        l = len(Nl) - 1
        mlmc_cost = np.dot(Nl,Cl)
        std_cost = var2[min(len(var2)-1,l)] * Cl[-1] / ((1.0 - theta) * eps ** 2)

        write(logfile, "%.3e %11.4e  %.3e  %.3e  %7.2f " % (eps, P, mlmc_cost, std_cost, std_cost/mlmc_cost))
        write(logfile, " ".join(["%9d" % n for n in Nl]))
        write(logfile, "\n")

    write(logfile, "\n")

def write(logfile, msg):
    """
    Write to a logfile.
    """
    logfile.write(msg)
    logfile.flush()