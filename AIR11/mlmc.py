import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle

def write(logfile, msg):
    """
    Write to a logfile.
    """
    logfile.write(msg)
    logfile.flush()

def mlmc(mlmc_l, N0, eps, Lmin, Lmax, alpha0, beta0, gamma0, Nlfile, *args):
    """
    Inputs:
      N0:   initial number of samples    >  0
      eps:  desired accuracy (rms error) >  0
      Lmin: minimum level of refinement  >= 2
      Lmax: maximum level of refinement  >= Lmin
      mlmc_l: the user low-level routine for level l estimator. 
      *args, **kwargs = optional additional user variables to be passed to mlmc_l

    Outputs:
      P:  value of the QoI
      Nl: number of samples at each level
      Cl: cost of samples at each level
    """

    # Convergence rates
    alpha = max(0, alpha0)
    beta  = max(0, beta0)
    gamma = max(0, gamma0)

    theta = 0.5     # coefficient for weak convergence inequalities
    
    L = Lmin        # initial maximum refinement level


    Nl = np.zeros(L+1)          # number of samples per level (initially set to zero, at the first iteration the number of samples N0 is represented by dNl) 
    costl = np.zeros(L+1)       # cost for each level
    dNl = np.ones(L+1) * N0     # delta samples at each level, at the first iteration of the while cycle it is equal to N0, then it becomes a difference

    cellsom1P             = [None] * (L+1);     cellsom2P             = [None] * (L+1)
    cellsom1_nd_elecMinus = [None] * (L+1);     cellsom2_nd_elecMinus = [None] * (L+1)
    cellsom1_nd_nPlus     = [None] * (L+1);     cellsom2_nd_nPlus     = [None] * (L+1)
    cellsom1_nd_oPlus     = [None] * (L+1);     cellsom2_nd_oPlus     = [None] * (L+1)
    cellsom1_nd_noPlus    = [None] * (L+1);     cellsom2_nd_noPlus    = [None] * (L+1)
    cellsom1_nd_n2Plus    = [None] * (L+1);     cellsom2_nd_n2Plus    = [None] * (L+1)
    cellsom1_nd_o2Plus    = [None] * (L+1);     cellsom2_nd_o2Plus    = [None] * (L+1)
    cellsom1elecMinus     = [None] * (L+1);     cellsom2elecMinus     = [None] * (L+1)
    cellsom1NPlus         = [None] * (L+1);     cellsom2NPlus         = [None] * (L+1)
    cellsom1OPlus         = [None] * (L+1);     cellsom2OPlus         = [None] * (L+1)
    cellsom1NOPlus        = [None] * (L+1);     cellsom2NOPlus        = [None] * (L+1)
    cellsom1N2Plus        = [None] * (L+1);     cellsom2N2Plus        = [None] * (L+1)
    cellsom1O2Plus        = [None] * (L+1);     cellsom2O2Plus        = [None] * (L+1)
    cellsom1N             = [None] * (L+1);     cellsom2N             = [None] * (L+1)
    cellsom1O             = [None] * (L+1);     cellsom2O             = [None] * (L+1)       
    cellsom1NO            = [None] * (L+1);     cellsom2NO            = [None] * (L+1)
    cellsom1N2            = [None] * (L+1);     cellsom2N2            = [None] * (L+1)
    cellsom1O2            = [None] * (L+1);     cellsom2O2            = [None] * (L+1)      
    cellsom1Ttr           = [None] * (L+1);     cellsom2Ttr           = [None] * (L+1)
    cellsom1Tve           = [None] * (L+1);     cellsom2Tve           = [None] * (L+1)
    cellsom1M             = [None] * (L+1);     cellsom2M             = [None] * (L+1)

    cellnodes = [None] * (L+1)


    while np.sum(dNl) > 0:

       # Update sample sums

        for l in range(L+1):

            if dNl[l] > 0:

                [x2, sums1, sums2, _, _, sums1ndelecMinus, sums2ndelecMinus, sums1ndNPlus, sums2ndNPlus, sums1ndOPlus, 
                 sums2ndOPlus, sums1ndNOPlus, sums2ndNOPlus, sums1ndN2Plus, sums2ndN2Plus, sums1ndO2Plus, sums2ndO2Plus, 
                 sums1elecMinus, sums2elecMinus, sums1NPlus, sums2NPlus, sums1OPlus, sums2OPlus, sums1NOPlus, sums2NOPlus, 
                 sums1N2Plus, sums2N2Plus, sums1O2Plus, sums2O2Plus, sums1N, sums2N, sums1O, sums2O, sums1NO, sums2NO, 
                 sums1N2, sums2N2, sums1O2, sums2O2, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost] = mlmc_l(l, int(dNl[l]), *args)
                
                Nl[l]    += dNl[l]
                costl[l] += cost

                if cellsom1P[l] is None:
                    cellsom1P[l]             = sums1;               cellsom2P[l]             = sums2
                    cellsom1_nd_elecMinus[l] = sums1ndelecMinus;    cellsom2_nd_elecMinus[l] = sums2ndelecMinus
                    cellsom1_nd_nPlus[l]     = sums1ndNPlus;        cellsom2_nd_nPlus[l]     = sums2ndNPlus
                    cellsom1_nd_oPlus[l]     = sums1ndOPlus;        cellsom2_nd_oPlus[l]     = sums2ndOPlus
                    cellsom1_nd_noPlus[l]    = sums1ndNOPlus;       cellsom2_nd_noPlus[l]    = sums2ndNOPlus
                    cellsom1_nd_n2Plus[l]    = sums1ndN2Plus;       cellsom2_nd_n2Plus[l]    = sums2ndN2Plus
                    cellsom1_nd_o2Plus[l]    = sums1ndO2Plus;       cellsom2_nd_o2Plus[l]    = sums2ndO2Plus
                    cellsom1elecMinus[l]     = sums1elecMinus;      cellsom2elecMinus[l]     = sums2elecMinus
                    cellsom1NPlus[l]         = sums1NPlus;          cellsom2NPlus[l]         = sums2NPlus
                    cellsom1OPlus[l]         = sums1OPlus;          cellsom2OPlus[l]         = sums2OPlus
                    cellsom1NOPlus[l]        = sums1NOPlus;         cellsom2NOPlus[l]        = sums2NOPlus
                    cellsom1N2Plus[l]        = sums1N2Plus;         cellsom2N2Plus[l]        = sums2N2Plus
                    cellsom1O2Plus[l]        = sums1O2Plus;         cellsom2O2Plus[l]        = sums2O2Plus
                    cellsom1N[l]             = sums1N;              cellsom2N[l]             = sums2N
                    cellsom1O[l]             = sums1O;              cellsom2O[l]             = sums2O
                    cellsom1NO[l]            = sums1NO;             cellsom2NO[l]            = sums2NO
                    cellsom1N2[l]            = sums1N2;             cellsom2N2[l]            = sums2N2
                    cellsom1O2[l]            = sums1O2;             cellsom2O2[l]            = sums2O2
                    cellsom1Ttr[l]           = sums1Ttr;            cellsom2Ttr[l]           = sums2Ttr
                    cellsom1Tve[l]           = sums1Tve;            cellsom2Tve[l]           = sums2Tve
                    cellsom1M[l]             = sums1M;              cellsom2M[l]             = sums2M
                    
                else:
                    cellsom1P[l]             += sums1;              cellsom2P[l]             += sums2
                    cellsom1_nd_elecMinus[l] += sums1ndelecMinus;   cellsom2_nd_elecMinus[l] += sums2ndelecMinus
                    cellsom1_nd_nPlus[l]     += sums1ndNPlus;       cellsom2_nd_nPlus[l]     += sums2ndNPlus
                    cellsom1_nd_oPlus[l]     += sums1ndOPlus;       cellsom2_nd_oPlus[l]     += sums2ndOPlus
                    cellsom1_nd_noPlus[l]    += sums1ndNOPlus;      cellsom2_nd_noPlus[l]    += sums2ndNOPlus
                    cellsom1_nd_n2Plus[l]    += sums1ndN2Plus;      cellsom2_nd_n2Plus[l]    += sums2ndN2Plus
                    cellsom1_nd_o2Plus[l]    += sums1ndO2Plus;      cellsom2_nd_o2Plus[l]    += sums2ndO2Plus
                    cellsom1elecMinus[l]     += sums1elecMinus;     cellsom2elecMinus[l]     += sums2elecMinus
                    cellsom1NPlus[l]         += sums1NPlus;         cellsom2NPlus[l]         += sums2NPlus
                    cellsom1OPlus[l]         += sums1OPlus;         cellsom2OPlus[l]         += sums2OPlus
                    cellsom1NOPlus[l]        += sums1NOPlus;        cellsom2NOPlus[l]        += sums2NOPlus
                    cellsom1N2Plus[l]        += sums1N2Plus;        cellsom2N2Plus[l]        += sums2N2Plus
                    cellsom1O2Plus[l]        += sums1O2Plus;        cellsom2O2Plus[l]        += sums2O2Plus
                    cellsom1N[l]             += sums1N;             cellsom2N[l]             += sums2N
                    cellsom1O[l]             += sums1O;             cellsom2O[l]             += sums2O
                    cellsom1NO[l]            += sums1NO;            cellsom2NO[l]            += sums2NO
                    cellsom1N2[l]            += sums1N2;            cellsom2N2[l]            += sums2N2
                    cellsom1O2[l]            += sums1O2;            cellsom2O2[l]            += sums2O2
                    cellsom1Ttr[l]           += sums1Ttr;           cellsom2Ttr[l]           += sums2Ttr
                    cellsom1Tve[l]           += sums1Tve;           cellsom2Tve[l]           += sums2Tve
                    cellsom1M[l]             += sums1M;             cellsom2M[l]             += sums2M                    
                       
                cellnodes[l] = x2
        
        # Interpolation on coarsest grid
        coarsest_grid = cellnodes[0]

        interpolated_cellsom1P             = [];    interpolated_cellsom2P             = []
        interpolated_cellsom1_nd_elecMinus = [];    interpolated_cellsom2_nd_elecMinus = []
        interpolated_cellsom1_nd_nPlus     = [];    interpolated_cellsom2_nd_nPlus     = []
        interpolated_cellsom1_nd_oPlus     = [];    interpolated_cellsom2_nd_oPlus     = []
        interpolated_cellsom1_nd_noPlus    = [];    interpolated_cellsom2_nd_noPlus    = []
        interpolated_cellsom1_nd_n2Plus    = [];    interpolated_cellsom2_nd_n2Plus    = []
        interpolated_cellsom1_nd_o2Plus    = [];    interpolated_cellsom2_nd_o2Plus    = []
        interpolated_cellsom1elecMinus     = [];    interpolated_cellsom2elecMinus     = []
        interpolated_cellsom1NPlus         = [];    interpolated_cellsom2NPlus         = []
        interpolated_cellsom1OPlus         = [];    interpolated_cellsom2OPlus         = []
        interpolated_cellsom1NOPlus        = [];    interpolated_cellsom2NOPlus        = []
        interpolated_cellsom1N2Plus        = [];    interpolated_cellsom2N2Plus        = []
        interpolated_cellsom1O2Plus        = [];    interpolated_cellsom2O2Plus        = []
        interpolated_cellsom1N             = [];    interpolated_cellsom2N             = []
        interpolated_cellsom1O             = [];    interpolated_cellsom2O             = []
        interpolated_cellsom1NO            = [];    interpolated_cellsom2NO            = []
        interpolated_cellsom1N2            = [];    interpolated_cellsom2N2            = []
        interpolated_cellsom1O2            = [];    interpolated_cellsom2O2            = []
        interpolated_cellsom1Ttr           = [];    interpolated_cellsom2Ttr           = []
        interpolated_cellsom1Tve           = [];    interpolated_cellsom2Tve           = []
        interpolated_cellsom1M             = [];    interpolated_cellsom2M             = []

        for i in range(len(cellsom1P)):

            current_arrayCS1P = cellsom1P[i]
            interpolated_cellsom1P.append(interp1d(cellnodes[i], current_arrayCS1P, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2P = cellsom2P[i]
            interpolated_cellsom2P.append(interp1d(cellnodes[i], current_arrayCS2P, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_elecMinus = cellsom1_nd_elecMinus[i]
            interpolated_cellsom1_nd_elecMinus.append(interp1d(cellnodes[i], current_arrayCS1_nd_elecMinus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_elecMinus = cellsom2_nd_elecMinus[i]
            interpolated_cellsom2_nd_elecMinus.append(interp1d(cellnodes[i], current_arrayCS2_nd_elecMinus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_nPlus = cellsom1_nd_nPlus[i]
            interpolated_cellsom1_nd_nPlus.append(interp1d(cellnodes[i], current_arrayCS1_nd_nPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_nPlus = cellsom2_nd_nPlus[i]
            interpolated_cellsom2_nd_nPlus.append(interp1d(cellnodes[i], current_arrayCS2_nd_nPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_oPlus = cellsom1_nd_oPlus[i]
            interpolated_cellsom1_nd_oPlus.append(interp1d(cellnodes[i], current_arrayCS1_nd_oPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_oPlus = cellsom2_nd_oPlus[i]
            interpolated_cellsom2_nd_oPlus.append(interp1d(cellnodes[i], current_arrayCS2_nd_oPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_noPlus = cellsom1_nd_noPlus[i]
            interpolated_cellsom1_nd_noPlus.append(interp1d(cellnodes[i], current_arrayCS1_nd_noPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_noPlus = cellsom2_nd_noPlus[i]
            interpolated_cellsom2_nd_noPlus.append(interp1d(cellnodes[i], current_arrayCS2_nd_noPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_n2Plus = cellsom1_nd_n2Plus[i]
            interpolated_cellsom1_nd_n2Plus.append(interp1d(cellnodes[i], current_arrayCS1_nd_n2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_n2Plus = cellsom2_nd_n2Plus[i]
            interpolated_cellsom2_nd_n2Plus.append(interp1d(cellnodes[i], current_arrayCS2_nd_n2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_o2Plus = cellsom1_nd_o2Plus[i]
            interpolated_cellsom1_nd_o2Plus.append(interp1d(cellnodes[i], current_arrayCS1_nd_o2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_o2Plus = cellsom2_nd_o2Plus[i]
            interpolated_cellsom2_nd_o2Plus.append(interp1d(cellnodes[i], current_arrayCS2_nd_o2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1elecMinus = cellsom1elecMinus[i]
            interpolated_cellsom1elecMinus.append(interp1d(cellnodes[i], current_arrayCS1elecMinus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2elecMinus = cellsom2elecMinus[i]
            interpolated_cellsom2elecMinus.append(interp1d(cellnodes[i], current_arrayCS2elecMinus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1NPlus = cellsom1NPlus[i]
            interpolated_cellsom1NPlus.append(interp1d(cellnodes[i], current_arrayCS1NPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2NPlus = cellsom2NPlus[i]
            interpolated_cellsom2NPlus.append(interp1d(cellnodes[i], current_arrayCS2NPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1OPlus = cellsom1OPlus[i]
            interpolated_cellsom1OPlus.append(interp1d(cellnodes[i], current_arrayCS1OPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2OPlus = cellsom2OPlus[i]
            interpolated_cellsom2OPlus.append(interp1d(cellnodes[i], current_arrayCS2OPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1NOPlus = cellsom1NOPlus[i]
            interpolated_cellsom1NOPlus.append(interp1d(cellnodes[i], current_arrayCS1NOPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2NOPlus = cellsom2NOPlus[i]
            interpolated_cellsom2NOPlus.append(interp1d(cellnodes[i], current_arrayCS2NOPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1N2Plus = cellsom1N2Plus[i]
            interpolated_cellsom1N2Plus.append(interp1d(cellnodes[i], current_arrayCS1N2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2N2Plus = cellsom2N2Plus[i]
            interpolated_cellsom2N2Plus.append(interp1d(cellnodes[i], current_arrayCS2N2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1O2Plus = cellsom1O2Plus[i]
            interpolated_cellsom1O2Plus.append(interp1d(cellnodes[i], current_arrayCS1O2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2O2Plus = cellsom2O2Plus[i]
            interpolated_cellsom2O2Plus.append(interp1d(cellnodes[i], current_arrayCS2O2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            
            current_arrayCS1N = cellsom1N[i]
            interpolated_cellsom1N.append(interp1d(cellnodes[i], current_arrayCS1N, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2N = cellsom2N[i]
            interpolated_cellsom2N.append(interp1d(cellnodes[i], current_arrayCS2N, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1O = cellsom1O[i]
            interpolated_cellsom1O.append(interp1d(cellnodes[i], current_arrayCS1O, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2O = cellsom2O[i]
            interpolated_cellsom2O.append(interp1d(cellnodes[i], current_arrayCS2O, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1NO = cellsom1NO[i]
            interpolated_cellsom1NO.append(interp1d(cellnodes[i], current_arrayCS1NO, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2NO = cellsom2NO[i]
            interpolated_cellsom2NO.append(interp1d(cellnodes[i], current_arrayCS2NO, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1N2 = cellsom1N2[i]
            interpolated_cellsom1N2.append(interp1d(cellnodes[i], current_arrayCS1N2, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2N2 = cellsom2N2[i]
            interpolated_cellsom2N2.append(interp1d(cellnodes[i], current_arrayCS2N2, kind='linear', fill_value='extrapolate')(coarsest_grid))
            
            current_arrayCS1O2 = cellsom1O2[i]
            interpolated_cellsom1O2.append(interp1d(cellnodes[i], current_arrayCS1O2, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2O2 = cellsom2O2[i]
            interpolated_cellsom2O2.append(interp1d(cellnodes[i], current_arrayCS2O2, kind='linear', fill_value='extrapolate')(coarsest_grid))
                       
            current_arrayCS1Ttr = cellsom1Ttr[i]
            interpolated_cellsom1Ttr.append(interp1d(cellnodes[i], current_arrayCS1Ttr, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2Ttr = cellsom2Ttr[i]
            interpolated_cellsom2Ttr.append(interp1d(cellnodes[i], current_arrayCS2Ttr, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1Tve = cellsom1Tve[i]
            interpolated_cellsom1Tve.append(interp1d(cellnodes[i], current_arrayCS1Tve, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2Tve = cellsom2Tve[i]
            interpolated_cellsom2Tve.append(interp1d(cellnodes[i], current_arrayCS2Tve, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1M = cellsom1M[i]
            interpolated_cellsom1M.append(interp1d(cellnodes[i], current_arrayCS1M, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2M = cellsom2M[i]
            interpolated_cellsom2M.append(interp1d(cellnodes[i], current_arrayCS2M, kind='linear', fill_value='extrapolate')(coarsest_grid))     
                        
        # Storing the lists of interpolated values as matrices
        CS1P_matrix             = np.transpose(np.array(interpolated_cellsom1P));               CS2P_matrix             = np.transpose(np.array(interpolated_cellsom2P))
        CS1_nd_elecMinus_matrix = np.transpose(np.array(interpolated_cellsom1_nd_elecMinus));   CS2_nd_elecMinus_matrix = np.transpose(np.array(interpolated_cellsom2_nd_elecMinus))
        CS1_nd_nPlus_matrix     = np.transpose(np.array(interpolated_cellsom1_nd_nPlus));       CS2_nd_nPlus_matrix     = np.transpose(np.array(interpolated_cellsom2_nd_nPlus))
        CS1_nd_oPlus_matrix     = np.transpose(np.array(interpolated_cellsom1_nd_oPlus));       CS2_nd_oPlus_matrix     = np.transpose(np.array(interpolated_cellsom2_nd_oPlus))
        CS1_nd_noPlus_matrix    = np.transpose(np.array(interpolated_cellsom1_nd_noPlus));      CS2_nd_noPlus_matrix    = np.transpose(np.array(interpolated_cellsom2_nd_noPlus))
        CS1_nd_n2Plus_matrix    = np.transpose(np.array(interpolated_cellsom1_nd_n2Plus));      CS2_nd_n2Plus_matrix    = np.transpose(np.array(interpolated_cellsom2_nd_n2Plus))
        CS1_nd_o2Plus_matrix    = np.transpose(np.array(interpolated_cellsom1_nd_o2Plus));      CS2_nd_o2Plus_matrix    = np.transpose(np.array(interpolated_cellsom2_nd_o2Plus))
        CS1elecMinus_matrix     = np.transpose(np.array(interpolated_cellsom1elecMinus));       CS2elecMinus_matrix     = np.transpose(np.array(interpolated_cellsom2elecMinus))
        CS1NPlus_matrix         = np.transpose(np.array(interpolated_cellsom1NPlus));           CS2NPlus_matrix         = np.transpose(np.array(interpolated_cellsom2NPlus))
        CS1OPlus_matrix         = np.transpose(np.array(interpolated_cellsom1OPlus));           CS2OPlus_matrix         = np.transpose(np.array(interpolated_cellsom2OPlus))
        CS1NOPlus_matrix        = np.transpose(np.array(interpolated_cellsom1NOPlus));          CS2NOPlus_matrix        = np.transpose(np.array(interpolated_cellsom2NOPlus))
        CS1N2Plus_matrix        = np.transpose(np.array(interpolated_cellsom1N2Plus));          CS2N2Plus_matrix        = np.transpose(np.array(interpolated_cellsom2N2Plus))
        CS1O2Plus_matrix        = np.transpose(np.array(interpolated_cellsom1O2Plus));          CS2O2Plus_matrix        = np.transpose(np.array(interpolated_cellsom2O2Plus))
        CS1N_matrix             = np.transpose(np.array(interpolated_cellsom1N));               CS2N_matrix             = np.transpose(np.array(interpolated_cellsom2N))
        CS1O_matrix             = np.transpose(np.array(interpolated_cellsom1O));               CS2O_matrix             = np.transpose(np.array(interpolated_cellsom2O))
        CS1NO_matrix            = np.transpose(np.array(interpolated_cellsom1NO));              CS2NO_matrix            = np.transpose(np.array(interpolated_cellsom2NO))
        CS1N2_matrix            = np.transpose(np.array(interpolated_cellsom1N2));              CS2N2_matrix            = np.transpose(np.array(interpolated_cellsom2N2))
        CS1O2_matrix            = np.transpose(np.array(interpolated_cellsom1O2));              CS2O2_matrix            = np.transpose(np.array(interpolated_cellsom2O2))
        CS1Ttr_matrix           = np.transpose(np.array(interpolated_cellsom1Ttr));             CS2Ttr_matrix           = np.transpose(np.array(interpolated_cellsom2Ttr))
        CS1Tve_matrix           = np.transpose(np.array(interpolated_cellsom1Tve));             CS2Tve_matrix           = np.transpose(np.array(interpolated_cellsom2Tve))
        CS1M_matrix             = np.transpose(np.array(interpolated_cellsom1M));               CS2M_matrix             = np.transpose(np.array(interpolated_cellsom2M))

        # Compute absolute average, variance and cost
        ml = np.abs(np.max(CS1P_matrix / Nl, axis=0)) # L-infinity norm of the average of each level
        Vl = np.maximum(0, np.max(CS2P_matrix / Nl - (CS1P_matrix / Nl)**2, axis=0))
        Cl = costl / Nl

        ml_vec = CS1P_matrix / Nl
        Vl_vec = CS2P_matrix / Nl - ml_vec**2
        Vl_vec[Vl_vec < 0] = 0
                
        # Other quantities mean and variance
        ml_nd_elecMinus_vec = CS1_nd_elecMinus_matrix / Nl
        Vl_nd_elecMinus_vec = CS2_nd_elecMinus_matrix / Nl - ml_nd_elecMinus_vec**2
        Vl_nd_elecMinus_vec[Vl_nd_elecMinus_vec < 0] = 0

        ml_nd_nPlus_vec = CS1_nd_nPlus_matrix / Nl
        Vl_nd_nPlus_vec = CS2_nd_nPlus_matrix / Nl - ml_nd_nPlus_vec**2
        Vl_nd_nPlus_vec[Vl_nd_nPlus_vec < 0] = 0

        ml_nd_oPlus_vec = CS1_nd_oPlus_matrix / Nl
        Vl_nd_oPlus_vec = CS2_nd_oPlus_matrix / Nl - ml_nd_oPlus_vec**2
        Vl_nd_oPlus_vec[Vl_nd_oPlus_vec < 0] = 0

        ml_nd_noPlus_vec = CS1_nd_noPlus_matrix / Nl
        Vl_nd_noPlus_vec = CS2_nd_noPlus_matrix / Nl - ml_nd_noPlus_vec**2
        Vl_nd_noPlus_vec[Vl_nd_noPlus_vec < 0] = 0

        ml_nd_n2Plus_vec = CS1_nd_n2Plus_matrix / Nl
        Vl_nd_n2Plus_vec = CS2_nd_n2Plus_matrix / Nl - ml_nd_n2Plus_vec**2
        Vl_nd_n2Plus_vec[Vl_nd_n2Plus_vec < 0] = 0

        ml_nd_o2Plus_vec = CS1_nd_o2Plus_matrix / Nl
        Vl_nd_o2Plus_vec = CS2_nd_o2Plus_matrix / Nl - ml_nd_o2Plus_vec**2
        Vl_nd_o2Plus_vec[Vl_nd_o2Plus_vec < 0] = 0

        mlElecMinus_vec = CS1elecMinus_matrix / Nl
        VlElecMinus_vec = CS2elecMinus_matrix / Nl - mlElecMinus_vec**2
        VlElecMinus_vec[VlElecMinus_vec < 0] = 0

        mlNPlus_vec = CS1NPlus_matrix / Nl
        VlNPlus_vec = CS2NPlus_matrix / Nl - mlNPlus_vec**2
        VlNPlus_vec[VlNPlus_vec < 0] = 0

        mlOPlus_vec = CS1OPlus_matrix / Nl
        VlOPlus_vec = CS2OPlus_matrix / Nl - mlOPlus_vec**2
        VlOPlus_vec[VlOPlus_vec < 0] = 0

        mlNOPlus_vec = CS1NOPlus_matrix / Nl
        VlNOPlus_vec = CS2NOPlus_matrix / Nl - mlNOPlus_vec**2
        VlNOPlus_vec[VlNOPlus_vec < 0] = 0

        mlN2Plus_vec = CS1N2Plus_matrix / Nl
        VlN2Plus_vec = CS2N2Plus_matrix / Nl - mlN2Plus_vec**2
        VlN2Plus_vec[VlN2Plus_vec < 0] = 0

        mlO2Plus_vec = CS1O2Plus_matrix / Nl
        VlO2Plus_vec = CS2O2Plus_matrix / Nl - mlO2Plus_vec**2
        VlO2Plus_vec[VlO2Plus_vec < 0] = 0

        mlN_vec = CS1N_matrix / Nl
        VlN_vec = CS2N_matrix / Nl - mlN_vec**2
        VlN_vec[VlN_vec < 0] = 0

        mlO_vec = CS1O_matrix / Nl
        VlO_vec = CS2O_matrix / Nl - mlO_vec**2
        VlO_vec[VlO_vec < 0] = 0

        mlNO_vec = CS1NO_matrix / Nl
        VlNO_vec = CS2NO_matrix / Nl - mlNO_vec**2
        VlNO_vec[VlNO_vec < 0] = 0

        mlN2_vec = CS1N2_matrix / Nl
        VlN2_vec = CS2N2_matrix / Nl - mlN2_vec**2
        VlN2_vec[VlN2_vec < 0] = 0

        mlO2_vec = CS1O2_matrix / Nl
        VlO2_vec = CS2O2_matrix / Nl - mlO2_vec**2
        VlO2_vec[VlO2_vec < 0] = 0

        mlTtr_vec = CS1Ttr_matrix / Nl
        VlTtr_vec = CS2Ttr_matrix / Nl - mlTtr_vec**2
        VlTtr_vec[VlTtr_vec < 0] = 0

        mlTve_vec = CS1Tve_matrix / Nl
        VlTve_vec = CS2Tve_matrix / Nl - mlTve_vec**2
        VlTve_vec[VlTve_vec < 0] = 0

        mlM_vec = CS1M_matrix / Nl
        VlM_vec = CS2M_matrix / Nl - mlM_vec**2
        VlM_vec[VlM_vec < 0] = 0

        # Set optimal number of additional samples
        Ns = np.ceil(np.sqrt(Vl / Cl) * np.sum(np.sqrt(Vl * Cl)) / ((1 - theta) * eps**2))
        dNl = np.maximum(0, Ns - Nl)

        dNl_str = str(dNl)
        write(Nlfile, f"dNl = {dNl_str}\n")

        # If (almost) converged, estimate remaining error and decide whether a new level is required
        if np.sum(dNl > 0.01 * Nl) == 0:

            rem = ml[L] / (2**alpha - 1)

            if rem > np.sqrt(theta) * eps:

                if L == Lmax:
                    write(Nlfile, "*** failed to achieve weak convergence ***\n")

                else:
                    # add another level
                    L += 1

                    # appending asymptotic estimates of Vl, Cl from weak convergence conditions to compute a preliminar Ns
                    Vl = np.append(Vl, Vl[-1] / 2**beta)
                    Vl_vec = np.append(Vl_vec, (Vl_vec[:, L-1] / (2 ** beta)).reshape(-1, 1), axis=1)
                    Cl = np.append(Cl, Cl[-1] * 2**gamma)

                    Nl = np.append(Nl, 0.0)
                    
                    # adding elements to list for new level quantities
                    cellsom1P.append(None);                 cellsom2P.append(None)
                    cellsom1_nd_elecMinus.append(None);     cellsom2_nd_elecMinus.append(None)
                    cellsom1_nd_nPlus.append(None);         cellsom2_nd_nPlus.append(None)
                    cellsom1_nd_oPlus.append(None);         cellsom2_nd_oPlus.append(None)
                    cellsom1_nd_noPlus.append(None);        cellsom2_nd_noPlus.append(None)
                    cellsom1_nd_n2Plus.append(None);        cellsom2_nd_n2Plus.append(None)
                    cellsom1_nd_o2Plus.append(None);        cellsom2_nd_o2Plus.append(None)
                    cellsom1elecMinus.append(None);         cellsom2elecMinus.append(None)
                    cellsom1NPlus.append(None);             cellsom2NPlus.append(None)
                    cellsom1OPlus.append(None);             cellsom2OPlus.append(None)
                    cellsom1NOPlus.append(None);            cellsom2NOPlus.append(None)
                    cellsom1N2Plus.append(None);            cellsom2N2Plus.append(None)
                    cellsom1O2Plus.append(None);            cellsom2O2Plus.append(None)
                    cellsom1N.append(None);                 cellsom2N.append(None)
                    cellsom1O.append(None);                 cellsom2O.append(None)
                    cellsom1NO.append(None);                cellsom2NO.append(None)
                    cellsom1N2.append(None);                cellsom2N2.append(None)
                    cellsom1O2.append(None);                cellsom2O2.append(None)
                    cellsom1Ttr.append(None);               cellsom2Ttr.append(None)
                    cellsom1Tve.append(None);               cellsom2Tve.append(None)
                    cellsom1M.append(None);                 cellsom2M.append(None)
                       
                    cellnodes.append(None)

                    costl = np.append(costl, 0)

                    # computing initial optimal samples number for new level
                    Ns = np.ceil(np.sqrt(Vl / Cl) * np.sum(np.sqrt(Vl * Cl)) / ((1 - theta) * eps**2))

                    dNl = np.maximum(0, Ns - Nl)

    # Evaluate multilevel estimator
    P = np.sum(np.mean(ml_vec, axis=0)) # mean is for the columns and sum is for the rows: it represents the mean of the solution vector

    # Compute vector QoIs sums of the mean and uncertainty bounds
    sum_ml = np.sum(ml_vec, axis=1) # sum the averages by rows
    sum_Vl = np.sum(Vl_vec, axis=1)
    upper_bound = sum_ml + np.sqrt(np.abs(sum_Vl))
    lower_bound = sum_ml - np.sqrt(np.abs(sum_Vl))

    sum_ml_nd_elecMinus = np.sum(ml_nd_elecMinus_vec, axis=1)
    sum_Vl_nd_elecMinus = np.sum(Vl_nd_elecMinus_vec, axis=1)
    upper_bound_nd_elecMinus = sum_ml_nd_elecMinus + np.sqrt(np.abs(sum_Vl_nd_elecMinus))
    lower_bound_nd_elecMinus = sum_ml_nd_elecMinus - np.sqrt(np.abs(sum_Vl_nd_elecMinus))

    sum_ml_nd_nPlus = np.sum(ml_nd_nPlus_vec, axis=1)
    sum_Vl_nd_nPlus = np.sum(Vl_nd_nPlus_vec, axis=1)
    upper_bound_nd_nPlus = sum_ml_nd_nPlus + np.sqrt(np.abs(sum_Vl_nd_nPlus))
    lower_bound_nd_nPlus = sum_ml_nd_nPlus - np.sqrt(np.abs(sum_Vl_nd_nPlus))

    sum_ml_nd_oPlus = np.sum(ml_nd_oPlus_vec, axis=1)
    sum_Vl_nd_oPlus = np.sum(Vl_nd_oPlus_vec, axis=1)
    upper_bound_nd_oPlus = sum_ml_nd_oPlus + np.sqrt(np.abs(sum_Vl_nd_oPlus))
    lower_bound_nd_oPlus = sum_ml_nd_oPlus - np.sqrt(np.abs(sum_Vl_nd_oPlus))

    sum_ml_nd_noPlus = np.sum(ml_nd_noPlus_vec, axis=1)
    sum_Vl_nd_noPlus = np.sum(Vl_nd_noPlus_vec, axis=1)
    upper_bound_nd_noPlus = sum_ml_nd_noPlus + np.sqrt(np.abs(sum_Vl_nd_noPlus))
    lower_bound_nd_noPlus = sum_ml_nd_noPlus - np.sqrt(np.abs(sum_Vl_nd_noPlus))

    sum_ml_nd_n2Plus = np.sum(ml_nd_n2Plus_vec, axis=1)
    sum_Vl_nd_n2Plus = np.sum(Vl_nd_n2Plus_vec, axis=1)
    upper_bound_nd_n2Plus = sum_ml_nd_n2Plus + np.sqrt(np.abs(sum_Vl_nd_n2Plus))
    lower_bound_nd_n2Plus = sum_ml_nd_n2Plus - np.sqrt(np.abs(sum_Vl_nd_n2Plus))

    sum_ml_nd_o2Plus = np.sum(ml_nd_o2Plus_vec, axis=1)
    sum_Vl_nd_o2Plus = np.sum(Vl_nd_o2Plus_vec, axis=1)
    upper_bound_nd_o2Plus = sum_ml_nd_o2Plus + np.sqrt(np.abs(sum_Vl_nd_o2Plus))
    lower_bound_nd_o2Plus = sum_ml_nd_o2Plus - np.sqrt(np.abs(sum_Vl_nd_o2Plus))

    sum_mlElecMinus = np.sum(mlElecMinus_vec, axis=1)
    sum_VlElecMinus = np.sum(VlElecMinus_vec, axis=1)
    upper_boundElecMinus = sum_mlElecMinus + np.sqrt(np.abs(sum_VlElecMinus))
    lower_boundElecMinus = sum_mlElecMinus - np.sqrt(np.abs(sum_VlElecMinus))

    sum_mlNPlus = np.sum(mlNPlus_vec, axis=1)
    sum_VlNPlus = np.sum(VlNPlus_vec, axis=1)
    upper_boundNPlus = sum_mlNPlus + np.sqrt(np.abs(sum_VlNPlus))
    lower_boundNPlus = sum_mlNPlus - np.sqrt(np.abs(sum_VlNPlus))

    sum_mlOPlus = np.sum(mlOPlus_vec, axis=1)
    sum_VlOPlus = np.sum(VlOPlus_vec, axis=1)
    upper_boundOPlus = sum_mlOPlus + np.sqrt(np.abs(sum_VlOPlus))
    lower_boundOPlus = sum_mlOPlus - np.sqrt(np.abs(sum_VlOPlus))

    sum_mlNOPlus = np.sum(mlNOPlus_vec, axis=1)
    sum_VlNOPlus = np.sum(VlNOPlus_vec, axis=1)
    upper_boundNOPlus = sum_mlNOPlus + np.sqrt(np.abs(sum_VlNOPlus))
    lower_boundNOPlus = sum_mlNOPlus - np.sqrt(np.abs(sum_VlNOPlus))

    sum_mlN2Plus = np.sum(mlN2Plus_vec, axis=1)
    sum_VlN2Plus = np.sum(VlN2Plus_vec, axis=1)
    upper_boundN2Plus = sum_mlN2Plus + np.sqrt(np.abs(sum_VlN2Plus))
    lower_boundN2Plus = sum_mlN2Plus - np.sqrt(np.abs(sum_VlN2Plus))

    sum_mlO2Plus = np.sum(mlO2Plus_vec, axis=1)
    sum_VlO2Plus = np.sum(VlO2Plus_vec, axis=1)
    upper_boundO2Plus = sum_mlO2Plus + np.sqrt(np.abs(sum_VlO2Plus))
    lower_boundO2Plus = sum_mlO2Plus - np.sqrt(np.abs(sum_VlO2Plus))
    
    sum_mlN = np.sum(mlN_vec, axis=1)
    sum_VlN = np.sum(VlN_vec, axis=1)
    upper_boundN = sum_mlN + np.sqrt(np.abs(sum_VlN))
    lower_boundN = sum_mlN - np.sqrt(np.abs(sum_VlN))    

    sum_mlO = np.sum(mlO_vec, axis=1)
    sum_VlO = np.sum(VlO_vec, axis=1)
    upper_boundO = sum_mlO + np.sqrt(np.abs(sum_VlO))
    lower_boundO = sum_mlO - np.sqrt(np.abs(sum_VlO))    

    sum_mlNO = np.sum(mlNO_vec, axis=1)
    sum_VlNO = np.sum(VlNO_vec, axis=1)
    upper_boundNO = sum_mlNO + np.sqrt(np.abs(sum_VlNO))
    lower_boundNO = sum_mlNO - np.sqrt(np.abs(sum_VlNO))    

    sum_mlN2 = np.sum(mlN2_vec, axis=1)
    sum_VlN2 = np.sum(VlN2_vec, axis=1)
    upper_boundN2 = sum_mlN2 + np.sqrt(np.abs(sum_VlN2))
    lower_boundN2 = sum_mlN2 - np.sqrt(np.abs(sum_VlN2))    

    sum_mlO2 = np.sum(mlO2_vec, axis=1)
    sum_VlO2 = np.sum(VlO2_vec, axis=1)
    upper_boundO2 = sum_mlO2 + np.sqrt(np.abs(sum_VlO2))
    lower_boundO2 = sum_mlO2 - np.sqrt(np.abs(sum_VlO2))    

    sum_mlTtr = np.sum(mlTtr_vec, axis=1)
    sum_VlTtr = np.sum(VlTtr_vec, axis=1)
    upper_boundTtr = sum_mlTtr + np.sqrt(np.abs(sum_VlTtr))
    lower_boundTtr = sum_mlTtr - np.sqrt(np.abs(sum_VlTtr))    

    sum_mlTve = np.sum(mlTve_vec, axis=1)
    sum_VlTve = np.sum(VlTve_vec, axis=1)
    upper_boundTve = sum_mlTve + np.sqrt(np.abs(sum_VlTve))
    lower_boundTve = sum_mlTve - np.sqrt(np.abs(sum_VlTve))    

    sum_mlM = np.sum(mlM_vec, axis=1)
    sum_VlM = np.sum(VlM_vec, axis=1)
    upper_boundM = sum_mlM + np.sqrt(np.abs(sum_VlM))
    lower_boundM = sum_mlM - np.sqrt(np.abs(sum_VlM))    
    
    # Plot 1: normalized mean pressure, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax = plt.gca()  
    ax.fill_between(coarsest_grid, upper_bound, lower_bound, color=shaded_color, alpha=0.7, edgecolor='none')
    ax.plot(coarsest_grid, sum_ml)
    ax.set_title(r'P with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  # Convert eps array to string and replace dots with underscore
    ax.figure.savefig(f'1_P_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'1_P_epsilon_{eps_str}.pkl'
    
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_bound, lower_bound, sum_ml), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()
    
    # Plot 2: mean number density for e-, 1sigma uncertainty region
    ax2 = plt.gca()
    ax2.fill_between(coarsest_grid, upper_bound_nd_elecMinus, lower_bound_nd_elecMinus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax2.plot(coarsest_grid, sum_ml_nd_elecMinus)
    ax2.set_title(r'Number density e- with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax2.figure.savefig(f'2_nd_elecMinus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'2_nd_elecMinus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax2, coarsest_grid, upper_bound_nd_elecMinus, lower_bound_nd_elecMinus, sum_ml_nd_elecMinus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 3: mean number density for N+, 1sigma uncertainty region
    ax3 = plt.gca()
    ax3.fill_between(coarsest_grid, upper_bound_nd_nPlus, lower_bound_nd_nPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax3.plot(coarsest_grid, sum_ml_nd_nPlus)
    ax3.set_title(r'Number density N+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax3.figure.savefig(f'3_nd_nPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'3_nd_nPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax3, coarsest_grid, upper_bound_nd_nPlus, lower_bound_nd_nPlus, sum_ml_nd_nPlus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 4: mean number density for O+, 1sigma uncertainty region
    ax4 = plt.gca()
    ax4.fill_between(coarsest_grid, upper_bound_nd_oPlus, lower_bound_nd_oPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax4.plot(coarsest_grid, sum_ml_nd_oPlus)
    ax4.set_title(r'Number density O+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax4.figure.savefig(f'4_nd_oPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'4_nd_oPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax4, coarsest_grid, upper_bound_nd_oPlus, lower_bound_nd_oPlus, sum_ml_nd_oPlus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 5: mean number density for NO+, 1sigma uncertainty region
    ax5 = plt.gca()
    ax5.fill_between(coarsest_grid, upper_bound_nd_noPlus, lower_bound_nd_noPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax5.plot(coarsest_grid, sum_ml_nd_noPlus)
    ax5.set_title(r'Number density NO+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax5.figure.savefig(f'5_nd_noPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'5_nd_noPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax5, coarsest_grid, upper_bound_nd_noPlus, lower_bound_nd_noPlus, sum_ml_nd_noPlus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 6: mean number density for N2+, 1sigma uncertainty region
    ax6 = plt.gca()
    ax6.fill_between(coarsest_grid, upper_bound_nd_n2Plus, lower_bound_nd_n2Plus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax6.plot(coarsest_grid, sum_ml_nd_n2Plus)
    ax6.set_title(r'Number density N2+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax6.figure.savefig(f'6_nd_n2Plus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'6_nd_n2Plus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax6, coarsest_grid, upper_bound_nd_n2Plus, lower_bound_nd_n2Plus, sum_ml_nd_n2Plus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 7: mean number density for O2+, 1sigma uncertainty region
    ax7 = plt.gca()
    ax7.fill_between(coarsest_grid, upper_bound_nd_o2Plus, lower_bound_nd_o2Plus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax7.plot(coarsest_grid, sum_ml_nd_o2Plus)
    ax7.set_title(r'Number density O2+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax7.figure.savefig(f'7_nd_o2Plus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'7_nd_o2Plus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax7, coarsest_grid, upper_bound_nd_o2Plus, lower_bound_nd_o2Plus, sum_ml_nd_o2Plus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()
    
    # Plot 8: e- gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax8 = plt.gca()  # Get the current axis
    ax8.fill_between(coarsest_grid, upper_boundElecMinus, lower_boundElecMinus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax8.plot(coarsest_grid, sum_mlElecMinus)
    ax8.set_title(r'ElecMinus with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax8.figure.savefig(f'8_ElecMinus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename_ElecMinus = f'8_ElecMinus_epsilon_{eps_str}.pkl'

    with open(plot_filename_ElecMinus, 'wb') as f_ElecMinus:
        pickle.dump((ax8, coarsest_grid, upper_boundElecMinus, lower_boundElecMinus, sum_mlElecMinus), f_ElecMinus)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 9: N+ gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax9 = plt.gca()  # Get the current axis
    ax9.fill_between(coarsest_grid, upper_boundNPlus, lower_boundNPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax9.plot(coarsest_grid, sum_mlNPlus)
    ax9.set_title(r'NPlus with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax9.figure.savefig(f'9_NPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename_NPlus = f'9_NPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename_NPlus, 'wb') as f_NPlus:
        pickle.dump((ax9, coarsest_grid, upper_boundNPlus, lower_boundNPlus, sum_mlNPlus), f_NPlus)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 10: O+ gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax10 = plt.gca()  # Get the current axis
    ax10.fill_between(coarsest_grid, upper_boundOPlus, lower_boundOPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax10.plot(coarsest_grid, sum_mlOPlus)
    ax10.set_title(r'OPlus with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax10.figure.savefig(f'10_OPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename_OPlus = f'10_OPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename_OPlus, 'wb') as f_OPlus:
        pickle.dump((ax10, coarsest_grid, upper_boundOPlus, lower_boundOPlus, sum_mlOPlus), f_OPlus)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 11: NO+ gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax11 = plt.gca()  # Get the current axis
    ax11.fill_between(coarsest_grid, upper_boundNOPlus, lower_boundNOPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax11.plot(coarsest_grid, sum_mlNOPlus)
    ax11.set_title(r'NOPlus with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax11.figure.savefig(f'11_NOPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename_NOPlus = f'11_NOPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename_NOPlus, 'wb') as f_NOPlus:
        pickle.dump((ax11, coarsest_grid, upper_boundNOPlus, lower_boundNOPlus, sum_mlNOPlus), f_NOPlus)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 12: N2+ gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax12 = plt.gca()  # Get the current axis
    ax12.fill_between(coarsest_grid, upper_boundN2Plus, lower_boundN2Plus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax12.plot(coarsest_grid, sum_mlN2Plus)
    ax12.set_title(r'N2Plus with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax12.figure.savefig(f'12_N2Plus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename_N2Plus = f'12_N2Plus_epsilon_{eps_str}.pkl'

    with open(plot_filename_N2Plus, 'wb') as f_N2Plus:
        pickle.dump((ax12, coarsest_grid, upper_boundN2Plus, lower_boundN2Plus, sum_mlN2Plus), f_N2Plus)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 13: O2+ gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax13 = plt.gca()  # Get the current axis
    ax13.fill_between(coarsest_grid, upper_boundO2Plus, lower_boundO2Plus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax13.plot(coarsest_grid, sum_mlO2Plus)
    ax13.set_title(r'O2Plus with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax13.figure.savefig(f'13_O2Plus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename_O2Plus = f'13_O2Plus_epsilon_{eps_str}.pkl'

    with open(plot_filename_O2Plus, 'wb') as f_O2Plus:
        pickle.dump((ax13, coarsest_grid, upper_boundO2Plus, lower_boundO2Plus, sum_mlO2Plus), f_O2Plus)

    plt.close("all")
    plt.cla()
    plt.clf()    
    
    # Plot 14: N gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax14 = plt.gca()  # Get the current axis
    ax14.fill_between(coarsest_grid, upper_boundN, lower_boundN, color=shaded_color, alpha=0.7, edgecolor='none')
    ax14.plot(coarsest_grid, sum_mlN)
    ax14.set_title(r'N with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax14.figure.savefig(f'14_N_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'14_N_epsilon_{eps_str}.pkl'
       
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax14, coarsest_grid, upper_boundN, lower_boundN, sum_mlN), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()    
    
    # Plot 15: O gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax15 = plt.gca()  # Get the current axis
    ax15.fill_between(coarsest_grid, upper_boundO, lower_boundO, color=shaded_color, alpha=0.7, edgecolor='none')
    ax15.plot(coarsest_grid, sum_mlO)
    ax15.set_title(r'O with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax15.figure.savefig(f'15_O_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'15_O_epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax15, coarsest_grid, upper_boundO, lower_boundO, sum_mlO), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()    
    
    # Plot 16: NO gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax16 = plt.gca()  # Get the current axis
    ax16.fill_between(coarsest_grid, upper_boundNO, lower_boundNO, color=shaded_color, alpha=0.7, edgecolor='none')
    ax16.plot(coarsest_grid, sum_mlNO)
    ax16.set_title(r'NO with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax16.figure.savefig(f'16_NO_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'16_NO_epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax16, coarsest_grid, upper_boundNO, lower_boundNO, sum_mlNO), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()      
    
    # Plot 17: N2 gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax17 = plt.gca()  # Get the current axis
    ax17.fill_between(coarsest_grid, upper_boundN2, lower_boundN2, color=shaded_color, alpha=0.7, edgecolor='none')
    ax17.plot(coarsest_grid, sum_mlN2)
    ax17.set_title(r'N2 with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax17.figure.savefig(f'17_N2_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'17_N2_epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax17, coarsest_grid, upper_boundN2, lower_boundN2, sum_mlN2), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()      

    # Plot 18: O2 gas compostion mass fraction, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax18 = plt.gca()  # Get the current axis
    ax18.fill_between(coarsest_grid, upper_boundO2, lower_boundO2, color=shaded_color, alpha=0.7, edgecolor='none')
    ax18.plot(coarsest_grid, sum_mlO2)
    ax18.set_title(r'O2 with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax18.figure.savefig(f'18_O2_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'18_O2_epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax18, coarsest_grid, upper_boundO2, lower_boundO2, sum_mlO2), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()      

    # Plot 19: Normalized rototranslational temperature, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax19 = plt.gca()  # Get the current axis
    ax19.fill_between(coarsest_grid, upper_boundTtr, lower_boundTtr, color=shaded_color, alpha=0.7, edgecolor='none')
    ax19.plot(coarsest_grid, sum_mlTtr)
    ax19.set_title(r'Ttr with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax19.figure.savefig(f'19_Ttr_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'19_Ttr_epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax19, coarsest_grid, upper_boundTtr, lower_boundTtr, sum_mlTtr), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()               
    
    # Plot 20: Normalized vibrational temperature, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax20 = plt.gca()  # Get the current axis
    ax20.fill_between(coarsest_grid, upper_boundTve, lower_boundTve, color=shaded_color, alpha=0.7, edgecolor='none')
    ax20.plot(coarsest_grid, sum_mlTve)
    ax20.set_title(r'Tve with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax20.figure.savefig(f'20_Tve_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'20_Tve_epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax20, coarsest_grid, upper_boundTve, lower_boundTve, sum_mlTve), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()           
    
    # Plot 21: Normalized mach, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax21 = plt.gca()  # Get the current axis
    ax21.fill_between(coarsest_grid, upper_boundM, lower_boundM, color=shaded_color, alpha=0.7, edgecolor='none')
    ax21.plot(coarsest_grid, sum_mlM)
    ax21.set_title(r'M with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax21.figure.savefig(f'21_M_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'21_M_epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax21, coarsest_grid, upper_boundM, lower_boundM, sum_mlM), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()       
    
    return P, Nl, Cl


def screening(mlmc_l, L, N, logfile, *args):
    '''
    Function to perform the MLMC screening to estimate alpha, beta, gamma (weak convergence rates)

    Inputs:
    - mlmc_l function to perform the MC simulation at a given level
    - L total number of levels
    - N number of samples at each level for the screening
    - logfile

    Outputs:
    - alpha
    - beta 
    - gamma
    '''

    del1 = []
    del2 = []
    var1 = []
    var2 = []
    cost = []

    for l in range(L + 1):
        sums1 = 0; sums2 = 0; sums5 = 0; sums6 = 0
        cst = 0
        
        _, sums1_j, sums2_j, sums5_j, sums6_j, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, cst_j = mlmc_l(l, int(N / 1), *args)

        sums1 += sums1_j / N
        sums2 += sums2_j / N
        sums5 += sums5_j / N
        sums6 += sums6_j / N
        cst += cst_j / N

        cost.append(cst)
        del1.append(np.linalg.norm(sums1, np.inf))  # Ave(Pf-Pc)
        del2.append(np.linalg.norm(sums5, np.inf))  # Ave(Pf)
        var1.append(np.linalg.norm((sums2 - sums1 ** 2), np.inf))  # Var(Pf-Pc)
        var2.append(np.linalg.norm((sums6 - sums5 ** 2), np.inf))  # Var(Pf)
        var2[-1] = max(var2[-1], 1e-10)  

        write(logfile, "%2d  %11.4e %11.4e  %.3e  %.3e %.2e \n" % \
                      (l, del1[l], del2[l], var1[l], var2[l], cst))

    # Linear regression to estimate alpha, beta and gamma
    L1 = 1
    L2 = L + 1
    pa    = np.polyfit(range(L1, L2), np.log2(np.abs(del1[L1:L2])), 1);  alpha = -pa[0]
    pb    = np.polyfit(range(L1, L2), np.log2(np.abs(var1[L1:L2])), 1);  beta  = -pb[0]
    pg    = np.polyfit(range(L1, L2), np.log2(np.abs(cost[L1:L2])), 1);  gamma =  pg[0]

    return del1, del2, var1, var2, cost, alpha, beta, gamma

