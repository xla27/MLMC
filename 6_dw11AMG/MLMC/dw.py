# Author: Nicolò Sarman, 10614962, thesis
# Description: Multi level Monte Carlo, hypersonic flow of air-11 over a 15-45° double wedge in thermal and chemical non equilibrium

import numpy as np
from scipy.interpolate import interp1d
import time
import matplotlib.pyplot as plt
from mlmc_test import mlmc_test
from mlmc_plot import mlmc_plot
from cfd_call_coarse import cfd_call_coarse
from cfd_call_fine import cfd_call_fine

def moving_average(data, window_size):
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1
    padded_data = np.pad(data, (pad_left, pad_right), mode='edge')
    ma_data = np.convolve(padded_data, np.ones(window_size), 'valid') / window_size
    return ma_data

def dw_l(l, N_samples):
    # Variables initialization
    start = time.time()
    sums1 = 0; sums2 = 0; sums5 = 0; sums6 = 0; # P
    sums1ndelecMinus = 0; sums2ndelecMinus = 0; # nd e-
    sums1ndNPlus = 0; sums2ndNPlus = 0; # nd N+
    sums1ndOPlus = 0; sums2ndOPlus = 0; # nd O+
    sums1ndNOPlus = 0; sums2ndNOPlus = 0; # nd NO+
    sums1ndN2Plus = 0; sums2ndN2Plus = 0; # nd N2+
    sums1ndO2Plus = 0; sums2ndO2Plus = 0; # nd O2+   
    sums1Ttr = 0; sums2Ttr = 0; # Ttr
    sums1Tve = 0; sums2Tve = 0; # Tve
    sums1M = 0; sums2M = 0; # M

    for i in range(1, N_samples  + 1):
        # Random variables definition
        Mmean = 9.0; M_max = 9;  M_min = 9; 
        Tmean = 1000; T_max = 1000; T_min = 1000; 
        Pmean = 390; P_max = 390; P_min = 390;
        Bn2mean = 0.79; Bn2_max = 0.79; Bn2_min = 0.79;
        
        mach = M_min + (M_max - M_min) * np.random.rand(); T = T_min + (T_max - T_min) * np.random.rand();
        P = P_min + (P_max - P_min) * np.random.rand(); Bn2 = Bn2_min + (Bn2_max - Bn2_min) * np.random.rand(); Bo2 = 1 - Bn2;
        
        mach = '{:.3f}'.format(mach); T = '{:.1f}'.format(T); P = '{:.1f}'.format(P);
        Bn2 = '{:.3f}'.format(Bn2); Bo2 = '{:.3f}'.format(Bo2);
        
        valIns_M = str(mach); valIns_T = str(T); valIns_P = str(P); valIns_Bn2 = str(Bn2); valIns_Bo2 = str(Bo2);

        SF = 0.0075

        if l == 0:
            # Call to CFD with fine mesh, coarse results set to zero as it is the starting level
            Pc = 0; 
            nd_elecMinusc=0; nd_nPlusc=0; nd_oPlusc=0; nd_noPlusc=0; nd_n2Plusc=0; nd_o2Plusc=0;
            Ttr_ic = 0; Tve_ic = 0; M_ic = 0;
            
            nd_elecMinusf, nd_nPlusf, nd_oPlusf, nd_noPlusf, nd_n2Plusf, nd_o2Plusf, p_if, Ttr_if, Tve_if, M_if, xnodesf = cfd_call_fine(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
            
            xnodesc = xnodesf;  Pf = p_if; ws = max(int(len(Pf) * SF), 1);

            Pf = moving_average(Pf, ws); 
            nd_elecMinusf = moving_average(nd_elecMinusf, ws); nd_nPlusf = moving_average(nd_nPlusf, ws); nd_oPlusf = moving_average(nd_oPlusf, ws); nd_noPlusf = moving_average(nd_noPlusf, ws); 
            nd_n2Plusf = moving_average(nd_n2Plusf, ws); nd_o2Plusf = moving_average(nd_o2Plusf, ws);            
            Ttr_if = moving_average(Ttr_if, ws); Tve_if = moving_average(Tve_if, ws); M_if = moving_average(M_if, ws);   

        else:
            # Call to CFD with coarse mesh
            nd_elecMinusc, nd_nPlusc, nd_oPlusc, nd_noPlusc, nd_n2Plusc, nd_o2Plusc, p_ic, Ttr_ic, Tve_ic, M_ic, xnodesc = cfd_call_coarse(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
            Pc = p_ic; ws = max(int(len(Pc) * SF), 1); 

            Pc = moving_average(Pc, ws); 
            nd_elecMinusc = moving_average(nd_elecMinusc, ws); nd_nPlusc = moving_average(nd_nPlusc, ws); nd_oPlusc = moving_average(nd_oPlusc, ws); nd_noPlusc = moving_average(nd_noPlusc, ws); 
            nd_n2Plusc = moving_average(nd_n2Plusc, ws); nd_o2Plusc = moving_average(nd_o2Plusc, ws);
            Ttr_ic = moving_average(Ttr_ic, ws); Tve_ic = moving_average(Tve_ic, ws); M_ic = moving_average(M_ic, ws);             

            # Call to CFD with fine mesh
            nd_elecMinusf, nd_nPlusf, nd_oPlusf, nd_noPlusf, nd_n2Plusf, nd_o2Plusf, p_if, Ttr_if, Tve_if, M_if, xnodesf = cfd_call_fine(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
            Pf = p_if

            # Results interpolation on the coarser grid            
            Pf = interp1d(xnodesf, Pf, kind='linear', fill_value='extrapolate')(xnodesc); 
            nd_elecMinusf = interp1d(xnodesf, nd_elecMinusf, kind='linear', fill_value='extrapolate')(xnodesc); nd_nPlusf = interp1d(xnodesf, nd_nPlusf, kind='linear', fill_value='extrapolate')(xnodesc); 
            nd_oPlusf = interp1d(xnodesf, nd_oPlusf, kind='linear', fill_value='extrapolate')(xnodesc); nd_noPlusf = interp1d(xnodesf, nd_noPlusf, kind='linear', fill_value='extrapolate')(xnodesc); 
            nd_n2Plusf = interp1d(xnodesf, nd_n2Plusf, kind='linear', fill_value='extrapolate')(xnodesc); nd_o2Plusf = interp1d(xnodesf, nd_o2Plusf, kind='linear', fill_value='extrapolate')(xnodesc);
            Ttr_if = interp1d(xnodesf, Ttr_if, kind='linear', fill_value='extrapolate')(xnodesc); Tve_if = interp1d(xnodesf, Tve_if, kind='linear', fill_value='extrapolate')(xnodesc); 
            M_if = interp1d(xnodesf, M_if, kind='linear', fill_value='extrapolate')(xnodesc); 

            Pf = moving_average(Pf, ws); nd_elecMinusf = moving_average(nd_elecMinusf, ws); nd_nPlusf = moving_average(nd_nPlusf, ws); nd_oPlusf = moving_average(nd_oPlusf, ws); 
            nd_noPlusf = moving_average(nd_noPlusf, ws); nd_n2Plusf = moving_average(nd_n2Plusf, ws); nd_o2Plusf = moving_average(nd_o2Plusf, ws);            
            Ttr_if = moving_average(Ttr_if, ws); Tve_if = moving_average(Tve_if, ws); M_if = moving_average(M_if, ws);   

       
        # Results update
        sums1 += Pf - Pc; sums2 += (Pf - Pc) ** 2; sums5 += Pf; sums6 += Pf ** 2; # P
        
        sums1ndelecMinus += nd_elecMinusf - nd_elecMinusc; sums2ndelecMinus += (nd_elecMinusf - nd_elecMinusc) ** 2; # nd e-
        sums1ndNPlus += nd_nPlusf - nd_nPlusc; sums2ndNPlus += (nd_nPlusf - nd_nPlusc) ** 2; # nd N+
        sums1ndOPlus += nd_oPlusf - nd_oPlusc; sums2ndOPlus += (nd_oPlusf - nd_oPlusc) ** 2; # nd O+
        sums1ndNOPlus += nd_noPlusf - nd_noPlusc; sums2ndNOPlus += (nd_noPlusf - nd_noPlusc) ** 2; # nd NO+
        sums1ndN2Plus += nd_n2Plusf - nd_n2Plusc; sums2ndN2Plus += (nd_n2Plusf - nd_n2Plusc) ** 2; # nd N2+
        sums1ndO2Plus += nd_o2Plusf - nd_o2Plusc; sums2ndO2Plus += (nd_o2Plusf - nd_o2Plusc) ** 2; # nd O2+  
        
        sums1Ttr += Ttr_if - Ttr_ic; sums2Ttr += (Ttr_if - Ttr_ic) ** 2; # Ttr
        sums1Tve += Tve_if - Tve_ic; sums2Tve += (Tve_if - Tve_ic) ** 2;  # Tve
        sums1M += M_if - M_ic; sums2M += (M_if - M_ic) ** 2;  # M
        
        end=time.time() 
        nproc=16
        cost=nproc*(end-start)
        
    return xnodesc, sums1, sums2, sums5, sums6, sums1ndelecMinus, sums2ndelecMinus, sums1ndNPlus, sums2ndNPlus, sums1ndOPlus, sums2ndOPlus, sums1ndNOPlus, sums2ndNOPlus, sums1ndN2Plus, sums2ndN2Plus, sums1ndO2Plus, sums2ndO2Plus, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost


if __name__ == "__main__":
    t_Start = time.time()

    N = 10   # samples for convergence tests
    L = 4       # levels for convergence tests

    N0 = 2      # initial samples on coarsest levels
    Lmin = 2    # minimum refinement level
    Lmax = 4   # maximum refinement level

    Eps = [0.05, 0.06, 0.07]

    filename = "dw.txt"; logfile = open(filename, "w");
    mlmc_test(dw_l, N, L, N0, Eps, Lmin, Lmax, logfile)
    time_elapsed = time.time() - t_Start; logfile.write('\nTime elapsed: {} seconds\n'.format(time_elapsed));
    logfile.close(); del logfile;
    plt.close()
    plt.cla()
    mlmc_plot(filename, nvert=3)
    plt.savefig(filename.replace('.txt', '.svg'))
