# Author: Nicolò Sarman, 10614962, thesis
# Description: Multi level Monte Carlo, hypersonic flow of air-5 over a 15-45° double wedge in thermal and chemical non equilibrium

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
    sums1N = 0; sums2N = 0; # N
    sums1O = 0; sums2O = 0; # O
    sums1NO = 0; sums2NO = 0; # NO
    sums1N2 = 0; sums2N2 = 0; # N2
    sums1O2 = 0; sums2O2 = 0; # O2
    sums1Ttr = 0; sums2Ttr = 0; # Ttr
    sums1Tve = 0; sums2Tve = 0; # Tve
    sums1M = 0; sums2M = 0; # M

    for i in range(1, N_samples  + 1):
        # Random variables definition
        Mmean = 9.0; M_max = 9.5;  M_min = 8.0; 
        Tmean = 1000; T_max = 1050; T_min = 850; 
        Pmean = 390; P_max = 600; P_min = 300;
        Bn2mean = 0.79; Bn2_max = 0.8; Bn2_min = 0.76;
        
        mach = M_min + (M_max - M_min) * np.random.rand(); T = T_min + (T_max - T_min) * np.random.rand();
        P = P_min + (P_max - P_min) * np.random.rand(); Bn2 = Bn2_min + (Bn2_max - Bn2_min) * np.random.rand(); Bo2 = 1 - Bn2;
        
        mach = '{:.3f}'.format(mach); T = '{:.1f}'.format(T); P = '{:.1f}'.format(P);
        Bn2 = '{:.3f}'.format(Bn2); Bo2 = '{:.3f}'.format(Bo2);
        
        valIns_M = str(mach); valIns_T = str(T); valIns_P = str(P); valIns_Bn2 = str(Bn2); valIns_Bo2 = str(Bo2);

        SF = 0.0075

        if l == 0:
            # Call to CFD with fine mesh, coarse results set to zero as it is the starting level
            Pc = 0; beta_nc = 0; beta_oc = 0; beta_noc = 0; beta_n2c = 0; beta_o2c = 0; Ttr_ic = 0; Tve_ic = 0; M_ic = 0;
            beta_nf, beta_of, beta_nof, beta_n2f, beta_o2f, p_if, Ttr_if, Tve_if, M_if, xnodesf = cfd_call_fine(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
            
            xnodesc = xnodesf;  Pf = p_if; ws = max(int(len(Pf) * SF), 1);

            Pf = moving_average(Pf, ws); beta_nf = moving_average(beta_nf, ws); beta_of = moving_average(beta_of, ws);
            beta_nof = moving_average(beta_nof, ws); beta_n2f = moving_average(beta_n2f, ws); beta_o2f = moving_average(beta_o2f, ws);
            Ttr_if = moving_average(Ttr_if, ws); Tve_if = moving_average(Tve_if, ws); M_if = moving_average(M_if, ws);   

        else:
            # Call to CFD with coarse mesh
            beta_nc, beta_oc, beta_noc, beta_n2c, beta_o2c, p_ic, Ttr_ic, Tve_ic, M_ic, xnodesc = cfd_call_coarse(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
            Pc = p_ic; ws = max(int(len(Pc) * SF), 1); 

            Pc = moving_average(Pc, ws); beta_nc = moving_average(beta_nc, ws); beta_oc = moving_average(beta_oc, ws);
            beta_noc = moving_average(beta_noc, ws); beta_n2c = moving_average(beta_n2c, ws); beta_o2c = moving_average(beta_o2c, ws);
            Ttr_ic = moving_average(Ttr_ic, ws); Tve_ic = moving_average(Tve_ic, ws); M_ic = moving_average(M_ic, ws);
    

            # Call to CFD with fine mesh
            beta_nf, beta_of, beta_nof, beta_n2f, beta_o2f, p_if, Ttr_if, Tve_if, M_if, xnodesf = cfd_call_fine(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
            Pf = p_if

            # Results interpolation on the coarser grid            
            Pf = interp1d(xnodesf, Pf, kind='linear', fill_value='extrapolate')(xnodesc); beta_nf = interp1d(xnodesf, beta_nf, kind='linear', fill_value='extrapolate')(xnodesc); 
            beta_of = interp1d(xnodesf, beta_of, kind='linear', fill_value='extrapolate')(xnodesc); beta_nof = interp1d(xnodesf, beta_nof, kind='linear', fill_value='extrapolate')(xnodesc);                   
            beta_n2f = interp1d(xnodesf, beta_n2f, kind='linear', fill_value='extrapolate')(xnodesc); beta_o2f = interp1d(xnodesf, beta_o2f, kind='linear', fill_value='extrapolate')(xnodesc); 
            Ttr_if = interp1d(xnodesf, Ttr_if, kind='linear', fill_value='extrapolate')(xnodesc); Tve_if = interp1d(xnodesf, Tve_if, kind='linear', fill_value='extrapolate')(xnodesc); 
            M_if = interp1d(xnodesf, M_if, kind='linear', fill_value='extrapolate')(xnodesc); 

            Pf = moving_average(Pf, ws); beta_nf = moving_average(beta_nf, ws); beta_of = moving_average(beta_of, ws);
            beta_nof = moving_average(beta_nof, ws); beta_n2f = moving_average(beta_n2f, ws); beta_o2f = moving_average(beta_o2f, ws);
            Ttr_if = moving_average(Ttr_if, ws); Tve_if = moving_average(Tve_if, ws); M_if = moving_average(M_if, ws);   

        
        # Results update
        sums1 += Pf - Pc; sums2 += (Pf - Pc) ** 2; sums5 += Pf; sums6 += Pf ** 2; # P
        sums1N += beta_nf - beta_nc; sums2N += (beta_nf - beta_nc) ** 2; # N
        sums1O += beta_of - beta_oc; sums2O += (beta_of - beta_oc) ** 2; # O
        sums1NO += beta_nof - beta_noc; sums2NO += (beta_nof - beta_noc) ** 2; # NO
        sums1N2 += beta_n2f - beta_n2c; sums2N2 += (beta_n2f - beta_n2c) ** 2; # N2
        sums1O2 += beta_o2f - beta_o2c; sums2O2 += (beta_o2f - beta_o2c) ** 2; # O2
        sums1Ttr += Ttr_if - Ttr_ic; sums2Ttr += (Ttr_if - Ttr_ic) ** 2; # Ttr
        sums1Tve += Tve_if - Tve_ic; sums2Tve += (Tve_if - Tve_ic) ** 2;  # Tve
        sums1M += M_if - M_ic; sums2M += (M_if - M_ic) ** 2;  # M
        
        end=time.time() 
        nproc=16
        cost=nproc*(end-start)
        
    return xnodesc, sums1, sums2, sums5, sums6, sums1N, sums2N, sums1O, sums2O, sums1NO, sums2NO, sums1N2, sums2N2, sums1O2, sums2O2, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost


if __name__ == "__main__":
    t_Start = time.time()

    N = 10   # samples for convergence tests
    L = 4       # levels for convergence tests

    N0 = 2      # initial samples on coarsest levels
    Lmin = 2    # minimum refinement level
    Lmax = 4   # maximum refinement level

    Eps = [0.03, 0.04, 0.07]

    filename = "dw.txt"; logfile = open(filename, "w");
    mlmc_test(dw_l, N, L, N0, Eps, Lmin, Lmax, logfile)
    time_elapsed = time.time() - t_Start; logfile.write('\nTime elapsed: {} seconds\n'.format(time_elapsed));
    logfile.close(); del logfile;
    plt.close()
    plt.cla()
    mlmc_plot(filename, nvert=3)
    plt.savefig(filename.replace('.txt', '.svg'))