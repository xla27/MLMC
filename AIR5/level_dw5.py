import numpy as np
from scipy.interpolate import interp1d
import time
from cfd_call import cfd_call

def dw_l(l, N_samples, *args):
    """
    Function handling the CFD simulations and the difference between fine and coarse at given level.

    l:          level
    N_samples:  number of samples
    
    """

    (nproc, _, _) = args

    # Start of time recording
    start = time.time()

    # P
    sums1 = 0;  # sum of difference between static Pressure at fine and coarse
    sums2 = 0;  # sum of square of difference between static Pressure at fine and coarse
    sums5 = 0;  # sum of static Pressure at fine 
    sums6 = 0;  # sum of square of static Pressure at fine 

    # N
    sums1N = 0; # sum of difference between atomic Nitrogen mass fraction at fine and coarse
    sums2N = 0; # sum of square of difference between atomic Nitrogen mass fraction at fine and coarse

    # O
    sums1O = 0; # sum of difference between atomic Oxygen mass fraction at fine and coarse
    sums2O = 0; # sum of square of difference between atomic Oxygen mass fraction at fine and coarse

    # NO
    sums1NO = 0; # sum of difference between Nitric Oxide mass fraction at fine and coarse
    sums2NO = 0; # sum of square of difference between Nitric Oxide mass fraction at fine and coarse

    # N2
    sums1N2 = 0; # sum of difference between diatomic Nitrogen mass fraction at fine and coarse
    sums2N2 = 0; # # sum of square of difference between diatomic Nitrogen mass fraction at fine and coarse

    # O2 
    sums1O2 = 0; # sum of difference between diatomic Oxygen mass fraction at fine and coarse
    sums2O2 = 0; # sum of square of difference between diatomic Oxygen mass fraction at fine and coarse

    # Temperatures 
    sums1Ttr = 0; # sum of difference between translational Temperature at fine and coarse
    sums2Ttr = 0; # sum of squares of difference between translational Temperature at fine and coarse
    sums1Tve = 0; # sum of difference between vibrational Temperature at fine and coarse
    sums2Tve = 0; # sum of squares of difference between vibrational Temperature at fine and coarse 

    # Mach
    sums1M = 0; # sum of difference between Mach at fine and coarse
    sums2M = 0; # sum of square of difference between Mach at fine and coarse

    # Ranges for aleatoric uncertainties (Freestream values)
    Mmean   = 9.0;   M_max   = 9.5;   M_min   = 8.0 
    Tmean   = 1000;  T_max   = 1050;  T_min   = 850 
    Pmean   = 390;   P_max   = 600;   P_min   = 300
    Bn2mean = 0.79;  Bn2_max = 0.8;   Bn2_min = 0.76

    # Looping over the samples
    for i in range(1, N_samples  + 1):
        
        # Sampling the aleatoric uncertainties
        M_inf = M_min + (M_max - M_min) * np.random.rand()
        T_inf    = T_min + (T_max - T_min) * np.random.rand()
        P_inf    = P_min + (P_max - P_min) * np.random.rand() 
        Bn2_inf  = Bn2_min + (Bn2_max - Bn2_min) * np.random.rand() 
        Bo2_inf  = 1 - Bn2_inf
        
        # Formatting for printing
        M_inf = '{:.3f}'.format(M_inf)
        T_inf    = '{:.1f}'.format(T_inf)
        P_inf    = '{:.1f}'.format(P_inf)
        Bn2_inf  = '{:.3f}'.format(Bn2_inf)
        Bo2_inf = '{:.3f}'.format(Bo2_inf)
        
        # Strings to be inserted in SU2 configs
        valIns_M = str(M_inf)
        valIns_T = str(T_inf)
        valIns_P = str(P_inf)
        valIns_Bn2 = str(Bn2_inf)
        valIns_Bo2 = str(Bo2_inf)

        SF = 0.0075 

        if l == 0:
            # No call to CFD with coearse mesh, coarse results set to zero as it is the starting level
            Pc       = 0 
            beta_Nc  = 0
            beta_Oc  = 0 
            beta_NOc = 0
            beta_N2c = 0
            beta_O2c = 0
            Ttr_ic   = 0
            Tve_ic   = 0
            M_ic     = 0

            # Call to CFD with fine mesh
            beta_Nf, beta_Of, beta_NOf, beta_N2f, beta_O2f, P_if, Ttr_if, Tve_if, M_if, xnodesf = cfd_call('FINE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)
            
            xnodesc = xnodesf    # No need for interpolation on coarse, xnodes put equal

            Pf = P_if
            ws = max(int(len(Pf) * SF), 1)    # window size

            # Moving average on wall quantities
            Pf       = moving_average(Pf,       ws)
            beta_Nf  = moving_average(beta_Nf,  ws)
            beta_Of  = moving_average(beta_Of,  ws)
            beta_NOf = moving_average(beta_NOf, ws)
            beta_N2f = moving_average(beta_N2f, ws)
            beta_O2f = moving_average(beta_O2f, ws)
            Ttr_if   = moving_average(Ttr_if,   ws)
            Tve_if   = moving_average(Tve_if,   ws)
            M_if     = moving_average(M_if,     ws)   

        else:
            # Call to CFD with coarse mesh
            beta_Nc, beta_Oc, beta_NOc, beta_N2c, beta_O2c, p_ic, Ttr_ic, Tve_ic, M_ic, xnodesc = cfd_call('COARSE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)

            Pc = p_ic
            ws = max(int(len(Pc) * SF), 1) 

            # Moving average on wall quantities
            Pc       = moving_average(Pc,       ws)
            beta_Nc  = moving_average(beta_Nc,  ws)
            beta_Oc  = moving_average(beta_Oc,  ws)
            beta_NOc = moving_average(beta_NOc, ws)
            beta_N2c = moving_average(beta_N2c, ws)
            beta_O2c = moving_average(beta_O2c, ws)
            Ttr_ic   = moving_average(Ttr_ic,   ws)
            Tve_ic   = moving_average(Tve_ic,   ws)
            M_ic     = moving_average(M_ic,     ws)
    

            # Call to CFD with fine mesh
            beta_Nf, beta_Of, beta_NOf, beta_N2f, beta_O2f, P_if, Ttr_if, Tve_if, M_if, xnodesf = cfd_call('FINE',valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)

            Pf = P_if

            # Results interpolation on the coarser grid            
            Pf       = interp1d(xnodesf, Pf,       kind='linear', fill_value='extrapolate')(xnodesc) 
            beta_Nf  = interp1d(xnodesf, beta_Nf,  kind='linear', fill_value='extrapolate')(xnodesc)
            beta_Of  = interp1d(xnodesf, beta_Of,  kind='linear', fill_value='extrapolate')(xnodesc)
            beta_NOf = interp1d(xnodesf, beta_NOf, kind='linear', fill_value='extrapolate')(xnodesc)                   
            beta_N2f = interp1d(xnodesf, beta_N2f, kind='linear', fill_value='extrapolate')(xnodesc)
            beta_O2f = interp1d(xnodesf, beta_O2f, kind='linear', fill_value='extrapolate')(xnodesc) 
            Ttr_if   = interp1d(xnodesf, Ttr_if,   kind='linear', fill_value='extrapolate')(xnodesc) 
            Tve_if   = interp1d(xnodesf, Tve_if,   kind='linear', fill_value='extrapolate')(xnodesc) 
            M_if     = interp1d(xnodesf, M_if,     kind='linear', fill_value='extrapolate')(xnodesc) 

            # Moving average on wall quantities
            Pf       = moving_average(Pf,       ws)
            beta_Nf  = moving_average(beta_Nf,  ws)
            beta_Of  = moving_average(beta_Of,  ws)
            beta_NOf = moving_average(beta_NOf, ws)
            beta_N2f = moving_average(beta_N2f, ws)
            beta_O2f = moving_average(beta_O2f, ws)
            Ttr_if   = moving_average(Ttr_if,   ws)
            Tve_if   = moving_average(Tve_if,   ws)
            M_if     = moving_average(M_if,     ws) 

        
        # RESULTS UPDATE

        # P
        sums1 += Pf - Pc 
        sums2 += (Pf - Pc) ** 2
        sums5 += Pf
        sums6 += Pf ** 2

        # N
        sums1N += beta_Nf - beta_Nc
        sums2N += (beta_Nf - beta_Nc) ** 2

        # O
        sums1O += beta_Of - beta_Oc
        sums2O += (beta_Of - beta_Oc) ** 2

        # NO
        sums1NO += beta_NOf - beta_NOc
        sums2NO += (beta_NOf - beta_NOc) ** 2

        # N2
        sums1N2 += beta_N2f - beta_N2c
        sums2N2 += (beta_N2f - beta_N2c) ** 2

        # O2
        sums1O2 += beta_O2f - beta_O2c
        sums2O2 += (beta_O2f - beta_O2c) ** 2
        
        # Temperatures
        sums1Ttr += Ttr_if - Ttr_ic
        sums2Ttr += (Ttr_if - Ttr_ic) ** 2
        sums1Tve += Tve_if - Tve_ic
        sums2Tve += (Tve_if - Tve_ic) ** 2

        # M
        sums1M += M_if - M_ic
        sums2M += (M_if - M_ic) ** 2
        
        # End of recording time
        end   = time.time() 
        cost  = nproc*(end-start)    # Cost computation
        
    return xnodesc, sums1, sums2, sums5, sums6, sums1N, sums2N, sums1O, sums2O, sums1NO, sums2NO, sums1N2, sums2N2, sums1O2, sums2O2, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost


def moving_average(data, window_size):
    """
    Function to perform the moving average.

    data:           np.array of data
    window_size:    size of the window
    """
    pad_left    = window_size // 2
    pad_right   = window_size - pad_left - 1
    padded_data = np.pad(data, (pad_left, pad_right), mode='edge')
    ma_data     = np.convolve(padded_data, np.ones(window_size), 'valid') / window_size
    return ma_data

