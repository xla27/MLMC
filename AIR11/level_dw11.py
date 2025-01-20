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
    sums1 = 0  # sum of difference between static Pressure at fine and coarse
    sums2 = 0  # sum of square of difference between static Pressure at fine and coarse
    sums5 = 0  # sum of static Pressure at fine 
    sums6 = 0  # sum of square of static Pressure at fine 

    # ND e-
    sums1ndelecMinus = 0 # sum of difference between electrons number density at fine and coarse
    sums2ndelecMinus = 0 # sum of square of difference between electrons number density at fine and coarse

    # ND N+
    sums1ndNPlus = 0 # sum of difference between atomic nitrogen ions number density at fine and coarse
    sums2ndNPlus = 0 # sum of square of difference between atomic nitrogen ions number density at fine and coarse

    # ND O+
    sums1ndOPlus = 0 # sum of difference between atomic oxygen ions number density at fine and coarse
    sums2ndOPlus = 0 # sum of square of difference between atomic oxygen ions number density at fine and coarse

    # ND NO+
    sums1ndNOPlus = 0 # sum of difference between nitric oxyde ions number density at fine and coarse
    sums2ndNOPlus = 0 # sum of square of difference between nitric oxyde ions number density at fine and coarse

    # ND N2+
    sums1ndN2Plus = 0 # sum of difference between nitrogen ions number density at fine and coarse
    sums2ndN2Plus = 0 # sum of square of difference between nitrogen ions number density at fine and coarse

    # ND O2+
    sums1ndO2Plus = 0 # sum of difference between oxygen ions number density at fine and coarse
    sums2ndO2Plus = 0 # sum of square of difference between oxygen ions number density at fine and coarse

    # e-   
    sums1elecMinus = 0 # sum of difference between electron mass fraction at fine and coarse
    sums2elecMinus = 0 # sum of square of difference between electron mass fraction at fine and coarse

    # N+
    sums1NPlus = 0 # sum of difference between atomic nitrogen ions mass fraction at fine and coarse
    sums2NPlus = 0 # sum of square of difference between atomic nitrogen ions mass fraction at fine and coarse

    # O+
    sums1OPlus = 0 # sum of difference between atomic oxygen ions mass fraction at fine and coarse
    sums2OPlus = 0 # sum of square of difference between atomic oxygen ions mass fraction at fine and coarse

    # NO+
    sums1NOPlus = 0 # sum of difference between nitric oxyde ions mass fraction at fine and coarse
    sums2NOPlus = 0 # sum of square of difference between nitric oxyde ions mass fraction at fine and coarse

    # N2+
    sums1N2Plus = 0 # sum of difference between nitrogen ions mass fraction at fine and coarse
    sums2N2Plus = 0 # sum of square of difference between nitrogen ions mass fraction at fine and coarse

    # O2+
    sums1O2Plus = 0 # sum of difference between oxygen ions mass fraction at fine and coarse
    sums2O2Plus = 0 # sum of square of difference between oxygen ions mass fraction at fine and coarse

    # N
    sums1N = 0 # sum of difference between atomic Nitrogen mass fraction at fine and coarse
    sums2N = 0 # sum of square of difference between atomic Nitrogen mass fraction at fine and coarse

    # O
    sums1O = 0 # sum of difference between atomic Oxygen mass fraction at fine and coarse
    sums2O = 0 # sum of square of difference between atomic Oxygen mass fraction at fine and coarse

    # NO
    sums1NO = 0 # sum of difference between Nitric Oxide mass fraction at fine and coarse
    sums2NO = 0 # sum of square of difference between Nitric Oxide mass fraction at fine and coarse

    # N2
    sums1N2 = 0 # sum of difference between diatomic Nitrogen mass fraction at fine and coarse
    sums2N2 = 0 # # sum of square of difference between diatomic Nitrogen mass fraction at fine and coarse

    # O2 
    sums1O2 = 0 # sum of difference between diatomic Oxygen mass fraction at fine and coarse
    sums2O2 = 0 # sum of square of difference between diatomic Oxygen mass fraction at fine and coarse

    # Temperatures 
    sums1Ttr = 0 # sum of difference between translational Temperature at fine and coarse
    sums2Ttr = 0 # sum of squares of difference between translational Temperature at fine and coarse
    sums1Tve = 0 # sum of difference between vibrational Temperature at fine and coarse
    sums2Tve = 0 # sum of squares of difference between vibrational Temperature at fine and coarse 

    # Mach
    sums1M = 0 # sum of difference between Mach at fine and coarse
    sums2M = 0 # sum of square of difference between Mach at fine and coarse

    # Ranges for aleatoric uncertainties (Freestream values)
    Mmean   = 9.0;   M_max   = 9.5;   M_min   = 8.0 
    Tmean   = 1000;  T_max   = 1050;  T_min   = 850 
    Pmean   = 390;   P_max   = 600;   P_min   = 300
    Bn2mean = 0.79;  Bn2_max = 0.8;   Bn2_min = 0.76

    # Looping over the samples
    for i in range(1, N_samples  + 1):
        
        # Sampling the aleatoric uncertainties
        M_inf   = M_min + (M_max - M_min) * np.random.rand()
        T_inf   = T_min + (T_max - T_min) * np.random.rand()
        P_inf   = P_min + (P_max - P_min) * np.random.rand() 
        Bn2_inf = Bn2_min + (Bn2_max - Bn2_min) * np.random.rand() 
        Bo2_inf = 1 - Bn2_inf
        
        # Formatting for printing
        M_inf   = '{:.3f}'.format(M_inf)
        T_inf   = '{:.1f}'.format(T_inf)
        P_inf   = '{:.1f}'.format(P_inf)
        Bn2_inf = '{:.3f}'.format(Bn2_inf)
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
            Pc              = 0 
            nd_elecMinusc   = 0
            nd_NPlusc       = 0 
            nd_OPlusc       = 0 
            nd_NOPlusc      = 0 
            nd_N2Plusc      = 0 
            nd_O2Plusc      = 0
            beta_elecMinusc = 0 
            beta_NPlusc     = 0
            beta_OPlusc     = 0 
            beta_NOPlusc    = 0 
            beta_N2Plusc    = 0 
            beta_O2Plusc    = 0
            beta_Nc         = 0
            beta_Oc         = 0 
            beta_NOc        = 0
            beta_N2c        = 0
            beta_O2c        = 0
            Ttr_ic          = 0
            Tve_ic          = 0
            M_ic            = 0

            # Call to CFD with fine mesh
            [nd_elecMinusf, nd_NPlusf, nd_OPlusf, nd_NOPlusf, nd_N2Plusf, nd_O2Plusf, beta_elecMinusf, 
             beta_NPlusf, beta_OPlusf, beta_NOPlusf, beta_N2Plusf, beta_O2Plusf, beta_Nf, beta_Of, beta_NOf, 
             beta_N2f, beta_O2f, P_if, Ttr_if, Tve_if, M_if, xnodesf] = cfd_call('FINE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)
            
            xnodesc = xnodesf    # No need for interpolation on coarse, xnodes put equal

            Pf = P_if
            ws = max(int(len(Pf) * SF), 1)    # window size

            # Moving average on wall quantities
            Pf              = moving_average(Pf,              ws)
            nd_elecMinusf   = moving_average(nd_elecMinusf,   ws) 
            nd_NPlusf       = moving_average(nd_NPlusf,       ws) 
            nd_OPlusf       = moving_average(nd_OPlusf,       ws) 
            nd_NOPlusf      = moving_average(nd_NOPlusf,      ws) 
            nd_N2Plusf      = moving_average(nd_N2Plusf,      ws)
            nd_O2Plusf      = moving_average(nd_O2Plusf,      ws)  
            beta_elecMinusf = moving_average(beta_elecMinusf, ws) 
            beta_NPlusf     = moving_average(beta_NPlusf,     ws) 
            beta_OPlusf     = moving_average(beta_OPlusf,     ws) 
            beta_NOPlusf    = moving_average(beta_NOPlusf,    ws) 
            beta_N2Plusf    = moving_average(beta_N2Plusf,    ws) 
            beta_O2Plusf    = moving_average(beta_O2Plusf,    ws) 
            beta_Nf         = moving_average(beta_Nf,         ws)
            beta_Of         = moving_average(beta_Of,         ws)
            beta_NOf        = moving_average(beta_NOf,        ws)
            beta_N2f        = moving_average(beta_N2f,        ws)
            beta_O2f        = moving_average(beta_O2f,        ws)
            Ttr_if          = moving_average(Ttr_if,          ws)
            Tve_if          = moving_average(Tve_if,          ws)
            M_if            = moving_average(M_if,            ws)   

        else:
            # Call to CFD with coarse mesh
            [nd_elecMinusc, nd_NPlusc, nd_OPlusc, nd_NOPlusc, nd_N2Plusc, nd_O2Plusc, beta_elecMinusc, 
             beta_NPlusc, beta_OPlusc, beta_NOPlusc, beta_N2Plusc, beta_O2Plusc, beta_Nc, beta_Oc, beta_NOc,
             beta_N2c, beta_O2c, P_ic, Ttr_ic, Tve_ic, M_ic, xnodesc] = cfd_call('COARSE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)

            Pc = P_ic
            ws = max(int(len(Pc) * SF), 1) 

            # Moving average on wall quantities
            Pc              = moving_average(Pc,              ws)
            nd_elecMinusc   = moving_average(nd_elecMinusc,   ws)
            nd_NPlusc       = moving_average(nd_NPlusc,       ws)
            nd_OPlusc       = moving_average(nd_OPlusc,       ws) 
            nd_NOPlusc      = moving_average(nd_NOPlusc,      ws) 
            nd_N2Plusc      = moving_average(nd_N2Plusc,      ws) 
            nd_O2Plusc      = moving_average(nd_O2Plusc,      ws)
            beta_elecMinusc = moving_average(beta_elecMinusc, ws)
            beta_NPlusc     = moving_average(beta_NPlusc,     ws) 
            beta_OPlusc     = moving_average(beta_OPlusc,     ws) 
            beta_NOPlusc    = moving_average(beta_NOPlusc,    ws) 
            beta_N2Plusc    = moving_average(beta_N2Plusc,    ws) 
            beta_O2Plusc    = moving_average(beta_O2Plusc,    ws) 
            beta_Nc         = moving_average(beta_Nc,         ws)
            beta_Oc         = moving_average(beta_Oc,         ws)
            beta_NOc        = moving_average(beta_NOc,        ws)
            beta_N2c        = moving_average(beta_N2c,        ws)
            beta_O2c        = moving_average(beta_O2c,        ws)
            Ttr_ic          = moving_average(Ttr_ic,          ws)
            Tve_ic          = moving_average(Tve_ic,          ws)
            M_ic            = moving_average(M_ic,            ws)
    

            # Call to CFD with fine mesh
            [nd_elecMinusf, nd_NPlusf, nd_OPlusf, nd_NOPlusf, nd_N2Plusf, nd_O2Plusf, beta_elecMinusf, 
             beta_NPlusf, beta_OPlusf, beta_NOPlusf, beta_N2Plusf, beta_O2Plusf, beta_Nf, beta_Of, beta_NOf, 
             beta_N2f, beta_O2f, P_if, Ttr_if, Tve_if, M_if, xnodesf] = cfd_call('FINE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)
            
            Pf = P_if

            # Results interpolation on the coarser grid            
            Pf              = interp1d(xnodesf, Pf,              kind='linear', fill_value='extrapolate')(xnodesc) 
            nd_elecMinusf   = interp1d(xnodesf, nd_elecMinusf,   kind='linear', fill_value='extrapolate')(xnodesc) 
            nd_NPlusf       = interp1d(xnodesf, nd_NPlusf,       kind='linear', fill_value='extrapolate')(xnodesc) 
            nd_OPlusf       = interp1d(xnodesf, nd_OPlusf,       kind='linear', fill_value='extrapolate')(xnodesc)
            nd_NOPlusf      = interp1d(xnodesf, nd_NOPlusf,      kind='linear', fill_value='extrapolate')(xnodesc) 
            nd_N2Plusf      = interp1d(xnodesf, nd_N2Plusf,      kind='linear', fill_value='extrapolate')(xnodesc) 
            nd_O2Plusf      = interp1d(xnodesf, nd_O2Plusf,      kind='linear', fill_value='extrapolate')(xnodesc)
            beta_elecMinusf = interp1d(xnodesf, beta_elecMinusf, kind='linear', fill_value='extrapolate')(xnodesc) 
            beta_NPlusf     = interp1d(xnodesf, beta_NPlusf,     kind='linear', fill_value='extrapolate')(xnodesc) 
            beta_OPlusf     = interp1d(xnodesf, beta_OPlusf,     kind='linear', fill_value='extrapolate')(xnodesc)  
            beta_NOPlusf    = interp1d(xnodesf, beta_NOPlusf,    kind='linear', fill_value='extrapolate')(xnodesc) 
            beta_N2Plusf    = interp1d(xnodesf, beta_N2Plusf,    kind='linear', fill_value='extrapolate')(xnodesc) 
            beta_O2Plusf    = interp1d(xnodesf, beta_O2Plusf,    kind='linear', fill_value='extrapolate')(xnodesc)
            beta_Nf         = interp1d(xnodesf, beta_Nf,         kind='linear', fill_value='extrapolate')(xnodesc)
            beta_Of         = interp1d(xnodesf, beta_Of,         kind='linear', fill_value='extrapolate')(xnodesc)
            beta_NOf        = interp1d(xnodesf, beta_NOf,        kind='linear', fill_value='extrapolate')(xnodesc)                   
            beta_N2f        = interp1d(xnodesf, beta_N2f,        kind='linear', fill_value='extrapolate')(xnodesc)
            beta_O2f        = interp1d(xnodesf, beta_O2f,        kind='linear', fill_value='extrapolate')(xnodesc) 
            Ttr_if          = interp1d(xnodesf, Ttr_if,          kind='linear', fill_value='extrapolate')(xnodesc) 
            Tve_if          = interp1d(xnodesf, Tve_if,          kind='linear', fill_value='extrapolate')(xnodesc) 
            M_if            = interp1d(xnodesf, M_if,            kind='linear', fill_value='extrapolate')(xnodesc) 

            # Moving average on wall quantities
            Pf              = moving_average(Pf,              ws)
            nd_elecMinusf   = moving_average(nd_elecMinusf,   ws) 
            nd_NPlusf       = moving_average(nd_NPlusf,       ws) 
            nd_OPlusf       = moving_average(nd_OPlusf,       ws) 
            nd_NOPlusf      = moving_average(nd_NOPlusf,      ws) 
            nd_N2Plusf      = moving_average(nd_N2Plusf,      ws)
            nd_O2Plusf      = moving_average(nd_O2Plusf,      ws)  
            beta_elecMinusf = moving_average(beta_elecMinusf, ws) 
            beta_NPlusf     = moving_average(beta_NPlusf,     ws) 
            beta_OPlusf     = moving_average(beta_OPlusf,     ws) 
            beta_NOPlusf    = moving_average(beta_NOPlusf,    ws) 
            beta_N2Plusf    = moving_average(beta_N2Plusf,    ws) 
            beta_O2Plusf    = moving_average(beta_O2Plusf,    ws) 
            beta_Nf         = moving_average(beta_Nf,         ws)
            beta_Of         = moving_average(beta_Of,         ws)
            beta_NOf        = moving_average(beta_NOf,        ws)
            beta_N2f        = moving_average(beta_N2f,        ws)
            beta_O2f        = moving_average(beta_O2f,        ws)
            Ttr_if          = moving_average(Ttr_if,          ws)
            Tve_if          = moving_average(Tve_if,          ws)
            M_if            = moving_average(M_if,            ws)  

        
        # RESULTS UPDATE

        # P
        sums1 += Pf - Pc 
        sums2 += (Pf - Pc) ** 2
        sums5 += Pf
        sums6 += Pf ** 2

        # ND e-
        sums1ndelecMinus += nd_elecMinusf - nd_elecMinusc
        sums2ndelecMinus += (nd_elecMinusf - nd_elecMinusc) ** 2

        # ND N+
        sums1ndNPlus += nd_NPlusf - nd_NPlusc
        sums2ndNPlus += (nd_NPlusf - nd_NPlusc) ** 2

        # ND O+ 
        sums1ndOPlus += nd_OPlusf - nd_OPlusc
        sums2ndOPlus += (nd_OPlusf - nd_OPlusc) ** 2

        # ND NO+
        sums1ndNOPlus += nd_NOPlusf - nd_NOPlusc
        sums2ndNOPlus += (nd_NOPlusf - nd_NOPlusc) ** 2

        # ND N2+
        sums1ndN2Plus += nd_N2Plusf - nd_N2Plusc
        sums2ndN2Plus += (nd_N2Plusf - nd_N2Plusc) ** 2

        # ND NO+
        sums1ndO2Plus += nd_O2Plusf - nd_O2Plusc
        sums2ndO2Plus += (nd_O2Plusf - nd_O2Plusc) ** 2 
        
        # e-
        sums1elecMinus  += beta_elecMinusf - beta_elecMinusc
        sums2elecMinus += (beta_elecMinusf - beta_elecMinusc) ** 2

        # N+
        sums1NPlus += beta_NPlusf - beta_NPlusc
        sums2NPlus += (beta_NPlusf - beta_NPlusc) ** 2

        # O+
        sums1OPlus += beta_OPlusf - beta_OPlusc
        sums2OPlus += (beta_OPlusf - beta_OPlusc) ** 2

        # NO+
        sums1NOPlus += beta_NOPlusf - beta_NOPlusc
        sums2NOPlus += (beta_NOPlusf - beta_NOPlusc) ** 2
        
        # N2+
        sums1N2Plus += beta_N2Plusf - beta_N2Plusc
        sums2N2Plus += (beta_N2Plusf - beta_N2Plusc) ** 2

        # O2+
        sums1O2Plus += beta_O2Plusf - beta_O2Plusc
        sums2O2Plus += (beta_O2Plusf - beta_O2Plusc) ** 2   

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
        
    return xnodesc, sums1, sums2, sums5, sums6, sums1ndelecMinus, sums2ndelecMinus, sums1ndNPlus, sums2ndNPlus, sums1ndOPlus, sums2ndOPlus, sums1ndNOPlus, sums2ndNOPlus, sums1ndN2Plus, sums2ndN2Plus, sums1ndO2Plus, sums2ndO2Plus, sums1elecMinus, sums2elecMinus, sums1NPlus, sums2NPlus, sums1OPlus, sums2OPlus, sums1NOPlus, sums2NOPlus, sums1N2Plus, sums2N2Plus, sums1O2Plus, sums2O2Plus, sums1N, sums2N, sums1O, sums2O, sums1NO, sums2NO, sums1N2, sums2N2, sums1O2, sums2O2, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost


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

