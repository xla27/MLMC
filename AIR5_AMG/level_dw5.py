import numpy as np
from scipy.interpolate import interp1d
import time

from AIR5_AMG import cfd_call_amg
from AIR5     import cfd_call

def dw_l(level, N_samples, *args):
    """
    Function handling the CFD simulations and the difference between fine and coarse at given level.

    l:          level
    N_samples:  number of samples
    
    """

    (nproc, _, _) = args

    # Start of time recording
    start = time.time()

    # P
    sums1_i = [None] * N_samples  # sum of difference between static Pressure at fine and coarse
    sums2_i = [None] * N_samples # sum of square of difference between static Pressure at fine and coarse
    sums5_i = [None] * N_samples  # sum of static Pressure at fine 
    sums6_i = [None] * N_samples  # sum of square of static Pressure at fine 

    # N
    sums1N_i = [None] * N_samples # sum of difference between atomic Nitrogen mass fraction at fine and coarse
    sums2N_i = [None] * N_samples # sum of square of difference between atomic Nitrogen mass fraction at fine and coarse

    # O
    sums1O_i = [None] * N_samples # sum of difference between atomic Oxygen mass fraction at fine and coarse
    sums2O_i = [None] * N_samples # sum of square of difference between atomic Oxygen mass fraction at fine and coarse

    # NO
    sums1NO_i = [None] * N_samples # sum of difference between Nitric Oxide mass fraction at fine and coarse
    sums2NO_i = [None] * N_samples # sum of square of difference between Nitric Oxide mass fraction at fine and coarse

    # N2
    sums1N2_i = [None] * N_samples # sum of difference between diatomic Nitrogen mass fraction at fine and coarse
    sums2N2_i = [None] * N_samples # # sum of square of difference between diatomic Nitrogen mass fraction at fine and coarse

    # O2 
    sums1O2_i = [None] * N_samples # sum of difference between diatomic Oxygen mass fraction at fine and coarse
    sums2O2_i = [None] * N_samples # sum of square of difference between diatomic Oxygen mass fraction at fine and coarse

    # Temperatures 
    sums1Ttr_i = [None] * N_samples # sum of difference between translational Temperature at fine and coarse
    sums2Ttr_i = [None] * N_samples # sum of squares of difference between translational Temperature at fine and coarse
    sums1Tve_i = [None] * N_samples # sum of difference between vibrational Temperature at fine and coarse
    sums2Tve_i = [None] * N_samples # sum of squares of difference between vibrational Temperature at fine and coarse 

    # Mach
    sums1M_i = [None] * N_samples # sum of difference between Mach at fine and coarse
    sums2M_i = [None] * N_samples # sum of square of difference between Mach at fine and coarse

    # Ranges for aleatoric uncertainties (Freestream values)
    Mmean   = 9.0;   M_max   = 9.5;   M_min   = 8.0 
    Tmean   = 1000;  T_max   = 1050;  T_min   = 850 
    Pmean   = 390;   P_max   = 600;   P_min   = 300
    Bn2mean = 0.79;  Bn2_max = 0.8;   Bn2_min = 0.76

    xnodesc_list = []

    # Looping over the samples
    for i in range(N_samples):
        
        # Sampling the aleatoric uncertainties
        M_inf    = M_min + (M_max - M_min) * np.random.rand()
        T_inf    = T_min + (T_max - T_min) * np.random.rand()
        P_inf    = P_min + (P_max - P_min) * np.random.rand() 
        Bn2_inf  = Bn2_min + (Bn2_max - Bn2_min) * np.random.rand() 
        Bo2_inf  = 1 - Bn2_inf
        
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

        # return of CFD calls [beta_n, beta_o, beta_no, beta_n2, beta_o2, P, Ttr, Tve, M, xnodes]

        if level == 0:

            # Call to CFD with fine mesh (no adaptation)
            (_, baseFolder, workingFolder) = args
            baseFolder2 = baseFolder.replace('AIR5_AMG','AIR5')
            args2 = (nproc, baseFolder2, workingFolder)
            QoI_fine = list(cfd_call('FINE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args2))

            # No call to CFD with coearse mesh, coarse results set to zero as it is the starting level
            xnodesc = QoI_fine[-1]
            QoI_coarse = [[0.] * len(xnodesc) for i in range(9)]
            QoI_coarse.append(xnodesc)

            # moving average for FINE results
            ws = max(int(len(xnodesc) * SF), 1)
            QoI_fine = [moving_average(QoI_fine[i], ws) for i in range(9)]

            xnodesc_list.append(xnodesc)

        else:

            # Call to CFD with fine mesh
            QoI_fine = list(cfd_call_amg('FINE',valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args))

            # Call to CFD with coarse mesh
            QoI_coarse = list(cfd_call_amg('COARSE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args))

            # moving average
            xnodesc = QoI_fine[-1]
            ws = max(int(len(xnodesc) * SF), 1)

            QoI_coarse = [moving_average(QoI_coarse[i], ws) for i in range(9)]
            QoI_fine   = [moving_average(QoI_fine[i],   ws) for i in range(9)]

            # interpolating on FINE on COARSE
            QoI_fine = [interp1d(QoI_fine[-1], QoI_fine[i], kind='linear', fill_value='extrapolate')(xnodesc) for i in range(9)]
            QoI_fine.append(xnodesc)

            xnodesc_list.append(xnodesc)

        sums1_i[i] = QoI_fine[5] - QoI_coarse[5]
        sums2_i[i] = (QoI_fine[5] - QoI_coarse[5])**2
        sums5_i[i] =  QoI_fine[5]
        sums6_i[i] = QoI_fine[5]**2

        sums1N_i[i] = QoI_fine[0] - QoI_coarse[0]
        sums2N_i[i] = (QoI_fine[0] - QoI_coarse[0]**2)

        sums1O_i[i] = QoI_fine[1] - QoI_coarse[1]
        sums2O_i[i] = (QoI_fine[1] - QoI_coarse[1])**2

        sums1NO_i[i] = QoI_fine[2] - QoI_coarse[2]
        sums2NO_i[i] = (QoI_fine[2] - QoI_coarse[2])**2

        sums1N2_i[i] = QoI_fine[3] - QoI_coarse[3]
        sums2N2_i[i] = (QoI_fine[3] - QoI_coarse[3])**2

        sums1O2_i[i] = QoI_fine[4] - QoI_coarse[4]
        sums2O2_i[i] = (QoI_fine[4] - QoI_coarse[4])**2

        sums1Ttr_i[i] = QoI_fine[6] - QoI_coarse[6]
        sums2Ttr_i[i] = (QoI_fine[6] - QoI_coarse[6]**2)

        sums1Tve_i[i] = QoI_fine[7] - QoI_coarse[7]
        sums2Tve_i[i] = (QoI_fine[7] - QoI_coarse[7])**2

        sums1M_i[i] = QoI_fine[8] - QoI_coarse[8]
        sums2M_i[i] = (QoI_fine[8] - QoI_coarse[8])**2

    # finding the smallest xnodesc
    xnodesc_list = [QoI_coarse[i][-1] for i in range(N_samples)]
    lengths      = np.array([len(xnodes) for xnodes in xnodesc_list])
    xnodesc_ref  = xnodesc_list[np.argmin(lengths)]

    # interpolating "finer" coarses on the coarsest COARSE if level is larger than 1
    if level >= 2:
        for i in range(N_samples):

            sums1_i[i] = interp1d(xnodesc_list[i], sums1_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2_i[i] = interp1d(xnodesc_list[i], sums2_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums5_i[i] = interp1d(xnodesc_list[i], sums5_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums6_i[i] = interp1d(xnodesc_list[i], sums6_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1N_i[i] = interp1d(xnodesc_list[i], sums1N_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2N_i[i] = interp1d(xnodesc_list[i], sums2N_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1O_i[i] = interp1d(xnodesc_list[i], sums1O_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2O_i[i] = interp1d(xnodesc_list[i], sums2O_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1NO_i[i] = interp1d(xnodesc_list[i], sums1NO_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2NO_i[i] = interp1d(xnodesc_list[i], sums2NO_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1N2_i[i] = interp1d(xnodesc_list[i], sums1N2_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2N2_i[i] = interp1d(xnodesc_list[i], sums2N2_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1O2_i[i] = interp1d(xnodesc_list[i], sums1O2_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2O2_i[i] = interp1d(xnodesc_list[i], sums2O2_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1Ttr_i[i] = interp1d(xnodesc_list[i], sums1Ttr_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2Ttr_i[i] = interp1d(xnodesc_list[i], sums2Ttr_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1Tve_i[i] = interp1d(xnodesc_list[i], sums1Tve_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2Tve_i[i] = interp1d(xnodesc_list[i], sums2Tve_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            sums1M_i[i] = interp1d(xnodesc_list[i], sums1M_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)
            sums2M_i[i] = interp1d(xnodesc_list[i], sums2M_i[i], kind='linear', fill_value='extrapolate')(xnodesc_ref)

        
    # RESULTS UPDATE

    # [beta_n, beta_o, beta_no, beta_n2, beta_o2, P, Ttr, Tve, M, xnodes]

    # P
    sums1 = np.sum(np.array(sums1_i), axis=0)
    sums2 = np.sum(np.array(sums2_i), axis=0)
    sums5 = np.sum(np.array(sums5_i), axis=0)
    sums6 = np.sum(np.array(sums6_i), axis=0)

    # N
    sums1N = np.sum(np.array(sums1N_i), axis=0)
    sums2N = np.sum(np.array(sums2N_i), axis=0)

    # O
    sums1O = np.sum(np.array(sums1O_i), axis=0)
    sums2O = np.sum(np.array(sums2O_i), axis=0)

    # NO
    sums1NO = np.sum(np.array(sums1NO_i), axis=0)
    sums2NO = np.sum(np.array(sums2NO_i), axis=0)

    # N2
    sums1N2 = np.sum(np.array(sums1N2_i), axis=0)
    sums2N2 = np.sum(np.array(sums2N2_i), axis=0)

    # O2
    sums1O2 = np.sum(np.array(sums1O2_i), axis=0)
    sums2O2 = np.sum(np.array(sums2O2_i), axis=0)
    
    # Temperatures
    sums1Ttr = np.sum(np.array(sums1Ttr_i), axis=0)
    sums2Ttr = np.sum(np.array(sums2Ttr_i), axis=0)
    sums1Tve = np.sum(np.array(sums1Tve_i), axis=0)
    sums2Tve = np.sum(np.array(sums2Tve_i), axis=0)

    # M
    sums1M = np.sum(np.array(sums1M_i), axis=0)
    sums2M = np.sum(np.array(sums2M_i), axis=0)
    
    # End of recording time
    end   = time.time() 
    cost  = nproc*(end-start)    # Cost computation
        
    return xnodesc_ref, sums1, sums2, sums5, sums6, sums1N, sums2N, sums1O, sums2O, sums1NO, sums2NO, sums1N2, sums2N2, sums1O2, sums2O2, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost


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

