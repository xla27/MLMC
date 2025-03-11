import numpy as np
from scipy.interpolate import interp1d
import time

from AIR11_AMG import cfd_call_amg
from AIR11     import cfd_call

def dw_l(level, N_samples, *args):
    """
    Function handling the CFD simulations and the difference between fine and coarse at given level.

    l:          level
    N_samples:  number of samples
    
    """

    (nproc, _, _) = args

    # Start of time recording
    start = time.time()

    # Ranges for aleatoric uncertainties (Freestream values)
    Mmean   = 9.0;   M_max   = 9.5;   M_min   = 8.0 
    Tmean   = 1000;  T_max   = 1050;  T_min   = 850 
    Pmean   = 390;   P_max   = 600;   P_min   = 300
    Bn2mean = 0.79;  Bn2_max = 0.8;   Bn2_min = 0.76

    xnodesc_list = []
    xnodesf_list = []
    QoI_fine   = [None] * N_samples; QoI_fine_avg   = [None] * N_samples; QoI_fine_interp   = [None] * N_samples
    QoI_coarse = [None] * N_samples; QoI_coarse_avg = [None] * N_samples; QoI_coarse_interp = [None] * N_samples

    SF = 0.015

    # Looping over the samples
    for i in range(N_samples):
        
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

        # return of CFD calls [nd_elecMinus, nd_nPlus, nd_oPlus, nd_noPlus, nd_n2Plus, nd_o2Plus, beta_elecMinus, beta_nPlus, beta_oPlus, 
        # beta_noPlus, beta_n2Plus, beta_o2Plus, beta_n, beta_o, beta_no, beta_n2, beta_o2, p_i, Ttr_i, Tve_i, M_i, xnodesf]

        if level == 0:

            # Call to CFD with fine mesh
            (_, baseFolder, workingFolder) = args
            baseFolder2 = baseFolder.replace('AIR11_AMG','AIR11')
            args2 = (nproc, baseFolder2, workingFolder)
            QoI_fine[i] = list(cfd_call('FINE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args2))
            xnodesf = QoI_fine[i][-1]

            # No call to CFD with coearse mesh, coarse results set to zero as it is the starting level
            xnodesc = QoI_fine[i][-1]
            QoI_coarse[i] = [[0.] * len(xnodesc) for i in range(0,21)]
            QoI_coarse[i].append(xnodesc)

        else:

            # Call to CFD with fine mesh
            QoI_fine[i] = list(cfd_call_amg('FINE',valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args))
            xnodesf = QoI_fine[i][-1]

            # Call to CFD with coarse mesh
            QoI_coarse[i] = list(cfd_call_amg('COARSE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args))
            xnodesc = QoI_coarse[i][-1]

        xnodesc_list.append(xnodesc)
        xnodesf_list.append(xnodesf)

    # finding the smallest xnodesc
    lengths      = np.array([len(xnodes) for xnodes in xnodesc_list])
    xnodesc_ref  = xnodesc_list[np.argmin(lengths)]

    # moving average and interpolation
    ws = max(int(len(xnodesc_ref) * SF), 1)

    for i in range(N_samples):
        QoI_coarse_avg[i] = [moving_average(QoI_coarse[i][qoi], ws) for qoi in range(21)]
        QoI_fine_avg[i]   = [moving_average(QoI_fine[i][qoi],   ws) for qoi in range(21)]

        if level == 0:
            QoI_coarse_interp[i] = QoI_coarse_avg[i]
            QoI_fine_interp[i]   = QoI_fine_avg[i]

        if level == 1:
            QoI_coarse_interp[i] = QoI_coarse_avg[i]
            QoI_fine_interp[i]   = [interp1d(xnodesf_list[i], QoI_fine_avg[i][qoi], kind='linear', )(xnodesc_ref) for qoi in range(21)]
        else:
            QoI_coarse_interp[i] = [interp1d(xnodesc_list[i], QoI_coarse_avg[i][qoi], kind='linear', )(xnodesc_ref) for qoi in range(21)]
            QoI_fine_interp[i]   = [interp1d(xnodesf_list[i], QoI_fine_avg[i][qoi],   kind='linear', )(xnodesc_ref) for qoi in range(21)]



    # RESULTS UPDATE
    QoI_coarse_interp = np.array(QoI_coarse_interp)
    QoI_fine_interp   = np.array(QoI_fine_interp)

    # P
    sums1 = np.sum( QoI_fine_interp[:,17,:] - QoI_coarse_interp[:,17,:], axis=0)
    sums2 = np.sum((QoI_fine_interp[:,17,:] - QoI_coarse_interp[:,17,:])**2, axis=0)
    sums5 = np.sum(QoI_fine_interp[:,17,:], axis=0)
    sums6 = np.sum(QoI_fine_interp[:,17,:]**2, axis=0)

    # ND e-
    sums1ndelecMinus = np.sum( QoI_fine_interp[:,0,:] - QoI_coarse_interp[:,0,:], axis=0)
    sums2ndelecMinus = np.sum((QoI_fine_interp[:,0,:] - QoI_coarse_interp[:,0,:])**2, axis=0)

    # ND N+
    sums1ndNPlus = np.sum( QoI_fine_interp[:,1,:] - QoI_coarse_interp[:,1,:], axis=0)
    sums2ndNPlus = np.sum((QoI_fine_interp[:,1,:] - QoI_coarse_interp[:,1,:])**2, axis=0)

    # ND O+ 
    sums1ndOPlus = np.sum( QoI_fine_interp[:,2,:] - QoI_coarse_interp[:,2,:], axis=0)
    sums2ndOPlus = np.sum((QoI_fine_interp[:,2,:] - QoI_coarse_interp[:,2,:])**2, axis=0)

    # ND NO+
    sums1ndNOPlus = np.sum( QoI_fine_interp[:,3,:] - QoI_coarse_interp[:,3,:], axis=0)
    sums2ndNOPlus = np.sum((QoI_fine_interp[:,3,:] - QoI_coarse_interp[:,3,:])**2, axis=0)

    # ND N2+
    sums1ndN2Plus = np.sum( QoI_fine_interp[:,4,:] - QoI_coarse_interp[:,4,:], axis=0)
    sums2ndN2Plus = np.sum((QoI_fine_interp[:,4,:] - QoI_coarse_interp[:,4,:])**2, axis=0)

    # ND NO+
    sums1ndO2Plus = np.sum( QoI_fine_interp[:,5,:] - QoI_coarse_interp[:,5,:], axis=0)
    sums2ndO2Plus = np.sum((QoI_fine_interp[:,5,:] - QoI_coarse_interp[:,5,:])**2, axis=0) 
    
    # e-
    sums1elecMinus = np.sum( QoI_fine_interp[:,6,:] - QoI_coarse_interp[:,6,:], axis=0)
    sums2elecMinus = np.sum((QoI_fine_interp[:,6,:] - QoI_coarse_interp[:,6,:])**2, axis=0)

    # N+
    sums1NPlus = np.sum( QoI_fine_interp[:,7,:] - QoI_coarse_interp[:,7,:], axis=0)
    sums2NPlus = np.sum((QoI_fine_interp[:,7,:] - QoI_coarse_interp[:,7,:])**2, axis=0)

    # O+
    sums1OPlus = np.sum( QoI_fine_interp[:,8,:] - QoI_coarse_interp[:,8,:], axis=0)
    sums2OPlus = np.sum((QoI_fine_interp[:,8,:] - QoI_coarse_interp[:,8,:])**2, axis=0)

    # NO+
    sums1NOPlus = np.sum( QoI_fine_interp[:,9,:] - QoI_coarse_interp[:,9,:], axis=0)
    sums2NOPlus = np.sum((QoI_fine_interp[:,9,:] - QoI_coarse_interp[:,9,:])**2, axis=0)
    
    # N2+
    sums1N2Plus = np.sum( QoI_fine_interp[:,10,:] - QoI_coarse_interp[:,10,:], axis=0)
    sums2N2Plus = np.sum((QoI_fine_interp[:,10,:] - QoI_coarse_interp[:,10,:])**2, axis=0)

    # O2+
    sums1O2Plus = np.sum( QoI_fine_interp[:,11,:] - QoI_coarse_interp[:,11,:], axis=0)
    sums2O2Plus = np.sum((QoI_fine_interp[:,11,:] - QoI_coarse_interp[:,11,:])**2, axis=0)   

    # N
    sums1N = np.sum( QoI_fine_interp[:,12,:] - QoI_coarse_interp[:,12,:], axis=0)
    sums2N = np.sum((QoI_fine_interp[:,12,:] - QoI_coarse_interp[:,12,:])**2, axis=0) 

    # O
    sums1O = np.sum( QoI_fine_interp[:,13,:] - QoI_coarse_interp[:,13,:], axis=0)
    sums2O = np.sum((QoI_fine_interp[:,13,:] - QoI_coarse_interp[:,13,:])**2, axis=0) 

    # NO
    sums1NO = np.sum( QoI_fine_interp[:,14,:] - QoI_coarse_interp[:,14,:], axis=0)
    sums2NO = np.sum((QoI_fine_interp[:,14,:] - QoI_coarse_interp[:,14,:])**2, axis=0) 

    # N2
    sums1N2 = np.sum( QoI_fine_interp[:,15,:] - QoI_coarse_interp[:,15,:], axis=0)
    sums2N2 = np.sum((QoI_fine_interp[:,15,:] - QoI_coarse_interp[:,15,:])**2, axis=0) 

    # O2
    sums1O2 = np.sum( QoI_fine_interp[:,16,:] - QoI_coarse_interp[:,16,:], axis=0)
    sums2O2 = np.sum((QoI_fine_interp[:,16,:] - QoI_coarse_interp[:,16,:])**2, axis=0) 
    
    # Temperatures
    sums1Ttr = np.sum( QoI_fine_interp[:,18,:] - QoI_coarse_interp[:,18,:], axis=0)
    sums2Ttr = np.sum((QoI_fine_interp[:,18,:] - QoI_coarse_interp[:,18,:])**2, axis=0)
    sums1Tve = np.sum( QoI_fine_interp[:,19,:] - QoI_coarse_interp[:,19,:], axis=0)
    sums2Tve = np.sum((QoI_fine_interp[:,19,:] - QoI_coarse_interp[:,19,:])**2, axis=0)

    # M
    sums1M = np.sum( QoI_fine_interp[:,20,:] - QoI_coarse_interp[:,20,:], axis=0)
    sums2M = np.sum((QoI_fine_interp[:,20,:] - QoI_coarse_interp[:,20,:])**2, axis=0)
    
    # End of recording time
    end   = time.time() 
    cost  = nproc*(end-start)    # Cost computation
        
    return xnodesc_ref, sums1, sums2, sums5, sums6, sums1ndelecMinus, sums2ndelecMinus, sums1ndNPlus, sums2ndNPlus, sums1ndOPlus, sums2ndOPlus, sums1ndNOPlus, sums2ndNOPlus, sums1ndN2Plus, sums2ndN2Plus, sums1ndO2Plus, sums2ndO2Plus, sums1elecMinus, sums2elecMinus, sums1NPlus, sums2NPlus, sums1OPlus, sums2OPlus, sums1NOPlus, sums2NOPlus, sums1N2Plus, sums2N2Plus, sums1O2Plus, sums2O2Plus, sums1N, sums2N, sums1O, sums2O, sums1NO, sums2NO, sums1N2, sums2N2, sums1O2, sums2O2, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost


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

