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
    sums1 = 0  # sum of difference between static Pressure at fine and coarse
    sums2 = 0  # sum of square of difference between static Pressure at fine and coarse
    sums5 = 0  # sum of static Pressure at fine 
    sums6 = 0  # sum of square of static Pressure at fine 

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

    QoI_coarse = [None] * N_samples
    QoI_fine   = [None] * N_samples

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
            QoI_fine[i] = list(cfd_call('FINE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args2))

            # No call to CFD with coearse mesh, coarse results set to zero as it is the starting level
            xnodesc = QoI_fine[i][-1]
            QoI_coarse[i] = [[0.] * len(xnodesc) for i in range(0,9)]
            QoI_coarse[i].append(xnodesc)

        else:

            # Call to CFD with fine mesh
            QoI_fine[i] = list(cfd_call_amg('FINE',valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args))

            # Call to CFD with coarse mesh
            QoI_coarse[i] = list(cfd_call_amg('COARSE', valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, i, *args))


    # Extracting the reference xnodes for the coarse level as the xnodes component of COARSE samples with the lowest number of elements
    xnodesc_list = [QoI_coarse[i][-1] for i in range(N_samples)]

    xnodesc_ref = min(xnodesc_list, key=len)

    # windows size for moving average
    ws = max(int(len(xnodesc_ref) * SF), 1)

    QoI_coarse_interp = np.zeros((N_samples, 9, len(xnodesc_ref)))
    QoI_fine_interp   = np.zeros((N_samples, 9, len(xnodesc_ref)))

    # Interpolating the coarse and fine on the reference coarse and doing the moving average of the samples
    for i in range(N_samples):

        for j in range(0,9):

            if level >= 2:  # the interpolation is needed for both coarse and fine on reference xnodesc

                QoI_coarse_interp[i,j,:] = interp1d(QoI_coarse[i][-1], QoI_coarse[i][j], kind='linear', fill_value='extrapolate')(xnodesc_ref)
                QoI_fine_interp[i,j,:]   = interp1d(QoI_fine[i][-1],   QoI_fine[i][j],   kind='linear', fill_value='extrapolate')(xnodesc_ref)

            elif level == 1: # the interpolation is needed for fine on reference xnodesc

                QoI_fine_interp[i,j,:] = interp1d(QoI_fine[i][-1], QoI_fine[i][j], kind='linear', fill_value='extrapolate')(xnodesc_ref)

            elif level == 0:

                QoI_fine_interp[i,j,:] = np.array(QoI_fine[i][j])

            # moving average
            QoI_coarse_interp[i,j,:] = moving_average(QoI_coarse_interp[i,j,:], ws)
            QoI_fine_interp[i,j,:]   = moving_average(QoI_fine_interp[i,j,:],   ws)

        
    # RESULTS UPDATE

    # [beta_n, beta_o, beta_no, beta_n2, beta_o2, P, Ttr, Tve, M, xnodes]

    # P
    sums1 = np.sum(QoI_fine_interp[:,5,:]    - QoI_coarse_interp[:,5,:], axis=0)
    sums2 = np.sum(QoI_fine_interp[:,5,:]**2 - QoI_coarse_interp[:,5,:]**2, axis=0)
    sums5 = np.sum(QoI_fine_interp[:,5,:], axis=0)
    sums6 = np.sum(QoI_fine_interp[:,5,:]**2, axis=0)

    # N
    sums1N = np.sum(QoI_fine_interp[:,0,:]    - QoI_coarse_interp[:,0,:], axis=0)
    sums2N = np.sum(QoI_fine_interp[:,0,:]**2 - QoI_coarse_interp[:,0,:]**2, axis=0)

    # O
    sums1O = np.sum(QoI_fine_interp[:,1,:]    - QoI_coarse_interp[:,1,:], axis=0)
    sums2O = np.sum(QoI_fine_interp[:,1,:]**2 - QoI_coarse_interp[:,1,:]**2, axis=0)

    # NO
    sums1NO = np.sum(QoI_fine_interp[:,2,:]    - QoI_coarse_interp[:,2,:], axis=0)
    sums2NO = np.sum(QoI_fine_interp[:,2,:]**2 - QoI_coarse_interp[:,2,:]**2, axis=0)

    # N2
    sums1N2 = np.sum(QoI_fine_interp[:,3,:]    - QoI_coarse_interp[:,3,:], axis=0)
    sums2N2 = np.sum(QoI_fine_interp[:,3,:]**2 - QoI_coarse_interp[:,3,:]**2, axis=0)

    # O2
    sums1O2 = np.sum(QoI_fine_interp[:,4,:]    - QoI_coarse_interp[:,4,:], axis=0)
    sums2O2 = np.sum(QoI_fine_interp[:,4,:]**2 - QoI_coarse_interp[:,4,:]**2, axis=0)
    
    # Temperatures
    sums1Ttr = np.sum(QoI_fine_interp[:,6,:]    - QoI_coarse_interp[:,6,:], axis=0)
    sums2Ttr = np.sum(QoI_fine_interp[:,6,:]**2 - QoI_coarse_interp[:,6,:]**2, axis=0)
    sums1Tve = np.sum(QoI_fine_interp[:,7,:]    - QoI_coarse_interp[:,7,:], axis=0)
    sums2Tve = np.sum(QoI_fine_interp[:,7,:]**2 - QoI_coarse_interp[:,7,:]**2, axis=0)

    # M
    sums1M = np.sum(QoI_fine_interp[:,8,:]    - QoI_coarse_interp[:,8,:], axis=0)
    sums2M = np.sum(QoI_fine_interp[:,8,:]**2 - QoI_coarse_interp[:,8,:]**2, axis=0)
    
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

