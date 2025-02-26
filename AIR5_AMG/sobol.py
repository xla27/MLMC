import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.analyze import sobol

from AIR5_AMG import cfd_call_amg
from AIR5     import cfd_call


# Function to perform Sobol analysis on a given level (fine or coarse)
def compute_sobol_indices(type, param_values, level, problem, *args):
    '''
    Function to compute Sobol indices at a given level.
    The function considers fine or coarse part depending on "type" input.    
    '''
    # Run the model for each sample
    model_outputsN   = []
    model_outputsO   = []
    model_outputsNO  = []
    model_outputsN2  = []
    model_outputsO2  = []
    model_outputsP   = []
    model_outputsTtr = []
    model_outputsTve = []
    model_outputsM   = []
    
    xnodes_list = []
    samples = 0

    # if l = 0 cfd_call does not need to be called as mesh adaptation
    if level == 0:
        cfd = cfd_call
        (nproc, baseFolder, workingFolder) = args
        baseFolder2 = baseFolder.replace('AIR5_AMG','AIR5')
        args2 = (nproc, baseFolder2, workingFolder)
    else:
        cfd = cfd_call_amg
        args2 = args

    # cycling on UQ variables samples
    for X in param_values:
        
        M   = '{:.3f}'.format(X[0])
        T   = '{:.1f}'.format(X[1])
        P   = '{:.1f}'.format(X[2])
        Bn2 = '{:.3f}'.format(X[3])
        Bo2 = '{:.3f}'.format(1 - X[3])
        
        valIns_M   = str(M)
        valIns_T   = str(T)
        valIns_P   = str(P)
        valIns_Bn2 = str(Bn2)
        valIns_Bo2 = str(Bo2)

        beta_N, beta_O, beta_NO, beta_N2, beta_O2, P_i, Ttr_i, Tve_i, M_i, xnodes = cfd(type, valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, samples, *args2)
        samples = samples + 1
        
        xnodes_list.append(xnodes)
        
        # storing samples observation
        model_outputsN.append(beta_N)
        model_outputsO.append(beta_O)
        model_outputsNO.append(beta_NO)
        model_outputsN2.append(beta_N2)
        model_outputsO2.append(beta_O2)
        model_outputsP.append(P_i)
        model_outputsTtr.append(Ttr_i)
        model_outputsTve.append(Tve_i)
        model_outputsM.append(M_i)

    # finding the coarsest xnode and interpolating over it al the other values
    xnodes_ref = max(xnodes_list, key=len)

    # moving average on the samples
    SF = 0.015
    ws = max(int(len(xnodes_ref) * SF), 1)
    for i in range(samples):
        model_outputsN[i]   = moving_average(model_outputsN[i],   ws)
        model_outputsO[i]   = moving_average(model_outputsO[i],   ws)
        model_outputsNO[i]  = moving_average(model_outputsNO[i],  ws)
        model_outputsN2[i]  = moving_average(model_outputsN2[i],  ws)
        model_outputsO2[i]  = moving_average(model_outputsO2[i],  ws)
        model_outputsP[i]   = moving_average(model_outputsP[i],   ws)
        model_outputsTtr[i] = moving_average(model_outputsTtr[i], ws)
        model_outputsTve[i] = moving_average(model_outputsTve[i], ws)
        model_outputsM[i]   = moving_average(model_outputsM[i],   ws)


    for i in range(samples):
        model_outputsN[i]   = interp1d(xnodes_list[i], model_outputsN[i],   kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsO[i]   = interp1d(xnodes_list[i], model_outputsO[i],   kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsNO[i]  = interp1d(xnodes_list[i], model_outputsNO[i],  kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsN2[i]  = interp1d(xnodes_list[i], model_outputsN2[i],  kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsO2[i]  = interp1d(xnodes_list[i], model_outputsO2[i],  kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsP[i]   = interp1d(xnodes_list[i], model_outputsP[i],   kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsTtr[i] = interp1d(xnodes_list[i], model_outputsTtr[i], kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsTve[i] = interp1d(xnodes_list[i], model_outputsTve[i], kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputsM[i]   = interp1d(xnodes_list[i], model_outputsM[i],   kind='linear', fill_value='extrapolate')(xnodes_ref)

    model_outputsN   = np.array(model_outputsN)
    model_outputsO   = np.array(model_outputsO)
    model_outputsNO  = np.array(model_outputsNO)
    model_outputsN2  = np.array(model_outputsN2)
    model_outputsO2  = np.array(model_outputsO2)
    model_outputsP   = np.array(model_outputsP)
    model_outputsTtr = np.array(model_outputsTtr)
    model_outputsTve = np.array(model_outputsTve)
    model_outputsM   = np.array(model_outputsM)

    # Compute Sobol indices for each x point (wall point)
    S1N   = np.zeros((4, len(xnodes_ref)))    # sobol indices with N as QoI
    S1O   = np.zeros((4, len(xnodes_ref)))    # sobol indices with O as QoI
    S1NO  = np.zeros((4, len(xnodes_ref)))    # sobol indices with NO as QoI
    S1N2  = np.zeros((4, len(xnodes_ref)))    # sobol indices with N2 as QoI
    S1O2  = np.zeros((4, len(xnodes_ref)))    # sobol indices with O2 as QoI
    S1P   = np.zeros((4, len(xnodes_ref)))  # sobol indices with P as QoI
    S1Ttr = np.zeros((4, len(xnodes_ref)))  # sobol indices with Ttr as QoI
    S1Tve = np.zeros((4, len(xnodes_ref)))  # sobol indices with Tve as QoI
    S1M   = np.zeros((4, len(xnodes_ref)))  # sobol indices with M as QoI
    
    # Compute the Sobol' indices wrt the other model_outputs (now there is not just one QoI)
    for j in range(len(xnodes_ref)):

        # computation of S related to N
        Si_N = sobol.analyze(problem, model_outputsN[:, j], calc_second_order=False) 
        S1N[:,j] = Si_N['S1']
        
        # computation of S related to O
        Si_O = sobol.analyze(problem, model_outputsO[:, j], calc_second_order=False) 
        S1O[:,j] = Si_O['S1']
        
        # computation of S related to NO
        Si_NO = sobol.analyze(problem, model_outputsNO[:, j], calc_second_order=False) 
        S1NO[:,j] = Si_NO['S1']
        
        # computation of S related to N2
        Si_N2 = sobol.analyze(problem, model_outputsN2[:, j], calc_second_order=False) 
        S1N2[:,j] = Si_N2['S1']
        
        # computation of S related to O2
        Si_O2 = sobol.analyze(problem, model_outputsO2[:, j], calc_second_order=False) 
        S1O2[:,j] = Si_O2['S1']
        
        # computation of S related to P 
        Si_P = sobol.analyze(problem, model_outputsP[:, j], calc_second_order=False) 
        S1P[:,j] = Si_P['S1']
        
        # computation of S related to Ttr
        Si_Ttr = sobol.analyze(problem, model_outputsTtr[:, j], calc_second_order=False) 
        S1Ttr[:,j] = Si_Ttr['S1']

        # computation of S related to Tve
        Si_Tve = sobol.analyze(problem, model_outputsTve[:, j], calc_second_order=False) 
        S1Tve[:,j] = Si_Tve['S1']

        # computation of S related to M
        Si_M = sobol.analyze(problem, model_outputsM[:, j], calc_second_order=False) 
        S1M[:,j] = Si_M['S1']
                
    return xnodes_ref, S1N, S1O, S1NO, S1N2, S1O2, S1P, S1Ttr, S1Tve, S1M
    

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
