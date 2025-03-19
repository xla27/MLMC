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
    model_outputs_nd_elecMinus   = []
    model_outputs_nd_NPlus       = [] 
    model_outputs_nd_OPlus       = [] 
    model_outputs_nd_NOPlus      = [] 
    model_outputs_nd_N2Plus      = [] 
    model_outputs_nd_O2Plus      = []
    model_outputs_beta_elecMinus = [] 
    model_outputs_beta_NPlus     = [] 
    model_outputs_beta_OPlus     = [] 
    model_outputs_beta_NOPlus    = [] 
    model_outputs_beta_N2Plus    = [] 
    model_outputs_beta_O2Plus    = []
    model_outputs_beta_N         = []
    model_outputs_beta_O         = []
    model_outputs_beta_NO        = []
    model_outputs_beta_N2        = []
    model_outputs_beta_O2        = []
    model_outputs_P              = []
    model_outputs_Ttr            = []
    model_outputs_Tve            = []
    model_outputs_M              = []
    
    xnodes_list = []
    samples = 0

    # if l = 0 cfd_call does not need to be called as mesh adaptation
    if level == 0:
        cfd = cfd_call
        (nproc, baseFolder, workingFolder) = args
        baseFolder2 = baseFolder.replace('AIR11_AMG','AIR11')
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

        [nd_elecMinus, nd_NPlus, nd_OPlus, nd_NOPlus, nd_N2Plus, nd_O2Plus, beta_elecMinus,
         beta_NPlus, beta_OPlus, beta_NOPlus, beta_N2Plus, beta_O2Plus, beta_N, beta_O, beta_NO, 
         beta_N2, beta_O2, P_i, Ttr_i, Tve_i, M_i, xnodes] = cfd(type, valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, level, samples, *args2)
        samples = samples + 1
        
        xnodes_list.append(xnodes)
        
        # storing samples observation
        model_outputs_nd_elecMinus.append(nd_elecMinus) 
        model_outputs_nd_NPlus.append(nd_NPlus)
        model_outputs_nd_OPlus.append(nd_OPlus)
        model_outputs_nd_NOPlus.append(nd_NOPlus)         
        model_outputs_nd_N2Plus.append(nd_N2Plus) 
        model_outputs_nd_O2Plus.append(nd_O2Plus)
        model_outputs_beta_elecMinus.append(beta_elecMinus)
        model_outputs_beta_NPlus.append(beta_NPlus) 
        model_outputs_beta_OPlus.append(beta_OPlus) 
        model_outputs_beta_NOPlus.append(beta_NOPlus)         
        model_outputs_beta_N2Plus.append(beta_N2Plus) 
        model_outputs_beta_O2Plus.append(beta_O2Plus)
        model_outputs_beta_N.append(beta_N)
        model_outputs_beta_O.append(beta_O)
        model_outputs_beta_NO.append(beta_NO)
        model_outputs_beta_N2.append(beta_N2)
        model_outputs_beta_O2.append(beta_O2)
        model_outputs_P.append(P_i)
        model_outputs_Ttr.append(Ttr_i)
        model_outputs_Tve.append(Tve_i)
        model_outputs_M.append(M_i)

    model_outputs = [model_outputs_nd_elecMinus, model_outputs_nd_NPlus, model_outputs_nd_OPlus, model_outputs_nd_NOPlus, model_outputs_nd_N2Plus, model_outputs_nd_O2Plus,
                     model_outputs_beta_elecMinus, model_outputs_beta_NPlus, model_outputs_beta_OPlus, model_outputs_beta_NOPlus, model_outputs_beta_N2Plus, model_outputs_beta_O2Plus,
                     model_outputs_beta_N, model_outputs_beta_O, model_outputs_beta_NO, model_outputs_beta_N2, model_outputs_beta_O2, 
                     model_outputs_P, model_outputs_Ttr, model_outputs_Tve, model_outputs_M]

    # finding the coarsest xnode and interpolating over it al the other values
    xnodes_ref = max(xnodes_list, key=len)

    # moving average on the samples
    SF = 0.015
    ws = max(int(len(xnodes_ref) * SF), 1)
    for qoi in range(21):
        for smp in range(samples):
            model_outputs[qoi][smp] = moving_average(model_outputs[qoi][smp], ws)

    # interpolating wall quantities to a common xref
    model_outputs_interp = np.zeros((21, samples, len(xnodes_ref)))

    for qoi in range(21):
        for smp in range(samples):
            if level!= 0 or (level == 1 and type == 'COARSE'):
                model_outputs_interp[qoi, smp, :] = model_outputs[qoi][smp]
            else:
                model_outputs_interp[qoi, smp, :] = interp1d(xnodes_list[smp], model_outputs[qoi][smp], kind='linear', fill_value='extrapolate')(xnodes_ref)

    # Compute Sobol indices for each x point (wall point)
    S1 = np.zeros((21, len(xnodes_ref), x))   
    
    # Compute the Sobol' indices wrt the other model_outputs (now there is not just one QoI)
    for qoi in range(21):
        for x in range(len(xnodes_ref)):
            
            Sx = sobol.analyze(problem, model_outputs_interp[qoi, :, x], calc_second_order=False) 
            S1[qoi, x, :] = Sx['S1']

    return xnodes_ref, S1


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

