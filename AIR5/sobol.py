import os, sys
import numpy as np
from SALib.analyze import sobol

from AIR5 import cfd_call


# Function to perform Sobol analysis on a given level (fine or coarse)
def compute_sobol_indices(type, param_values, l, problem, *args):
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
    
    xnode_vec = None
    i = 0

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

        beta_N, beta_O, beta_NO, beta_N2, beta_O2, P_i, Ttr_i, Tve_i, M_i, xnodes = cfd_call(type, valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)
        i = i + 1
        if xnode_vec is None:
            x_vec = xnodes
        
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
    S1N_M  = []; S1N_T  = []; S1N_P  = []; S1N_beta  = []   # sobol indices with N as QoI
    S1O_M  = []; S1O_T  = []; S1O_P  = []; S1O_beta  = []   # sobol indices with O as QoI
    S1NO_M = []; S1NO_T = []; S1NO_P = []; S1NO_beta = []   # sobol indices with NO as QoI
    S1N2_M = []; S1N2_T = []; S1N2_P = []; S1N2_beta = []   # sobol indices with N2 as QoI
    S1O2_M = []; S1O2_T = []; S1O2_P = []; S1O2_beta = []   # sobol indices with O2 as QoI
    
    S1P_M   = []; S1P_T   = []; S1P_P   = []; S1P_beta   = []; # sobol indices with P as QoI
    S1Ttr_M = []; S1Ttr_T = []; S1Ttr_P = []; S1Ttr_beta = []; # sobol indices with Ttr as QoI
    S1Tve_M = []; S1Tve_T = []; S1Tve_P = []; S1Tve_beta = []; # sobol indices with Tve as QoI
    S1M_M   = []; S1M_T   = []; S1M_P   = []; S1M_beta   = []; # sobol indices with M as QoI
    
    # Compute the Sobol' indices wrt the other model_outputs (now there is not just one QoI)
    for j in range(len(x_vec)):

        # computation of S related to N
        Si_N = sobol.analyze(problem, model_outputsN[:, j], calc_second_order=False) 
        S1N_M.append(   Si_N['S1'][0])
        S1N_T.append(   Si_N['S1'][1])
        S1N_P.append(   Si_N['S1'][2])
        S1N_beta.append(Si_N['S1'][3])
        
        # computation of S related to O
        Si_O = sobol.analyze(problem, model_outputsO[:, j], calc_second_order=False) 
        S1O_M.append(   Si_O['S1'][0])
        S1O_T.append(   Si_O['S1'][1])
        S1O_P.append(   Si_O['S1'][2])
        S1O_beta.append(Si_O['S1'][3])
        
        # computation of S related to NO
        Si_NO = sobol.analyze(problem, model_outputsNO[:, j], calc_second_order=False) 
        S1NO_M.append(   Si_NO['S1'][0])
        S1NO_T.append(   Si_NO['S1'][1])
        S1NO_P.append(   Si_NO['S1'][2])
        S1NO_beta.append(Si_NO['S1'][3])  
        
        # computation of S related to N2
        Si_N2 = sobol.analyze(problem, model_outputsN2[:, j], calc_second_order=False) 
        S1N2_M.append(   Si_N2['S1'][0])
        S1N2_T.append(   Si_N2['S1'][1])
        S1N2_P.append(   Si_N2['S1'][2])
        S1N2_beta.append(Si_N2['S1'][3]) 
        
        # computation of S related to O2
        Si_O2 = sobol.analyze(problem, model_outputsO2[:, j], calc_second_order=False) 
        S1O2_M.append(   Si_O2['S1'][0])
        S1O2_T.append(   Si_O2['S1'][1])
        S1O2_P.append(   Si_O2['S1'][2])
        S1O2_beta.append(Si_O2['S1'][3])   
        
        # computation of S related to P 
        Si_P = sobol.analyze(problem, model_outputsP[:, j], calc_second_order=False) 
        S1P_M.append(   Si_P['S1'][0])
        S1P_T.append(   Si_P['S1'][1])
        S1P_P.append(   Si_P['S1'][2])
        S1P_beta.append(Si_P['S1'][3])      
        
        # computation of S related to Ttr
        Si_Ttr = sobol.analyze(problem, model_outputsTtr[:, j], calc_second_order=False) 
        S1Ttr_M.append(   Si_Ttr['S1'][0])
        S1Ttr_T.append(   Si_Ttr['S1'][1])
        S1Ttr_P.append(   Si_Ttr['S1'][2])
        S1Ttr_beta.append(Si_Ttr['S1'][3])                   
        
        # computation of S related to Tve
        Si_Tve = sobol.analyze(problem, model_outputsTve[:, j], calc_second_order=False) 
        S1Tve_M.append(   Si_Tve['S1'][0])
        S1Tve_T.append(   Si_Tve['S1'][1])
        S1Tve_P.append(   Si_Tve['S1'][2])
        S1Tve_beta.append(Si_Tve['S1'][3])    
        
        # computation of S related to M
        Si_M = sobol.analyze(problem, model_outputsM[:, j], calc_second_order=False) 
        S1M_M.append(   Si_M['S1'][0])
        S1M_T.append(   Si_M['S1'][1])
        S1M_P.append(   Si_M['S1'][2])
        S1M_beta.append(Si_M['S1'][3])         
                
    return x_vec, S1N_M, S1N_T, S1N_P, S1N_beta, S1O_M, S1O_T, S1O_P, S1O_beta, S1NO_M, S1NO_T, S1NO_P, S1NO_beta, S1N2_M, S1N2_T, S1N2_P, S1N2_beta, S1O2_M, S1O2_T, S1O2_P, S1O2_beta, S1P_M, S1P_T, S1P_P, S1P_beta, S1Ttr_M, S1Ttr_T, S1Ttr_P, S1Ttr_beta, S1Tve_M, S1Tve_T, S1Tve_P, S1Tve_beta, S1M_M, S1M_T, S1M_P, S1M_beta
