import os, sys
import numpy as np
from SALib.analyze import sobol

from cfd_call import cfd_call


# Function to perform Sobol analysis on a given level (fine or coarse)
def compute_sobol_indices(type, param_values, l, problem, *args):
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
    model_outputs_bta_O2         = []
    model_outputsP               = []
    model_outputsTtr             = []
    model_outputsTve             = []
    model_outputsM               = []
    
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

        [xnodes, nd_elecMinus, nd_NPlus, nd_OPlus, nd_NOPlus, nd_N2Plus, nd_O2Plus, beta_elecMinus,
         beta_NPlus, beta_OPlus, beta_NOPlus, beta_N2Plus, beta_O2Plus, beta_N, beta_O, beta_NO, 
         beta_N2, beta_O2, P_i, Ttr_i, Tve_i, M_i] = cfd_call(type, valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args)
        
        i = i + 1
        if xnode_vec is None:
            x_vec = xnodes
        
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
        model_outputs_bta_O2.append(beta_O2)
        model_outputsP.append(P_i)
        model_outputsTtr.append(Ttr_i)
        model_outputsTve.append(Tve_i)
        model_outputsM.append(M_i)

    model_outputs_nd_elecMinus   = np.array(model_outputs_nd_elecMinus)
    model_outputs_nd_NPlus       = np.array(model_outputs_nd_NPlus)
    model_outputs_nd_OPlus       = np.array(model_outputs_nd_OPlus)
    model_outputs_nd_NOPlus      = np.array(model_outputs_nd_NOPlus)        
    model_outputs_nd_N2Plus      = np.array(model_outputs_nd_N2Plus)
    model_outputs_nd_O2Plus      = np.array(model_outputs_nd_O2Plus)
    model_outputs_beta_elecMinus = np.array(model_outputs_beta_elecMinus)
    model_outputs_beta_NPlus     = np.array(model_outputs_beta_NPlus) 
    model_outputs_beta_OPlus     = np.array(model_outputs_beta_OPlus)
    model_outputs_beta_NOPlus    = np.array(model_outputs_beta_NOPlus)        
    model_outputs_beta_N2Plus    = np.array(model_outputs_beta_N2Plus) 
    model_outputs_beta_O2Plus    = np.array(model_outputs_beta_O2Plus)
    model_outputs_beta_N         = np.array(model_outputs_beta_N)
    model_outputs_beta_O         = np.array(model_outputs_beta_O)
    model_outputs_beta_NO        = np.array(model_outputs_beta_NO)
    model_outputs_beta_N2        = np.array(model_outputs_beta_N2)
    model_outputs_bta_O2         = np.array(model_outputs_bta_O2)
    model_outputsP               = np.array(model_outputsP)
    model_outputsTtr             = np.array(model_outputsTtr)
    model_outputsTve             = np.array(model_outputsTve)
    model_outputsM               = np.array(model_outputsM)

    # Compute Sobol indices for each x point (wall point)
    S1nd_elecMinus_M = []; S1nd_elecMinus_T = []; S1nd_elecMinus_P = []; S1nd_elecMinus_beta = [] # sobol indices with ND e- as QoI
    S1nd_NPlus_M     = []; S1nd_NPlus_T     = []; S1nd_NPlus_P     = []; S1nd_NPlus_beta     = [] # sobol indices with ND N+ as QoI 
    S1nd_OPlus_M     = []; S1nd_OPlus_T     = []; S1nd_OPlus_P     = []; S1nd_OPlus_beta     = [] # sobol indices with ND O+ as QoI
    S1nd_NOPlus_M    = []; S1nd_NOPlus_T    = []; S1nd_NOPlus_P    = []; S1nd_NOPlus_beta    = [] # sobol indices with ND NO+ as QoI
    S1nd_N2Plus_M    = []; S1nd_N2Plus_T    = []; S1nd_N2Plus_P    = []; S1nd_N2Plus_beta    = [] # sobol indices with ND N2+ as QoI
    S1nd_O2Plus_M    = []; S1nd_O2Plus_T    = []; S1nd_O2Plus_P    = []; S1nd_O2Plus_beta    = [] # sobol indices with ND O2+ as QoI

    S1beta_elecMinus_M = []; S1beta_elecMinus_T = []; S1beta_elecMinus_P = []; S1beta_elecMinus_beta = [] # sobol indices with e- as QoI
    S1beta_NPlus_M     = []; S1beta_NPlus_T     = []; S1beta_NPlus_P     = []; S1beta_NPlus_beta     = [] # sobol indices with N+ as QoI
    S1beta_OPlus_M     = []; S1beta_OPlus_T     = []; S1beta_OPlus_P     = []; S1beta_OPlus_beta     = [] # sobol indices with O+ as QoI
    S1beta_NOPlus_M    = []; S1beta_NOPlus_T    = []; S1beta_NOPlus_P    = []; S1beta_NOPlus_beta    = [] # sobol indices with NO+ as QoI
    S1beta_N2Plus_M    = []; S1beta_N2Plus_T    = []; S1beta_N2Plus_P    = []; S1beta_N2Plus_beta    = [] # sobol indices with N2+ as QoI
    S1beta_O2Plus_M    = []; S1beta_O2Plus_T    = []; S1beta_O2Plus_P    = []; S1beta_O2Plus_beta    = [] # sobol indices with O2+ QoI
    S1betaN_M          = []; S1betaN_T          = []; S1betaN_P          = []; S1betaN_beta          = [] # sobol indices with N as QoI
    S1betaO_M          = []; S1betaO_T          = []; S1betaO_P          = []; S1betaO_beta          = [] # sobol indices with O as QoI
    S1betaNO_M         = []; S1betaNO_T         = []; S1betaNO_P         = []; S1betaNO_beta         = [] # sobol indices with NO as QoI
    S1betaN2_M         = []; S1betaN2_T         = []; S1betaN2_P         = []; S1betaN2_beta         = [] # sobol indices with N2 as QoI
    S1betaO2_M         = []; S1betaO2_T         = []; S1betaO2_P         = []; S1betaO2_beta         = [] # sobol indices with O2 as QoI
    
    S1P_M   = []; S1P_T   = []; S1P_P   = []; S1P_beta   = [] # sobol indices with P as QoI
    S1Ttr_M = []; S1Ttr_T = []; S1Ttr_P = []; S1Ttr_beta = [] # sobol indices with Ttr as QoI
    S1Tve_M = []; S1Tve_T = []; S1Tve_P = []; S1Tve_beta = [] # sobol indices with Tve as QoI
    S1M_M   = []; S1M_T   = []; S1M_P   = []; S1M_beta   = [] # sobol indices with M as QoI
    
    # Compute the Sobol' indices wrt the other model_outputs (now there is not just one QoI)
    for j in range(len(x_vec)):

        # computation of S related to ND e-
        Si_nd_elecMinus = sobol.analyze(problem, model_outputs_nd_elecMinus[:, j], calc_second_order=False)
        S1nd_elecMinus_M.append(   Si_nd_elecMinus['S1'][0]) 
        S1nd_elecMinus_T.append(   Si_nd_elecMinus['S1'][1]) 
        S1nd_elecMinus_P.append(   Si_nd_elecMinus['S1'][2])
        S1nd_elecMinus_beta.append(Si_nd_elecMinus['S1'][3])

        # computation of S related to ND N+
        Si_nd_NPlus = sobol.analyze(problem, model_outputs_nd_NPlus[:, j], calc_second_order=False)
        S1nd_NPlus_M.append(   Si_nd_NPlus['S1'][0]) 
        S1nd_NPlus_T.append(   Si_nd_NPlus['S1'][1]) 
        S1nd_NPlus_P.append(   Si_nd_NPlus['S1'][2]) 
        S1nd_NPlus_beta.append(Si_nd_NPlus['S1'][3])

        # computation of S related to ND O+
        Si_nd_OPlus = sobol.analyze(problem, model_outputs_nd_OPlus[:, j], calc_second_order=False)
        S1nd_OPlus_M.append(   Si_nd_OPlus['S1'][0]) 
        S1nd_OPlus_T.append(   Si_nd_OPlus['S1'][1])
        S1nd_OPlus_P.append(   Si_nd_OPlus['S1'][2]) 
        S1nd_OPlus_beta.append(Si_nd_OPlus['S1'][3])

        # computation of S related to ND NO+
        Si_nd_NOPlus = sobol.analyze(problem, model_outputs_nd_NOPlus[:, j], calc_second_order=False)
        S1nd_NOPlus_M.append(   Si_nd_NOPlus['S1'][0])
        S1nd_NOPlus_T.append(   Si_nd_NOPlus['S1'][1]) 
        S1nd_NOPlus_P.append(   Si_nd_NOPlus['S1'][2]) 
        S1nd_NOPlus_beta.append(Si_nd_NOPlus['S1'][3])

        # computation of S related to ND N2+
        Si_nd_N2Plus = sobol.analyze(problem, model_outputs_nd_N2Plus[:, j], calc_second_order=False)
        S1nd_N2Plus_M.append(   Si_nd_N2Plus['S1'][0]) 
        S1nd_N2Plus_T.append(   Si_nd_N2Plus['S1'][1]) 
        S1nd_N2Plus_P.append(   Si_nd_N2Plus['S1'][2]) 
        S1nd_N2Plus_beta.append(Si_nd_N2Plus['S1'][3])

        # computation of S related to ND O2+
        Si_nd_O2Plus = sobol.analyze(problem, model_outputs_nd_O2Plus[:, j], calc_second_order=False)
        S1nd_O2Plus_M.append(   Si_nd_O2Plus['S1'][0])
        S1nd_O2Plus_T.append(   Si_nd_O2Plus['S1'][1]) 
        S1nd_O2Plus_P.append(   Si_nd_O2Plus['S1'][2]) 
        S1nd_O2Plus_beta.append(Si_nd_O2Plus['S1'][3])

        # computation of S related to e-
        Si_beta_elecMinus = sobol.analyze(problem, model_outputs_beta_elecMinus[:, j], calc_second_order=False)
        S1beta_elecMinus_M.append(   Si_beta_elecMinus['S1'][0]) 
        S1beta_elecMinus_T.append(   Si_beta_elecMinus['S1'][1])
        S1beta_elecMinus_P.append(   Si_beta_elecMinus['S1'][2]) 
        S1beta_elecMinus_beta.append(Si_beta_elecMinus['S1'][3])

        # computation of S related to N+
        Si_beta_NPlus = sobol.analyze(problem, model_outputs_beta_NPlus[:, j], calc_second_order=False)
        S1beta_NPlus_M.append(   Si_beta_NPlus['S1'][0]) 
        S1beta_NPlus_T.append(   Si_beta_NPlus['S1'][1]) 
        S1beta_NPlus_P.append(   Si_beta_NPlus['S1'][2]) 
        S1beta_NPlus_beta.append(Si_beta_NPlus['S1'][3])

        # computation of S related to O+
        Si_beta_OPlus = sobol.analyze(problem, model_outputs_beta_OPlus[:, j], calc_second_order=False)
        S1beta_OPlus_M.append(   Si_beta_OPlus['S1'][0]) 
        S1beta_OPlus_T.append(   Si_beta_OPlus['S1'][1])
        S1beta_OPlus_P.append(   Si_beta_OPlus['S1'][2]) 
        S1beta_OPlus_beta.append(Si_beta_OPlus['S1'][3])

        # computation of S related to NO+
        Si_beta_NOPlus = sobol.analyze(problem, model_outputs_beta_NOPlus[:, j], calc_second_order=False) 
        S1beta_NOPlus_M.append(   Si_beta_NOPlus['S1'][0]); 
        S1beta_NOPlus_T.append(   Si_beta_NOPlus['S1'][1]); 
        S1beta_NOPlus_P.append(   Si_beta_NOPlus['S1'][2]); 
        S1beta_NOPlus_beta.append(Si_beta_NOPlus['S1'][3]);

        # computation of S related to N2+
        Si_beta_N2Plus = sobol.analyze(problem, model_outputs_beta_N2Plus[:, j], calc_second_order=False)
        S1beta_N2Plus_M.append(   Si_beta_N2Plus['S1'][0]) 
        S1beta_N2Plus_T.append(   Si_beta_N2Plus['S1'][1]) 
        S1beta_N2Plus_P.append(   Si_beta_N2Plus['S1'][2]) 
        S1beta_N2Plus_beta.append(Si_beta_N2Plus['S1'][3])

        # computation of S related to O2+
        Si_beta_O2Plus = sobol.analyze(problem, model_outputs_beta_O2Plus[:, j], calc_second_order=False)
        S1beta_O2Plus_M.append(Si_beta_O2Plus['S1'][0]);
        S1beta_O2Plus_T.append(Si_beta_O2Plus['S1'][1]); 
        S1beta_O2Plus_P.append(Si_beta_O2Plus['S1'][2]); 
        S1beta_O2Plus_beta.append(Si_beta_O2Plus['S1'][3]);

        # computation of S related to N
        Si_N = sobol.analyze(problem, model_outputs_beta_N[:, j], calc_second_order=False) 
        S1betaN_M.append(   Si_N['S1'][0])
        S1betaN_T.append(   Si_N['S1'][1])
        S1betaN_P.append(   Si_N['S1'][2])
        S1betaN_beta.append(Si_N['S1'][3])
        
        # computation of S related to O
        Si_O = sobol.analyze(problem, model_outputs_beta_O[:, j], calc_second_order=False) 
        S1betaO_M.append(   Si_O['S1'][0])
        S1betaO_T.append(   Si_O['S1'][1])
        S1betaO_P.append(   Si_O['S1'][2])
        S1betaO_beta.append(Si_O['S1'][3])
        
        # computation of S related to NO
        Si_NO = sobol.analyze(problem, model_outputs_beta_NO[:, j], calc_second_order=False) 
        S1betaNO_M.append(   Si_NO['S1'][0])
        S1betaNO_T.append(   Si_NO['S1'][1])
        S1betaNO_P.append(   Si_NO['S1'][2])
        S1betaNO_beta.append(Si_NO['S1'][3])  
        
        # computation of S related to N2
        Si_N2 = sobol.analyze(problem, model_outputs_beta_N2[:, j], calc_second_order=False) 
        S1betaN2_M.append(   Si_N2['S1'][0])
        S1betaN2_T.append(   Si_N2['S1'][1])
        S1betaN2_P.append(   Si_N2['S1'][2])
        S1betaN2_beta.append(Si_N2['S1'][3]) 
        
        # computation of S related to O2
        Si_O2 = sobol.analyze(problem, model_outputs_bta_O2[:, j], calc_second_order=False) 
        S1betaO2_M.append(   Si_O2['S1'][0])
        S1betaO2_T.append(   Si_O2['S1'][1])
        S1betaO2_P.append(   Si_O2['S1'][2])
        S1betaO2_beta.append(Si_O2['S1'][3])   
        
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
                
    return x_vec, S1nd_elecMinus_M, S1nd_elecMinus_T, S1nd_elecMinus_P, S1nd_elecMinus_beta, S1nd_NPlus_M, S1nd_NPlus_T, S1nd_NPlus_P, S1nd_NPlus_beta, S1nd_OPlus_M, S1nd_OPlus_T, S1nd_OPlus_P, S1nd_OPlus_beta, S1nd_NOPlus_M, S1nd_NOPlus_T, S1nd_NOPlus_P, S1nd_NOPlus_beta, S1nd_N2Plus_M, S1nd_N2Plus_T, S1nd_N2Plus_P, S1nd_N2Plus_beta, S1nd_O2Plus_M, S1nd_O2Plus_T, S1nd_O2Plus_P, S1nd_O2Plus_beta, S1beta_elecMinus_M, S1beta_elecMinus_T, S1beta_elecMinus_P, S1beta_elecMinus_beta, S1beta_NPlus_M, S1beta_NPlus_T, S1beta_NPlus_P, S1beta_NPlus_beta, S1beta_OPlus_M, S1beta_OPlus_T, S1beta_OPlus_P, S1beta_OPlus_beta, S1beta_NOPlus_M, S1beta_NOPlus_T, S1beta_NOPlus_P, S1beta_NOPlus_beta, S1beta_N2Plus_M, S1beta_N2Plus_T, S1beta_N2Plus_P, S1beta_N2Plus_beta, S1beta_O2Plus_M, S1beta_O2Plus_T, S1beta_O2Plus_P, S1beta_O2Plus_beta, S1betaN_M, S1betaN_T, S1betaN_P, S1betaN_beta, S1betaO_M, S1betaO_T, S1betaO_P, S1betaO_beta, S1betaNO_M, S1betaNO_T, S1betaNO_P, S1betaNO_beta, S1betaN2_M, S1betaN2_T, S1betaN2_P, S1betaN2_beta, S1betaO2_M, S1betaO2_T, S1betaO2_P, S1betaO2_beta, S1P_M, S1P_T, S1P_P, S1P_beta, S1Ttr_M, S1Ttr_T, S1Ttr_P, S1Ttr_beta, S1Tve_M, S1Tve_T, S1Tve_P, S1Tve_beta, S1M_M, S1M_T, S1M_P, S1M_beta
