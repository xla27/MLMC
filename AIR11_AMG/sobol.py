import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.analyze import sobol

from AIR5_AMG import cfd_call_amg
from AIR5     import cfd_call


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
    model_outputs_beta_O2        = []
    model_outputs_P              = []
    model_outputs_Ttr            = []
    model_outputs_Tve            = []
    model_outputs_M              = []
    
    xnodes_list = []
    samples = 0

    # if l = 0 cfd_call does not need to be called as mesh adaptation
    if l == 0:
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

        [nd_elecMinus, nd_NPlus, nd_OPlus, nd_NOPlus, nd_N2Plus, nd_O2Plus, beta_elecMinus,
         beta_NPlus, beta_OPlus, beta_NOPlus, beta_N2Plus, beta_O2Plus, beta_N, beta_O, beta_NO, 
         beta_N2, beta_O2, P_i, Ttr_i, Tve_i, M_i, xnodes] = cfd(type, valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args2)
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

    # finding the coarsest xnode and interpolating over it al the other values
    xnodes_ref = min(xnodes_list, key=len)

    for i in range(samples):
        model_outputs_nd_elecMinus[i]   = interp1d(xnodes_list[i], model_outputs_nd_elecMinus[i],   kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_nd_NPlus[i]       = interp1d(xnodes_list[i], model_outputs_nd_NPlus[i],       kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_nd_OPlus[i]       = interp1d(xnodes_list[i], model_outputs_nd_OPlus[i],       kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_nd_NOPlus[i]      = interp1d(xnodes_list[i], model_outputs_nd_NOPlus[i],      kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_nd_N2Plus[i]      = interp1d(xnodes_list[i], model_outputs_nd_N2Plus[i],      kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_nd_O2Plus[i]      = interp1d(xnodes_list[i], model_outputs_nd_O2Plus[i],      kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_elecMinus[i] = interp1d(xnodes_list[i], model_outputs_beta_elecMinus[i], kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_NPlus[i]     = interp1d(xnodes_list[i], model_outputs_beta_NPlus[i],     kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_OPlus[i]     = interp1d(xnodes_list[i], model_outputs_beta_OPlus[i],     kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_NOPlus[i]    = interp1d(xnodes_list[i], model_outputs_beta_NOPlus[i],    kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_N2Plus[i]    = interp1d(xnodes_list[i], model_outputs_beta_N2Plus[i],    kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_O2Plus[i]    = interp1d(xnodes_list[i], model_outputs_beta_O2Plus[i],    kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_N[i]         = interp1d(xnodes_list[i], model_outputs_beta_N[i],         kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_O[i]         = interp1d(xnodes_list[i], model_outputs_beta_O[i],         kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_NO[i]        = interp1d(xnodes_list[i], model_outputs_beta_NO[i],        kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_N2[i]        = interp1d(xnodes_list[i], model_outputs_beta_N2[i],        kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_beta_O2[i]        = interp1d(xnodes_list[i], model_outputs_beta_O2[i],        kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_P[i]              = interp1d(xnodes_list[i], model_outputs_P[i],              kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_Ttr[i]            = interp1d(xnodes_list[i], model_outputs_Ttr[i],            kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_Tve[i]            = interp1d(xnodes_list[i], model_outputs_Tve[i],            kind='linear', fill_value='extrapolate')(xnodes_ref)
        model_outputs_M[i]              = interp1d(xnodes_list[i], model_outputs_M[i],              kind='linear', fill_value='extrapolate')(xnodes_ref)

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
    model_outputs_beta_O2        = np.array(model_outputs_beta_O2)
    model_outputs_P              = np.array(model_outputs_P)
    model_outputs_Ttr            = np.array(model_outputs_Ttr)
    model_outputs_Tve            = np.array(model_outputs_Tve)
    model_outputs_M              = np.array(model_outputs_M)

    # Compute Sobol indices for each x point (wall point)
    S1nd_elecMinus = np.zeros((4, len(xnodes_ref))) # sobol indices with ND e- as QoI
    S1nd_NPlus     = np.zeros((4, len(xnodes_ref))) # sobol indices with ND N+ as QoI 
    S1nd_OPlus     = np.zeros((4, len(xnodes_ref))) # sobol indices with ND O+ as QoI
    S1nd_NOPlus    = np.zeros((4, len(xnodes_ref))) # sobol indices with ND NO+ as QoI
    S1nd_N2Plus    = np.zeros((4, len(xnodes_ref))) # sobol indices with ND N2+ as QoI
    S1nd_O2Plus    = np.zeros((4, len(xnodes_ref))) # sobol indices with ND O2+ as QoI

    S1beta_elecMinus = np.zeros((4, len(xnodes_ref))) # sobol indices with e- as QoI
    S1beta_NPlus     = np.zeros((4, len(xnodes_ref))) # sobol indices with N+ as QoI
    S1beta_OPlus     = np.zeros((4, len(xnodes_ref))) # sobol indices with O+ as QoI
    S1beta_NOPlus    = np.zeros((4, len(xnodes_ref))) # sobol indices with NO+ as QoI
    S1beta_N2Plus    = np.zeros((4, len(xnodes_ref))) # sobol indices with N2+ as QoI
    S1beta_O2Plus    = np.zeros((4, len(xnodes_ref))) # sobol indices with O2+ QoI
    S1beta_N         = np.zeros((4, len(xnodes_ref))) # sobol indices with N as QoI
    S1beta_O         = np.zeros((4, len(xnodes_ref))) # sobol indices with O as QoI
    S1beta_NO        = np.zeros((4, len(xnodes_ref))) # sobol indices with NO as QoI
    S1beta_N2        = np.zeros((4, len(xnodes_ref))) # sobol indices with N2 as QoI
    S1beta_O2        = np.zeros((4, len(xnodes_ref))) # sobol indices with O2 as QoI
    
    S1P   = np.zeros((4, len(xnodes_ref))) # sobol indices with P as QoI
    S1Ttr = np.zeros((4, len(xnodes_ref))) # sobol indices with Ttr as QoI
    S1Tve = np.zeros((4, len(xnodes_ref))) # sobol indices with Tve as QoI
    S1M   = np.zeros((4, len(xnodes_ref))) # sobol indices with M as QoI
    
    # Compute the Sobol' indices wrt the other model_outputs (now there is not just one QoI)
    for j in range(len(xnodes_ref)):

        # computation of S related to ND e-
        Si_nd_elecMinus = sobol.analyze(problem, model_outputs_nd_elecMinus[:, j], calc_second_order=False)
        S1nd_elecMinus[:,j] = Si_nd_elecMinus['S1']

        # computation of S related to ND N+
        Si_nd_NPlus = sobol.analyze(problem, model_outputs_nd_NPlus[:, j], calc_second_order=False)
        S1nd_NPlus[:,j] = Si_nd_NPlus['S1']

        # computation of S related to ND O+
        Si_nd_OPlus = sobol.analyze(problem, model_outputs_nd_OPlus[:, j], calc_second_order=False)
        S1nd_OPlus[:,j] = Si_nd_OPlus['S1']

        # computation of S related to ND NO+
        Si_nd_NOPlus = sobol.analyze(problem, model_outputs_nd_NOPlus[:, j], calc_second_order=False)
        S1nd_NOPlus[:,j] = Si_nd_NOPlus['S1']

        # computation of S related to ND N2+
        Si_nd_N2Plus = sobol.analyze(problem, model_outputs_nd_N2Plus[:, j], calc_second_order=False)
        S1nd_N2Plus[:,j] = Si_nd_N2Plus['S1']

        # computation of S related to ND O2+
        Si_nd_O2Plus = sobol.analyze(problem, model_outputs_nd_O2Plus[:, j], calc_second_order=False)
        S1nd_O2Plus[:,j] = Si_nd_O2Plus['S1']

        # computation of S related to e-
        Si_beta_elecMinus = sobol.analyze(problem, model_outputs_beta_elecMinus[:, j], calc_second_order=False)
        S1beta_elecMinus[:,j] = Si_beta_elecMinus['S1']

        # computation of S related to N+
        Si_beta_NPlus = sobol.analyze(problem, model_outputs_beta_NPlus[:, j], calc_second_order=False)
        S1beta_NPlus[:,j] = Si_beta_NPlus['S1']

        # computation of S related to O+
        Si_beta_OPlus = sobol.analyze(problem, model_outputs_beta_OPlus[:, j], calc_second_order=False)
        S1beta_OPlus[:,j] = Si_beta_OPlus['S1']

        # computation of S related to NO+
        Si_beta_NOPlus = sobol.analyze(problem, model_outputs_beta_NOPlus[:, j], calc_second_order=False) 
        S1beta_NOPlus[:,j] = Si_beta_NOPlus['S1']

        # computation of S related to N2+
        Si_beta_N2Plus = sobol.analyze(problem, model_outputs_beta_N2Plus[:, j], calc_second_order=False)
        S1beta_N2Plus[:,j] = Si_beta_N2Plus['S1']

        # computation of S related to O2+
        Si_beta_O2Plus = sobol.analyze(problem, model_outputs_beta_O2Plus[:, j], calc_second_order=False)
        S1beta_O2Plus[:,j] = Si_beta_O2Plus['S1']

        # computation of S related to N
        Si_N = sobol.analyze(problem, model_outputs_beta_N[:, j], calc_second_order=False) 
        S1beta_N[:,j] = Si_N['S1']
        
        # computation of S related to O
        Si_O = sobol.analyze(problem, model_outputs_beta_O[:, j], calc_second_order=False) 
        S1beta_O[:,j] = Si_O['S1']
        
        # computation of S related to NO
        Si_NO = sobol.analyze(problem, model_outputs_beta_NO[:, j], calc_second_order=False) 
        S1beta_NO[:,j] = Si_NO['S1']
        
        # computation of S related to N2
        Si_N2 = sobol.analyze(problem, model_outputs_beta_N2[:, j], calc_second_order=False) 
        S1beta_N2[:,j] = Si_N2['S1']
        
        # computation of S related to O2
        Si_O2 = sobol.analyze(problem, model_outputs_beta_O2[:, j], calc_second_order=False) 
        S1beta_O2[:,j] = Si_O2['S1']
        
        # computation of S related to P 
        Si_P = sobol.analyze(problem, model_outputs_P[:, j], calc_second_order=False) 
        S1P[:,j] = Si_P['S1']     
        
        # computation of S related to Ttr
        Si_Ttr = sobol.analyze(problem, model_outputs_Ttr[:, j], calc_second_order=False) 
        S1Ttr[:,j] = Si_Ttr['S1']                 
        
        # computation of S related to Tve
        Si_Tve = sobol.analyze(problem, model_outputs_Tve[:, j], calc_second_order=False) 
        S1Tve[:,j] = Si_Tve['S1']   
        
        # computation of S related to M
        Si_M = sobol.analyze(problem, model_outputs_M[:, j], calc_second_order=False) 
        S1M[:,j] = Si_M['S1']        
                
    return xnodes_ref, S1nd_elecMinus, S1nd_NPlus, S1nd_OPlus, S1nd_NOPlus, S1nd_N2Plus, S1nd_O2Plus, S1beta_elecMinus, S1beta_NPlus, S1beta_OPlus, S1beta_NOPlus, S1beta_N2Plus, S1beta_O2Plus, S1beta_N, S1beta_O, S1beta_NO, S1beta_N2, S1beta_O2, S1P, S1Ttr, S1Tve, S1M