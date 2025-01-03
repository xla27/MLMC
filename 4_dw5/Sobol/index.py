from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import pickle
from cfd_call_fine import cfd_call_fine
from cfd_call_coarse import cfd_call_coarse


def modelCallFine(M, T, P, beta, l, i):
    Bo2 = 1 - beta
    
    M = '{:.3f}'.format(M)
    T = '{:.1f}'.format(T)
    P = '{:.1f}'.format(P)
    Bn2 = '{:.3f}'.format(beta)
    Bo2 = '{:.3f}'.format(Bo2)
    
    valIns_M = str(M)
    valIns_T = str(T)
    valIns_P = str(P)
    valIns_Bn2 = str(Bn2)
    valIns_Bo2 = str(Bo2)
    
    beta_nf, beta_of, beta_nof, beta_n2f, beta_o2f, p_if, Ttr_if, Tve_if, M_if, xnodesf = cfd_call_fine(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
    return xnodesf, beta_nf, beta_of, beta_nof, beta_n2f, beta_o2f, p_if, Ttr_if, Tve_if, M_if

def modelCallCoarse(M, T, P, beta, l, i):
    Bo2 = 1 - beta
    
    M = '{:.3f}'.format(M)
    T = '{:.1f}'.format(T)
    P = '{:.1f}'.format(P)
    Bn2 = '{:.3f}'.format(beta)
    Bo2 = '{:.3f}'.format(Bo2)
    
    valIns_M = str(M)
    valIns_T = str(T)
    valIns_P = str(P)
    valIns_Bn2 = str(Bn2)
    valIns_Bo2 = str(Bo2)
    beta_nc, beta_oc, beta_noc, beta_n2c, beta_o2c, p_ic, Ttr_ic, Tve_ic, M_ic, xnodesc = cfd_call_coarse(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i)
    
    return xnodesc, beta_nc, beta_oc, beta_noc, beta_n2c, beta_o2c, p_ic, Ttr_ic, Tve_ic, M_ic

def interpolate_to_fine(x_fine, x_coarse, u_coarse):
    interp_func = interp1d(x_coarse, u_coarse, kind='linear', fill_value='extrapolate')
    u_interpolated = interp_func(x_fine)
    return u_interpolated

# Function to perform Sobol analysis on a given level
def compute_sobol_indicesFine(param_values, l, problem):
    # Run the model for each sample
    model_outputsN = []; model_outputsO = []; model_outputsNO = []; model_outputsN2 = []; model_outputsO2 = [];
    model_outputsP = []; model_outputsTtr = []; model_outputsTve = []; model_outputsM = [];
    
    xnode_vec = None
    i = 0
    for X in param_values:
        xnodes, beta_n, beta_o, beta_no, beta_n2, beta_o2, p_i, Ttr_i, Tve_i, M_i = modelCallFine(X[0], X[1], X[2], X[3], l, i)
        i = i + 1
        if xnode_vec is None:
            x_vec = xnodes
        
        model_outputsN.append(beta_n); model_outputsO.append(beta_o); model_outputsNO.append(beta_no); model_outputsN2.append(beta_n2); model_outputsO2.append(beta_o2);
        model_outputsP.append(p_i); model_outputsTtr.append(Ttr_i); model_outputsTve.append(Tve_i); model_outputsM.append(M_i);

    model_outputsN = np.array(model_outputsN); model_outputsO = np.array(model_outputsO); model_outputsNO = np.array(model_outputsNO); model_outputsN2 = np.array(model_outputsN2);
    model_outputsO2 = np.array(model_outputsO2);
    
    model_outputsP = np.array(model_outputsP); model_outputsTtr = np.array(model_outputsTtr); model_outputsTve = np.array(model_outputsTve); model_outputsM = np.array(model_outputsM);

    # Compute Sobol indices for each x point
    S1n_M = []; S1n_T = []; S1n_P = []; S1n_beta = []; # sobol indices with N as QoI
    S1o_M = []; S1o_T = []; S1o_P = []; S1o_beta = []; # sobol indices with O as QoI
    S1no_M = []; S1no_T = []; S1no_P = []; S1no_beta = []; # sobol indices with NO as QoI
    S1n2_M = []; S1n2_T = []; S1n2_P = []; S1n2_beta = []; # sobol indices with N2 as QoI
    S1o2_M = []; S1o2_T = []; S1o2_P = []; S1o2_beta = []; # sobol indices with O2 as QoI
    
    S1p_M = []; S1p_T = []; S1p_P = []; S1p_beta = []; # sobol indices with P as QoI
    S1ttr_M = []; S1ttr_T = []; S1ttr_P = []; S1ttr_beta = []; # sobol indices with Ttr as QoI
    S1tve_M = []; S1tve_T = []; S1tve_P = []; S1tve_beta = []; # sobol indices with Tve as QoI
    S1m_M = []; S1m_T = []; S1m_P = []; S1m_beta = []; # sobol indices with M as QoI
    
    # Compute the Sobol' indices wrt the other model_outputs (now there is not just one QoI)
    for j in range(len(x_vec)):
        Si_n = sobol.analyze(problem, model_outputsN[:, j], calc_second_order=False) 
        S1n_M.append(Si_n['S1'][0]);  S1n_T.append(Si_n['S1'][1]); S1n_P.append(Si_n['S1'][2]); S1n_beta.append(Si_n['S1'][3]); # computation of S related to N
        
        Si_o = sobol.analyze(problem, model_outputsO[:, j], calc_second_order=False) 
        S1o_M.append(Si_o['S1'][0]);  S1o_T.append(Si_o['S1'][1]); S1o_P.append(Si_o['S1'][2]); S1o_beta.append(Si_o['S1'][3]); # computation of S related to O
        
        Si_no = sobol.analyze(problem, model_outputsNO[:, j], calc_second_order=False) 
        S1no_M.append(Si_no['S1'][0]);  S1no_T.append(Si_no['S1'][1]); S1no_P.append(Si_no['S1'][2]); S1no_beta.append(Si_no['S1'][3]); # computation of S related to NO  
        
        Si_n2 = sobol.analyze(problem, model_outputsN2[:, j], calc_second_order=False) 
        S1n2_M.append(Si_n2['S1'][0]);  S1n2_T.append(Si_n2['S1'][1]); S1n2_P.append(Si_n2['S1'][2]); S1n2_beta.append(Si_n2['S1'][3]); # computation of S related to N2
        
        Si_o2 = sobol.analyze(problem, model_outputsO2[:, j], calc_second_order=False) 
        S1o2_M.append(Si_o2['S1'][0]);  S1o2_T.append(Si_o2['S1'][1]); S1o2_P.append(Si_o2['S1'][2]); S1o2_beta.append(Si_o2['S1'][3]); # computation of S related to O2   
        
        Si_p = sobol.analyze(problem, model_outputsP[:, j], calc_second_order=False) 
        S1p_M.append(Si_p['S1'][0]);  S1p_T.append(Si_p['S1'][1]); S1p_P.append(Si_p['S1'][2]); S1p_beta.append(Si_p['S1'][3]); # computation of S related to P   
        
        Si_ttr = sobol.analyze(problem, model_outputsTtr[:, j], calc_second_order=False) 
        S1ttr_M.append(Si_ttr['S1'][0]);  S1ttr_T.append(Si_ttr['S1'][1]); S1ttr_P.append(Si_ttr['S1'][2]); S1ttr_beta.append(Si_ttr['S1'][3]); # computation of S related to Ttr                   
        
        Si_tve = sobol.analyze(problem, model_outputsTve[:, j], calc_second_order=False) 
        S1tve_M.append(Si_tve['S1'][0]);  S1tve_T.append(Si_tve['S1'][1]); S1tve_P.append(Si_tve['S1'][2]); S1tve_beta.append(Si_tve['S1'][3]); # computation of S related to Tve   
        
        Si_m = sobol.analyze(problem, model_outputsM[:, j], calc_second_order=False) 
        S1m_M.append(Si_m['S1'][0]);  S1m_T.append(Si_m['S1'][1]); S1m_P.append(Si_m['S1'][2]); S1m_beta.append(Si_m['S1'][3]); # computation of S related to M           
                
    return x_vec, S1n_M, S1n_T, S1n_P, S1n_beta, S1o_M, S1o_T, S1o_P, S1o_beta, S1no_M, S1no_T, S1no_P, S1no_beta, S1n2_M, S1n2_T, S1n2_P, S1n2_beta, S1o2_M, S1o2_T, S1o2_P, S1o2_beta, S1p_M, S1p_T, S1p_P, S1p_beta, S1ttr_M, S1ttr_T, S1ttr_P, S1ttr_beta, S1tve_M, S1tve_T, S1tve_P, S1tve_beta, S1m_M, S1m_T, S1m_P, S1m_beta

def compute_sobol_indicesCoarse(param_values, l, problem):
    # Run the model for each sample
    model_outputsN = []; model_outputsO = []; model_outputsNO = []; model_outputsN2 = []; model_outputsO2 = [];
    model_outputsP = []; model_outputsTtr = []; model_outputsTve = []; model_outputsM = [];
    
    xnode_vec = None
    i = 0
    for X in param_values:
        xnodes, beta_n, beta_o, beta_no, beta_n2, beta_o2, p_i, Ttr_i, Tve_i, M_i = modelCallCoarse(X[0], X[1], X[2], X[3], l, i)
        i = i + 1
        if xnode_vec is None:
            x_vec = xnodes
        
        model_outputsN.append(beta_n); model_outputsO.append(beta_o); model_outputsNO.append(beta_no); model_outputsN2.append(beta_n2); model_outputsO2.append(beta_o2);
        model_outputsP.append(p_i); model_outputsTtr.append(Ttr_i); model_outputsTve.append(Tve_i); model_outputsM.append(M_i);

    model_outputsN = np.array(model_outputsN); model_outputsO = np.array(model_outputsO); model_outputsNO = np.array(model_outputsNO); model_outputsN2 = np.array(model_outputsN2);
    model_outputsO2 = np.array(model_outputsO2);
    
    model_outputsP = np.array(model_outputsP); model_outputsTtr = np.array(model_outputsTtr); model_outputsTve = np.array(model_outputsTve); model_outputsM = np.array(model_outputsM);

    # Compute Sobol indices for each x point
    S1n_M = []; S1n_T = []; S1n_P = []; S1n_beta = []; # sobol indices with N as QoI
    S1o_M = []; S1o_T = []; S1o_P = []; S1o_beta = []; # sobol indices with O as QoI
    S1no_M = []; S1no_T = []; S1no_P = []; S1no_beta = []; # sobol indices with NO as QoI
    S1n2_M = []; S1n2_T = []; S1n2_P = []; S1n2_beta = []; # sobol indices with N2 as QoI
    S1o2_M = []; S1o2_T = []; S1o2_P = []; S1o2_beta = []; # sobol indices with O2 as QoI
    
    S1p_M = []; S1p_T = []; S1p_P = []; S1p_beta = []; # sobol indices with P as QoI
    S1ttr_M = []; S1ttr_T = []; S1ttr_P = []; S1ttr_beta = []; # sobol indices with Ttr as QoI
    S1tve_M = []; S1tve_T = []; S1tve_P = []; S1tve_beta = []; # sobol indices with Tve as QoI
    S1m_M = []; S1m_T = []; S1m_P = []; S1m_beta = []; # sobol indices with M as QoI
    
    # Compute the Sobol' indices wrt the other model_outputs (now there is not just one QoI)
    for j in range(len(x_vec)):
        Si_n = sobol.analyze(problem, model_outputsN[:, j], calc_second_order=False) 
        S1n_M.append(Si_n['S1'][0]);  S1n_T.append(Si_n['S1'][1]); S1n_P.append(Si_n['S1'][2]); S1n_beta.append(Si_n['S1'][3]); # computation of S related to N
        
        Si_o = sobol.analyze(problem, model_outputsO[:, j], calc_second_order=False) 
        S1o_M.append(Si_o['S1'][0]);  S1o_T.append(Si_o['S1'][1]); S1o_P.append(Si_o['S1'][2]); S1o_beta.append(Si_o['S1'][3]); # computation of S related to O
        
        Si_no = sobol.analyze(problem, model_outputsNO[:, j], calc_second_order=False) 
        S1no_M.append(Si_no['S1'][0]);  S1no_T.append(Si_no['S1'][1]); S1no_P.append(Si_no['S1'][2]); S1no_beta.append(Si_no['S1'][3]); # computation of S related to NO  
        
        Si_n2 = sobol.analyze(problem, model_outputsN2[:, j], calc_second_order=False) 
        S1n2_M.append(Si_n2['S1'][0]);  S1n2_T.append(Si_n2['S1'][1]); S1n2_P.append(Si_n2['S1'][2]); S1n2_beta.append(Si_n2['S1'][3]); # computation of S related to N2
        
        Si_o2 = sobol.analyze(problem, model_outputsO2[:, j], calc_second_order=False) 
        S1o2_M.append(Si_o2['S1'][0]);  S1o2_T.append(Si_o2['S1'][1]); S1o2_P.append(Si_o2['S1'][2]); S1o2_beta.append(Si_o2['S1'][3]); # computation of S related to O2   
        
        Si_p = sobol.analyze(problem, model_outputsP[:, j], calc_second_order=False) 
        S1p_M.append(Si_p['S1'][0]);  S1p_T.append(Si_p['S1'][1]); S1p_P.append(Si_p['S1'][2]); S1p_beta.append(Si_p['S1'][3]); # computation of S related to P   
        
        Si_ttr = sobol.analyze(problem, model_outputsTtr[:, j], calc_second_order=False) 
        S1ttr_M.append(Si_ttr['S1'][0]);  S1ttr_T.append(Si_ttr['S1'][1]); S1ttr_P.append(Si_ttr['S1'][2]); S1ttr_beta.append(Si_ttr['S1'][3]); # computation of S related to Ttr                   
        
        Si_tve = sobol.analyze(problem, model_outputsTve[:, j], calc_second_order=False) 
        S1tve_M.append(Si_tve['S1'][0]);  S1tve_T.append(Si_tve['S1'][1]); S1tve_P.append(Si_tve['S1'][2]); S1tve_beta.append(Si_tve['S1'][3]); # computation of S related to Tve   
        
        Si_m = sobol.analyze(problem, model_outputsM[:, j], calc_second_order=False) 
        S1m_M.append(Si_m['S1'][0]);  S1m_T.append(Si_m['S1'][1]); S1m_P.append(Si_m['S1'][2]); S1m_beta.append(Si_m['S1'][3]); # computation of S related to M       
                
    return x_vec, S1n_M, S1n_T, S1n_P, S1n_beta, S1o_M, S1o_T, S1o_P, S1o_beta, S1no_M, S1no_T, S1no_P, S1no_beta, S1n2_M, S1n2_T, S1n2_P, S1n2_beta, S1o2_M, S1o2_T, S1o2_P, S1o2_beta, S1p_M, S1p_T, S1p_P, S1p_beta, S1ttr_M, S1ttr_T, S1ttr_P, S1ttr_beta, S1tve_M, S1tve_T, S1tve_P, S1tve_beta, S1m_M, S1m_T, S1m_P, S1m_beta

# - MAIN

# Set up the problem for SALib
problem = {
    'num_vars': 4,
    'names': ['M', 'T','P','beta'],
    'bounds': [[8.0, 9.5], [850, 1050], [300, 600], [0.76, 0.8]]
}

# Generate samples using Saltelli's sampling
sample_sizes = [1]  # Adjust sample sizes as needed for each level
lev = [0]
d = problem['num_vars']
param_values0 = saltelli.sample(problem, sample_sizes[0], calc_second_order=False)

# First level computations (l = 0)
x_vec0, S1n_M0, S1n_T0, S1n_P0, S1n_beta0, S1o_M0, S1o_T0, S1o_P0, S1o_beta0, S1no_M0, S1no_T0, S1no_P0, S1no_beta0, S1n2_M0, S1n2_T0, S1n2_P0, S1n2_beta0, S1o2_M0, S1o2_T0, S1o2_P0, S1o2_beta0, S1p_M0, S1p_T0, S1p_P0, S1p_beta0, S1ttr_M0, S1ttr_T0, S1ttr_P0, S1ttr_beta0, S1tve_M0, S1tve_T0, S1tve_P0, S1tve_beta0, S1m_M0, S1m_T0, S1m_P0, S1m_beta0 = compute_sobol_indicesFine(param_values0, lev[0], problem)

# Finer grids corrections (l>0)
Lmax = len(sample_sizes) - 1
total_S1n_M = np.array(S1n_M0); total_S1n_T = np.array(S1n_T0); total_S1n_P = np.array(S1n_P0); total_S1n_beta = np.array(S1n_beta0); # level 0 indices for N
total_S1o_M = np.array(S1o_M0); total_S1o_T = np.array(S1o_T0); total_S1o_P = np.array(S1o_P0); total_S1o_beta = np.array(S1o_beta0); # level 0 indices for O
total_S1no_M = np.array(S1no_M0); total_S1no_T = np.array(S1no_T0); total_S1no_P = np.array(S1no_P0); total_S1no_beta = np.array(S1no_beta0); # level 0 indices for NO
total_S1n2_M = np.array(S1n2_M0); total_S1n2_T = np.array(S1n2_T0); total_S1n2_P = np.array(S1n2_P0); total_S1n2_beta = np.array(S1n2_beta0); # level 0 indices for N2
total_S1o2_M = np.array(S1o2_M0); total_S1o2_T = np.array(S1o2_T0); total_S1o2_P = np.array(S1o2_P0); total_S1o2_beta = np.array(S1o2_beta0); # level 0 indices for O2
total_S1p_M = np.array(S1p_M0); total_S1p_T = np.array(S1p_T0); total_S1p_P = np.array(S1p_P0); total_S1p_beta = np.array(S1p_beta0); # level 0 indices for P
total_S1ttr_M = np.array(S1ttr_M0); total_S1ttr_T = np.array(S1ttr_T0); total_S1ttr_P = np.array(S1ttr_P0); total_S1ttr_beta = np.array(S1ttr_beta0); # level 0 indices for Ttr
total_S1tve_M = np.array(S1tve_M0); total_S1tve_T = np.array(S1tve_T0); total_S1tve_P = np.array(S1tve_P0); total_S1tve_beta = np.array(S1tve_beta0); # level 0 indices for Tve
total_S1m_M = np.array(S1m_M0); total_S1m_T = np.array(S1m_T0); total_S1m_P = np.array(S1m_P0); total_S1m_beta = np.array(S1m_beta0); # level 0 indices for M

for l in range(1, Lmax + 1):
    # Finer grid computations
    param_valuesf = saltelli.sample(problem, sample_sizes[l], calc_second_order=False)
    x_vecf, S1n_Mf, S1n_Tf, S1n_Pf, S1n_betaf, S1o_Mf, S1o_Tf, S1o_Pf, S1o_betaf, S1no_Mf, S1no_Tf, S1no_Pf, S1no_betaf, S1n2_Mf, S1n2_Tf, S1n2_Pf, S1n2_betaf, S1o2_Mf, S1o2_Tf, S1o2_Pf, S1o2_betaf, S1p_Mf, S1p_Tf, S1p_Pf, S1p_betaf, S1ttr_Mf, S1ttr_Tf, S1ttr_Pf, S1ttr_betaf, S1tve_Mf, S1tve_Tf, S1tve_Pf, S1tve_betaf, S1m_Mf, S1m_Tf, S1m_Pf, S1m_betaf = compute_sobol_indicesFine(param_valuesf, lev[l], problem)
    
    
    # Coarser grid computations
    param_valuesc = saltelli.sample(problem, sample_sizes[l], calc_second_order=False)
    x_vecc, S1n_Mc, S1n_Tc, S1n_Pc, S1n_betac, S1o_Mc, S1o_Tc, S1o_Pc, S1o_betac, S1no_Mc, S1no_Tc, S1no_Pc, S1no_betac, S1n2_Mc, S1n2_Tc, S1n2_Pc, S1n2_betac, S1o2_Mc, S1o2_Tc, S1o2_Pc, S1o2_betac, S1p_Mc, S1p_Tc, S1p_Pc, S1p_betac, S1ttr_Mc, S1ttr_Tc, S1ttr_Pc, S1ttr_betac, S1tve_Mc, S1tve_Tc, S1tve_Pc, S1tve_betac, S1m_Mc, S1m_Tc, S1m_Pc, S1m_betac = compute_sobol_indicesCoarse(param_valuesc, lev[l], problem)
    
    
    # Interpolate Sobol indices of the coarser grid to the finer grid
    S1n_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1n_Mc); S1n_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1n_Tc);
    S1n_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1n_Pc); S1n_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1n_betac);
    
    S1o_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1o_Mc); S1o_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1o_Tc);
    S1o_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1o_Pc); S1o_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1o_betac);
    
    S1no_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1no_Mc); S1no_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1no_Tc);
    S1no_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1no_Pc); S1no_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1no_betac);
    
    S1n2_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1n2_Mc); S1n2_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1n2_Tc);
    S1n2_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1n2_Pc); S1n2_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1n2_betac);
    
    S1o2_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1o2_Mc); S1o2_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1o2_Tc);
    S1o2_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1o2_Pc); S1o2_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1o2_betac);
    
    S1p_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1p_Mc); S1p_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1p_Tc);
    S1p_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1p_Pc); S1p_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1p_betac);
    
    S1ttr_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1ttr_Mc); S1ttr_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1ttr_Tc);
    S1ttr_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1ttr_Pc); S1ttr_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1ttr_betac);
    
    S1tve_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1tve_Mc); S1tve_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1tve_Tc);
    S1tve_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1tve_Pc); S1tve_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1tve_betac);
    
    S1m_Mc_interp = interpolate_to_fine(x_vecf, x_vecc, S1m_Mc); S1m_Tc_interp = interpolate_to_fine(x_vecf, x_vecc, S1m_Tc);
    S1m_Pc_interp = interpolate_to_fine(x_vecf, x_vecc, S1m_Pc); S1m_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1m_betac);

    # Compute the difference between finer and coarser grid
    S1n_M_diff = np.array(S1n_Mf) - np.array(S1n_Mc_interp); S1n_T_diff = np.array(S1n_Tf) - np.array(S1n_Tc_interp);
    S1n_P_diff = np.array(S1n_Pf) - np.array(S1n_Pc_interp); S1n_beta_diff = np.array(S1n_betaf) - np.array(S1n_betac_interp);
    
    S1o_M_diff = np.array(S1o_Mf) - np.array(S1o_Mc_interp); S1o_T_diff = np.array(S1o_Tf) - np.array(S1o_Tc_interp);
    S1o_P_diff = np.array(S1o_Pf) - np.array(S1o_Pc_interp); S1o_beta_diff = np.array(S1o_betaf) - np.array(S1o_betac_interp);
    
    S1no_M_diff = np.array(S1no_Mf) - np.array(S1no_Mc_interp); S1no_T_diff = np.array(S1no_Tf) - np.array(S1no_Tc_interp);
    S1no_P_diff = np.array(S1no_Pf) - np.array(S1no_Pc_interp); S1no_beta_diff = np.array(S1no_betaf) - np.array(S1no_betac_interp);
    
    S1n2_M_diff = np.array(S1n2_Mf) - np.array(S1n2_Mc_interp); S1n2_T_diff = np.array(S1n2_Tf) - np.array(S1n2_Tc_interp);
    S1n2_P_diff = np.array(S1n2_Pf) - np.array(S1n2_Pc_interp); S1n2_beta_diff = np.array(S1n2_betaf) - np.array(S1n2_betac_interp);
    
    S1o2_M_diff = np.array(S1o2_Mf) - np.array(S1o2_Mc_interp); S1o2_T_diff = np.array(S1o2_Tf) - np.array(S1o2_Tc_interp);
    S1o2_P_diff = np.array(S1o2_Pf) - np.array(S1o2_Pc_interp); S1o2_beta_diff = np.array(S1o2_betaf) - np.array(S1o2_betac_interp);
    
    S1p_M_diff = np.array(S1p_Mf) - np.array(S1p_Mc_interp); S1p_T_diff = np.array(S1p_Tf) - np.array(S1p_Tc_interp);
    S1p_P_diff = np.array(S1p_Pf) - np.array(S1p_Pc_interp); S1p_beta_diff = np.array(S1p_betaf) - np.array(S1p_betac_interp);
    
    S1ttr_M_diff = np.array(S1ttr_Mf) - np.array(S1ttr_Mc_interp); S1ttr_T_diff = np.array(S1ttr_Tf) - np.array(S1ttr_Tc_interp);
    S1ttr_P_diff = np.array(S1ttr_Pf) - np.array(S1ttr_Pc_interp); S1ttr_beta_diff = np.array(S1ttr_betaf) - np.array(S1ttr_betac_interp);
    
    S1tve_M_diff = np.array(S1tve_Mf) - np.array(S1tve_Mc_interp); S1tve_T_diff = np.array(S1tve_Tf) - np.array(S1tve_Tc_interp);
    S1tve_P_diff = np.array(S1tve_Pf) - np.array(S1tve_Pc_interp); S1tve_beta_diff = np.array(S1tve_betaf) - np.array(S1tve_betac_interp);
    
    S1m_M_diff = np.array(S1m_Mf) - np.array(S1m_Mc_interp); S1m_T_diff = np.array(S1m_Tf) - np.array(S1m_Tc_interp);
    S1m_P_diff = np.array(S1m_Pf) - np.array(S1m_Pc_interp); S1m_beta_diff = np.array(S1m_betaf) - np.array(S1m_betac_interp);

    
    # Add the correction to the total Sobol indices 
    total_S1n_M = interpolate_to_fine(x_vecf, x_vec0, total_S1n_M); total_S1n_M += S1n_M_diff;
    total_S1n_T = interpolate_to_fine(x_vecf, x_vec0, total_S1n_T); total_S1n_T += S1n_T_diff;
    total_S1n_P = interpolate_to_fine(x_vecf, x_vec0, total_S1n_P); total_S1n_P += S1n_P_diff;
    total_S1n_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1n_beta); total_S1n_beta += S1n_beta_diff;

    total_S1o_M = interpolate_to_fine(x_vecf, x_vec0, total_S1o_M); total_S1o_M += S1o_M_diff;
    total_S1o_T = interpolate_to_fine(x_vecf, x_vec0, total_S1o_T); total_S1o_T += S1o_T_diff;
    total_S1o_P = interpolate_to_fine(x_vecf, x_vec0, total_S1o_P); total_S1o_P += S1o_P_diff;
    total_S1o_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1o_beta); total_S1o_beta += S1o_beta_diff;

    total_S1no_M = interpolate_to_fine(x_vecf, x_vec0, total_S1no_M); total_S1no_M += S1no_M_diff;
    total_S1no_T = interpolate_to_fine(x_vecf, x_vec0, total_S1no_T); total_S1no_T += S1no_T_diff;
    total_S1no_P = interpolate_to_fine(x_vecf, x_vec0, total_S1no_P); total_S1no_P += S1no_P_diff;
    total_S1no_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1no_beta); total_S1no_beta += S1no_beta_diff;

    total_S1n2_M = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_M); total_S1n2_M += S1n2_M_diff;
    total_S1n2_T = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_T); total_S1n2_T += S1n2_T_diff;
    total_S1n2_P = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_P); total_S1n2_P += S1n2_P_diff;
    total_S1n2_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_beta); total_S1n2_beta += S1n2_beta_diff;

    total_S1o2_M = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_M); total_S1o2_M += S1o2_M_diff;
    total_S1o2_T = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_T); total_S1o2_T += S1o2_T_diff;
    total_S1o2_P = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_P); total_S1o2_P += S1o2_P_diff;
    total_S1o2_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_beta); total_S1o2_beta += S1o2_beta_diff;

    total_S1p_M = interpolate_to_fine(x_vecf, x_vec0, total_S1p_M); total_S1p_M += S1p_M_diff;
    total_S1p_T = interpolate_to_fine(x_vecf, x_vec0, total_S1p_T); total_S1p_T += S1p_T_diff;
    total_S1p_P = interpolate_to_fine(x_vecf, x_vec0, total_S1p_P); total_S1p_P += S1p_P_diff;
    total_S1p_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1p_beta); total_S1p_beta += S1p_beta_diff;

    total_S1ttr_M = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_M); total_S1ttr_M += S1ttr_M_diff;
    total_S1ttr_T = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_T); total_S1ttr_T += S1ttr_T_diff;
    total_S1ttr_P = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_P); total_S1ttr_P += S1ttr_P_diff;
    total_S1ttr_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_beta); total_S1ttr_beta += S1ttr_beta_diff;

    total_S1tve_M = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_M); total_S1tve_M += S1tve_M_diff;
    total_S1tve_T = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_T); total_S1tve_T += S1tve_T_diff;
    total_S1tve_P = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_P); total_S1tve_P += S1tve_P_diff;
    total_S1tve_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_beta); total_S1tve_beta += S1tve_beta_diff;

    total_S1m_M = interpolate_to_fine(x_vecf, x_vec0, total_S1m_M); total_S1m_M += S1m_M_diff;
    total_S1m_T = interpolate_to_fine(x_vecf, x_vec0, total_S1m_T); total_S1m_T += S1m_T_diff;
    total_S1m_P = interpolate_to_fine(x_vecf, x_vec0, total_S1m_P); total_S1m_P += S1m_P_diff;
    total_S1m_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1m_beta); total_S1m_beta += S1m_beta_diff;

    # Update xnode_vec0 to be the finer grid for the next iteration
    x_vec0 = x_vecf

threshold = 0.193192
x_vec0fil = np.array(x_vec0)
filter_indices = x_vec0fil > threshold
x_vec0_filtered = x_vec0fil[filter_indices]

# Plot 1: first-order Sobol indices of N against space coordinate
plt.figure(figsize=(13, 11))

total_S1n_M = np.array(total_S1n_M)
total_S1n_T = np.array(total_S1n_T)
total_S1n_P = np.array(total_S1n_P)
total_S1n_beta = np.array(total_S1n_beta)

total_S1n_M_filtered = total_S1n_M[filter_indices]
total_S1n_T_filtered = total_S1n_T[filter_indices]
total_S1n_P_filtered = total_S1n_P[filter_indices]
total_S1n_beta_filtered = total_S1n_beta[filter_indices]

plt.plot(x_vec0_filtered, total_S1n_M_filtered, label='S1n(M)')
plt.plot(x_vec0_filtered, total_S1n_T_filtered, label='S1n(T)')
plt.plot(x_vec0_filtered, total_S1n_P_filtered, label='S1n(P)')
plt.plot(x_vec0_filtered, total_S1n_beta_filtered, label='S1n(beta)')
plt.title('First-order Sobol Indices N against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '1_Nfirst_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0_filtered': x_vec0_filtered,
    'total_S1n_M': total_S1n_M_filtered,
    'total_S1n_T': total_S1n_T_filtered,
    'total_S1n_P': total_S1n_P_filtered,
    'total_S1n_beta': total_S1n_beta_filtered,
}

pickle_filename = '1_Nfirst_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 2: first-order Sobol indices of O against space coordinate
plt.figure(figsize=(13, 11))

total_S1o_M = np.array(total_S1o_M)
total_S1o_T = np.array(total_S1o_T)
total_S1o_P = np.array(total_S1o_P)
total_S1o_beta = np.array(total_S1o_beta)

total_S1o_M_filtered = total_S1o_M[filter_indices]
total_S1o_T_filtered = total_S1o_T[filter_indices]
total_S1o_P_filtered = total_S1o_P[filter_indices]
total_S1o_beta_filtered = total_S1o_beta[filter_indices]

plt.plot(x_vec0_filtered, total_S1o_M_filtered, label='S1o(M)')
plt.plot(x_vec0_filtered, total_S1o_T_filtered, label='S1o(T)')
plt.plot(x_vec0_filtered, total_S1o_P_filtered, label='S1o(P)')
plt.plot(x_vec0_filtered, total_S1o_beta_filtered, label='S1o(beta)')
plt.title('First-order Sobol Indices O against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '2_Ofirst_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0_filtered': x_vec0_filtered,
    'total_S1o_M': total_S1o_M_filtered,
    'total_S1o_T': total_S1o_T_filtered,
    'total_S1o_P': total_S1o_P_filtered,
    'total_S1o_beta': total_S1o_beta_filtered,
}

pickle_filename = '2_Ofirst_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 3: first-order Sobol indices of NO against space coordinate
plt.figure(figsize=(13, 11))

total_S1no_M = np.array(total_S1no_M)
total_S1no_T = np.array(total_S1no_T)
total_S1no_P = np.array(total_S1no_P)
total_S1no_beta = np.array(total_S1no_beta)

total_S1no_M_filtered = total_S1no_M[filter_indices]
total_S1no_T_filtered = total_S1no_T[filter_indices]
total_S1no_P_filtered = total_S1no_P[filter_indices]
total_S1no_beta_filtered = total_S1no_beta[filter_indices]

plt.plot(x_vec0_filtered, total_S1no_M_filtered, label='S1no(M)')
plt.plot(x_vec0_filtered, total_S1no_T_filtered, label='S1no(T)')
plt.plot(x_vec0_filtered, total_S1no_P_filtered, label='S1no(P)')
plt.plot(x_vec0_filtered, total_S1no_beta_filtered, label='S1no(beta)')
plt.title('First-order Sobol Indices NO against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '3_NOfirst_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0_filtered': x_vec0_filtered,
    'total_S1no_M': total_S1no_M_filtered,
    'total_S1no_T': total_S1no_T_filtered,
    'total_S1no_P': total_S1no_P_filtered,
    'total_S1no_beta': total_S1no_beta_filtered,
}

pickle_filename = '3_NOfirst_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 4: first-order Sobol indices of N2 against space coordinate
plt.figure(figsize=(13, 11))

plt.plot(x_vec0, total_S1n2_M, label='S1n2(M)')
plt.plot(x_vec0, total_S1n2_T, label='S1n2(T)')
plt.plot(x_vec0, total_S1n2_P, label='S1n2(P)')
plt.plot(x_vec0, total_S1n2_beta, label='S1n2(beta)')
plt.title('First-order Sobol Indices N2 against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '4_N2first_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0': x_vec0,
    'total_S1n2_M': total_S1n2_M,
    'total_S1n2_T': total_S1n2_T,
    'total_S1n2_P': total_S1n2_P,
    'total_S1n2_beta': total_S1n2_beta,
}

pickle_filename = '4_N2first_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 5: first-order Sobol indices of O2 against space coordinate
plt.figure(figsize=(13, 11))

plt.plot(x_vec0, total_S1o2_M, label='S1o2(M)')
plt.plot(x_vec0, total_S1o2_T, label='S1o2(T)')
plt.plot(x_vec0, total_S1o2_P, label='S1o2(P)')
plt.plot(x_vec0, total_S1o2_beta, label='S1o2(beta)')
plt.title('First-order Sobol Indices O2 against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '5_O2first_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0': x_vec0,
    'total_S1o2_M': total_S1o2_M,
    'total_S1o2_T': total_S1o2_T,
    'total_S1o2_P': total_S1o2_P,
    'total_S1o2_beta': total_S1o2_beta,
}

pickle_filename = '5_O2first_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 6: first-order Sobol indices of P against space coordinate
plt.figure(figsize=(13, 11))

plt.plot(x_vec0, total_S1p_M, label='S1p(M)')
plt.plot(x_vec0, total_S1p_T, label='S1p(T)')
plt.plot(x_vec0, total_S1p_P, label='S1p(P)')
plt.plot(x_vec0, total_S1p_beta, label='S1p(beta)')
plt.title('First-order Sobol Indices P against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '6_Pfirst_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0': x_vec0,
    'total_S1p_M': total_S1p_M,
    'total_S1p_T': total_S1p_T,
    'total_S1p_P': total_S1p_P,
    'total_S1p_beta': total_S1p_beta,
}

pickle_filename = '6_Pfirst_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 7: first-order Sobol indices of Ttr against space coordinate
plt.figure(figsize=(13, 11))

plt.plot(x_vec0, total_S1ttr_M, label='S1ttr(M)')
plt.plot(x_vec0, total_S1ttr_T, label='S1ttr(T)')
plt.plot(x_vec0, total_S1ttr_P, label='S1ttr(P)')
plt.plot(x_vec0, total_S1ttr_beta, label='S1ttr(beta)')
plt.title('First-order Sobol Indices Ttr against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '7_Ttrfirst_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0': x_vec0,
    'total_S1ttr_M': total_S1ttr_M,
    'total_S1ttr_T': total_S1ttr_T,
    'total_S1ttr_P': total_S1ttr_P,
    'total_S1ttr_beta': total_S1ttr_beta,
}

pickle_filename = '7_Ttrfirst_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 8: first-order Sobol indices of Tve against space coordinate
plt.figure(figsize=(13, 11))

plt.plot(x_vec0, total_S1tve_M, label='S1tve(M)')
plt.plot(x_vec0, total_S1tve_T, label='S1tve(T)')
plt.plot(x_vec0, total_S1tve_P, label='S1tve(P)')
plt.plot(x_vec0, total_S1tve_beta, label='S1tve(beta)')
plt.title('First-order Sobol Indices Tve against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '8_Tvefirst_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0': x_vec0,
    'total_S1tve_M': total_S1tve_M,
    'total_S1tve_T': total_S1tve_T,
    'total_S1tve_P': total_S1tve_P,
    'total_S1tve_beta': total_S1tve_beta,
}

pickle_filename = '8_Tvefirst_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()

# Plot 9: first-order Sobol indices of M against space coordinate
plt.figure(figsize=(13, 11))

plt.plot(x_vec0, total_S1m_M, label='S1m(M)')
plt.plot(x_vec0, total_S1m_T, label='S1m(T)')
plt.plot(x_vec0, total_S1m_P, label='S1m(P)')
plt.plot(x_vec0, total_S1m_beta, label='S1m(beta)')
plt.title('First-order Sobol Indices M against X coordinate $\\varepsilon_r = 0.003$')
plt.xlabel('X coordinate')
plt.ylabel('Sobol Index Value')
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_filename = '9_Mfirst_order_sobol_indices'
plt.savefig(plot_filename, format='svg')

plot_data = {
    'x_vec0': x_vec0,
    'total_S1m_M': total_S1m_M,
    'total_S1m_T': total_S1m_T,
    'total_S1m_P': total_S1m_P,
    'total_S1m_beta': total_S1m_beta,
}

pickle_filename = '9_Mfirst_order_sobol_indices.pkl'
with open(pickle_filename, 'wb') as f:
    pickle.dump((plt.gcf(), plot_data), f)

f.close()
plt.close("all")
plt.cla()
plt.clf()