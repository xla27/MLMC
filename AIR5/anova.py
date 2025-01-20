import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.sample import saltelli

from AIR5.sobol import compute_sobol_indices

def anova(problem, lev, sample_sizes, *varargin):

    # First level computations (l = 0)
    param_values0 = saltelli.sample(problem, sample_sizes[0], calc_second_order=False)
    x_vec0, S1n_M0, S1n_T0, S1n_P0, S1n_beta0, S1o_M0, S1o_T0, S1o_P0, S1o_beta0, S1no_M0, S1no_T0, S1no_P0, S1no_beta0, S1n2_M0, S1n2_T0, S1n2_P0, S1n2_beta0, S1o2_M0, S1o2_T0, S1o2_P0, S1o2_beta0, S1p_M0, S1p_T0, S1p_P0, S1p_beta0, S1ttr_M0, S1ttr_T0, S1ttr_P0, S1ttr_beta0, S1tve_M0, S1tve_T0, S1tve_P0, S1tve_beta0, S1m_M0, S1m_T0, S1m_P0, S1m_beta0 = compute_sobol_indices('FINE', param_values0, lev[0], problem, *varargin)

    # Finer grids corrections (l>0)
    Lmax = len(sample_sizes) - 1
    total_S1n_M   = np.array(S1n_M0);   total_S1n_T   = np.array(S1n_T0);   total_S1n_P   = np.array(S1n_P0);   total_S1n_beta   = np.array(S1n_beta0);   # level 0 indices for N
    total_S1o_M   = np.array(S1o_M0);   total_S1o_T   = np.array(S1o_T0);   total_S1o_P   = np.array(S1o_P0);   total_S1o_beta   = np.array(S1o_beta0);   # level 0 indices for O
    total_S1no_M  = np.array(S1no_M0);  total_S1no_T  = np.array(S1no_T0);  total_S1no_P  = np.array(S1no_P0);  total_S1no_beta  = np.array(S1no_beta0);  # level 0 indices for NO
    total_S1n2_M  = np.array(S1n2_M0);  total_S1n2_T  = np.array(S1n2_T0);  total_S1n2_P  = np.array(S1n2_P0);  total_S1n2_beta  = np.array(S1n2_beta0);  # level 0 indices for N2
    total_S1o2_M  = np.array(S1o2_M0);  total_S1o2_T  = np.array(S1o2_T0);  total_S1o2_P  = np.array(S1o2_P0);  total_S1o2_beta  = np.array(S1o2_beta0);  # level 0 indices for O2
    total_S1p_M   = np.array(S1p_M0);   total_S1p_T   = np.array(S1p_T0);   total_S1p_P   = np.array(S1p_P0);   total_S1p_beta   = np.array(S1p_beta0);   # level 0 indices for P
    total_S1ttr_M = np.array(S1ttr_M0); total_S1ttr_T = np.array(S1ttr_T0); total_S1ttr_P = np.array(S1ttr_P0); total_S1ttr_beta = np.array(S1ttr_beta0); # level 0 indices for Ttr
    total_S1tve_M = np.array(S1tve_M0); total_S1tve_T = np.array(S1tve_T0); total_S1tve_P = np.array(S1tve_P0); total_S1tve_beta = np.array(S1tve_beta0); # level 0 indices for Tve
    total_S1m_M   = np.array(S1m_M0);   total_S1m_T   = np.array(S1m_T0);   total_S1m_P   = np.array(S1m_P0);   total_S1m_beta   = np.array(S1m_beta0);   # level 0 indices for M

    for l in range(1, Lmax + 1):
        
        # Finer grid computations
        param_valuesf = saltelli.sample(problem, sample_sizes[l], calc_second_order=False)
        x_vecf, S1n_Mf, S1n_Tf, S1n_Pf, S1n_betaf, S1o_Mf, S1o_Tf, S1o_Pf, S1o_betaf, S1no_Mf, S1no_Tf, S1no_Pf, S1no_betaf, S1n2_Mf, S1n2_Tf, S1n2_Pf, S1n2_betaf, S1o2_Mf, S1o2_Tf, S1o2_Pf, S1o2_betaf, S1p_Mf, S1p_Tf, S1p_Pf, S1p_betaf, S1ttr_Mf, S1ttr_Tf, S1ttr_Pf, S1ttr_betaf, S1tve_Mf, S1tve_Tf, S1tve_Pf, S1tve_betaf, S1m_Mf, S1m_Tf, S1m_Pf, S1m_betaf = compute_sobol_indices('FINE', param_valuesf, lev[l], problem, *varargin)
        
        
        # Coarser grid computations
        param_valuesc = saltelli.sample(problem, sample_sizes[l], calc_second_order=False)
        x_vecc, S1n_Mc, S1n_Tc, S1n_Pc, S1n_betac, S1o_Mc, S1o_Tc, S1o_Pc, S1o_betac, S1no_Mc, S1no_Tc, S1no_Pc, S1no_betac, S1n2_Mc, S1n2_Tc, S1n2_Pc, S1n2_betac, S1o2_Mc, S1o2_Tc, S1o2_Pc, S1o2_betac, S1p_Mc, S1p_Tc, S1p_Pc, S1p_betac, S1ttr_Mc, S1ttr_Tc, S1ttr_Pc, S1ttr_betac, S1tve_Mc, S1tve_Tc, S1tve_Pc, S1tve_betac, S1m_Mc, S1m_Tc, S1m_Pc, S1m_betac = compute_sobol_indices('COARSE', param_valuesc, lev[l], problem, *varargin)
        
        
        # Interpolate Sobol indices of the coarser grid to the finer grid
        S1n_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1n_Mc) 
        S1n_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1n_Tc)
        S1n_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1n_Pc)
        S1n_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1n_betac)
        
        S1o_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1o_Mc)
        S1o_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1o_Tc)
        S1o_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1o_Pc)
        S1o_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1o_betac)
        
        S1no_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1no_Mc)
        S1no_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1no_Tc)
        S1no_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1no_Pc)
        S1no_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1no_betac)
        
        S1n2_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1n2_Mc)
        S1n2_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1n2_Tc)
        S1n2_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1n2_Pc)
        S1n2_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1n2_betac)

        S1o2_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1o2_Mc)
        S1o2_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1o2_Tc)
        S1o2_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1o2_Pc)
        S1o2_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1o2_betac)
        
        S1p_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1p_Mc)
        S1p_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1p_Tc)
        S1p_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1p_Pc)
        S1p_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1p_betac)
        
        S1ttr_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1ttr_Mc)
        S1ttr_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1ttr_Tc)
        S1ttr_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1ttr_Pc)
        S1ttr_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1ttr_betac)
        
        S1tve_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1tve_Mc)
        S1tve_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1tve_Tc)
        S1tve_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1tve_Pc)
        S1tve_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1tve_betac)
        
        S1m_Mc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1m_Mc)
        S1m_Tc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1m_Tc)
        S1m_Pc_interp    = interpolate_to_fine(x_vecf, x_vecc, S1m_Pc)
        S1m_betac_interp = interpolate_to_fine(x_vecf, x_vecc, S1m_betac)

        # Compute the difference between finer and coarser grid
        S1n_M_diff    = np.array(S1n_Mf) - np.array(S1n_Mc_interp)
        S1n_T_diff    = np.array(S1n_Tf) - np.array(S1n_Tc_interp)
        S1n_P_diff    = np.array(S1n_Pf) - np.array(S1n_Pc_interp)
        S1n_beta_diff = np.array(S1n_betaf) - np.array(S1n_betac_interp)
        
        S1o_M_diff    = np.array(S1o_Mf) - np.array(S1o_Mc_interp)
        S1o_T_diff    = np.array(S1o_Tf) - np.array(S1o_Tc_interp)
        S1o_P_diff    = np.array(S1o_Pf) - np.array(S1o_Pc_interp)
        S1o_beta_diff = np.array(S1o_betaf) - np.array(S1o_betac_interp)
        
        S1no_M_diff    = np.array(S1no_Mf) - np.array(S1no_Mc_interp)
        S1no_T_diff    = np.array(S1no_Tf) - np.array(S1no_Tc_interp)
        S1no_P_diff    = np.array(S1no_Pf) - np.array(S1no_Pc_interp)
        S1no_beta_diff = np.array(S1no_betaf) - np.array(S1no_betac_interp)
        
        S1n2_M_diff    = np.array(S1n2_Mf) - np.array(S1n2_Mc_interp)
        S1n2_T_diff    = np.array(S1n2_Tf) - np.array(S1n2_Tc_interp)
        S1n2_P_diff    = np.array(S1n2_Pf) - np.array(S1n2_Pc_interp)
        S1n2_beta_diff = np.array(S1n2_betaf) - np.array(S1n2_betac_interp)
        
        S1o2_M_diff    = np.array(S1o2_Mf) - np.array(S1o2_Mc_interp)
        S1o2_T_diff    = np.array(S1o2_Tf) - np.array(S1o2_Tc_interp)
        S1o2_P_diff    = np.array(S1o2_Pf) - np.array(S1o2_Pc_interp)
        S1o2_beta_diff = np.array(S1o2_betaf) - np.array(S1o2_betac_interp)
        
        S1p_M_diff    = np.array(S1p_Mf) - np.array(S1p_Mc_interp)
        S1p_T_diff    = np.array(S1p_Tf) - np.array(S1p_Tc_interp)
        S1p_P_diff    = np.array(S1p_Pf) - np.array(S1p_Pc_interp)
        S1p_beta_diff = np.array(S1p_betaf) - np.array(S1p_betac_interp)
        
        S1ttr_M_diff    = np.array(S1ttr_Mf) - np.array(S1ttr_Mc_interp)
        S1ttr_T_diff    = np.array(S1ttr_Tf) - np.array(S1ttr_Tc_interp)
        S1ttr_P_diff    = np.array(S1ttr_Pf) - np.array(S1ttr_Pc_interp)
        S1ttr_beta_diff = np.array(S1ttr_betaf) - np.array(S1ttr_betac_interp)
        
        S1tve_M_diff    = np.array(S1tve_Mf) - np.array(S1tve_Mc_interp)
        S1tve_T_diff    = np.array(S1tve_Tf) - np.array(S1tve_Tc_interp)
        S1tve_P_diff    = np.array(S1tve_Pf) - np.array(S1tve_Pc_interp)
        S1tve_beta_diff = np.array(S1tve_betaf) - np.array(S1tve_betac_interp)
        
        S1m_M_diff    = np.array(S1m_Mf) - np.array(S1m_Mc_interp)
        S1m_T_diff    = np.array(S1m_Tf) - np.array(S1m_Tc_interp)
        S1m_P_diff    = np.array(S1m_Pf) - np.array(S1m_Pc_interp)
        S1m_beta_diff = np.array(S1m_betaf) - np.array(S1m_betac_interp)

        
        # Add the correction to the total Sobol indices 
        total_S1n_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1n_M);    total_S1n_M += S1n_M_diff
        total_S1n_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1n_T);    total_S1n_T += S1n_T_diff
        total_S1n_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1n_P);    total_S1n_P += S1n_P_diff
        total_S1n_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1n_beta); total_S1n_beta += S1n_beta_diff

        total_S1o_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1o_M);    total_S1o_M += S1o_M_diff
        total_S1o_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1o_T);    total_S1o_T += S1o_T_diff
        total_S1o_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1o_P);    total_S1o_P += S1o_P_diff
        total_S1o_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1o_beta); total_S1o_beta += S1o_beta_diff

        total_S1no_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1no_M);    total_S1no_M += S1no_M_diff
        total_S1no_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1no_T);    total_S1no_T += S1no_T_diff
        total_S1no_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1no_P);    total_S1no_P += S1no_P_diff
        total_S1no_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1no_beta); total_S1no_beta += S1no_beta_diff

        total_S1n2_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_M);    total_S1n2_M += S1n2_M_diff
        total_S1n2_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_T);    total_S1n2_T += S1n2_T_diff
        total_S1n2_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_P);    total_S1n2_P += S1n2_P_diff
        total_S1n2_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1n2_beta); total_S1n2_beta += S1n2_beta_diff

        total_S1o2_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_M);    total_S1o2_M += S1o2_M_diff
        total_S1o2_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_T);    total_S1o2_T += S1o2_T_diff
        total_S1o2_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_P);    total_S1o2_P += S1o2_P_diff
        total_S1o2_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1o2_beta); total_S1o2_beta += S1o2_beta_diff

        total_S1p_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1p_M);    total_S1p_M += S1p_M_diff
        total_S1p_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1p_T);    total_S1p_T += S1p_T_diff
        total_S1p_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1p_P);    total_S1p_P += S1p_P_diff
        total_S1p_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1p_beta); total_S1p_beta += S1p_beta_diff

        total_S1ttr_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_M);    total_S1ttr_M += S1ttr_M_diff
        total_S1ttr_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_T);    total_S1ttr_T += S1ttr_T_diff
        total_S1ttr_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_P);    total_S1ttr_P += S1ttr_P_diff
        total_S1ttr_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1ttr_beta); total_S1ttr_beta += S1ttr_beta_diff

        total_S1tve_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_M);    total_S1tve_M += S1tve_M_diff
        total_S1tve_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_T);    total_S1tve_T += S1tve_T_diff
        total_S1tve_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_P);    total_S1tve_P += S1tve_P_diff
        total_S1tve_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1tve_beta); total_S1tve_beta += S1tve_beta_diff

        total_S1m_M    = interpolate_to_fine(x_vecf, x_vec0, total_S1m_M);    total_S1m_M += S1m_M_diff
        total_S1m_T    = interpolate_to_fine(x_vecf, x_vec0, total_S1m_T);    total_S1m_T += S1m_T_diff
        total_S1m_P    = interpolate_to_fine(x_vecf, x_vec0, total_S1m_P);    total_S1m_P += S1m_P_diff
        total_S1m_beta = interpolate_to_fine(x_vecf, x_vec0, total_S1m_beta); total_S1m_beta += S1m_beta_diff

        # Update xnode_vec0 to be the finer grid for the next iteration
        x_vec0 = x_vecf

    # SAVING RESULTS

    total_S = {}

    total_S['total_S1n_M']    = total_S1n_M
    total_S['total_S1n_T']    = total_S1n_T
    total_S['total_S1n_P']    = total_S1n_P
    total_S['total_S1n_beta'] = total_S1n_beta

    total_S['total_S1o_M']    = total_S1o_M
    total_S['total_S1o_T']    = total_S1o_T
    total_S['total_S1o_P']    = total_S1o_P
    total_S['total_S1o_beta'] = total_S1o_beta

    total_S['total_S1no_M']    = total_S1no_M
    total_S['total_S1no_T']    = total_S1no_T
    total_S['total_S1no_P']    = total_S1no_P
    total_S['total_S1no_beta'] = total_S1no_beta

    total_S['total_S1n2_M']    = total_S1n2_M
    total_S['total_S1n2_T']    = total_S1n2_T
    total_S['total_S1n2_P']    = total_S1n2_P
    total_S['total_S1n2_beta'] = total_S1n2_beta

    total_S['total_S1o2_M']    = total_S1o2_M
    total_S['total_S1o2_T']    = total_S1o2_T
    total_S['total_S1o2_P']    = total_S1o2_P
    total_S['total_S1o2_beta'] = total_S1o2_beta

    total_S['total_S1p_M']    = total_S1p_M
    total_S['total_S1p_T']    = total_S1p_T
    total_S['total_S1p_P']    = total_S1p_P
    total_S['total_S1p_beta'] = total_S1p_beta

    total_S['total_S1ttr_M']    = total_S1ttr_M
    total_S['total_S1ttr_T']    = total_S1ttr_T
    total_S['total_S1ttr_P']    = total_S1ttr_P
    total_S['total_S1ttr_beta'] = total_S1ttr_beta

    total_S['total_S1tve_M']    = total_S1tve_M
    total_S['total_S1tve_T']    = total_S1tve_T
    total_S['total_S1tve_P']    = total_S1tve_P
    total_S['total_S1tve_beta'] = total_S1tve_beta

    total_S['total_S1m_M']    = total_S1m_M
    total_S['total_S1m_T']    = total_S1m_T
    total_S['total_S1m_P']    = total_S1m_P
    total_S['total_S1m_beta'] = total_S1m_beta

    return x_vec0, total_S


def interpolate_to_fine(x_fine, x_coarse, u_coarse):
    interp_func = interp1d(x_coarse, u_coarse, kind='linear', fill_value='extrapolate')
    u_interpolated = interp_func(x_fine)
    return u_interpolated