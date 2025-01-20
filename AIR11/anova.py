import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.sample import saltelli

from AIR5.sobol import compute_sobol_indices

def anova(problem, lev, sample_sizes, *varargin):

    # First level computations (l = 0)
    param_values0 = saltelli.sample(problem, sample_sizes[0], calc_second_order=False)
    [x_vec0, S1nd_elecMinus_0, S1nd_NPlus_0, S1nd_OPlus_0, S1nd_NOPlus_0, S1nd_N2Plus_0, S1nd_O2Plus_0, S1beta_elecMinus_0, S1beta_NPlus_0, S1beta_OPlus_0, S1beta_NOPlus_0, 
     S1beta_N2Plus_0, S1beta_O2Plus_0, S1beta_N_0, S1beta_O_0, S1beta_NO_0, S1beta_N2_0, S1beta_O2_0, S1P_0, S1Ttr_0, S1Tve_0, S1M_0] = compute_sobol_indices('FINE', param_values0, lev[0], problem, *varargin)

    # Finer grids corrections (l>0)
    Lmax = len(sample_sizes) - 1

    # level 0 indices 
    total_S1nd_elecMinus = np.array(S1nd_elecMinus_0) 
    total_S1nd_Nplus     = np.array(S1nd_NPlus_0) 
    total_S1nd_Oplus     = np.array(S1nd_OPlus_0) 
    total_S1nd_NOplus    = np.array(S1nd_NOPlus_0) 
    total_S1nd_N2plus    = np.array(S1nd_N2Plus_0) 
    total_S1nd_O2plus    = np.array(S1nd_O2Plus_0) 


    total_S1beta_elecMinus = np.array(S1beta_elecMinus_0)
    total_S1beta_NPlus     = np.array(S1beta_NPlus_0)
    total_S1beta_OPlus     = np.array(S1beta_OPlus_0)
    total_S1beta_NOPlus    = np.array(S1beta_NOPlus_0)
    total_S1beta_N2Plus    = np.array(S1beta_N2Plus_0)
    total_S1beta_O2Plus    = np.array(S1beta_O2Plus_0)
    total_S1beta_N         = np.array(S1beta_N_0)
    total_S1beta_O         = np.array(S1beta_O_0)
    total_S1beta_NO        = np.array(S1beta_NO_0)
    total_S1beta_N2        = np.array(S1beta_N2_0)
    total_S1beta_O2        = np.array(S1beta_O2_0)

    total_S1P   = np.array(S1P_0)
    total_S1Ttr = np.array(S1Ttr_0)
    total_S1Tve = np.array(S1Tve_0)
    total_S1M   = np.array(S1M_0)


    for l in range(1, Lmax + 1):
        
        # Finer grid computations
        param_valuesf = saltelli.sample(problem, sample_sizes[l], calc_second_order=False)
        [x_vecf, S1nd_elecMinus_f, S1nd_NPlus_f, S1nd_OPlus_f, S1nd_NOPlus_f, S1nd_N2Plus_f, S1nd_O2Plus_f, S1beta_elecMinus_f, S1beta_NPlus_f, S1beta_OPlus_f, S1beta_NOPlus_f, 
        S1beta_N2Plus_f, S1beta_O2Plus_f, S1beta_N_f, S1beta_O_f, S1beta_NO_f, S1beta_N2_f, S1beta_O2_f, S1P_f, S1Ttr_f, S1Tve_f, S1M_f] = compute_sobol_indices('FINE', param_valuesf, lev[l], problem, *varargin)
        
        
        # Coarser grid computations
        param_valuesc = saltelli.sample(problem, sample_sizes[l], calc_second_order=False)
        [x_vecc, S1nd_elecMinus_c, S1nd_NPlus_c, S1nd_OPlus_c, S1nd_NOPlus_c, S1nd_N2Plus_c, S1nd_O2Plus_c, S1beta_elecMinus_c, S1beta_NPlus_c, S1beta_OPlus_c, S1beta_NOPlus_c, 
        S1beta_N2Plus_c, S1beta_O2Plus_c, S1beta_N_c, S1beta_O_c, S1beta_NO_c, S1beta_N2_c, S1beta_O2_c, S1P_c, S1Ttr_c, S1Tve_c, S1M_c] = compute_sobol_indices('COARSE', param_valuesc, lev[l], problem, *varargin)
        
        
        # Interpolate Sobol indices of the coarser grid to the finer grid
        S1nd_elecMinus_interp = np.array([interpolate_to_fine(x_vecf, x_vecc, S1nd_elecMinus_c[i]) for i in range(4)])
        S1nd_NPlus_interp     = np.array([interpolate_to_fine(x_vecf, x_vecc, S1nd_NPlus_c[i])     for i in range(4)])
        S1nd_OPlus_interp     = np.array([interpolate_to_fine(x_vecf, x_vecc, S1nd_OPlus_c[i])     for i in range(4)])
        S1nd_NOPlus_interp    = np.array([interpolate_to_fine(x_vecf, x_vecc, S1nd_NOPlus_c[i])    for i in range(4)])
        S1nd_N2Plus_interp    = np.array([interpolate_to_fine(x_vecf, x_vecc, S1nd_N2Plus_c[i])    for i in range(4)])
        S1nd_O2Plus_interp    = np.array([interpolate_to_fine(x_vecf, x_vecc, S1nd_O2Plus_c[i])    for i in range(4)])

        S1beta_elecMinus_interp = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_elecMinus_c[i]) for i in range(4)])
        S1beta_NPlus_interp     = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_NPlus_c[i])     for i in range(4)])
        S1beta_OPlus_interp     = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_OPlus_c[i])     for i in range(4)])
        S1beta_NOPlus_interp    = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_NOPlus_c[i])    for i in range(4)])
        S1beta_N2Plus_interp    = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_N2Plus_c[i])    for i in range(4)])
        S1beta_O2Plus_interp    = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_O2Plus_c[i])    for i in range(4)])
        S1beta_N_interp         = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_N_c[i])         for i in range(4)])
        S1beta_O_interp         = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_O_c[i])         for i in range(4)])
        S1beta_NO_interp        = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_NO_c[i])        for i in range(4)])
        S1beta_N2_interp        = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_N2_c[i])        for i in range(4)])
        S1beta_O2_interp        = np.array([interpolate_to_fine(x_vecf, x_vecc, S1beta_O2_c[i])        for i in range(4)])

        S1P_interp   = np.array([interpolate_to_fine(x_vecf, x_vecc, S1P_c[i])   for i in range(4)])
        S1Ttr_interp = np.array([interpolate_to_fine(x_vecf, x_vecc, S1Ttr_c[i]) for i in range(4)])
        S1Tve_interp = np.array([interpolate_to_fine(x_vecf, x_vecc, S1Tve_c[i]) for i in range(4)])
        S1M_interp   = np.array([interpolate_to_fine(x_vecf, x_vecc, S1M_c[i])   for i in range(4)])        

        # Compute the difference between finer and coarser grid
        S1nd_elecMinus_diff = np.array([np.array(S1nd_elecMinus_f[i]) - S1nd_elecMinus_interp[i] for i in range(4)])
        S1nd_NPlus_diff     = np.array([np.array(S1nd_NPlus_f[i])     - S1nd_NPlus_interp[i]     for i in range(4)])
        S1nd_OPlus_diff     = np.array([np.array(S1nd_OPlus_f[i])     - S1nd_OPlus_interp[i]     for i in range(4)])
        S1nd_NOPlus_diff    = np.array([np.array(S1nd_NOPlus_f[i])    - S1nd_NOPlus_interp[i]    for i in range(4)])
        S1nd_N2Plus_diff    = np.array([np.array(S1nd_N2Plus_f[i])    - S1nd_N2Plus_interp[i]    for i in range(4)])
        S1nd_O2Plus_diff    = np.array([np.array(S1nd_O2Plus_f[i])    - S1nd_O2Plus_interp[i]    for i in range(4)])

        S1beta_elecMinus_diff = np.array([np.array(S1beta_elecMinus_f[i]) - S1beta_elecMinus_interp[i] for i in range(4)])
        S1beta_NPlus_diff     = np.array([np.array(S1beta_NPlus_f[i])     - S1beta_NPlus_interp[i]     for i in range(4)])
        S1beta_OPlus_diff     = np.array([np.array(S1beta_OPlus_f[i])     - S1beta_OPlus_interp[i]     for i in range(4)])
        S1beta_NOPlus_diff    = np.array([np.array(S1beta_NOPlus_f[i])    - S1beta_NOPlus_interp[i]    for i in range(4)])
        S1beta_N2Plus_diff    = np.array([np.array(S1beta_N2Plus_f[i])    - S1beta_N2Plus_interp[i]    for i in range(4)])
        S1beta_O2Plus_diff    = np.array([np.array(S1beta_O2Plus_f[i])    - S1beta_O2Plus_interp[i]    for i in range(4)])
        S1beta_N_diff         = np.array([np.array(S1beta_N_f[i])         - S1beta_N_interp[i]         for i in range(4)])
        S1beta_O_diff         = np.array([np.array(S1beta_O_f[i])         - S1beta_O_interp[i]         for i in range(4)])
        S1beta_NO_diff        = np.array([np.array(S1beta_NO_f[i])        - S1beta_NO_interp[i]        for i in range(4)])
        S1beta_N2_diff        = np.array([np.array(S1beta_N2_f[i])        - S1beta_N2_interp[i]        for i in range(4)])
        S1beta_O2_diff        = np.array([np.array(S1beta_O2_f[i])        - S1beta_O2_interp[i]        for i in range(4)])
  
        S1P_diff   = np.array([np.array(S1P_f[i])   - S1P_interp[i]   for i in range(4)])
        S1Ttr_diff = np.array([np.array(S1Ttr_f[i]) - S1Ttr_interp[i] for i in range(4)])
        S1Tve_diff = np.array([np.array(S1Tve_f[i]) - S1Tve_interp[i] for i in range(4)])
        S1M_diff   = np.array([np.array(S1M_f[i])   - S1M_interp[i]   for i in range(4)])

        # Add the correction to the total Sobol indices 
        total_S1nd_elecMinus = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1nd_elecMinus[i]) for i in range(3)]);    total_S1nd_elecMinus += S1nd_elecMinus_diff
        total_S1nd_NPlus     = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1nd_NPlus[i])     for i in range(3)]);    total_S1nd_NPlus     += S1nd_NPlus_diff
        total_S1nd_OPlus     = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1nd_OPlus[i])     for i in range(3)]);    total_S1nd_OPlus     += S1nd_OPlus_diff
        total_S1nd_NOPlus    = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1nd_NOPlus[i])    for i in range(3)]);    total_S1nd_NOPlus    += S1nd_NOPlus_diff
        total_S1nd_N2Plus    = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1nd_N2Plus[i])    for i in range(3)]);    total_S1nd_N2Plus    += S1nd_N2Plus_diff
        total_S1nd_O2Plus    = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1nd_O2Plus[i])    for i in range(3)]);    total_S1nd_O2Plus    += S1nd_O2Plus_diff

        total_S1beta_elecMinus = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_elecMinus[i]) for i in range(3)]);    total_S1beta_elecMinus += S1beta_elecMinus_diff
        total_S1beta_NPlus     = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_NPlus[i])     for i in range(3)]);    total_S1beta_NPlus     += S1beta_NPlus_diff
        total_S1beta_OPlus     = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_OPlus[i])     for i in range(3)]);    total_S1beta_OPlus     += S1beta_OPlus_diff
        total_S1beta_NOPlus    = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_NOPlus[i])    for i in range(3)]);    total_S1beta_NOPlus    += S1beta_NOPlus_diff
        total_S1beta_N2Plus    = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_N2Plus[i])    for i in range(3)]);    total_S1beta_N2Plus    += S1beta_N2Plus_diff
        total_S1beta_O2Plus    = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_O2Plus[i])    for i in range(3)]);    total_S1beta_O2Plus    += S1beta_O2Plus_diff
        total_S1beta_N         = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_N[i])         for i in range(3)]);    total_S1beta_N         += S1beta_N_diff
        total_S1beta_O         = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_O[i])         for i in range(3)]);    total_S1beta_O         += S1beta_O_diff
        total_S1beta_NO        = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_NO[i])        for i in range(3)]);    total_S1beta_NO        += S1beta_NO_diff
        total_S1beta_N2        = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_N2[i])        for i in range(3)]);    total_S1beta_N2        += S1beta_N2_diff
        total_S1beta_O2        = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1beta_O2[i])        for i in range(3)]);    total_S1beta_O2        += S1beta_O2_diff

        total_S1P   = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1P[i])   for i in range(3)]);    total_S1P   += S1P_diff
        total_S1Ttr = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1Ttr[i]) for i in range(3)]);    total_S1Ttr += S1Ttr_diff
        total_S1Tve = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1Tve[i]) for i in range(3)]);    total_S1Tve += S1Tve_diff
        total_S1M   = np.array([interpolate_to_fine(x_vecf, x_vec0, total_S1M[i])   for i in range(3)]);    total_S1M   += S1M_diff

        # Update xnode_vec0 to be the finer grid for the next iteration
        x_vec0 = x_vecf

    # SAVING RESULTS
    total_S = {}

    total_S['total_S1nd_elecMinus'] = total_S1nd_elecMinus
    total_S['total_S1nd_NPlus']     = total_S1nd_NPlus
    total_S['total_S1nd_OPlus']     = total_S1nd_OPlus
    total_S['total_S1nd_NOPlus']    = total_S1nd_NOPlus
    total_S['total_S1nd_N2Plus']    = total_S1nd_N2Plus
    total_S['total_S1nd_O2Plus']    = total_S1nd_O2Plus

    total_S['total_S1beta_elecMinus'] = total_S1beta_elecMinus
    total_S['total_S1beta_NPlus']     = total_S1beta_NPlus
    total_S['total_S1beta_OPlus']     = total_S1beta_OPlus
    total_S['total_S1beta_NOPlus']    = total_S1beta_NOPlus
    total_S['total_S1beta_N2Plus']    = total_S1beta_N2Plus
    total_S['total_S1beta_O2Plus']    = total_S1beta_O2Plus
    total_S['total_S1beta_N']         = total_S1beta_N
    total_S['total_S1beta_O']         = total_S1beta_O
    total_S['total_S1beta_NO']        = total_S1beta_NO
    total_S['total_S1beta_N2']        = total_S1beta_N2
    total_S['total_S1beta_O2']        = total_S1beta_O2

    total_S['total_S1P']   = total_S1P
    total_S['total_S1Ttr'] = total_S1Ttr
    total_S['total_S1Tve'] = total_S1Tve
    total_S['total_S1M']   = total_S1M

    return x_vec0, total_S


def interpolate_to_fine(x_fine, x_coarse, u_coarse):
    interp_func = interp1d(x_coarse, u_coarse, kind='linear', fill_value='extrapolate')
    u_interpolated = interp_func(x_fine)
    return u_interpolated