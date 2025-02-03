import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.sample import saltelli

from AIR5_AMG import compute_sobol_indices

def anova(problem, lev, sample_sizes, *varargin):

    # First level computations (l = 0)
    param_values0 = saltelli.sample(problem, sample_sizes[0], calc_second_order=False)
    x_vec0, total_S1N, total_S1O, total_S1NO, total_S1N2, total_S1O2, total_S1P, total_S1Ttr, total_S1Tve, total_S1M = compute_sobol_indices('FINE', param_values0, 0, problem, *varargin)

    # Finer grids corrections (l>0)
    Lmax = len(sample_sizes) - 1

    for level in range(1, Lmax + 1):
        
        # Finer grid computations
        param_valuesf = saltelli.sample(problem, sample_sizes[level], calc_second_order=False)
        x_vecf, S1N_f, S1O_f, S1NO_f, S1N2_f, S1O2_f, S1P_f, S1Ttr_f, S1Tve_f, S1M_f = compute_sobol_indices('FINE', param_valuesf, level, problem, *varargin)
        
        
        # Coarser grid computations
        param_valuesc = saltelli.sample(problem, sample_sizes[level], calc_second_order=False)
        x_vecc, S1N_c, S1O_c, S1NO_c, S1N2_c, S1O2_c, S1P_c, S1Ttr_c, S1Tve_c, S1M_c = compute_sobol_indices('FINE', param_valuesc, level-1, problem, *varargin)
        # Note that imposing 'FINE' with level-1 makes the cfd_call perform a new simulation from scratch instead of reading the results as if 'COARSE' with lev.
        # This is due to the fact that the COARSE call happens with different parameters than the 'FINE' call, leading to different freestream conditions
        
        
        S1N_c_interp   = np.zeros((4, len(x_vecf))); S1N_diff   = np.zeros((4, len(x_vecf)))
        S1O_c_interp   = np.zeros((4, len(x_vecf))); S1O_diff   = np.zeros((4, len(x_vecf)))
        S1NO_c_interp  = np.zeros((4, len(x_vecf))); S1NO_diff  = np.zeros((4, len(x_vecf)))
        S1N2_c_interp  = np.zeros((4, len(x_vecf))); S1N2_diff  = np.zeros((4, len(x_vecf)))
        S1O2_c_interp  = np.zeros((4, len(x_vecf))); S1O2_diff  = np.zeros((4, len(x_vecf)))
        S1P_c_interp   = np.zeros((4, len(x_vecf))); S1P_diff   = np.zeros((4, len(x_vecf)))
        S1Ttr_c_interp = np.zeros((4, len(x_vecf))); S1Ttr_diff = np.zeros((4, len(x_vecf)))
        S1Tve_c_interp = np.zeros((4, len(x_vecf))); S1Tve_diff = np.zeros((4, len(x_vecf)))
        S1M_c_interp   = np.zeros((4, len(x_vecf))); S1M_diff   = np.zeros((4, len(x_vecf)))

        for var in range(4):

            # Interpolate Sobol indices of the coarser grid to the finer grid
            S1N_c_interp[var,:]   = interpolate_to_fine(x_vecf, x_vecc, S1N_c[var,:])
            S1O_c_interp[var,:]   = interpolate_to_fine(x_vecf, x_vecc, S1O_c[var,:])
            S1NO_c_interp[var,:]  = interpolate_to_fine(x_vecf, x_vecc, S1NO_c[var,:])
            S1N2_c_interp[var,:]  = interpolate_to_fine(x_vecf, x_vecc, S1N2_c[var,:])
            S1O2_c_interp[var,:]  = interpolate_to_fine(x_vecf, x_vecc, S1O2_c[var,:])
            S1P_c_interp[var,:]   = interpolate_to_fine(x_vecf, x_vecc, S1P_c[var,:])
            S1Ttr_c_interp[var,:] = interpolate_to_fine(x_vecf, x_vecc, S1Ttr_c[var,:])
            S1Tve_c_interp[var,:] = interpolate_to_fine(x_vecf, x_vecc, S1Tve_c[var,:])
            S1M_c_interp[var,:]   = interpolate_to_fine(x_vecf, x_vecc, S1M_c[var,:])

            # Compute the difference between finer and coarser grid
            S1N_diff[var,:]   = S1N_f[var,:]   - S1N_c_interp[var,:]
            S1O_diff[var,:]   = S1O_f[var,:]   - S1O_c_interp[var,:]
            S1NO_diff[var,:]  = S1NO_f[var,:]  - S1NO_c_interp[var,:]
            S1N2_diff[var,:]  = S1N2_f[var,:]  - S1N2_c_interp[var,:]
            S1O2_diff[var,:]  = S1O2_f[var,:]  - S1O2_c_interp[var,:]
            S1P_diff[var,:]   = S1P_f[var,:]   - S1P_c_interp[var,:]
            S1Ttr_diff[var,:] = S1Ttr_f[var,:] - S1Ttr_c_interp[var,:]
            S1Tve_diff[var,:] = S1Tve_f[var,:] - S1Tve_c_interp[var,:]
            S1M_diff[var,:]   = S1M_f[var,:]   - S1M_c_interp[var,:]

        
        # Add the correction to the total Sobol indices
        total_S1N_interp   = np.zeros((4, len(x_vecf)))
        total_S1O_interp   = np.zeros((4, len(x_vecf)))
        total_S1NO_interp  = np.zeros((4, len(x_vecf)))
        total_S1N2_interp  = np.zeros((4, len(x_vecf)))
        total_S1O2_interp  = np.zeros((4, len(x_vecf)))
        total_S1P_interp   = np.zeros((4, len(x_vecf)))
        total_S1Ttr_interp = np.zeros((4, len(x_vecf)))
        total_S1Tve_interp = np.zeros((4, len(x_vecf)))
        total_S1M_interp   = np.zeros((4, len(x_vecf)))

        for var in range(4):

            total_S1N_interp[var,:]   = interpolate_to_fine(x_vecf, x_vec0, total_S1N[var,:]);   total_S1N_interp[var,:]   += S1N_diff[var,:]
            total_S1O_interp[var,:]   = interpolate_to_fine(x_vecf, x_vec0, total_S1O[var,:]);   total_S1O_interp[var,:]   += S1O_diff[var,:]
            total_S1NO_interp[var,:]  = interpolate_to_fine(x_vecf, x_vec0, total_S1NO[var,:]);  total_S1NO_interp[var,:]  += S1NO_diff[var,:]
            total_S1N2_interp[var,:]  = interpolate_to_fine(x_vecf, x_vec0, total_S1N2[var,:]);  total_S1N2_interp[var,:]  += S1N2_diff[var,:]
            total_S1O2_interp[var,:]  = interpolate_to_fine(x_vecf, x_vec0, total_S1O2[var,:]);  total_S1O2_interp[var,:]  += S1O2_diff[var,:]
            total_S1P_interp[var,:]   = interpolate_to_fine(x_vecf, x_vec0, total_S1P[var,:]);   total_S1P_interp[var,:]   += S1P_diff[var,:]
            total_S1Ttr_interp[var,:] = interpolate_to_fine(x_vecf, x_vec0, total_S1Ttr[var,:]); total_S1Ttr_interp[var,:] += S1Ttr_diff[var,:]
            total_S1Tve_interp[var,:] = interpolate_to_fine(x_vecf, x_vec0, total_S1Tve[var,:]); total_S1Tve_interp[var,:] += S1Tve_diff[var,:]
            total_S1M_interp[var,:]   = interpolate_to_fine(x_vecf, x_vec0, total_S1M[var,:]);   total_S1M_interp[var,:]   += S1M_diff[var,:]

        total_S1N   = total_S1N_interp
        total_S1O   = total_S1O_interp
        total_S1NO  = total_S1NO_interp
        total_S1N2  = total_S1N2_interp
        total_S1O2  = total_S1O2_interp
        total_S1P   = total_S1P_interp
        total_S1Ttr = total_S1Ttr_interp
        total_S1Tve = total_S1Tve_interp
        total_S1M   = total_S1M_interp

        # Update the finer grid to be x_vec0 for the next iteration
        x_vec0 = x_vecf

    # SAVING RESULTS

    total_S = {}

    total_S['total_S1n_M']    = total_S1N[0,:]
    total_S['total_S1n_T']    = total_S1N[1,:]
    total_S['total_S1n_P']    = total_S1N[2,:]
    total_S['total_S1n_beta'] = total_S1N[3,:]

    total_S['total_S1o_M']    = total_S1O[0,:]
    total_S['total_S1o_T']    = total_S1O[1,:]
    total_S['total_S1o_P']    = total_S1O[2,:]
    total_S['total_S1o_beta'] = total_S1O[3,:]

    total_S['total_S1no_M']    = total_S1NO[0,:]
    total_S['total_S1no_T']    = total_S1NO[1,:]
    total_S['total_S1no_P']    = total_S1NO[2,:]
    total_S['total_S1no_beta'] = total_S1NO[3,:]

    total_S['total_S1n2_M']    = total_S1N2[0,:]
    total_S['total_S1n2_T']    = total_S1N2[1,:]
    total_S['total_S1n2_P']    = total_S1N2[2,:]
    total_S['total_S1n2_beta'] = total_S1N2[3,:]

    total_S['total_S1o2_M']    = total_S1O2[0,:]
    total_S['total_S1o2_T']    = total_S1O2[1,:]
    total_S['total_S1o2_P']    = total_S1O2[2,:]
    total_S['total_S1o2_beta'] = total_S1O2[3,:]

    total_S['total_S1p_M']    = total_S1P[0,:]
    total_S['total_S1p_T']    = total_S1P[1,:]
    total_S['total_S1p_P']    = total_S1P[2,:]
    total_S['total_S1p_beta'] = total_S1P[3,:]

    total_S['total_S1ttr_M']    = total_S1Ttr[0,:]
    total_S['total_S1ttr_T']    = total_S1Ttr[1,:]
    total_S['total_S1ttr_P']    = total_S1Ttr[2,:]
    total_S['total_S1ttr_beta'] = total_S1Ttr[3,:]

    total_S['total_S1tve_M']    = total_S1Tve[0,:]
    total_S['total_S1tve_T']    = total_S1Tve[1,:]
    total_S['total_S1tve_P']    = total_S1Tve[2,:]
    total_S['total_S1tve_beta'] = total_S1Tve[3,:]

    total_S['total_S1m_M']    = total_S1M[0,:]
    total_S['total_S1m_T']    = total_S1M[1,:]
    total_S['total_S1m_P']    = total_S1M[2,:]
    total_S['total_S1m_beta'] = total_S1M[3,:]

    return x_vec0, total_S


def interpolate_to_fine(x_fine, x_coarse, u_coarse):
    interp_func = interp1d(x_coarse, u_coarse, kind='linear', fill_value='extrapolate')
    u_interpolated = interp_func(x_fine)
    return u_interpolated