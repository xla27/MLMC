import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.sample import saltelli

from AIR5_AMG import compute_sobol_indices

def anova(problem, lev, sample_sizes, *varargin):

    # First level computations (l = 0)
    param_values0 = saltelli.sample(problem, sample_sizes[0], calc_second_order=False)
    x_vec0, S1_total = compute_sobol_indices('FINE', param_values0, 0, problem, *varargin)

    # Finer grids corrections (l>0)
    Lmax = len(sample_sizes) - 1

    for level in range(1, Lmax + 1):
        
        # Finer grid computations
        param_valuesf = saltelli.sample(problem, sample_sizes[level], calc_second_order=False)
        x_vecf, S1_f = compute_sobol_indices('FINE', param_valuesf, level, problem, *varargin)
        
        
        # Coarser grid computations
        param_valuesc = saltelli.sample(problem, sample_sizes[level], calc_second_order=False)
        x_vecc, S1_c = compute_sobol_indices('COARSE', param_valuesc, level, problem, *varargin)
        
        
        S1_c_interp = np.zeros((9, len(x_vecf), 4))
        S1_diff     = np.zeros((9, len(x_vecf), 4))
        
        for qoi in range(9):
            for dof in range(4):

                # Interpolate Sobol indices of the coarser grid to the finer grid
                S1_c_interp[qoi, :, dof] = interp1d(x_vecc, S1_c[qoi, :, dof], kind='linear', fill_value='extrapolate')(x_vecf)

                # Compute the difference between finer and coarser grid
                S1_diff[qoi, :, dof] = S1_f[qoi, :, dof] - S1_c_interp[qoi, :, dof]

        
        # Add the correction to the total Sobol indices
        S1_total_interp = np.zeros((9, len(x_vecf), 4))

        for qoi in range(9):
            for dof in range(4):

                S1_total_interp[qoi, :, dof]  = interp1d(x_vec0, S1_total[qoi, :, dof], kind='linear', fill_value='extrapolate')(x_vecf)
                S1_total_interp[qoi, :, dof] += S1_diff[qoi, :, dof]

        S1_total = S1_total_interp

        # Update the finer grid to be x_vec0 for the next iteration
        x_vec0 = x_vecf

    # SAVING RESULTS

    total_S = {}

    total_S['total_S1n_M']    = S1_total[0, :, 0]
    total_S['total_S1n_T']    = S1_total[0, :, 1]
    total_S['total_S1n_P']    = S1_total[0, :, 2]
    total_S['total_S1n_beta'] = S1_total[0, :, 3]

    total_S['total_S1o_M']    = S1_total[1, :, 0]
    total_S['total_S1o_T']    = S1_total[1, :, 1]
    total_S['total_S1o_P']    = S1_total[1, :, 2]
    total_S['total_S1o_beta'] = S1_total[1, :, 3]

    total_S['total_S1no_M']    = S1_total[2, :, 0]
    total_S['total_S1no_T']    = S1_total[2, :, 1]
    total_S['total_S1no_P']    = S1_total[2, :, 2]
    total_S['total_S1no_beta'] = S1_total[2, :, 3]

    total_S['total_S1n2_M']    = S1_total[3, :, 0]
    total_S['total_S1n2_T']    = S1_total[3, :, 1]
    total_S['total_S1n2_P']    = S1_total[3, :, 2]
    total_S['total_S1n2_beta'] = S1_total[3, :, 3]

    total_S['total_S1o2_M']    = S1_total[4, :, 0]
    total_S['total_S1o2_T']    = S1_total[4, :, 1]
    total_S['total_S1o2_P']    = S1_total[4, :, 2]
    total_S['total_S1o2_beta'] = S1_total[4, :, 3]

    total_S['total_S1p_M']    = S1_total[5, :, 0]
    total_S['total_S1p_T']    = S1_total[5, :, 1]
    total_S['total_S1p_P']    = S1_total[5, :, 2]
    total_S['total_S1p_beta'] = S1_total[5, :, 3]

    total_S['total_S1ttr_M']    = S1_total[6, :, 0]
    total_S['total_S1ttr_T']    = S1_total[6, :, 1]
    total_S['total_S1ttr_P']    = S1_total[6, :, 2]
    total_S['total_S1ttr_beta'] = S1_total[6, :, 3]

    total_S['total_S1tve_M']    = S1_total[7, :, 0]
    total_S['total_S1tve_T']    = S1_total[7, :, 1]
    total_S['total_S1tve_P']    = S1_total[7, :, 2]
    total_S['total_S1tve_beta'] = S1_total[7, :, 3]

    total_S['total_S1m_M']    = S1_total[8, :, 0]
    total_S['total_S1m_T']    = S1_total[8, :, 1]
    total_S['total_S1m_P']    = S1_total[8, :, 2]
    total_S['total_S1m_beta'] = S1_total[8, :, 3]

    return x_vec0, total_S

