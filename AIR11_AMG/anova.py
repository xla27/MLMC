import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.sample import saltelli

from AIR11_AMG import compute_sobol_indices

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

        S1_c_interp = np.zeros((21, len(x_vecf), 4))
        S1_diff     = np.zeros((21, len(x_vecf), 4))

        for qoi in range(21):
            for dof in range(4):

                # Interpolate Sobol indices of the coarser grid to the finer grid
                S1_c_interp[qoi, :, dof] = interp1d(x_vecc, S1_c[qoi, :, dof], kind='linear', fill_value='extrapolate')(x_vecf)

                # Compute the difference between finer and coarser grid
                S1_diff[qoi, :, dof] = S1_f[qoi, :, dof] - S1_c_interp[qoi, :, dof]

        
        # Add the correction to the total Sobol indices
        S1_total_interp = np.zeros((21, len(x_vecf), 4))

        for qoi in range(21):
            for dof in range(4):

                S1_total_interp[qoi, :, dof]  = interp1d(x_vec0, S1_total[qoi, :, dof], kind='linear', fill_value='extrapolate')(x_vecf)
                S1_total_interp[qoi, :, dof] += S1_diff[qoi, :, dof]

        S1_total = S1_total_interp

        # Update the finer grid to be x_vec0 for the next iteration
        x_vec0 = x_vecf

    # SAVING RESULTS
    total_S = {}

    total_S['total_S1nd_elecMinus'] = S1_total[0, :, :]
    total_S['total_S1nd_NPlus']     = S1_total[1, :, :]
    total_S['total_S1nd_OPlus']     = S1_total[2, :, :]
    total_S['total_S1nd_NOPlus']    = S1_total[3, :, :]
    total_S['total_S1nd_N2Plus']    = S1_total[4, :, :]
    total_S['total_S1nd_O2Plus']    = S1_total[5, :, :]

    total_S['total_S1beta_elecMinus'] = S1_total[6, :, :]
    total_S['total_S1beta_NPlus']     = S1_total[7, :, :]
    total_S['total_S1beta_OPlus']     = S1_total[8, :, :]
    total_S['total_S1beta_NOPlus']    = S1_total[9, :, :]
    total_S['total_S1beta_N2Plus']    = S1_total[10, :, :]
    total_S['total_S1beta_O2Plus']    = S1_total[11, :, :]
    total_S['total_S1beta_N']         = S1_total[12, :, :]
    total_S['total_S1beta_O']         = S1_total[13, :, :]
    total_S['total_S1beta_NO']        = S1_total[14, :, :]
    total_S['total_S1beta_N2']        = S1_total[15, :, :]
    total_S['total_S1beta_O2']        = S1_total[16, :, :]

    total_S['total_S1P']   = S1_total[17, :, :]
    total_S['total_S1Ttr'] = S1_total[18, :, :]
    total_S['total_S1Tve'] = S1_total[19, :, :]
    total_S['total_S1M']   = S1_total[20, :, :]

    return x_vec0, total_S


def interpolate_to_fine(x_fine, x_coarse, u_coarse):
    interp_func = interp1d(x_coarse, u_coarse, kind='linear', fill_value='extrapolate')
    u_interpolated = interp_func(x_fine)
    return u_interpolated