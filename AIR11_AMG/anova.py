import os, sys
import numpy as np
from scipy.interpolate import interp1d
from SALib.sample import saltelli

from AIR11_AMG import compute_sobol_indices

def anova(problem, lev, sample_sizes, *varargin):

    # First level computations (l = 0)
    param_values0 = saltelli.sample(problem, sample_sizes[0], calc_second_order=False)
    [x_vec0, total_S1nd_elecMinus, total_S1nd_NPlus, total_S1nd_OPlus, total_S1nd_NOPlus, 
     total_S1nd_N2Plus, total_S1nd_O2Plus, total_S1beta_elecMinus, total_S1beta_NPlus, 
     total_S1beta_OPlus, total_S1beta_NOPlus, total_S1beta_N2Plus, total_S1beta_O2Plus,
     total_S1beta_N, total_S1beta_O, total_S1beta_NO, total_S1beta_N2, total_S1beta_O2, 
     total_S1P, total_S1Ttr, total_S1Tve, total_S1M] = compute_sobol_indices('FINE', param_values0, 0, problem, *varargin)

    # Finer grids corrections (l>0)
    Lmax = len(sample_sizes) - 1

    for level in range(1, Lmax + 1):
        
        # Finer grid computations
        param_valuesf = saltelli.sample(problem, sample_sizes[level], calc_second_order=False)
        [x_vecf, S1nd_elecMinus_f, S1nd_NPlus_f, S1nd_OPlus_f, S1nd_NOPlus_f, S1nd_N2Plus_f, 
         S1nd_O2Plus_f, S1beta_elecMinus_f, S1beta_NPlus_f, S1beta_OPlus_f, S1beta_NOPlus_f, 
        S1beta_N2Plus_f, S1beta_O2Plus_f, S1beta_N_f, S1beta_O_f, S1beta_NO_f, S1beta_N2_f, 
        S1beta_O2_f, S1P_f, S1Ttr_f, S1Tve_f, S1M_f] = compute_sobol_indices('FINE', param_valuesf, level, problem, *varargin)
        
        
        # Coarser grid computations
        param_valuesc = saltelli.sample(problem, sample_sizes[level], calc_second_order=False)
        [x_vecc, S1nd_elecMinus_c, S1nd_NPlus_c, S1nd_OPlus_c, S1nd_NOPlus_c, S1nd_N2Plus_c, 
         S1nd_O2Plus_c, S1beta_elecMinus_c, S1beta_NPlus_c, S1beta_OPlus_c, S1beta_NOPlus_c, 
        S1beta_N2Plus_c, S1beta_O2Plus_c, S1beta_N_c, S1beta_O_c, S1beta_NO_c, S1beta_N2_c, 
        S1beta_O2_c, S1P_c, S1Ttr_c, S1Tve_c, S1M_c] = compute_sobol_indices('FINE', param_valuesc, level-1, problem, *varargin)
        # Note that imposing 'FINE' with level-1 makes the cfd_call perform a new simulation from scratch instead of reading the results as if 'COARSE' with lev.
        # This is due to the fact that the COARSE call happens with different parameters than the 'FINE' call, leading to different freestream conditions        
        

        S1nd_elecMinus_interp   = np.zeros((4, len(x_vecf))); S1nd_elecMinus_diff   = np.zeros((4, len(x_vecf)))
        S1nd_NPlus_interp       = np.zeros((4, len(x_vecf))); S1nd_NPlus_diff       = np.zeros((4, len(x_vecf)))
        S1nd_OPlus_interp       = np.zeros((4, len(x_vecf))); S1nd_OPlus_diff       = np.zeros((4, len(x_vecf)))
        S1nd_NOPlus_interp      = np.zeros((4, len(x_vecf))); S1nd_NOPlus_diff      = np.zeros((4, len(x_vecf)))
        S1nd_N2Plus_interp      = np.zeros((4, len(x_vecf))); S1nd_N2Plus_diff      = np.zeros((4, len(x_vecf)))
        S1nd_O2Plus_interp      = np.zeros((4, len(x_vecf))); S1nd_O2Plus_diff      = np.zeros((4, len(x_vecf)))
        S1beta_elecMinus_interp = np.zeros((4, len(x_vecf))); S1beta_elecMinus_diff = np.zeros((4, len(x_vecf)))
        S1beta_NPlus_interp     = np.zeros((4, len(x_vecf))); S1beta_NPlus_diff     = np.zeros((4, len(x_vecf)))
        S1beta_OPlus_interp     = np.zeros((4, len(x_vecf))); S1beta_OPlus_diff     = np.zeros((4, len(x_vecf)))
        S1beta_NOPlus_interp    = np.zeros((4, len(x_vecf))); S1beta_NOPlus_diff    = np.zeros((4, len(x_vecf)))
        S1beta_N2Plus_interp    = np.zeros((4, len(x_vecf))); S1beta_N2Plus_diff    = np.zeros((4, len(x_vecf)))
        S1beta_O2Plus_interp    = np.zeros((4, len(x_vecf))); S1beta_O2Plus_diff    = np.zeros((4, len(x_vecf)))
        S1beta_N_interp         = np.zeros((4, len(x_vecf))); S1beta_N_diff         = np.zeros((4, len(x_vecf)))
        S1beta_O_interp         = np.zeros((4, len(x_vecf))); S1beta_O_diff         = np.zeros((4, len(x_vecf)))
        S1beta_NO_interp        = np.zeros((4, len(x_vecf))); S1beta_NO_diff        = np.zeros((4, len(x_vecf)))
        S1beta_N2_interp        = np.zeros((4, len(x_vecf))); S1beta_N2_diff        = np.zeros((4, len(x_vecf)))
        S1beta_O2_interp        = np.zeros((4, len(x_vecf))); S1beta_O2_diff        = np.zeros((4, len(x_vecf)))
        S1P_interp              = np.zeros((4, len(x_vecf))); S1P_diff              = np.zeros((4, len(x_vecf)))
        S1Ttr_interp            = np.zeros((4, len(x_vecf))); S1Ttr_diff            = np.zeros((4, len(x_vecf)))
        S1Tve_interp            = np.zeros((4, len(x_vecf))); S1Tve_diff            = np.zeros((4, len(x_vecf)))
        S1M_interp              = np.zeros((4, len(x_vecf))); S1M_diff              = np.zeros((4, len(x_vecf)))

        for var in range(4):

            # Interpolate Sobol indices of the coarser grid to the finer grid
            S1nd_elecMinus_interp[var,:]   = interpolate_to_fine(x_vecf, x_vecc, S1nd_elecMinus_c[var,:]) 
            S1nd_NPlus_interp[var,:]       = interpolate_to_fine(x_vecf, x_vecc, S1nd_NPlus_c[var,:])
            S1nd_OPlus_interp[var,:]       = interpolate_to_fine(x_vecf, x_vecc, S1nd_OPlus_c[var,:])    
            S1nd_NOPlus_interp[var,:]      = interpolate_to_fine(x_vecf, x_vecc, S1nd_NOPlus_c[var,:])  
            S1nd_N2Plus_interp[var,:]      = interpolate_to_fine(x_vecf, x_vecc, S1nd_N2Plus_c[var,:])   
            S1nd_O2Plus_interp[var,:]      = interpolate_to_fine(x_vecf, x_vecc, S1nd_O2Plus_c[var,:])  
            S1beta_elecMinus_interp[var,:] = interpolate_to_fine(x_vecf, x_vecc, S1beta_elecMinus_c[var,:]) 
            S1beta_NPlus_interp[var,:]     = interpolate_to_fine(x_vecf, x_vecc, S1beta_NPlus_c[var,:])     
            S1beta_OPlus_interp[var,:]     = interpolate_to_fine(x_vecf, x_vecc, S1beta_OPlus_c[var,:])    
            S1beta_NOPlus_interp[var,:]    = interpolate_to_fine(x_vecf, x_vecc, S1beta_NOPlus_c[var,:]) 
            S1beta_N2Plus_interp[var,:]    = interpolate_to_fine(x_vecf, x_vecc, S1beta_N2Plus_c[var,:])    
            S1beta_O2Plus_interp[var,:]    = interpolate_to_fine(x_vecf, x_vecc, S1beta_O2Plus_c[var,:])    
            S1beta_N_interp[var,:]         = interpolate_to_fine(x_vecf, x_vecc, S1beta_N_c[var,:])       
            S1beta_O_interp[var,:]         = interpolate_to_fine(x_vecf, x_vecc, S1beta_O_c[var,:])        
            S1beta_NO_interp[var,:]        = interpolate_to_fine(x_vecf, x_vecc, S1beta_NO_c[var,:])    
            S1beta_N2_interp[var,:]        = interpolate_to_fine(x_vecf, x_vecc, S1beta_N2_c[var,:])    
            S1beta_O2_interp[var,:]        = interpolate_to_fine(x_vecf, x_vecc, S1beta_O2_c[var,:])    
            S1P_interp[var,:]              = interpolate_to_fine(x_vecf, x_vecc, S1P_c[var,:])  
            S1Ttr_interp[var,:]            = interpolate_to_fine(x_vecf, x_vecc, S1Ttr_c[var,:]) 
            S1Tve_interp[var,:]            = interpolate_to_fine(x_vecf, x_vecc, S1Tve_c[var,:]) 
            S1M_interp[var,:]              = interpolate_to_fine(x_vecf, x_vecc, S1M_c[var,:])         

            # Compute the difference between finer and coarser grid
            S1nd_elecMinus_diff[var,:]   = S1nd_elecMinus_f[var,:]   - S1nd_elecMinus_interp[var,:] 
            S1nd_NPlus_diff[var,:]       = S1nd_NPlus_f[var,:]       - S1nd_NPlus_interp[var,:]   
            S1nd_OPlus_diff[var,:]       = S1nd_OPlus_f[var,:]       - S1nd_OPlus_interp[var,:]    
            S1nd_NOPlus_diff[var,:]      = S1nd_NOPlus_f[var,:]      - S1nd_NOPlus_interp[var,:] 
            S1nd_N2Plus_diff[var,:]      = S1nd_N2Plus_f[var,:]      - S1nd_N2Plus_interp[var,:]   
            S1nd_O2Plus_diff[var,:]      = S1nd_O2Plus_f[var,:]      - S1nd_O2Plus_interp[var,:]    
            S1beta_elecMinus_diff[var,:] = S1beta_elecMinus_f[var,:] - S1beta_elecMinus_interp[var,:]
            S1beta_NPlus_diff[var,:]     = S1beta_NPlus_f[var,:]     - S1beta_NPlus_interp[var,:]   
            S1beta_OPlus_diff[var,:]     = S1beta_OPlus_f[var,:]     - S1beta_OPlus_interp[var,:] 
            S1beta_NOPlus_diff[var,:]    = S1beta_NOPlus_f[var,:]    - S1beta_NOPlus_interp[var,:]    
            S1beta_N2Plus_diff[var,:]    = S1beta_N2Plus_f[var,:]    - S1beta_N2Plus_interp[var,:]   
            S1beta_O2Plus_diff[var,:]    = S1beta_O2Plus_f[var,:]    - S1beta_O2Plus_interp[var,:]  
            S1beta_N_diff[var,:]         = S1beta_N_f[var,:]         - S1beta_N_interp[var,:]         
            S1beta_O_diff[var,:]         = S1beta_O_f[var,:]         - S1beta_O_interp[var,:]      
            S1beta_NO_diff[var,:]        = S1beta_NO_f[var,:]        - S1beta_NO_interp[var,:]      
            S1beta_N2_diff[var,:]        = S1beta_N2_f[var,:]        - S1beta_N2_interp[var,:]      
            S1beta_O2_diff[var,:]        = S1beta_O2_f[var,:]        - S1beta_O2_interp[var,:]        
            S1P_diff[var,:]              = S1P_f[var,:]              - S1P_interp[var,:]
            S1Ttr_diff[var,:]            = S1Ttr_f[var,:]            - S1Ttr_interp[var,:] 
            S1Tve_diff[var,:]            = S1Tve_f[var,:]            - S1Tve_interp[var,:] 
            S1M_diff[var,:]              = S1M_f[var,:]              - S1M_interp[var,:]  


        # Add the correction to the total Sobol indices 
        total_S1nd_elecMinus_interp    = np.zeros((4, len(x_vecf)))
        total_S1nd_NPlus_interp        = np.zeros((4, len(x_vecf)))
        total_S1nd_OPlus_interp        = np.zeros((4, len(x_vecf)))
        total_S1nd_NOPlus_interp       = np.zeros((4, len(x_vecf)))
        total_S1nd_N2Plus_interp       = np.zeros((4, len(x_vecf)))
        total_S1nd_O2Plus_interp       = np.zeros((4, len(x_vecf)))
        total_S1beta_elecMinus_interp  = np.zeros((4, len(x_vecf)))
        total_S1beta_NPlus_interp      = np.zeros((4, len(x_vecf)))
        total_S1beta_OPlus_interp      = np.zeros((4, len(x_vecf)))
        total_S1beta_NOPlus_interp     = np.zeros((4, len(x_vecf)))
        total_S1beta_N2Plus_interp     = np.zeros((4, len(x_vecf)))
        total_S1beta_O2Plus_interp     = np.zeros((4, len(x_vecf)))
        total_S1beta_N_interp          = np.zeros((4, len(x_vecf)))
        total_S1beta_O_interp          = np.zeros((4, len(x_vecf)))
        total_S1beta_NO_interp         = np.zeros((4, len(x_vecf)))
        total_S1beta_N2_interp         = np.zeros((4, len(x_vecf)))
        total_S1beta_O2_interp         = np.zeros((4, len(x_vecf)))
        total_S1P_interp               = np.zeros((4, len(x_vecf)))
        total_S1Ttr_interp             = np.zeros((4, len(x_vecf)))
        total_S1Tve_interp             = np.zeros((4, len(x_vecf)))
        total_S1M_interp               = np.zeros((4, len(x_vecf)))

        for var in range(4):

            total_S1nd_elecMinus_interp[var,:]   = interpolate_to_fine(x_vecf, x_vec0, total_S1nd_elecMinus[var,:]);   total_S1nd_elecMinus_interp[var,:]   += S1nd_elecMinus_diff[var,:]
            total_S1nd_NPlus_interp[var,:]       = interpolate_to_fine(x_vecf, x_vec0, total_S1nd_NPlus[var,:]);       total_S1nd_NPlus_interp[var,:]       += S1nd_NPlus_diff[var,:]
            total_S1nd_OPlus_interp[var,:]       = interpolate_to_fine(x_vecf, x_vec0, total_S1nd_OPlus[var,:]);       total_S1nd_OPlus_interp[var,:]       += S1nd_OPlus_diff[var,:]
            total_S1nd_NOPlus_interp[var,:]      = interpolate_to_fine(x_vecf, x_vec0, total_S1nd_NOPlus[var,:]);      total_S1nd_NOPlus_interp[var,:]      += S1nd_NOPlus_diff[var,:]
            total_S1nd_N2Plus_interp[var,:]      = interpolate_to_fine(x_vecf, x_vec0, total_S1nd_N2Plus[var,:]);      total_S1nd_N2Plus_interp[var,:]      += S1nd_N2Plus_diff[var,:]
            total_S1nd_O2Plus_interp[var,:]      = interpolate_to_fine(x_vecf, x_vec0, total_S1nd_O2Plus[var,:]);      total_S1nd_O2Plus_interp[var,:]      += S1nd_O2Plus_diff[var,:]
            total_S1beta_elecMinus_interp[var,:] = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_elecMinus[var,:]); total_S1beta_elecMinus_interp[var,:] += S1beta_elecMinus_diff[var,:]
            total_S1beta_NPlus_interp[var,:]     = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_NPlus[var,:]);     total_S1beta_NPlus_interp[var,:]     += S1beta_NPlus_diff[var,:]
            total_S1beta_OPlus_interp[var,:]     = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_OPlus[var,:]);     total_S1beta_OPlus_interp[var,:]     += S1beta_OPlus_diff[var,:]
            total_S1beta_NOPlus_interp[var,:]    = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_NOPlus[var,:]);    total_S1beta_NOPlus_interp[var,:]    += S1beta_NOPlus_diff[var,:]
            total_S1beta_N2Plus_interp[var,:]    = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_N2Plus[var,:]);    total_S1beta_N2Plus_interp[var,:]    += S1beta_N2Plus_diff[var,:]
            total_S1beta_O2Plus_interp[var,:]    = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_O2Plus[var,:]);    total_S1beta_O2Plus_interp[var,:]    += S1beta_O2Plus_diff[var,:]
            total_S1beta_N_interp[var,:]         = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_N[var,:]);         total_S1beta_N_interp[var,:]         += S1beta_N_diff[var,:]
            total_S1beta_O_interp[var,:]         = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_O[var,:]);         total_S1beta_O_interp[var,:]         += S1beta_O_diff[var,:]
            total_S1beta_NO_interp[var,:]        = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_NO[var,:]);        total_S1beta_NO_interp[var,:]        += S1beta_NO_diff[var,:]
            total_S1beta_N2_interp[var,:]        = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_N2[var,:]);        total_S1beta_N2_interp[var,:]        += S1beta_N2_diff[var,:]
            total_S1beta_O2_interp[var,:]        = interpolate_to_fine(x_vecf, x_vec0, total_S1beta_O2[var,:]);        total_S1beta_O2_interp[var,:]        += S1beta_O2_diff[var,:]
            total_S1P_interp[var,:]              = interpolate_to_fine(x_vecf, x_vec0, total_S1P[var,:]);              total_S1P_interp[var,:]              += S1P_diff[var,:]
            total_S1Ttr_interp[var,:]            = interpolate_to_fine(x_vecf, x_vec0, total_S1Ttr[var,:]);            total_S1Ttr_interp[var,:]            += S1Ttr_diff[var,:]
            total_S1Tve_interp[var,:]            = interpolate_to_fine(x_vecf, x_vec0, total_S1Tve[var,:]);            total_S1Tve_interp[var,:]            += S1Tve_diff[var,:]
            total_S1M_interp[var,:]              = interpolate_to_fine(x_vecf, x_vec0, total_S1M[var,:]);              total_S1M_interp[var,:]              += S1M_diff[var,:]

        total_S1nd_elecMinus    = total_S1nd_elecMinus_interp
        total_S1nd_NPlus        = total_S1nd_NPlus_interp
        total_S1nd_OPlus        = total_S1nd_OPlus_interp
        total_S1nd_NOPlus       = total_S1nd_NOPlus_interp
        total_S1nd_N2Plus       = total_S1nd_N2Plus_interp
        total_S1nd_O2Plus       = total_S1nd_O2Plus_interp
        total_S1beta_elecMinus  = total_S1beta_elecMinus_interp
        total_S1beta_NPlus      = total_S1beta_NPlus_interp
        total_S1beta_OPlus      = total_S1beta_OPlus_interp
        total_S1beta_NOPlus     = total_S1beta_NOPlus_interp
        total_S1beta_N2Plus     = total_S1beta_N2Plus_interp
        total_S1beta_O2Plus     = total_S1beta_O2Plus_interp
        total_S1beta_N          = total_S1beta_N_interp
        total_S1beta_O          = total_S1beta_O_interp
        total_S1beta_NO         = total_S1beta_NO_interp
        total_S1beta_N2         = total_S1beta_N2_interp
        total_S1beta_O2         = total_S1beta_O2_interp
        total_S1P               = total_S1P_interp
        total_S1Ttr             = total_S1Ttr_interp
        total_S1Tve             = total_S1Tve_interp
        total_S1M               = total_S1M_interp

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