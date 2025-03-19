import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from SALib.analyze import sobol
from SALib.sample import saltelli
import csv
from skimage.restoration import denoise_tv_chambolle

fieldnames_level0 = ["PointID","x","y","Density_0","Density_1","Density_2","Density_3","Density_4",
                    "Momentum_x","Momentum_y","Energy","Energy_ve","MassFrac_0","MassFrac_1","MassFrac_2",
                    "MassFrac_3","MassFrac_4","Pressure","Temperature_tr","Temperature_ve","Velocity_x",
                    "Velocity_y","Mach","Pressure_Coefficient"]

fieldnames_levels = ["PointID","x","y","Density_0","Density_1","Density_2","Density_3","Density_4",
                    "Momentum_x","Momentum_y","Energy","Energy_ve","MassFrac_0","MassFrac_1","MassFrac_2",
                    "MassFrac_3","MassFrac_4","Pressure","Temperature_tr","Temperature_ve","Velocity_x",
                    "Velocity_y","Mach","Pressure_Coefficient","Metric_xx","Metric_xy","Metric_yy"]


def csv2dict(filename, fieldnames, delimiter=','):
    data = dict(zip(fieldnames, [None]*len(fieldnames)))
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter)
        for i, item in enumerate(reader):
            if i == 0:
                continue
            elif i == 1:
                for key in fieldnames:
                    data[key] = np.array([float(item.get(key))])
            else:
                for key in fieldnames:
                    data[key] = np.append(data[key], float(item.get(key)))

    # sorting accoridng to "x"
    sorted_indexes = np.argsort(data["x"])
    for key in fieldnames:
        data[key] = data[key][sorted_indexes]

    return data


def moving_average(data, window_size):
    """
    Function to perform the moving average.

    data:           np.array of data
    window_size:    size of the window
    """
    pad_left    = window_size // 2
    pad_right   = window_size - pad_left - 1
    padded_data = np.pad(data, (pad_left, pad_right), mode='edge')
    ma_data     = np.convolve(padded_data, np.ones(window_size), 'valid') / window_size
    return ma_data


def level_surfaces(folder):

    ref_dir = os.getcwd()

    os.chdir(os.path.join(ref_dir,folder))

    surface_files_list = sorted(os.listdir('.'))

    model_outputsN   = []
    model_outputsO   = []
    model_outputsNO  = []
    model_outputsN2  = []
    model_outputsO2  = []
    model_outputsP   = []
    model_outputsTtr = []
    model_outputsTve = []
    model_outputsM   = []

    xnodes = []

    if folder in ['level0_surf', 'level1_surf/coarse']:
        fields = fieldnames_level0
    else:
        fields = fieldnames_levels

    for i_surf, surf_name in enumerate(surface_files_list):

        surf = csv2dict(surf_name, fieldnames=fields)

        model_outputsN.append(surf['MassFrac_0'].tolist())
        model_outputsO.append(surf['MassFrac_1'].tolist())
        model_outputsNO.append(surf['MassFrac_2'].tolist())
        model_outputsN2.append(surf['MassFrac_3'].tolist())
        model_outputsO2.append(surf['MassFrac_4'].tolist())
        model_outputsP.append((surf['Pressure'] / 390).tolist())
        model_outputsTtr.append((surf['Temperature_tr'] / 1000).tolist())
        model_outputsTve.append((surf['Temperature_ve'] / 1000).tolist())
        model_outputsM.append((surf['Mach'] / 9.0).tolist())
        xnodes.append(surf['x'].tolist())

    os.chdir(ref_dir)

    return xnodes, model_outputsN, model_outputsO, model_outputsNO, model_outputsN2, model_outputsO2, model_outputsP, model_outputsTtr, model_outputsTve, model_outputsM 


def compute_sobol_indices(level, problem):

    if level == 0:

        xnodes_f, *model_outputs_f = level_surfaces('level0_surf')
        xnodes_c = [[0.0] * len(xnodes_f[0])] * len(xnodes_f)
        model_outputs_c = [[[0.0] * len(xnodes_f[0])] * len(xnodes_f) ] * len(model_outputs_f)

    elif level == 1:

        xnodes_f, *model_outputs_f = level_surfaces('level1_surf/fine')
        xnodes_c, *model_outputs_c = level_surfaces('level1_surf/coarse_v2')

    elif level == 2:

        xnodes_f, *model_outputs_f = level_surfaces('level2_surf/fine')
        xnodes_c, *model_outputs_c = level_surfaces('level2_surf/coarse_v2')

    # smoothing the wall quantities
    SF = 0.015
    xref_c = max(xnodes_c, key=len)
    xref_f = max(xnodes_f, key=len)
    N_smp = len(xnodes_c)
    ws_c = max(int(len(xref_c) * SF), 1)
    ws_f = max(int(len(xref_f) * SF), 1)


    for qoi in range(9):
        for smp in range(len(model_outputs_f[qoi])):
            model_outputs_c[qoi][smp] = moving_average(model_outputs_c[qoi][smp], ws_c)
            model_outputs_f[qoi][smp] = moving_average(model_outputs_f[qoi][smp], ws_f)

    # interpolating wall quantities to a common xref
    model_outputs_c_interp = np.zeros((N_smp, 9, len(xref_c)))
    model_outputs_f_interp = np.zeros((N_smp, 9, len(xref_f)))

    for smp in range(N_smp):
        for qoi in range(9):
            if level == 0:
                model_outputs_c_interp[smp, qoi, :] = model_outputs_c[qoi][smp]
                model_outputs_f_interp[smp, qoi, :] = model_outputs_f[qoi][smp]
            elif level == 1:
                model_outputs_c_interp[smp, qoi, :] = model_outputs_c[qoi][smp]
                model_outputs_f_interp[smp, qoi, :] = interp1d(xnodes_f[smp], model_outputs_f[qoi][smp], kind='linear', fill_value='extrapolate')(xref_f)
            elif level == 2:
                model_outputs_c_interp[smp, qoi, :] = interp1d(xnodes_c[smp], model_outputs_c[qoi][smp], kind='linear', fill_value='extrapolate')(xref_c)
                model_outputs_f_interp[smp, qoi, :] = interp1d(xnodes_f[smp], model_outputs_f[qoi][smp], kind='linear', fill_value='extrapolate')(xref_f)

    # Sobol indices
    SOBOL_c = np.zeros((9, len(xref_c), 4))
    SOBOL_f = np.zeros((9, len(xref_f), 4))

    for qoi in range(9):

        if level != 0:
            for xc in range(len(xref_c)):
                S_c = sobol.analyze(problem,  model_outputs_c_interp[:, qoi, xc], calc_second_order=False)
                SOBOL_c[qoi, xc, :] = S_c['S1']

        for xf in range(len(xref_f)):
            S_f = sobol.analyze(problem,  model_outputs_f_interp[:, qoi, xf], calc_second_order=False)
            SOBOL_f[qoi, xf, :] = S_f['S1']

    for qoi in range(9):
        for dof in range(4):
            SOBOL_f[qoi, :, dof] = denoise_tv_chambolle(SOBOL_f[qoi, :, dof], weight=1.0)
            SOBOL_c[qoi, :, dof] = denoise_tv_chambolle(SOBOL_c[qoi, :, dof], weight=1.0)

    return xref_f, SOBOL_f, xref_c, SOBOL_c


def dump_csv(xref_0, SOBOL_total):

    # Wedge starting position
    threshold = 0.193192
    xref = np.array(xref_0)

    qoi_names = ['N', 'O', 'NO', 'N2', 'O2', 'P', 'Ttr', 'Tve', 'M']
    fieldnames = ['x','S1_M','S1_P','S1_T','S1_beta']

    x = xref
    S = SOBOL_total

    # transformation
    for qoi in range(9):
        for j in range(len(x)):
            S[qoi, j, :] = np.abs(S[qoi, j, :])
            summation = np.sum(S[qoi, j, :])
            if summation > 1.0:
                for k in range(4):
                    S[qoi, j, k] = S[qoi, j, k] / (summation)

    # dump
    for qoi in range(9):

        with open(qoi_names[qoi]+'.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for j in range(len(x)):
                writer.writerow({'x': x[j],
                                 'S1_M':    S[qoi, j, 0], 
                                 'S1_T':    S[qoi, j, 1], 
                                 'S1_P':    S[qoi, j, 2], 
                                 'S1_beta': S[qoi, j, 3]})


def main():

    # Set up the problem for SALib
    problem = {
        'num_vars': 4,
        'names': ['M', 'T','P','beta'],
        'bounds': [[8.0, 9.5], [850, 1050], [300, 600], [0.76, 0.8]]
    }

    sample_sizes  = [352, 39, 23]               # number of samples per level

    # level0
    print('LEVEL 0 SOBOL')
    xref_0, SOBOL_total, _, _ = compute_sobol_indices(0, problem)

    # higher levels:
    for level in range(1,3):
        print('LEVEL {} SOBOL'.format(level))

        xref_f, SOBOL_f, xref_c, SOBOL_c = compute_sobol_indices(level, problem)

        # interpolating the fine and coarse SOBOL to ref0
        SOBOL_c_interp = np.zeros((9, len(xref_f), 4))
        SOBOL_diff     = np.zeros((9, len(xref_f), 4))

        for qoi in range(9):
            for dof in range(4):
                SOBOL_c_interp[qoi, :, dof] = interp1d(xref_c, SOBOL_c[qoi, :, dof], kind='linear', fill_value='extrapolate')(xref_f)
                SOBOL_diff[qoi, :, dof]     = SOBOL_f[qoi, :, dof] - SOBOL_c_interp[qoi, :, dof]


        # adding the correction to the total indices
        SOBOL_total_interp = np.zeros((9, len(xref_f), 4))
        for qoi in range(9):
            for dof in range(4):
                SOBOL_total_interp[qoi, :, dof] = interp1d(xref_0, SOBOL_total[qoi, :, dof], kind='linear', fill_value='extrapolate')(xref_f)
                SOBOL_total_interp[qoi, :, dof] += SOBOL_diff[qoi, :, dof]

        # updating the cycle
        SOBOL_total = SOBOL_total_interp
        xref_0      = xref_f

    # file dump
    print('FILE DUMP')
    dump_csv(xref_0, SOBOL_total)

if __name__ == '__main__':
    main()



    
