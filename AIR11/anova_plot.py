import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

def anova_plot(x_vec0, total_S):

    # LOADING RESULTS
    total_S1nd_elecMinus = np.array(total_S['total_S1nd_elecMinus'])
    total_S1nd_NPlus     = np.array(total_S['total_S1nd_NPlus'])
    total_S1nd_OPlus     = np.array(total_S['total_S1nd_OPlus'])
    total_S1nd_NOPlus    = np.array(total_S['total_S1nd_NOPlus'])
    total_S1nd_N2Plus    = np.array(total_S['total_S1nd_N2Plus'])
    total_S1nd_O2Plus    = np.array(total_S['total_S1nd_O2Plus'])

    total_S1beta_elecMinus = np.array(total_S['total_S1beta_elecMinus'])
    total_S1beta_NPlus     = np.array(total_S['total_S1beta_NPlus'])
    total_S1beta_OPlus     = np.array(total_S['total_S1beta_OPlus'])
    total_S1beta_NOPlus    = np.array(total_S['total_S1beta_NOPlus'])
    total_S1beta_N2Plus    = np.array(total_S['total_S1beta_N2Plus'])
    total_S1beta_O2Plus    = np.array(total_S['total_S1beta_O2Plus'])
    total_S1beta_N         = np.array(total_S['total_S1beta_N'])
    total_S1beta_O         = np.array(total_S['total_S1beta_O'])
    total_S1beta_NO        = np.array(total_S['total_S1beta_NO'])
    total_S1beta_N2        = np.array(total_S['total_S1beta_N2'])
    total_S1beta_O2        = np.array(total_S['total_S1beta_O2'])

    total_S1P   = np.array(total_S['total_S1P'])
    total_S1Ttr = np.array(total_S['total_S1Ttr'])
    total_S1Tve = np.array(total_S['total_S1Tve'])
    total_S1M   = np.array(total_S['total_S1M'])

    # Wedge starting position
    threshold = 0.193192
    x_vec0fil = np.array(x_vec0)
    filter_indices = x_vec0fil > threshold
    x_vec0_filtered = x_vec0fil[filter_indices]

    # Plot 1: first-order Sobol indices of e- number density
    plt.figure(figsize=(13, 11))

    total_S1nd_elecMinus_filtered = total_S1nd_elecMinus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1nd_elecMinus_filtered[0], label='S1nd_elecMinus_M')
    plt.plot(x_vec0_filtered, total_S1nd_elecMinus_filtered[1], label='S1nd_elecMinus_T')
    plt.plot(x_vec0_filtered, total_S1nd_elecMinus_filtered[2], label='S1nd_elecMinus_P')
    plt.plot(x_vec0_filtered, total_S1nd_elecMinus_filtered[3], label='S1nd_elecMinus_beta')

    plt.title('First-order Sobol Indices of e- number density against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '1_S1nd_elecMinus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1nd_elecMinus': total_S1nd_elecMinus_filtered,
    }

    pickle_filename = '1_S1nd_elecMinus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 2: first-order Sobol indices of N+ number density
    plt.figure(figsize=(13, 11))

    total_S1nd_NPlus_filtered = total_S1nd_NPlus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1nd_NPlus_filtered[0], label='S1nd_nPlus_M')
    plt.plot(x_vec0_filtered, total_S1nd_NPlus_filtered[1], label='S1nd_nPlus_T')
    plt.plot(x_vec0_filtered, total_S1nd_NPlus_filtered[2], label='S1nd_nPlus_P')
    plt.plot(x_vec0_filtered, total_S1nd_NPlus_filtered[3], label='S1nd_nPlus_beta')

    plt.title('First-order Sobol Indices of N+ number density against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '2_S1nd_NPlus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1nd_NPlus': total_S1nd_NPlus_filtered,
    }

    pickle_filename = '2_S1nd_NPlus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 3: first-order Sobol indices of O+ number density
    plt.figure(figsize=(13, 11))

    total_S1nd_OPlus_filtered = total_S1nd_OPlus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1nd_OPlus_filtered[0], label='S1nd_oPlus_M')
    plt.plot(x_vec0_filtered, total_S1nd_OPlus_filtered[1], label='S1nd_oPlus_T')
    plt.plot(x_vec0_filtered, total_S1nd_OPlus_filtered[2], label='S1nd_oPlus_P')
    plt.plot(x_vec0_filtered, total_S1nd_OPlus_filtered[3], label='S1nd_oPlus_beta')

    plt.title('First-order Sobol Indices of O+ number density against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '3_S1nd_OPlus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1nd_OPlus': total_S1nd_OPlus_filtered,
    }

    pickle_filename = '3_S1nd_OPlus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()


    # Plot 4: first-order Sobol indices of NO+ number density
    plt.figure(figsize=(13, 11))

    total_S1nd_NOPlus_filtered = total_S1nd_NOPlus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1nd_NOPlus_filtered[0], label='S1nd_noPlus_M')
    plt.plot(x_vec0_filtered, total_S1nd_NOPlus_filtered[1], label='S1nd_noPlus_T')
    plt.plot(x_vec0_filtered, total_S1nd_NOPlus_filtered[2], label='S1nd_noPlus_P')
    plt.plot(x_vec0_filtered, total_S1nd_NOPlus_filtered[3], label='S1nd_noPlus_beta')

    plt.title('First-order Sobol Indices of NO+ number density against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '4_S1nd_NOPlus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1nd_NOPlus': total_S1nd_NOPlus_filtered,
    }

    pickle_filename = '4_S1nd_NOPlus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 5: first-order Sobol indices of N2+ number density
    plt.figure(figsize=(13, 11))

    total_S1nd_N2Plus_filtered = total_S1nd_N2Plus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1nd_N2Plus_filtered[0], label='S1nd_n2Plus_M')
    plt.plot(x_vec0_filtered, total_S1nd_N2Plus_filtered[1], label='S1nd_n2Plus_T')
    plt.plot(x_vec0_filtered, total_S1nd_N2Plus_filtered[2], label='S1nd_n2Plus_P')
    plt.plot(x_vec0_filtered, total_S1nd_N2Plus_filtered[3], label='S1nd_n2Plus_beta')

    plt.title('First-order Sobol Indices of N2+ number density against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '5_S1nd_N2Plus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1nd_N2Plus': total_S1nd_N2Plus_filtered,
    }

    pickle_filename = '5_S1nd_N2Plus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 6: first-order Sobol indices of O2+ number density
    plt.figure(figsize=(13, 11))

    total_S1nd_O2Plus_filtered = total_S1nd_O2Plus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1nd_O2Plus_filtered[0], label='S1nd_o2Plus_M')
    plt.plot(x_vec0_filtered, total_S1nd_O2Plus_filtered[1], label='S1nd_o2Plus_T')
    plt.plot(x_vec0_filtered, total_S1nd_O2Plus_filtered[2], label='S1nd_o2Plus_P')
    plt.plot(x_vec0_filtered, total_S1nd_O2Plus_filtered[3], label='S1nd_o2Plus_beta')

    plt.title('First-order Sobol Indices of O2+ number density against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '6_S1nd_O2Plus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1nd_O2Plus': total_S1nd_O2Plus_filtered,
    }

    pickle_filename = '6_S1nd_O2Plus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 7: first-order Sobol indices of e- mass fraction
    plt.figure(figsize=(13, 11))

    total_S1beta_elecMinus_filtered = total_S1beta_elecMinus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_elecMinus_filtered[0], label='S1beta_elecMinus_M')
    plt.plot(x_vec0_filtered, total_S1beta_elecMinus_filtered[1], label='S1beta_elecMinus_T')
    plt.plot(x_vec0_filtered, total_S1beta_elecMinus_filtered[2], label='S1beta_elecMinus_P')
    plt.plot(x_vec0_filtered, total_S1beta_elecMinus_filtered[3], label='S1beta_elecMinus_beta')

    plt.title('First-order Sobol Indices of e- mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '7_S1beta_elecMinus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_elecMinus': total_S1beta_elecMinus_filtered,
    }

    pickle_filename = '7_S1beta_elecMinus_first_order_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 8: first-order Sobol indices of N+ mass fraction
    plt.figure(figsize=(13, 11))

    total_S1beta_NPlus_filtered = total_S1beta_NPlus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_NPlus_filtered[0], label='S1beta_nPlus_M')
    plt.plot(x_vec0_filtered, total_S1beta_NPlus_filtered[1], label='S1beta_nPlus_T')
    plt.plot(x_vec0_filtered, total_S1beta_NPlus_filtered[2], label='S1beta_nPlus_P')
    plt.plot(x_vec0_filtered, total_S1beta_NPlus_filtered[3], label='S1beta_nPlus_beta')

    plt.title('First-order Sobol Indices of N+ mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '8_S1beta_NPlus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_NPlus': total_S1beta_NPlus_filtered,
    }

    pickle_filename = '8_S1beta_NPlus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 9: first-order Sobol indices of O+ mass fraction
    plt.figure(figsize=(13, 11))

    total_S1beta_OPlus_filtered = total_S1beta_OPlus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_OPlus_filtered[0], label='S1beta_oPlus_M')
    plt.plot(x_vec0_filtered, total_S1beta_OPlus_filtered[1], label='S1beta_oPlus_T')
    plt.plot(x_vec0_filtered, total_S1beta_OPlus_filtered[2], label='S1beta_oPlus_P')
    plt.plot(x_vec0_filtered, total_S1beta_OPlus_filtered[3], label='S1beta_oPlus_beta')

    plt.title('First-order Sobol Indices of O+ mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '9_S1beta_OPlus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_OPlus': total_S1beta_OPlus_filtered,
    }

    pickle_filename = '9_S1beta_OPlus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 10: first-order Sobol indices of NO+ mass fraction
    plt.figure(figsize=(13, 11))

    total_S1beta_NOPlus_filtered = total_S1beta_NOPlus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_NOPlus_filtered[0], label='S1beta_noPlus_M')
    plt.plot(x_vec0_filtered, total_S1beta_NOPlus_filtered[1], label='S1beta_noPlus_T')
    plt.plot(x_vec0_filtered, total_S1beta_NOPlus_filtered[2], label='S1beta_noPlus_P')
    plt.plot(x_vec0_filtered, total_S1beta_NOPlus_filtered[3], label='S1beta_noPlus_beta')

    plt.title('First-order Sobol Indices of NO+ mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '10_S1beta_NOPlus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_NOPlus': total_S1beta_NOPlus_filtered,
    }

    pickle_filename = '10_S1beta_NOPlus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 11: first-order Sobol indices of N2+ mass fraction
    plt.figure(figsize=(13, 11))

    total_S1beta_N2Plus_filtered = total_S1beta_N2Plus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_N2Plus_filtered[0], label='S1beta_n2Plus_M')
    plt.plot(x_vec0_filtered, total_S1beta_N2Plus_filtered[1], label='S1beta_n2Plus_T')
    plt.plot(x_vec0_filtered, total_S1beta_N2Plus_filtered[2], label='S1beta_n2Plus_P')
    plt.plot(x_vec0_filtered, total_S1beta_N2Plus_filtered[3], label='S1beta_n2Plus_beta')

    plt.title('First-order Sobol Indices of N2+ mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '11_S1beta_N2Plus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_N2Plus': total_S1beta_N2Plus_filtered,
    }

    pickle_filename = '11_S1beta_N2Plus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 12: first-order Sobol indices of O2+ mass fraction
    plt.figure(figsize=(13, 11))

    total_S1beta_O2Plus_filtered = total_S1beta_O2Plus[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_O2Plus_filtered[0], label='S1beta_o2Plus_M')
    plt.plot(x_vec0_filtered, total_S1beta_O2Plus_filtered[1], label='S1beta_o2Plus_T')
    plt.plot(x_vec0_filtered, total_S1beta_O2Plus_filtered[2], label='S1beta_o2Plus_P')
    plt.plot(x_vec0_filtered, total_S1beta_O2Plus_filtered[3], label='S1beta_o2Plus_beta')

    plt.title('First-order Sobol Indices of O2+ mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '12_S1beta_O2Plus_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_O2Plus': total_S1beta_O2Plus_filtered,
    }

    pickle_filename = '12_S1beta_O2Plus_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    plt.close("all")
    plt.cla()
    plt.clf()


    # Plot 13: first-order Sobol indices of N mass fraction against space coordinate
    plt.figure(figsize=(13, 11))

    total_S1beta_N_filtered = total_S1beta_N[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_N_filtered[0], label='S1n(M)')
    plt.plot(x_vec0_filtered, total_S1beta_N_filtered[1], label='S1n(T)')
    plt.plot(x_vec0_filtered, total_S1beta_N_filtered[2], label='S1n(P)')
    plt.plot(x_vec0_filtered, total_S1beta_N_filtered[3], label='S1n(beta)')
    plt.title('First-order Sobol Indices of N mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '13_N_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_N': total_S1beta_N_filtered,
    }

    pickle_filename = '13_N_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 14: first-order Sobol indices of O mass fraction against space coordinate
    plt.figure(figsize=(13, 11))

    total_S1beta_O_filtered = total_S1beta_O[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_O_filtered[0], label='S1o(M)')
    plt.plot(x_vec0_filtered, total_S1beta_O_filtered[1], label='S1o(T)')
    plt.plot(x_vec0_filtered, total_S1beta_O_filtered[2], label='S1o(P)')
    plt.plot(x_vec0_filtered, total_S1beta_O_filtered[3], label='S1o(beta)')
    plt.title('First-order Sobol Indices of O mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '14_O_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_O': total_S1beta_O_filtered,
    }

    pickle_filename = '14_O_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 15: first-order Sobol indices of NO mass fraction against space coordinate
    plt.figure(figsize=(13, 11))

    total_S1beta_NO_filtered = total_S1beta_NO[filter_indices]

    plt.plot(x_vec0_filtered, total_S1beta_NO_filtered[0], label='S1no(M)')
    plt.plot(x_vec0_filtered, total_S1beta_NO_filtered[1], label='S1no(T)')
    plt.plot(x_vec0_filtered, total_S1beta_NO_filtered[2], label='S1no(P)')
    plt.plot(x_vec0_filtered, total_S1beta_NO_filtered[3], label='S1no(beta)')
    plt.title('First-order Sobol Indices of NO mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '15_NO_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0_filtered': x_vec0_filtered,
        'total_S1beta_NO': total_S1beta_NO_filtered,
    }

    pickle_filename = '15_NO_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 16: first-order Sobol indices of N2 mass fraction against space coordinate
    plt.figure(figsize=(13, 11))

    plt.plot(x_vec0, total_S1beta_N2[0], label='S1n2(M)')
    plt.plot(x_vec0, total_S1beta_N2[1], label='S1n2(T)')
    plt.plot(x_vec0, total_S1beta_N2[2], label='S1n2(P)')
    plt.plot(x_vec0, total_S1beta_N2[3], label='S1n2(beta)')
    plt.title('First-order Sobol Indices of N2 mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '16_N2_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0': x_vec0,
        'total_S1n2_M': total_S1beta_N2,
    }

    pickle_filename = '16_N2_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 17: first-order Sobol indices of O2 mass fraction against space coordinate
    plt.figure(figsize=(13, 11))

    plt.plot(x_vec0, total_S1beta_O2[0], label='S1o2(M)')
    plt.plot(x_vec0, total_S1beta_O2[1], label='S1o2(T)')
    plt.plot(x_vec0, total_S1beta_O2[2], label='S1o2(P)')
    plt.plot(x_vec0, total_S1beta_O2[3], label='S1o2(beta)')
    plt.title('First-order Sobol Indices of O2 mass fraction against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '17_O2_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0': x_vec0,
        'total_S1beta_O2': total_S1beta_O2,
    }

    pickle_filename = '17_O2_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 18: first-order Sobol indices of P against space coordinate
    plt.figure(figsize=(13, 11))

    plt.plot(x_vec0, total_S1P[0], label='S1p(M)')
    plt.plot(x_vec0, total_S1P[1], label='S1p(T)')
    plt.plot(x_vec0, total_S1P[2], label='S1p(P)')
    plt.plot(x_vec0, total_S1P[3], label='S1p(beta)')
    plt.title('First-order Sobol Indices P against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '18_P_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0': x_vec0,
        'total_S1P': total_S1P,
    }

    pickle_filename = '18_P_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 19: first-order Sobol indices of Ttr against space coordinate
    plt.figure(figsize=(13, 11))

    plt.plot(x_vec0, total_S1Ttr[0], label='S1ttr(M)')
    plt.plot(x_vec0, total_S1Ttr[1], label='S1ttr(T)')
    plt.plot(x_vec0, total_S1Ttr[2], label='S1ttr(P)')
    plt.plot(x_vec0, total_S1Ttr[3], label='S1ttr(beta)')
    plt.title('First-order Sobol Indices Ttr against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '19_Ttr_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0': x_vec0,
        'total_S1Ttr': total_S1Ttr,
    }

    pickle_filename = '19_Ttr_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 20: first-order Sobol indices of Tve against space coordinate
    plt.figure(figsize=(13, 11))

    plt.plot(x_vec0, total_S1Tve[0], label='S1tve(M)')
    plt.plot(x_vec0, total_S1Tve[1], label='S1tve(T)')
    plt.plot(x_vec0, total_S1Tve[2], label='S1tve(P)')
    plt.plot(x_vec0, total_S1Tve[3], label='S1tve(beta)')
    plt.title('First-order Sobol Indices Tve against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '20_Tve_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0': x_vec0,
        'total_S1Tve': total_S1Tve,
    }

    pickle_filename = '20_Tve_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 21: first-order Sobol indices of M against space coordinate
    plt.figure(figsize=(13, 11))

    plt.plot(x_vec0, total_S1M[0], label='S1m(M)')
    plt.plot(x_vec0, total_S1M[1], label='S1m(T)')
    plt.plot(x_vec0, total_S1M[2], label='S1m(P)')
    plt.plot(x_vec0, total_S1M[3], label='S1m(beta)')
    plt.title('First-order Sobol Indices M against X coordinate $\\varepsilon_r = 0.003$')
    plt.xlabel('X coordinate')
    plt.ylabel('Sobol Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_filename = '21_Mf_sobol_indices'
    plt.savefig(plot_filename, format='svg')

    plot_data = {
        'x_vec0': x_vec0,
        'total_S1M': total_S1M,
    }

    pickle_filename = '21_M_sobol_indices.pkl'
    with open(pickle_filename, 'wb') as f:
        pickle.dump((plt.gcf(), plot_data), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()