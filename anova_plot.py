import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

def anova_plot(x_vec0, total_S):

    # LOADING RESULTS

    total_S1n_M    = total_S['total_S1n_M']
    total_S1n_T    = total_S['total_S1n_T']
    total_S1n_P    = total_S['total_S1n_P']
    total_S1n_beta = total_S['total_S1n_beta']

    total_S1o_M    = total_S['total_S1o_M']
    total_S1o_T    = total_S['total_S1o_T']
    total_S1o_P    = total_S['total_S1o_P']
    total_S1o_beta = total_S['total_S1o_beta']

    total_S1no_M    = total_S['total_S1no_M']
    total_S1no_T    = total_S['total_S1no_T']
    total_S1no_P    = total_S['total_S1no_P']
    total_S1no_beta = total_S['total_S1no_beta']

    total_S1n2_M    = total_S['total_S1n2_M']
    total_S1n2_T    = total_S['total_S1n2_T']
    total_S1n2_P    = total_S['total_S1n2_P']
    total_S1n2_beta = total_S['total_S1n2_beta']

    total_S1o2_M    = total_S['total_S1o2_M']
    total_S1o2_T    = total_S['total_S1o2_T']
    total_S1o2_P    = total_S['total_S1o2_P']
    total_S1o2_beta = total_S['total_S1o2_beta']

    total_S1p_M    = total_S['total_S1p_M']
    total_S1p_T    = total_S['total_S1p_T']
    total_S1p_P    = total_S['total_S1p_P']
    total_S1p_beta = total_S['total_S1p_beta']

    total_S1ttr_M    = total_S['total_S1ttr_M']
    total_S1ttr_T    = total_S['total_S1ttr_T']
    total_S1ttr_P    = total_S['total_S1ttr_P']
    total_S1ttr_beta = total_S['total_S1ttr_beta']

    total_S1tve_M    = total_S['total_S1tve_M']
    total_S1tve_T    = total_S['total_S1tve_T']
    total_S1tve_P    = total_S['total_S1tve_P']
    total_S1tve_beta = total_S['total_S1tve_beta']

    total_S1m_M    = total_S['total_S1m_M']
    total_S1m_T    = total_S['total_S1m_T']
    total_S1m_P    = total_S['total_S1m_P']
    total_S1m_beta = total_S['total_S1m_beta']

    # Wedge starting position
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