import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle

def write(logfile, msg):
    """
    Write to a logfile.
    """
    logfile.write(msg)
    logfile.flush()

def mlmc(mlmc_l, N0, eps, Lmin, Lmax, alpha0, beta0, gamma0, Nlfile, *args):
    """
    Inputs:
      N0:   initial number of samples    >  0
      eps:  desired accuracy (rms error) >  0
      Lmin: minimum level of refinement  >= 2
      Lmax: maximum level of refinement  >= Lmin
      mlmc_l: the user low-level routine for level l estimator. 
      *args, **kwargs = optional additional user variables to be passed to mlmc_l

    Outputs:
      P:  value of the QoI
      Nl: number of samples at each level
      Cl: cost of samples at each level
    """

    alpha = max(0, alpha0); beta = max(0, beta0); gamma = max(0, gamma0); theta = 0.5; L = Lmin;

    Nl = np.zeros(L+1); costl = np.zeros(L+1); dNl = np.ones(L+1) * N0;

    cellsom1P = [None] * (L+1); cellsom2P = [None] * (L+1);
    cellsom1_nd_elecMinus = [None] * (L + 1); cellsom2_nd_elecMinus = [None] * (L + 1)
    cellsom1_nd_nPlus = [None] * (L + 1); cellsom2_nd_nPlus = [None] * (L + 1)
    cellsom1_nd_oPlus = [None] * (L + 1); cellsom2_nd_oPlus = [None] * (L + 1)
    cellsom1_nd_noPlus = [None] * (L + 1); cellsom2_nd_noPlus = [None] * (L + 1)
    cellsom1_nd_n2Plus = [None] * (L + 1); cellsom2_nd_n2Plus = [None] * (L + 1)
    cellsom1_nd_o2Plus = [None] * (L + 1); cellsom2_nd_o2Plus = [None] * (L + 1)
    cellsom1Ttr = [None] * (L+1); cellsom2Ttr = [None] * (L+1);
    cellsom1Tve = [None] * (L+1); cellsom2Tve = [None] * (L+1);
    cellsom1M = [None] * (L+1); cellsom2M = [None] * (L+1);
    cellnodes = [None] * (L+1);
            
    while np.sum(dNl) > 0:
       # Update sample sums

        for l in range(L+1):
            if dNl[l] > 0:
                x2, sums1, sums2, _, _, sums1ndelecMinus, sums2ndelecMinus, sums1ndNPlus, sums2ndNPlus, sums1ndOPlus, sums2ndOPlus, sums1ndNOPlus, sums2ndNOPlus, sums1ndN2Plus, sums2ndN2Plus, sums1ndO2Plus, sums2ndO2Plus, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost = mlmc_l(l, int(dNl[l]), *args)
                Nl[l] += dNl[l]
                costl[l] += cost
                if cellsom1P[l] is None:
                    cellsom1P[l] = sums1; cellsom2P[l] = sums2;
                    cellsom1_nd_elecMinus[l] = sums1ndelecMinus; cellsom2_nd_elecMinus[l] = sums2ndelecMinus;
                    cellsom1_nd_nPlus[l] = sums1ndNPlus; cellsom2_nd_nPlus[l] = sums2ndNPlus;
                    cellsom1_nd_oPlus[l] = sums1ndOPlus; cellsom2_nd_oPlus[l] = sums2ndOPlus;
                    cellsom1_nd_noPlus[l] = sums1ndNOPlus; cellsom2_nd_noPlus[l] = sums2ndNOPlus;
                    cellsom1_nd_n2Plus[l] = sums1ndN2Plus; cellsom2_nd_n2Plus[l] = sums2ndN2Plus;
                    cellsom1_nd_o2Plus[l] = sums1ndO2Plus; cellsom2_nd_o2Plus[l] = sums2ndO2Plus;
                    cellsom1Ttr[l] = sums1Ttr; cellsom2Ttr[l] = sums2Ttr;
                    cellsom1Tve[l] = sums1Tve; cellsom2Tve[l] = sums2Tve;
                    cellsom1M[l] = sums1M; cellsom2M[l] = sums2M;
                    
                else:
                    cellsom1P[l] += sums1; cellsom2P[l] += sums2;
                    cellsom1_nd_elecMinus[l] += sums1ndelecMinus; cellsom2_nd_elecMinus[l] += sums2ndelecMinus;
                    cellsom1_nd_nPlus[l] += sums1ndNPlus; cellsom2_nd_nPlus[l] += sums2ndNPlus;
                    cellsom1_nd_oPlus[l] += sums1ndOPlus; cellsom2_nd_oPlus[l] += sums2ndOPlus;
                    cellsom1_nd_noPlus[l] += sums1ndNOPlus; cellsom2_nd_noPlus[l] += sums2ndNOPlus;
                    cellsom1_nd_n2Plus[l] += sums1ndN2Plus; cellsom2_nd_n2Plus[l] += sums2ndN2Plus;
                    cellsom1_nd_o2Plus[l] += sums1ndO2Plus; cellsom2_nd_o2Plus[l] += sums2ndO2Plus;
                    cellsom1Ttr[l] += sums1Ttr; cellsom2Ttr[l] += sums2Ttr;
                    cellsom1Tve[l] += sums1Tve; cellsom2Tve[l] += sums2Tve;
                    cellsom1M[l] += sums1M; cellsom2M[l] += sums2M;                    
                       
                cellnodes[l] = x2
        
        # Interpolation on coarsest grid
        coarsest_grid = cellnodes[0]

        interpolated_cellsom1P = []; interpolated_cellsom2P = [];
        interpolated_cellsom1_nd_elecMinus = []; interpolated_cellsom2_nd_elecMinus = [];
        interpolated_cellsom1_nd_nPlus = []; interpolated_cellsom2_nd_nPlus = [];
        interpolated_cellsom1_nd_oPlus = []; interpolated_cellsom2_nd_oPlus = [];
        interpolated_cellsom1_nd_noPlus = []; interpolated_cellsom2_nd_noPlus = [];
        interpolated_cellsom1_nd_n2Plus = []; interpolated_cellsom2_nd_n2Plus = [];
        interpolated_cellsom1_nd_o2Plus = []; interpolated_cellsom2_nd_o2Plus = [];
        interpolated_cellsom1Ttr = []; interpolated_cellsom2Ttr = [];
        interpolated_cellsom1Tve = []; interpolated_cellsom2Tve = [];
        interpolated_cellsom1M = []; interpolated_cellsom2M = [];

        for i in range(len(cellsom1P)):
            current_arrayCS1P = cellsom1P[i]
            interpolated_cellsom1P.append(interp1d(cellnodes[i], current_arrayCS1P, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2P = cellsom2P[i]
            interpolated_cellsom2P.append(interp1d(cellnodes[i], current_arrayCS2P, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_elecMinus = cellsom1_nd_elecMinus[i]
            interpolated_cellsom1_nd_elecMinus.append(interp1d(cellnodes[i], current_arrayCS1_nd_elecMinus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_elecMinus = cellsom2_nd_elecMinus[i]
            interpolated_cellsom2_nd_elecMinus.append(interp1d(cellnodes[i], current_arrayCS2_nd_elecMinus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_nPlus = cellsom1_nd_nPlus[i]
            interpolated_cellsom1_nd_nPlus.append(interp1d(cellnodes[i], current_arrayCS1_nd_nPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_nPlus = cellsom2_nd_nPlus[i]
            interpolated_cellsom2_nd_nPlus.append(interp1d(cellnodes[i], current_arrayCS2_nd_nPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_oPlus = cellsom1_nd_oPlus[i]
            interpolated_cellsom1_nd_oPlus.append(interp1d(cellnodes[i], current_arrayCS1_nd_oPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_oPlus = cellsom2_nd_oPlus[i]
            interpolated_cellsom2_nd_oPlus.append(interp1d(cellnodes[i], current_arrayCS2_nd_oPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_noPlus = cellsom1_nd_noPlus[i]
            interpolated_cellsom1_nd_noPlus.append(interp1d(cellnodes[i], current_arrayCS1_nd_noPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_noPlus = cellsom2_nd_noPlus[i]
            interpolated_cellsom2_nd_noPlus.append(interp1d(cellnodes[i], current_arrayCS2_nd_noPlus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_n2Plus = cellsom1_nd_n2Plus[i]
            interpolated_cellsom1_nd_n2Plus.append(interp1d(cellnodes[i], current_arrayCS1_nd_n2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_n2Plus = cellsom2_nd_n2Plus[i]
            interpolated_cellsom2_nd_n2Plus.append(interp1d(cellnodes[i], current_arrayCS2_nd_n2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1_nd_o2Plus = cellsom1_nd_o2Plus[i]
            interpolated_cellsom1_nd_o2Plus.append(interp1d(cellnodes[i], current_arrayCS1_nd_o2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2_nd_o2Plus = cellsom2_nd_o2Plus[i]
            interpolated_cellsom2_nd_o2Plus.append(interp1d(cellnodes[i], current_arrayCS2_nd_o2Plus, kind='linear', fill_value='extrapolate')(coarsest_grid))

            current_arrayCS1Ttr = cellsom1Ttr[i]
            interpolated_cellsom1Ttr.append(interp1d(cellnodes[i], current_arrayCS1Ttr, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2Ttr = cellsom2Ttr[i]
            interpolated_cellsom2Ttr.append(interp1d(cellnodes[i], current_arrayCS2Ttr, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1Tve = cellsom1Tve[i]
            interpolated_cellsom1Tve.append(interp1d(cellnodes[i], current_arrayCS1Tve, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2Tve = cellsom2Tve[i]
            interpolated_cellsom2Tve.append(interp1d(cellnodes[i], current_arrayCS2Tve, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1M = cellsom1M[i]
            interpolated_cellsom1M.append(interp1d(cellnodes[i], current_arrayCS1M, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2M = cellsom2M[i]
            interpolated_cellsom2M.append(interp1d(cellnodes[i], current_arrayCS2M, kind='linear', fill_value='extrapolate')(coarsest_grid))     
                        
        CS1P_matrix = np.transpose(np.array(interpolated_cellsom1P)); CS2P_matrix = np.transpose(np.array(interpolated_cellsom2P));
        CS1_nd_elecMinus_matrix = np.transpose(np.array(interpolated_cellsom1_nd_elecMinus)); CS2_nd_elecMinus_matrix = np.transpose(np.array(interpolated_cellsom2_nd_elecMinus));
        CS1_nd_nPlus_matrix = np.transpose(np.array(interpolated_cellsom1_nd_nPlus)); CS2_nd_nPlus_matrix = np.transpose(np.array(interpolated_cellsom2_nd_nPlus));
        CS1_nd_oPlus_matrix = np.transpose(np.array(interpolated_cellsom1_nd_oPlus)); CS2_nd_oPlus_matrix = np.transpose(np.array(interpolated_cellsom2_nd_oPlus));
        CS1_nd_noPlus_matrix = np.transpose(np.array(interpolated_cellsom1_nd_noPlus)); CS2_nd_noPlus_matrix = np.transpose(np.array(interpolated_cellsom2_nd_noPlus));
        CS1_nd_n2Plus_matrix = np.transpose(np.array(interpolated_cellsom1_nd_n2Plus)); CS2_nd_n2Plus_matrix = np.transpose(np.array(interpolated_cellsom2_nd_n2Plus));
        CS1_nd_o2Plus_matrix = np.transpose(np.array(interpolated_cellsom1_nd_o2Plus)); CS2_nd_o2Plus_matrix = np.transpose(np.array(interpolated_cellsom2_nd_o2Plus));
        CS1Ttr_matrix = np.transpose(np.array(interpolated_cellsom1Ttr)); CS2Ttr_matrix = np.transpose(np.array(interpolated_cellsom2Ttr));
        CS1Tve_matrix = np.transpose(np.array(interpolated_cellsom1Tve)); CS2Tve_matrix = np.transpose(np.array(interpolated_cellsom2Tve));
        CS1M_matrix = np.transpose(np.array(interpolated_cellsom1M)); CS2M_matrix = np.transpose(np.array(interpolated_cellsom2M));

        # Compute absolute average, variance and cost
        ml = np.abs(np.max(CS1P_matrix / Nl, axis=0)) # L-infinity norm of the average of each level
        Vl = np.maximum(0, np.max(CS2P_matrix / Nl - (CS1P_matrix / Nl)**2, axis=0))
        Cl = costl / Nl

        ml_vec = CS1P_matrix / Nl
        Vl_vec = CS2P_matrix / Nl - ml_vec**2
        Vl_vec[Vl_vec < 0] = 0
                
        # Other quantities mean and variance
        ml_nd_elecMinus_vec = CS1_nd_elecMinus_matrix / Nl
        Vl_nd_elecMinus_vec = CS2_nd_elecMinus_matrix / Nl - ml_nd_elecMinus_vec**2
        Vl_nd_elecMinus_vec[Vl_nd_elecMinus_vec < 0] = 0

        ml_nd_nPlus_vec = CS1_nd_nPlus_matrix / Nl
        Vl_nd_nPlus_vec = CS2_nd_nPlus_matrix / Nl - ml_nd_nPlus_vec**2
        Vl_nd_nPlus_vec[Vl_nd_nPlus_vec < 0] = 0

        ml_nd_oPlus_vec = CS1_nd_oPlus_matrix / Nl
        Vl_nd_oPlus_vec = CS2_nd_oPlus_matrix / Nl - ml_nd_oPlus_vec**2
        Vl_nd_oPlus_vec[Vl_nd_oPlus_vec < 0] = 0

        ml_nd_noPlus_vec = CS1_nd_noPlus_matrix / Nl
        Vl_nd_noPlus_vec = CS2_nd_noPlus_matrix / Nl - ml_nd_noPlus_vec**2
        Vl_nd_noPlus_vec[Vl_nd_noPlus_vec < 0] = 0

        ml_nd_n2Plus_vec = CS1_nd_n2Plus_matrix / Nl
        Vl_nd_n2Plus_vec = CS2_nd_n2Plus_matrix / Nl - ml_nd_n2Plus_vec**2
        Vl_nd_n2Plus_vec[Vl_nd_n2Plus_vec < 0] = 0

        ml_nd_o2Plus_vec = CS1_nd_o2Plus_matrix / Nl
        Vl_nd_o2Plus_vec = CS2_nd_o2Plus_matrix / Nl - ml_nd_o2Plus_vec**2
        Vl_nd_o2Plus_vec[Vl_nd_o2Plus_vec < 0] = 0

        mlTtr_vec = CS1Ttr_matrix / Nl
        VlTtr_vec = CS2Ttr_matrix / Nl - mlTtr_vec**2
        VlTtr_vec[VlTtr_vec < 0] = 0

        mlTve_vec = CS1Tve_matrix / Nl
        VlTve_vec = CS2Tve_matrix / Nl - mlTve_vec**2
        VlTve_vec[VlTve_vec < 0] = 0

        mlM_vec = CS1M_matrix / Nl
        VlM_vec = CS2M_matrix / Nl - mlM_vec**2
        VlM_vec[VlM_vec < 0] = 0

        # Set optimal number of additional samples
        Ns = np.ceil(np.sqrt(Vl / Cl) * np.sum(np.sqrt(Vl * Cl)) / ((1 - theta) * eps**2))
        dNl = np.maximum(0, Ns - Nl)
        dNl_str = str(dNl)
        write(Nlfile, f"dNl = {dNl_str}\n")

        # If (almost) converged, estimate remaining error and decide whether a new level is required
        if np.sum(dNl > 0.01 * Nl) == 0:
            rem = ml[L] / (2**alpha - 1)

            if rem > np.sqrt(theta) * eps:
                if L == Lmax:
                    write(Nlfile, "*** failed to achieve weak convergence ***\n")
                else:
                    L += 1
                    Vl = np.append(Vl, Vl[-1] / 2**beta)
                    Vl_vec = np.append(Vl_vec, (Vl_vec[:, L-1] / (2 ** beta)).reshape(-1, 1), axis=1)
                    Cl = np.append(Cl, Cl[-1] * 2**gamma)
                    Nl = np.append(Nl, 0.0)
                    
                    cellsom1P.append(None); cellsom2P.append(None);
                    cellsom1_nd_elecMinus.append(None); cellsom2_nd_elecMinus.append(None);
                    cellsom1_nd_nPlus.append(None); cellsom2_nd_nPlus.append(None);
                    cellsom1_nd_oPlus.append(None); cellsom2_nd_oPlus.append(None);
                    cellsom1_nd_noPlus.append(None); cellsom2_nd_noPlus.append(None);
                    cellsom1_nd_n2Plus.append(None); cellsom2_nd_n2Plus.append(None);
                    cellsom1_nd_o2Plus.append(None); cellsom2_nd_o2Plus.append(None);
                    cellsom1Ttr.append(None); cellsom2Ttr.append(None);
                    cellsom1Tve.append(None); cellsom2Tve.append(None);
                    cellsom1M.append(None); cellsom2M.append(None);
                       
                    cellnodes.append(None)
                    costl = np.append(costl, 0)

                    Ns = np.ceil(np.sqrt(Vl / Cl) * np.sum(np.sqrt(Vl * Cl)) / ((1 - theta) * eps**2))
                    dNl = np.maximum(0, Ns - Nl)

    # Evaluate multilevel estimator
    P = np.sum(np.mean(ml_vec, axis=0)) # mean is for the columns and sum is for the rows: it represents the mean of the solution vector

    # Compute vector QoIs sums of the mean and uncertainty bounds
    sum_ml = np.sum(ml_vec, axis=1) # sum the averages by rows
    sum_Vl = np.sum(Vl_vec, axis=1)
    upper_bound = sum_ml + np.sqrt(np.abs(sum_Vl))
    lower_bound = sum_ml - np.sqrt(np.abs(sum_Vl))

    sum_ml_nd_elecMinus = np.sum(ml_nd_elecMinus_vec, axis=1)
    sum_Vl_nd_elecMinus = np.sum(Vl_nd_elecMinus_vec, axis=1)
    upper_bound_nd_elecMinus = sum_ml_nd_elecMinus + np.sqrt(np.abs(sum_Vl_nd_elecMinus))
    lower_bound_nd_elecMinus = sum_ml_nd_elecMinus - np.sqrt(np.abs(sum_Vl_nd_elecMinus))

    sum_ml_nd_nPlus = np.sum(ml_nd_nPlus_vec, axis=1)
    sum_Vl_nd_nPlus = np.sum(Vl_nd_nPlus_vec, axis=1)
    upper_bound_nd_nPlus = sum_ml_nd_nPlus + np.sqrt(np.abs(sum_Vl_nd_nPlus))
    lower_bound_nd_nPlus = sum_ml_nd_nPlus - np.sqrt(np.abs(sum_Vl_nd_nPlus))

    sum_ml_nd_oPlus = np.sum(ml_nd_oPlus_vec, axis=1)
    sum_Vl_nd_oPlus = np.sum(Vl_nd_oPlus_vec, axis=1)
    upper_bound_nd_oPlus = sum_ml_nd_oPlus + np.sqrt(np.abs(sum_Vl_nd_oPlus))
    lower_bound_nd_oPlus = sum_ml_nd_oPlus - np.sqrt(np.abs(sum_Vl_nd_oPlus))

    sum_ml_nd_noPlus = np.sum(ml_nd_noPlus_vec, axis=1)
    sum_Vl_nd_noPlus = np.sum(Vl_nd_noPlus_vec, axis=1)
    upper_bound_nd_noPlus = sum_ml_nd_noPlus + np.sqrt(np.abs(sum_Vl_nd_noPlus))
    lower_bound_nd_noPlus = sum_ml_nd_noPlus - np.sqrt(np.abs(sum_Vl_nd_noPlus))

    sum_ml_nd_n2Plus = np.sum(ml_nd_n2Plus_vec, axis=1)
    sum_Vl_nd_n2Plus = np.sum(Vl_nd_n2Plus_vec, axis=1)
    upper_bound_nd_n2Plus = sum_ml_nd_n2Plus + np.sqrt(np.abs(sum_Vl_nd_n2Plus))
    lower_bound_nd_n2Plus = sum_ml_nd_n2Plus - np.sqrt(np.abs(sum_Vl_nd_n2Plus))

    sum_ml_nd_o2Plus = np.sum(ml_nd_o2Plus_vec, axis=1)
    sum_Vl_nd_o2Plus = np.sum(Vl_nd_o2Plus_vec, axis=1)
    upper_bound_nd_o2Plus = sum_ml_nd_o2Plus + np.sqrt(np.abs(sum_Vl_nd_o2Plus))
    lower_bound_nd_o2Plus = sum_ml_nd_o2Plus - np.sqrt(np.abs(sum_Vl_nd_o2Plus))

    sum_mlTtr = np.sum(mlTtr_vec, axis=1)
    sum_VlTtr = np.sum(VlTtr_vec, axis=1)
    upper_boundTtr = sum_mlTtr + np.sqrt(np.abs(sum_VlTtr))
    lower_boundTtr = sum_mlTtr - np.sqrt(np.abs(sum_VlTtr))    

    sum_mlTve = np.sum(mlTve_vec, axis=1)
    sum_VlTve = np.sum(VlTve_vec, axis=1)
    upper_boundTve = sum_mlTve + np.sqrt(np.abs(sum_VlTve))
    lower_boundTve = sum_mlTve - np.sqrt(np.abs(sum_VlTve))    

    sum_mlM = np.sum(mlM_vec, axis=1)
    sum_VlM = np.sum(VlM_vec, axis=1)
    upper_boundM = sum_mlM + np.sqrt(np.abs(sum_VlM))
    lower_boundM = sum_mlM - np.sqrt(np.abs(sum_VlM))    
    
    # Plot 1: normalized mean pressure, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax = plt.gca()  
    ax.fill_between(coarsest_grid, upper_bound, lower_bound, color=shaded_color, alpha=0.7, edgecolor='none')
    ax.plot(coarsest_grid, sum_ml)
    ax.set_title(r'P with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  # Convert eps array to string and replace dots with underscore
    ax.figure.savefig(f'1_Pepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'1_Pepsilon_{eps_str}.pkl'
    
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_bound, lower_bound, sum_ml), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()
    
    # Plot 2: mean number density for e-, 1sigma uncertainty region
    ax2 = plt.gca()
    ax2.fill_between(coarsest_grid, upper_bound_nd_elecMinus, lower_bound_nd_elecMinus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax2.plot(coarsest_grid, sum_ml_nd_elecMinus)
    ax2.set_title(r'Number density e- with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax2.figure.savefig(f'2_nd_elecMinus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'2_nd_elecMinus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax2, coarsest_grid, upper_bound_nd_elecMinus, lower_bound_nd_elecMinus, sum_ml_nd_elecMinus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 3: mean number density for N+, 1sigma uncertainty region
    ax3 = plt.gca()
    ax3.fill_between(coarsest_grid, upper_bound_nd_nPlus, lower_bound_nd_nPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax3.plot(coarsest_grid, sum_ml_nd_nPlus)
    ax3.set_title(r'Number density N+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax3.figure.savefig(f'3_nd_nPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'3_nd_nPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax3, coarsest_grid, upper_bound_nd_nPlus, lower_bound_nd_nPlus, sum_ml_nd_nPlus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 4: mean number density for O+, 1sigma uncertainty region
    ax4 = plt.gca()
    ax4.fill_between(coarsest_grid, upper_bound_nd_oPlus, lower_bound_nd_oPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax4.plot(coarsest_grid, sum_ml_nd_oPlus)
    ax4.set_title(r'Number density O+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax4.figure.savefig(f'4_nd_oPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'4_nd_oPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax4, coarsest_grid, upper_bound_nd_oPlus, lower_bound_nd_oPlus, sum_ml_nd_oPlus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 5: mean number density for NO+, 1sigma uncertainty region
    ax5 = plt.gca()
    ax5.fill_between(coarsest_grid, upper_bound_nd_noPlus, lower_bound_nd_noPlus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax5.plot(coarsest_grid, sum_ml_nd_noPlus)
    ax5.set_title(r'Number density NO+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax5.figure.savefig(f'5_nd_noPlus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'5_nd_noPlus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax5, coarsest_grid, upper_bound_nd_noPlus, lower_bound_nd_noPlus, sum_ml_nd_noPlus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 6: mean number density for N2+, 1sigma uncertainty region
    ax6 = plt.gca()
    ax6.fill_between(coarsest_grid, upper_bound_nd_n2Plus, lower_bound_nd_n2Plus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax6.plot(coarsest_grid, sum_ml_nd_n2Plus)
    ax6.set_title(r'Number density N2+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax6.figure.savefig(f'6_nd_n2Plus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'6_nd_n2Plus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax6, coarsest_grid, upper_bound_nd_n2Plus, lower_bound_nd_n2Plus, sum_ml_nd_n2Plus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 7: mean number density for O2+, 1sigma uncertainty region
    ax7 = plt.gca()
    ax7.fill_between(coarsest_grid, upper_bound_nd_o2Plus, lower_bound_nd_o2Plus, color=shaded_color, alpha=0.7, edgecolor='none')
    ax7.plot(coarsest_grid, sum_ml_nd_o2Plus)
    ax7.set_title(r'Number density O2+ with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')
    ax7.figure.savefig(f'7_nd_o2Plus_epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'7_nd_o2Plus_epsilon_{eps_str}.pkl'

    with open(plot_filename, 'wb') as f:
        pickle.dump((ax7, coarsest_grid, upper_bound_nd_o2Plus, lower_bound_nd_o2Plus, sum_ml_nd_o2Plus), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()

    # Plot 19: Normalized rototranslational temperature, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax19 = plt.gca()  # Get the current axis
    ax19.fill_between(coarsest_grid, upper_boundTtr, lower_boundTtr, color=shaded_color, alpha=0.7, edgecolor='none')
    ax19.plot(coarsest_grid, sum_mlTtr)
    ax19.set_title(r'Ttr with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax19.figure.savefig(f'19_Ttrepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'19_Ttrepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax19, coarsest_grid, upper_boundTtr, lower_boundTtr, sum_mlTtr), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()               
    
    # Plot 20: Normalized vibrational temperature, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax20 = plt.gca()  # Get the current axis
    ax20.fill_between(coarsest_grid, upper_boundTve, lower_boundTve, color=shaded_color, alpha=0.7, edgecolor='none')
    ax20.plot(coarsest_grid, sum_mlTve)
    ax20.set_title(r'Tve with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax20.figure.savefig(f'20_Tveepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'20_Tveepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax20, coarsest_grid, upper_boundTve, lower_boundTve, sum_mlTve), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()           
    
    # Plot 21: Normalized mach, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax21 = plt.gca()  # Get the current axis
    ax21.fill_between(coarsest_grid, upper_boundM, lower_boundM, color=shaded_color, alpha=0.7, edgecolor='none')
    ax21.plot(coarsest_grid, sum_mlM)
    ax21.set_title(r'M with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax21.figure.savefig(f'21_Mepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'21_Mepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax21, coarsest_grid, upper_boundM, lower_boundM, sum_mlM), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()       
    
    return P, Nl, Cl
