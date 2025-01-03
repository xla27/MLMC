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
    cellsom1N = [None] * (L+1); cellsom2N = [None] * (L+1);
    cellsom1O = [None] * (L+1); cellsom2O = [None] * (L+1);       
    cellsom1NO = [None] * (L+1); cellsom2NO = [None] * (L+1);
    cellsom1N2 = [None] * (L+1); cellsom2N2 = [None] * (L+1);
    cellsom1O2 = [None] * (L+1); cellsom2O2 = [None] * (L+1);      
    cellsom1Ttr = [None] * (L+1); cellsom2Ttr = [None] * (L+1);
    cellsom1Tve = [None] * (L+1); cellsom2Tve = [None] * (L+1);
    cellsom1M = [None] * (L+1); cellsom2M = [None] * (L+1);
    cellnodes = [None] * (L+1);
            
    while np.sum(dNl) > 0:
       # Update sample sums

        for l in range(L+1):
            if dNl[l] > 0:
                x2, sums1, sums2, _, _, sums1N, sums2N, sums1O, sums2O, sums1NO, sums2NO, sums1N2, sums2N2, sums1O2, sums2O2, sums1Ttr, sums2Ttr, sums1Tve, sums2Tve, sums1M, sums2M, cost = mlmc_l(l, int(dNl[l]), *args)
                Nl[l] += dNl[l]
                costl[l] += cost
                if cellsom1P[l] is None:
                    cellsom1P[l] = sums1; cellsom2P[l] = sums2;
                    cellsom1N[l] = sums1N; cellsom2N[l] = sums2N;
                    cellsom1O[l] = sums1O; cellsom2O[l] = sums2O;
                    cellsom1NO[l] = sums1NO; cellsom2NO[l] = sums2NO;
                    cellsom1N2[l] = sums1N2; cellsom2N2[l] = sums2N2;
                    cellsom1O2[l] = sums1O2; cellsom2O2[l] = sums2O2;
                    cellsom1Ttr[l] = sums1Ttr; cellsom2Ttr[l] = sums2Ttr;
                    cellsom1Tve[l] = sums1Tve; cellsom2Tve[l] = sums2Tve;
                    cellsom1M[l] = sums1M; cellsom2M[l] = sums2M;
                    
                else:
                    cellsom1P[l] += sums1; cellsom2P[l] += sums2;
                    cellsom1N[l] += sums1N; cellsom2N[l] += sums2N;
                    cellsom1O[l] += sums1O; cellsom2O[l] += sums2O;
                    cellsom1NO[l] += sums1NO; cellsom2NO[l] += sums2NO;
                    cellsom1N2[l] += sums1N2; cellsom2N2[l] += sums2N2;
                    cellsom1O2[l] += sums1O2; cellsom2O2[l] += sums2O2;
                    cellsom1Ttr[l] += sums1Ttr; cellsom2Ttr[l] += sums2Ttr;
                    cellsom1Tve[l] += sums1Tve; cellsom2Tve[l] += sums2Tve;
                    cellsom1M[l] += sums1M; cellsom2M[l] += sums2M;                    
                       
                cellnodes[l] = x2
        
        # Interpolation on coarsest grid
        coarsest_grid = cellnodes[0]

        interpolated_cellsom1P = []; interpolated_cellsom2P = [];
        interpolated_cellsom1N = []; interpolated_cellsom2N = [];
        interpolated_cellsom1O = []; interpolated_cellsom2O = [];
        interpolated_cellsom1NO = []; interpolated_cellsom2NO = [];
        interpolated_cellsom1N2 = []; interpolated_cellsom2N2 = [];
        interpolated_cellsom1O2 = []; interpolated_cellsom2O2 = [];
        interpolated_cellsom1Ttr = []; interpolated_cellsom2Ttr = [];
        interpolated_cellsom1Tve = []; interpolated_cellsom2Tve = [];
        interpolated_cellsom1M = []; interpolated_cellsom2M = [];

        for i in range(len(cellsom1P)):
            current_arrayCS1P = cellsom1P[i]
            interpolated_cellsom1P.append(interp1d(cellnodes[i], current_arrayCS1P, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2P = cellsom2P[i]
            interpolated_cellsom2P.append(interp1d(cellnodes[i], current_arrayCS2P, kind='linear', fill_value='extrapolate')(coarsest_grid))
            
            current_arrayCS1N = cellsom1N[i]
            interpolated_cellsom1N.append(interp1d(cellnodes[i], current_arrayCS1N, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2N = cellsom2N[i]
            interpolated_cellsom2N.append(interp1d(cellnodes[i], current_arrayCS2N, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1O = cellsom1O[i]
            interpolated_cellsom1O.append(interp1d(cellnodes[i], current_arrayCS1O, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2O = cellsom2O[i]
            interpolated_cellsom2O.append(interp1d(cellnodes[i], current_arrayCS2O, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1NO = cellsom1NO[i]
            interpolated_cellsom1NO.append(interp1d(cellnodes[i], current_arrayCS1NO, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2NO = cellsom2NO[i]
            interpolated_cellsom2NO.append(interp1d(cellnodes[i], current_arrayCS2NO, kind='linear', fill_value='extrapolate')(coarsest_grid))
                        
            current_arrayCS1N2 = cellsom1N2[i]
            interpolated_cellsom1N2.append(interp1d(cellnodes[i], current_arrayCS1N2, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2N2 = cellsom2N2[i]
            interpolated_cellsom2N2.append(interp1d(cellnodes[i], current_arrayCS2N2, kind='linear', fill_value='extrapolate')(coarsest_grid))
            
            current_arrayCS1O2 = cellsom1O2[i]
            interpolated_cellsom1O2.append(interp1d(cellnodes[i], current_arrayCS1O2, kind='linear', fill_value='extrapolate')(coarsest_grid))
            current_arrayCS2O2 = cellsom2O2[i]
            interpolated_cellsom2O2.append(interp1d(cellnodes[i], current_arrayCS2O2, kind='linear', fill_value='extrapolate')(coarsest_grid))
                       
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
        CS1N_matrix = np.transpose(np.array(interpolated_cellsom1N)); CS2N_matrix = np.transpose(np.array(interpolated_cellsom2N));
        CS1O_matrix = np.transpose(np.array(interpolated_cellsom1O)); CS2O_matrix = np.transpose(np.array(interpolated_cellsom2O));
        CS1NO_matrix = np.transpose(np.array(interpolated_cellsom1NO)); CS2NO_matrix = np.transpose(np.array(interpolated_cellsom2NO));
        CS1N2_matrix = np.transpose(np.array(interpolated_cellsom1N2)); CS2N2_matrix = np.transpose(np.array(interpolated_cellsom2N2));
        CS1O2_matrix = np.transpose(np.array(interpolated_cellsom1O2)); CS2O2_matrix = np.transpose(np.array(interpolated_cellsom2O2));
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
        mlN_vec = CS1N_matrix / Nl
        VlN_vec = CS2N_matrix / Nl - mlN_vec**2
        VlN_vec[VlN_vec < 0] = 0

        mlO_vec = CS1O_matrix / Nl
        VlO_vec = CS2O_matrix / Nl - mlO_vec**2
        VlO_vec[VlO_vec < 0] = 0

        mlNO_vec = CS1NO_matrix / Nl
        VlNO_vec = CS2NO_matrix / Nl - mlNO_vec**2
        VlNO_vec[VlNO_vec < 0] = 0

        mlN2_vec = CS1N2_matrix / Nl
        VlN2_vec = CS2N2_matrix / Nl - mlN2_vec**2
        VlN2_vec[VlN2_vec < 0] = 0

        mlO2_vec = CS1O2_matrix / Nl
        VlO2_vec = CS2O2_matrix / Nl - mlO2_vec**2
        VlO2_vec[VlO2_vec < 0] = 0

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
                    cellsom1N.append(None); cellsom2N.append(None);
                    cellsom1O.append(None); cellsom2O.append(None);
                    cellsom1NO.append(None); cellsom2NO.append(None);
                    cellsom1N2.append(None); cellsom2N2.append(None);
                    cellsom1O2.append(None); cellsom2O2.append(None);
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
    
    sum_mlN = np.sum(mlN_vec, axis=1)
    sum_VlN = np.sum(VlN_vec, axis=1)
    upper_boundN = sum_mlN + np.sqrt(np.abs(sum_VlN))
    lower_boundN = sum_mlN - np.sqrt(np.abs(sum_VlN))    

    sum_mlO = np.sum(mlO_vec, axis=1)
    sum_VlO = np.sum(VlO_vec, axis=1)
    upper_boundO = sum_mlO + np.sqrt(np.abs(sum_VlO))
    lower_boundO = sum_mlO - np.sqrt(np.abs(sum_VlO))    

    sum_mlNO = np.sum(mlNO_vec, axis=1)
    sum_VlNO = np.sum(VlNO_vec, axis=1)
    upper_boundNO = sum_mlNO + np.sqrt(np.abs(sum_VlNO))
    lower_boundNO = sum_mlNO - np.sqrt(np.abs(sum_VlNO))    

    sum_mlN2 = np.sum(mlN2_vec, axis=1)
    sum_VlN2 = np.sum(VlN2_vec, axis=1)
    upper_boundN2 = sum_mlN2 + np.sqrt(np.abs(sum_VlN2))
    lower_boundN2 = sum_mlN2 - np.sqrt(np.abs(sum_VlN2))    

    sum_mlO2 = np.sum(mlO2_vec, axis=1)
    sum_VlO2 = np.sum(VlO2_vec, axis=1)
    upper_boundO2 = sum_mlO2 + np.sqrt(np.abs(sum_VlO2))
    lower_boundO2 = sum_mlO2 - np.sqrt(np.abs(sum_VlO2))    

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
    
    # Plot 2: N gas compostion, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax2 = plt.gca()  # Get the current axis
    ax2.fill_between(coarsest_grid, upper_boundN, lower_boundN, color=shaded_color, alpha=0.7, edgecolor='none')
    ax2.plot(coarsest_grid, sum_mlN)
    ax2.set_title(r'N with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax2.figure.savefig(f'2_Nepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'2_Nepsilon_{eps_str}.pkl'
       
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundN, lower_boundN, sum_mlN), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()    
    
    # Plot 3: O gas compostion, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax3 = plt.gca()  # Get the current axis
    ax3.fill_between(coarsest_grid, upper_boundO, lower_boundO, color=shaded_color, alpha=0.7, edgecolor='none')
    ax3.plot(coarsest_grid, sum_mlO)
    ax3.set_title(r'O with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax3.figure.savefig(f'3_Oepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'3_Oepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundO, lower_boundO, sum_mlO), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()    
    
    # Plot 4: NO gas compostion, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax4 = plt.gca()  # Get the current axis
    ax4.fill_between(coarsest_grid, upper_boundNO, lower_boundNO, color=shaded_color, alpha=0.7, edgecolor='none')
    ax4.plot(coarsest_grid, sum_mlNO)
    ax4.set_title(r'NO with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax4.figure.savefig(f'4_NOepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'4_NOepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundNO, lower_boundNO, sum_mlNO), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()      
    
    # Plot 5: N2 gas compostion, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax5 = plt.gca()  # Get the current axis
    ax5.fill_between(coarsest_grid, upper_boundN2, lower_boundN2, color=shaded_color, alpha=0.7, edgecolor='none')
    ax5.plot(coarsest_grid, sum_mlN2)
    ax5.set_title(r'N2 with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax5.figure.savefig(f'5_N2epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'5_N2epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundN2, lower_boundN2, sum_mlN2), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()      

    # Plot 6: O2 gas compostion, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax6 = plt.gca()  # Get the current axis
    ax6.fill_between(coarsest_grid, upper_boundO2, lower_boundO2, color=shaded_color, alpha=0.7, edgecolor='none')
    ax6.plot(coarsest_grid, sum_mlO2)
    ax6.set_title(r'O2 with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax6.figure.savefig(f'6_O2epsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'6_O2epsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundO2, lower_boundO2, sum_mlO2), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()      

    # Plot 7: Normalized rototranslational temperature, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax7 = plt.gca()  # Get the current axis
    ax7.fill_between(coarsest_grid, upper_boundTtr, lower_boundTtr, color=shaded_color, alpha=0.7, edgecolor='none')
    ax7.plot(coarsest_grid, sum_mlTtr)
    ax7.set_title(r'Ttr with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax7.figure.savefig(f'7_Ttrepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'7_Ttrepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundTtr, lower_boundTtr, sum_mlTtr), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()               
    
    # Plot 8: Normalized vibrational temperature, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax8 = plt.gca()  # Get the current axis
    ax8.fill_between(coarsest_grid, upper_boundTve, lower_boundTve, color=shaded_color, alpha=0.7, edgecolor='none')
    ax8.plot(coarsest_grid, sum_mlTve)
    ax8.set_title(r'Tve with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax8.figure.savefig(f'8_Tveepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'8_Tveepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundTve, lower_boundTve, sum_mlTve), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()           
    
    # Plot 9: Normalized mach, 1sigma uncertainty region
    shaded_color = [0.6, 0.8, 1]
    ax9 = plt.gca()  # Get the current axis
    ax9.fill_between(coarsest_grid, upper_boundM, lower_boundM, color=shaded_color, alpha=0.7, edgecolor='none')
    ax9.plot(coarsest_grid, sum_mlM)
    ax9.set_title(r'M with $\varepsilon_r$ = {:.3f}'.format(eps), fontsize=14)

    eps_str = str(eps).replace('.', '_')  
    ax9.figure.savefig(f'9_Mepsilon_{eps_str}.svg', format='svg', dpi=1200)
    plt.tight_layout()
    plot_filename = f'9_Mepsilon_{eps_str}.pkl'
        
    with open(plot_filename, 'wb') as f:
    	pickle.dump((ax, coarsest_grid, upper_boundM, lower_boundM, sum_mlM), f)

    f.close()
    plt.close("all")
    plt.cla()
    plt.clf()       
    
    return P, Nl, Cl