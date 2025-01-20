import os, sys

def main(nproc, workingFolder):

    # folder for with MLMC code
    baseFolder = '/home/xla27/FAR-ESA/MLMC'

    if testcase == 'AIR5':

        baseFolder = baseFolder + '/AIR5'

        from AIR5 import anova, anova_plot

    varargin = (nproc, baseFolder, workingFolder) 

    # Set up the problem for SALib
    problem = {
        'num_vars': 4,
        'names': ['M', 'T','P','beta'],
        'bounds': [[8.0, 9.5], [850, 1050], [300, 600], [0.76, 0.8]]
    }

    # Generate samples using Saltelli's sampling
    Nl  = [2, 2, 2]               # number of samples per level
    lev = [i for i in range(len(Nl))]
    d   = problem['num_vars']

    # Computing the number of required samples per level
    sample_sizes = [(d+2) * N for N in Nl]

    # performing the analysis of variance
    x_vec0, total_S = anova(problem, lev, sample_sizes, *varargin)

    # plotting results
    anova_plot(x_vec0, total_S)

    return

if __name__ == '__main__':

    nproc = int(sys.argv[1])    # number of processors for CFD
    testcase = sys.argv[2]
    workingFolder = os.getcwd() # working directory for data saving

    main(nproc, testcase, workingFolder)
