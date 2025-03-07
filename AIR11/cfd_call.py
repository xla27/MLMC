import os
import shutil
import re
import subprocess
import csv
import numpy as np

fieldnames = ["PointID","x","y","Density_0","Density_1","Density_2","Density_3","Density_4",
              "Density_5","Density_6","Density_7","Density_8","Density_9","Density_10",
              "Momentum_x","Momentum_y","Energy","Energy_ve","MassFrac_0","MassFrac_1",
              "MassFrac_2","MassFrac_3","MassFrac_4","MassFrac_5","MassFrac_6","MassFrac_7",
              "MassFrac_8","MassFrac_9","MassFrac_10","Pressure","Temperature_tr",
              "Temperature_ve","Velocity_x","Velocity_y","Mach","Pressure_Coefficient"]


def cfd_call(type, valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i, *args):
    
    (nproc, baseFolder, workingFolder) = args

    stringPart = 'level'

    if type == 'COARSE':
        stringPart2 = '/Template_prova_coarse'
        stringIter  = "Nc_" + str(i).zfill(3)
    elif type == 'FINE':
        stringPart2 = '/Template_prova_fine'
        stringIter  = "Nf_" + str(i).zfill(3)

       
    lValue = l 

    #sourceFolder = os.path.join(baseFolder, f"{stringPart}{lValue}{stringPart2}")
    meshFolder   = os.path.join(baseFolder, f"mesh/{stringPart}{lValue}{stringPart2}")
    destinationFolder = os.path.join(workingFolder, f"{stringPart}{lValue}")

    # Creating the Destination folder
    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)

    # Creating the Destination subfolder (or removing if already existent)
    destinationSubfolder = os.path.join(destinationFolder, stringIter) # nome cartella + iterazione i-esima
    if os.path.exists(destinationSubfolder):
        #print(os.path.exists(destinationSubfolder))
        shutil.rmtree(destinationSubfolder) # Remove existing folder
    os.mkdir(destinationSubfolder)
    shutil.copy(f'{baseFolder}/config.cfg', destinationSubfolder)
    filePath = os.path.join(destinationFolder, stringIter, 'config.cfg') # percorso del file .cfg da modificare

    if os.path.isfile(filePath): # controlla se il file esiste
        # Modifica il file .cfg e inserisci i valori perturbati
        with open(filePath, 'r') as file:
            fileContents = file.read()

        fileContents = re.sub(r'MACH_NUMBER= 9.0', f'MACH_NUMBER= {valIns_M}', fileContents)
        fileContents = re.sub(r'FREESTREAM_TEMPERATURE= 1000.0', f'FREESTREAM_TEMPERATURE= {valIns_T}', fileContents)
        fileContents = re.sub(r'FREESTREAM_TEMPERATURE_VE= 1000.0', f'FREESTREAM_TEMPERATURE_VE= {valIns_T}', fileContents)
        fileContents = re.sub(r'FREESTREAM_PRESSURE= 390.0', f'FREESTREAM_PRESSURE= {valIns_P}', fileContents)
        fileContents = re.sub(r'GAS_COMPOSITION= \((\d+\.\d+, ){10}\d+\.\d+\)', 
                            f'GAS_COMPOSITION= (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {valIns_Bn2}, {valIns_Bo2})', fileContents)
        
        fileContents = re.sub(r'MESH_FILENAME= ', f'MESH_FILENAME= {meshFolder}/mesh.su2', fileContents)

        with open(filePath, 'w') as file:
            file.write(fileContents)

        # Lancio SU2
        os.chdir(os.path.join(destinationFolder, stringIter))
        su2Command = 'mpirun -n ' + str(nproc) + ' SU2_CFD config.cfg > log.out'
        result = subprocess.run(su2Command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output = result.stdout
        error = result.stderr

        # Read file to get qois
        csv_path_surf = os.path.join(destinationFolder, stringIter, 'surface_flow.csv')
        if os.path.isfile(csv_path_surf):

            data_surf = csv2dict(csv_path_surf, fieldnames)

            # Save number density of charged gas species
            nd_elecMinus = (data_surf['Density_0'] / 9.10938E-31).tolist()
            nd_nPlus     = (data_surf['Density_1'] / 2.33E-26).tolist()
            nd_oPlus     = (data_surf['Density_2'] / 2.66E-26).tolist()
            nd_noPlus    = (data_surf['Density_3'] / 4.98E-26).tolist()
            nd_n2Plus    = (data_surf['Density_4'] / 4.65E-26).tolist()
            nd_o2Plus    = (data_surf['Density_5'] / 5.31E-26).tolist()
            
            # Save gas composition at the wall
            beta_elecMinus = data_surf['MassFrac_0'].tolist()
            beta_nPlus     = data_surf['MassFrac_1'].tolist()
            beta_oPlus     = data_surf['MassFrac_2'].tolist()
            beta_noPlus    = data_surf['MassFrac_3'].tolist()
            beta_n2Plus    = data_surf['MassFrac_4'].tolist()
            beta_o2Plus    = data_surf['MassFrac_5'].tolist()
            
            beta_n  = data_surf['MassFrac_6'].tolist()
            beta_o  = data_surf['MassFrac_7'].tolist()
            beta_no = data_surf['MassFrac_8'].tolist()
            beta_n2 = data_surf['MassFrac_9'].tolist()
            beta_o2 = data_surf['MassFrac_10'].tolist()
            
            # Save wall flow quantities
            p_i   = (data_surf['Pressure'] / 390).tolist() # pressure normalized w.r.t asymptotic pressure
            Ttr_i = (data_surf['Temperature_tr'] / 1000).tolist() # rototranslational temperature normalized w.r.t asymptotic rototranslational temperature
            Tve_i = (data_surf['Temperature_ve'] / 1000).tolist() # vibrational temperature normalized w.r.t asymptotic vibrational temperature
            M_i   = (data_surf['Mach'] / 9).tolist() # mach normalized w.r.t asymptotic mach
       
            # Save grid
            xnodesf = data_surf['x'].tolist()

            os.remove('history.csv')

            os.chdir(workingFolder)  # Return to the starting directory
            return nd_elecMinus, nd_nPlus, nd_oPlus, nd_noPlus, nd_n2Plus, nd_o2Plus, beta_elecMinus, beta_nPlus, beta_oPlus, beta_noPlus, beta_n2Plus, beta_o2Plus, beta_n, beta_o, beta_no, beta_n2, beta_o2, p_i, Ttr_i, Tve_i, M_i, xnodesf
        
    return None, None


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

    return data