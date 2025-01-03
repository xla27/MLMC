import os
import shutil
import re
import subprocess
import pandas as pd

def cfd_call_fine(valIns_M, valIns_T, valIns_P, valIns_Bn2, valIns_Bo2, l, i):
    baseFolder = '/global-scratch/bulk_pool/nsarman/mlmc_dw11AMG/'
    stringPart = 'level'
    stringPart2 = '/Template_prova_fine'
    
    lValue = l 

    sourceFolder = os.path.join(baseFolder, f"{stringPart}{lValue}{stringPart2}")
    destinationFolder = os.path.join(baseFolder, f"{stringPart}{lValue}")

    # Copy template folder
    destinationSubfolder = os.path.join(destinationFolder, f"N{i}") # nome cartella + iterazione i-esima
    if os.path.exists(destinationSubfolder):
            shutil.rmtree(destinationSubfolder) # Remove existing folder
    shutil.copytree(sourceFolder, destinationSubfolder) # copia la cartella sorgente nella cartella di destinazione
    filePath = os.path.join(destinationFolder, f"N{i}", 'config.cfg') # percorso del file .cfg da modificare

    if os.path.isfile(filePath): # controlla se il file esiste
        # Modifica il file .cfg e inserisci i valori perturbati
        with open(filePath, 'r') as file:
            fileContents = file.read()

        fileContents = re.sub(r'MACH_NUMBER= 9.0', f'MACH_NUMBER= {valIns_M}', fileContents)
        fileContents = re.sub(r'FREESTREAM_TEMPERATURE= 1000.0', f'FREESTREAM_TEMPERATURE= {valIns_T}', fileContents)
        fileContents = re.sub(r'FREESTREAM_TEMPERATURE_VE= 1000.0', f'FREESTREAM_TEMPERATURE_VE= {valIns_T}', fileContents)
        fileContents = re.sub(r'FREESTREAM_PRESSURE= 390.0', f'FREESTREAM_PRESSURE= {valIns_P}', fileContents)
        fileContents = re.sub(r'GAS_COMPOSITION= \((\d+\.\d+, ){10}\d+\.\d+\)', 
    f'GAS_COMPOSITION= (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {valIns_Bn2}, {valIns_Bo2})', 
    fileContents)


        with open(filePath, 'w') as file:
            file.write(fileContents)

        # Lancio SU2
        os.chdir(os.path.join(destinationFolder, f"N{i}"))
        su2Command = 'mpirun -n 16 SU2_CFD config.cfg'
        result = subprocess.run(su2Command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output = result.stdout
        error = result.stderr

        # Read file to get qois
        csv_path_surf = os.path.join(destinationFolder, f"N{i}", 'adap', f"ite{l}", 'surface_flow.csv')
        if os.path.isfile(csv_path_surf):
            data_surf = pd.read_csv(csv_path_surf, header=None)
            data_surf = data_surf.apply(pd.to_numeric, errors='coerce').dropna() # any non-numeric values are coerced into NaN, and then rows with any NaN values are dropped
            data_surf.columns = [f'Var{j}' for j in range(1, len(data_surf.columns) + 1)] # renames the columns of the DataFrame to 'Var1', 'Var2', 'Var3', ..., 'VarN'
            data_surf = data_surf.sort_values(by='Var2')
            
            # Save number density of charged gas species
            nd_elecMinus = (data_surf['Var4'] / 9.10938E-31).tolist()
            nd_nPlus = (data_surf['Var5'] / 2.33E-26).tolist()
            nd_oPlus = (data_surf['Var6'] / 2.66E-26).tolist()
            nd_noPlus = (data_surf['Var7'] / 4.98E-26).tolist()
            nd_n2Plus = (data_surf['Var8'] / 4.65E-26).tolist()
            nd_o2Plus = (data_surf['Var9'] / 5.31E-26).tolist()
            
            # Save wall flow quantities
            p_i = (data_surf['Var19'] / 390).tolist() # pressure normalized w.r.t nominal pressure
            Ttr_i = (data_surf['Var20'] / 1000).tolist() # rototranslational temperature normalized w.r.t nominal rototranslational temperature
            Tve_i = (data_surf['Var21'] / 1000).tolist() # vibrational temperature normalized w.r.t nominal vibrational temperature
            M_i = (data_surf['Var24'] / 9).tolist() # mach normalized w.r.t nominal mach
       
            # Save grid
            xnodesf = data_surf['Var2'].tolist()

            os.chdir(baseFolder)  # Return to the starting directory
            return nd_elecMinus, nd_nPlus, nd_oPlus, nd_noPlus, nd_n2Plus, nd_o2Plus, p_i, Ttr_i, Tve_i, M_i, xnodesf
    return None, None
