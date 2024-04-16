import numpy as np
import os
import hickle

# Initialize dictionaries for each item
data = {
    'fock': [],
    'ovlp': [],
    'dm': [],
    'pop': [],
    'chg': [],
    'ao_labels': [],
    'dip_moment': [],
    'energy_elec': [],
    'energy_tot': [],
    'converged': [],
    'polarisability': [],
    }

# Iterate through each npz file
npz_files = sorted([file for file in os.listdir() if file.endswith('.npz')])

for file in npz_files:
    print(f"Processing file: {file}")
    # Load the npz file
    npz_data = np.load(file)
    
    # Append items from the npz file to respective arrays
    data['fock'].append(npz_data['fock'])
    data['ovlp'].append(npz_data['ovlp'])
    data['dm'].append(npz_data['dm'])
    data['pop'].append(npz_data['pop'])
    data['chg'].append(npz_data['chg'])
    data['ao_labels'].append(npz_data['ao_labels'])
    data['dip_moment'].append(npz_data['dip_moment'])
    data['energy_elec'].append(npz_data['energy_elec'])
    data['energy_tot'].append(npz_data['energy_tot'])
    data['converged'].append(npz_data['converged'])
    #data['polarisability'].append(npz_data['polarisability'])

# Save the combined data as a new npz file

hickle.dump(np.array(data['fock']), "fock.hkl")
hickle.dump(np.array(data['ovlp']), "ovlp.hkl")
hickle.dump(np.array(data['dm']), "dm.hkl")
hickle.dump(np.array(data['pop']), "pop.hkl")
hickle.dump(np.array(data['chg']), "chg.hkl")
hickle.dump(np.array(data['ao_labels']), "orbs.hkl")
hickle.dump(np.array(data['dip_moment']), "dipole_moment.hkl")
hickle.dump(np.array(data['energy_elec']), "energy_elec.hkl")
hickle.dump(np.array(data['energy_tot']), "energy_tot.hkl")
#hickle.dump(np.array(data['polarisability']), "polarisability.hkl")

