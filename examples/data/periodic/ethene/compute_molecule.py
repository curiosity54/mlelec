import hickle
from ase.io import read
import pyscf.gto as gto
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
#from pyscf.pbc.tools.k2gamma import get_phase, kpts_to_kmesh
import numpy as np 

import pyscf.scf as scf
frames = read("ethene.xyz", ":")

for ifr, frame in enumerate(frames[:]):
    print(ifr)
    mol = gto.Mole()
    mol.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
    
    mol.basis = "def2svp" #'sto-3g'
    mol.verbose = 5
    mol.build()
    
    print(mol.ao_labels())
    
    print("Transform k-point integrals to supermol integral")
    #smol, phase = get_phase(cell, kpts)
    #NR, Nk = phase.shape
    nao = mol.nao
    #smol.basis = "sto-3g"
    #smol.verbose = 5
    #smol.build()
    mf = scf.RHF(mol) 
    
    #mf = mf.density_fit()
    mf.max_cycle = 500
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-8
    
    mf.kernel()
    print("converged:", mf.converged)
    if not mf.converged:
        raise RuntimeError("SCF calculation not converged")
    
    fock = mf.get_fock()
    over = mf.get_ovlp()
    np.save('fock_ethene.npy', fock) 
    np.save('over_ethene.npy', over) 
