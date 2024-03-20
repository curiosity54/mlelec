from ase.atoms import Atoms
import hickle
from ase.io import read
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from pyscf.pbc.tools.k2gamma import get_phase, kpts_to_kmesh
import numpy as np 
import pyscf.pbc.dft as pbcdft
from pyscf.pbc import scf
from ase.dft.kpoints import sc_special_points as special_points, get_bandpath

a=1.42
frame=Atoms(symbols=['C','C'], cell=a/2*np.array([[3,-np.sqrt(3),0],[3,np.sqrt(3),0],[0,0,100]]), 
            scaled_positions=[[1/3,1/3,0],[2/3,2/3,0]], pbc=[True,True,False])

###########################################################################

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
cell.basis = 'sto-3g'
cell.a = frame.cell
cell.verbose = 3
cell.precision=1e-8
#cell.dimension=2
cell.build()

kpt=12
kmesh = [kpt,kpt,1]
print('KMESH', kmesh)
kpts = cell.make_kpts(kmesh)

print("Transform k-point integrals to supercell integral")
nao = cell.nao
kmf = scf.KRHF(cell, kpts = cell.make_kpts(kmesh))
#kmf = scf.addons.smearing_(kmf, sigma=0.01, method='fermi').run()
kmf = kmf.density_fit()
kmf.max_cycle = 500
kmf.conv_tol = 1e-8 # ?? 
kmf.conv_tol_grad = 1e-8

kmf.kernel()
print("converged:", kmf.converged)
if not kmf.converged:
    raise RuntimeError("SCF calculation not converged")

fock = kmf.get_fock()
over = kmf.get_ovlp()
np.save(f"fock_121212_3d_norun.npy", fock)
np.save(f"over_121212_3d_norun.npy", over) 

###########################################################################

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
cell.basis = 'sto-3g'
cell.a = frame.cell
cell.verbose = 3
cell.precision = 1e-8
cell.build()

kpt = 12
kmesh = [kpt,kpt,1]
print('KMESH', kmesh)
kpts = cell.make_kpts(kmesh)

print("Transform k-point integrals to supercell integral")
nao = cell.nao
kmf = scf.KRHF(cell, kpts = cell.make_kpts(kmesh))
#kmf = scf.addons.smearing_(kmf, sigma=0.01, method='fermi').run()
kmf = kmf.density_fit().run()
kmf.max_cycle = 500
kmf.conv_tol = 1e-8 # ?? 
kmf.conv_tol_grad = 1e-8

kmf.kernel()
print("converged:", kmf.converged)
if not kmf.converged:
    raise RuntimeError("SCF calculation not converged")

fock = kmf.get_fock()
over = kmf.get_ovlp()
np.save(f"fock_121212_3d_run.npy", fock)
np.save(f"over_121212_3d_run.npy", over) 

###########################################################################

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
cell.basis = 'sto-3g'
cell.a = frame.cell
cell.verbose = 3
cell.precision=1e-8
cell.dimension=2
cell.build()

kpt=12
kmesh = [kpt,kpt,1]
print('KMESH', kmesh)
kpts = cell.make_kpts(kmesh)

print("Transform k-point integrals to supercell integral")
nao = cell.nao
kmf = scf.KRHF(cell, kpts = cell.make_kpts(kmesh))
#kmf = scf.addons.smearing_(kmf, sigma=0.01, method='fermi').run()
kmf = kmf.density_fit() #.run()
kmf.max_cycle = 500
kmf.conv_tol = 1e-8 # ?? 
kmf.conv_tol_grad = 1e-8

kmf.kernel()
print("converged:", kmf.converged)
if not kmf.converged:
    raise RuntimeError("SCF calculation not converged")

fock = kmf.get_fock()
over = kmf.get_ovlp()
np.save(f"fock_121212_2d_norun.npy", fock)
np.save(f"over_121212_2d_norun.npy", over) 

