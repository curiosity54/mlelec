import numpy as np
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.tools.pyscf_ase as pyscf_ase
from ase.atoms import Atoms
from ase.io import write
from pyscf.pbc import scf

a = 1.42
graphene = Atoms(
    symbols=["C", "C"],
    cell=a / 2 * np.array([[3, -np.sqrt(3), 0], [3, np.sqrt(3), 0], [0, 0, 20]]),
    scaled_positions=[[1 / 3, 1 / 3, 0], [2 / 3, 2 / 3, 0]],
    pbc=[True, True, False],
)

###########################################################################

E2g_graphene = []
for distortion in np.linspace(0, 0.05, 50, endpoint=True):
    g = graphene.copy()
    p1 = g.positions[0]
    p2 = g.positions[1]
    d = g.get_distance(0, 1, mic=True)
    g.set_positions(
        [p1 + np.array([distortion * d, 0, 0]), p2 - np.array([distortion * d, 0, 0])]
    )
    E2g_graphene.append(g)


for ifr, frame in enumerate(E2g_graphene):
    cell = pbcgto.Cell()
    cell.atom = pyscf_ase.ase_atoms_to_pyscf(frame)
    cell.basis = "sto-3g"
    cell.a = frame.cell
    cell.verbose = 3
    cell.precision = 1e-8
    cell.dimension = 2
    cell.build()

    kpt = 8
    kmesh = [kpt, kpt, 1]
    print("KMESH", kmesh)
    kpts = cell.make_kpts(kmesh)

    print("Transform k-point integrals to supercell integral")
    nao = cell.nao
    kmf = scf.KRHF(cell, kpts=cell.make_kpts(kmesh))
    # kmf = scf.addons.smearing_(kmf, sigma=0.01, method='fermi').run()
    kmf = kmf.density_fit().run()
    kmf.max_cycle = 500
    kmf.conv_tol = 1e-8  # ??
    kmf.conv_tol_grad = 1e-8

    e = kmf.kernel()
    print("converged:", kmf.converged)
    if not kmf.converged:
        raise RuntimeError("SCF calculation not converged")

    fock = kmf.get_fock()
    over = kmf.get_ovlp()
    np.save(f"E2g_fock_{ifr}.npy", fock)
    np.save(f"E2g_over_{ifr}.npy", over)

    frame.info["energy"] = e

write("E2g_graphene.xyz", E2g_graphene, write_info=True)
