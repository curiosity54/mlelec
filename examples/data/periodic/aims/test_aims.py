# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +


import metatensor
import numpy as np
import torch
from ase.io import read
from ase.visualize import view
from metatensor import Labels, TensorBlock, TensorMap, save

from mlelec.data.dataset import QMDataset
from mlelec.features.acdc import *
from mlelec.features.acdc_utils import *

# -
from mlelec.models.linear import LinearModelPeriodic
from mlelec.utils.plot_utils import plot_hamiltonian
from mlelec.utils.twocenter_utils import fix_orbital_order

device='cpu'
filename = "C2_rotated"
frames = read('graphene174_fhiaims/single_graphene_sheet_vac.extxyz', ':') # will be automated from filename  
for f in frames:
    f.pbc = [True, True, True]
kfock = np.load('H_k.npz'.format(filename), allow_pickle=True)
kfock= np.array([kfock['arr_{}'.format(i)] for i in range(len(kfock))])
kmesh = [12,12,1]

orbitals = {'sto-3g': {6: [[1,0,0],[2,0,0],[2,1,1], [2,1,-1],[2,1,0]]}, 
            'def2svp': {6: [[1,0,0],[2,0,0],[3,0,0],[2,1,1], [2,1,-1],[2,1,0], [3,1,1], [3,1,-1],[3,1,0], [3,2,-2], [3,2,-1],[3,2,0], [3,2,1],[3,2,2]]}
           }
ORBS = 'sto-3g'


# +
for f in frames:
    f.pbc = [True, True, True]


dataset = QMDataset(frames = frames[:5], kgrid=kmesh, matrices_kpoint = kfock[:5], target=["real_translation"] ,device = "cpu", orbs = orbitals[ORBS], orbs_name = ORBS) 


from mlelec.utils.pbc_utils import matrix_to_blocks

# +
from mlelec.utils.twocenter_utils import _to_coupled_basis

# from mlelec.utils.twocenter_utils import _to_blocks  

def get_targets(dataset, device ="cpu"):
    blocks = matrix_to_blocks(dataset)
    coupled_blocks = _to_coupled_basis(blocks, skip_symmetry=True, device= device, translations=True)
    

    blocks = blocks.keys_to_samples('cell_shift_a')
    blocks = blocks.keys_to_samples('cell_shift_b')
    blocks = blocks.keys_to_samples('cell_shift_c')

    coupled_blocks = coupled_blocks.keys_to_samples('cell_shift_a')
    coupled_blocks = coupled_blocks.keys_to_samples('cell_shift_b')
    coupled_blocks = coupled_blocks.keys_to_samples('cell_shift_c')
    return blocks , coupled_blocks


# +
hypers_pair = {'cutoff': 4,
          'max_radial':10, 
          'max_angular':4,
          'atomic_gaussian_width':0.6,
          'center_atom_weight':1,
          "radial_basis": {"Gto": {}},
          "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
}

hypers_atom = {'cutoff': 4,
          'max_radial':10, 
          'max_angular':4,
          'atomic_gaussian_width':0.3,
          'center_atom_weight':1,
          "radial_basis": {"Gto": {}},
          "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
}
device = "cpu"
# -

LCUT = 3 #2*np.max([np.max( np.asarray(orbitals[ORBS][k])[:,1]) for k in orbitals[ORBS]])
print("LCUT", LCUT)
both_centers = True
rhoij = pair_features(dataset.structures, hypers_atom, hypers_pair, order_nu = 1, all_pairs = True, both_centers=both_centers,
                      max_shift = dataset.kmesh[0] ,  desired_shifts = dataset.desired_shifts_sup, mic=True, 
                      kmesh = dataset.kmesh[0], device="cpu", lcut = LCUT)


save('rhoij', rhoij)

if both_centers: 
    NU = 3
else: 
    NU = 2
rhonui = single_center_features(dataset.structures, hypers_atom, order_nu=NU, lcut=LCUT, device = device,
                                feature_names = rhoij.property_names) 


save('rhonui', rhonui)

hfeat_tc= twocenter_features_periodic_NH(single_center=rhonui, pair= rhoij) 

save('hfeat', hfeat_tc)


def train_ridge(model, target_blocks, set_bias=False):
    block_losses = {}
    loss = 0
    pred, ridges, kernels = model.fit_ridge_analytical(return_matrix = False, set_bias = set_bias)

    for (key, block) in pred.items():
        block_loss=torch.norm(block.values - target_blocks[key].values)**2
        loss += block_loss
        
        block_losses[tuple(key.values)] = block_loss

    # print(np.sum(list(block_losses.values())))
    return loss, pred, ridges, block_losses#, kernels 


def train_linear(model, target_blocks, nepochs, optimizer= None, log_interval =1):

    losses = []
    for epoch in range(nepochs):
        optimizer.zero_grad()
        pred = model(return_matrix = False)


        loss = 0
        for s in pred:
            for (key, block) in pred[s].items():
                loss+=torch.sum(block.values - target_blocks[s][key].values)**2
        losses.append(loss.item())
        if optimizer is None: 
            print(loss)
            return losses, pred, model
        
        loss.backward()
        optimizer.step()
        if epoch % log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))
            

    return losses, pred, model, optimizer 

model_ridge = LinearModelPeriodic(twocfeat=hfeat_tc, target_blocks=target_coupled_blocks, frames = dataset.structures, orbitals= dataset.basis, cell_shifts=dataset.desired_shifts[:], device = device)

loss_ridge_bias, pred_ridge_bias, ridges_bias, loss_blocks = train_ridge(model_ridge, target_coupled_blocks, set_bias=True)

np.save('ridges_bias.npy', ridges_bias)

print(loss_ridge_bias)

# +
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 500
x=[','.join([str(lbl[i]) for i in [0,2,3,5,6,7]]) for lbl in target_coupled_blocks.keys.values.tolist()]
fs = plt.rcParams['figure.figsize']
fig, ax = plt.subplots(figsize = (fs[0]*5, fs[1]))
ax_loss = ax.twinx()
# s = (0,0,0)
prediction_ = np.array([torch.linalg.norm(b.values) for b in pred_ridge_bias])
target_ = np.array([torch.linalg.norm(b.values) for b in target_coupled_blocks)
loss_ = np.array([torch.linalg.norm(b.values-b1.values)**2 for b,b1 in zip(target_coupled_blocks,pred_ridge_bias)])
print(np.sum(loss_))
# loss_ = np.array(list(loss_blocks.values()))
#MASKING LOSS
# mask = loss_ < 1e-6
# loss_[mask] = 0
# ax.bar(range(len(loss_blocks.keys())),list(loss_blocks.values()));

x_ = 3.5*np.arange(len(loss_blocks))

labels = []
handles = []
pl = ax.bar(x_, prediction_, label = 'pred', width = 1, color = 'tab:blue');
handles.append(pl)
labels.append('Prediction')
pl = ax.bar(x_+1, target_, alpha = 1, label = 'target', width = 1, color = 'tab:orange');
handles.append(pl)
labels.append('Target')

pl = ax_loss.bar(x_+2, loss_, alpha = 1, label = 'target', width = 1, color = 'tab:red');
handles.append(pl)
labels.append('Loss')

ax.set_ylim(1e-7, 1000)
ax.set_xticks(3.5*np.arange(len(loss_blocks))+3.5/3-0.5)
ax.set_xticklabels(x, rotation=90);
ax.legend(handles, labels, loc = 'best')
ax.set_ylabel('|H|')
ax_loss.set_ylabel('Loss')
ax_loss.set_yscale('log')
ax_loss.set_ylim(1e-10)
ax.set_yscale('log')
fig.savefig('block-errrors-c2-5.pdf')
# -







# +
# kpoints = np.load('kpoints.npz'.format(filename), allow_pickle=True) 
# kpoints= np.array([kpoints['arr_{}'.format(i)] for i in range(len(kpoints))])
# PYSCF 
# import pyscf.pbc.gto as pbcgto
# import pyscf.pbc.tools.pyscf_ase as pyscf_ase
# from pyscf.pbc.tools.k2gamma import get_phase, kpts_to_kmesh, k2gamma
# for ifr, frame in enumerate(frames[:1]):
#         print('frame', ifr)

#         cell = pbcgto.Cell()
#         cell.atom = pyscf_ase.ase_atoms_to_pyscf(frame)

#         cell.basis = 'sto-3g' #'def2-svp'
#         cell.a = frame.cell
#         cell.verbose = 3
#         print('-------------------------------------------------------------------------------', flush= True)
#         #print(cell.symmetry, flush = True)
#         #cell.symmetry = True
#         #print(cell.symmetry, flush = True)
#         cell.build()

#         kmesh = [12,12,1]
   
#         kpts = cell.make_kpts(kmesh)
#         #kmesh = [np.int32(np.max([1, np.ceil(2*np.pi*np.linalg.norm(vec)/kspacing)])) for vec in frame.cell.reciprocal().array]

#         kpts_1 = cell.make_kpts(kmesh)

# from pyscf.pbc.lib.kpts import KPoints
# kpts = KPoints(cell, kpts_1)
# kpts.build(space_group_symmetry=False)
# print(kpts.kpts_scaled)
# #(kmesh)
# assert np.isclose(kpts.kpts_scaled,  kpoints[0], atol=1e-5).all()
