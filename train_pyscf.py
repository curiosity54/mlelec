#!/usr/bin/env python
# coding: utf-8

from ase.io import read
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path

import torch 
torch.set_default_dtype(torch.float64) # <<< FIXME: a bug of torch requires it. Verbose workarounds are possible 

import metatensor 
from metatensor import Labels, TensorBlock, TensorMap, load, slice as mts_slice 

from mlelec.data.dataset import QMDataset
from mlelec.features.acdc import pair_features, single_center_features, twocenter_features_periodic_NH
from mlelec.models.linear import LinearModelPeriodic
from mlelec.metrics import L2_kspace_loss
from mlelec.utils.twocenter_utils import fix_orbital_order, _to_coupled_basis 
from mlelec.utils.pbc_utils import matrix_to_blocks, blocks_to_matrix
from mlelec.utils.symmetry import ClebschGordanReal
#============================================================================================================================================================
def get_targets(dataset, device ="cpu", cutoff = None, target='fock', all_pairs= True):
    if target.lower() == 'fock':
        matrices_negative = dataset._fock_realspace_negative_translations
    elif target.lower() == 'overlap':
        matrices_negative = dataset._overlap_realspace_negative_translations
    else: 
        raise ValueError('target must be fock or overlap')
    blocks = matrix_to_blocks(dataset, matrices_negative , device = device, cutoff = cutoff, all_pairs = all_pairs, target= target)
    coupled_blocks = _to_coupled_basis(blocks, skip_symmetry = True, device = device, translations = True)

    blocks = blocks.keys_to_samples('cell_shift_a')
    blocks = blocks.keys_to_samples('cell_shift_b')
    blocks = blocks.keys_to_samples('cell_shift_c')

    coupled_blocks = coupled_blocks.keys_to_samples('cell_shift_a')
    coupled_blocks = coupled_blocks.keys_to_samples('cell_shift_b')
    coupled_blocks = coupled_blocks.keys_to_samples('cell_shift_c')
    return blocks , coupled_blocks

def compute_pairfeatures(hypers_pair, hypers_atom, dataset, return_rho0ij = False, both_centers = False, all_pairs = False, LCUT = 3, device = 'cuda'):

    rhoij = pair_features(dataset.structures, 
                          hypers_atom, 
                          hypers_pair, 
                          order_nu = 1, 
                          all_pairs = all_pairs, 
                          both_centers = both_centers, 
                          mic = False,
                          kmesh = dataset.kmesh, 
                          device = device, 
                          lcut = LCUT, 
                          return_rho0ij = return_rho0ij, 
                          counter = dataset._translation_counter,
                          T_dict = dataset._translation_dict)

    if both_centers and not return_rho0ij:
        NU = 3
    else:
        NU = 2
    rhonui = single_center_features(dataset.structures, 
                                    hypers_atom, 
                                    order_nu = NU, 
                                    lcut = LCUT, 
                                    device = device,
                                    feature_names = rhoij.property_names)
    

    return rhonui, rhoij
    
#============================================================================================================================================================
# USER INPUT
device = 'cpu'

## dataset
ORBS = 'sto-3g'
START = 0
STOP = 1
kmesh = [8,8,1]
root = f'examples/data/periodic'
workdir = f"pyscf_{''.join([str(i) for i in kmesh])}_{START}-{STOP}"
Path(workdir).mkdir(exist_ok = True)
frames = read(f'{root}/c2/C2_174.extxyz', slice(START, STOP))

## features
cutoff = 6 # Angstrom
hypers_pair = {'cutoff': cutoff,
               'max_radial': 4,
               'max_angular': 4,
               'atomic_gaussian_width': 0.3,
               'center_atom_weight': 1,
               "radial_basis": {"Gto": {}},
               "cutoff_function": {"ShiftedCosine": {"width": 0.1}}}

hypers_atom = {'cutoff': 4,
               'max_radial': 10,
               'max_angular': 4,
               'atomic_gaussian_width': 0.3,
               'center_atom_weight': 1,
               "radial_basis": {"Gto": {}},
               "cutoff_function": {"ShiftedCosine": {"width": 0.1}}}

return_rho0ij = False
both_centers = False
all_pairs = False
LCUT = 3

## training
max_lr = 1e-2
scheduler_factor = 0.6
scheduler_patience = 50
nepoch = 10
seed = 1
np.random.seed(seed)
nhidden = 128
nlayers = 2
#============================================================================================================================================================
# DATASET
orbitals = {'sto-3g': {6: [[1,0,0],[2,0,0],[2,1,1], [2,1,-1],[2,1,0]]}, 
            'def2svp': {6: [[1,0,0],[2,0,0],[3,0,0],[2,1,1], [2,1,-1],[2,1,0], [3,1,1], [3,1,-1],[3,1,0], [3,2,-2], [3,2,-1],[3,2,0], [3,2,1],[3,2,2]]},
            'gthszvmolopt': {6: [[2, 0, 0], [2, 1, -1], [2, 1, 0], [2, 1, 1]]
                            }}

for f in frames: 
    f.pbc = True

kfock = [1/kmesh[0]*np.load(f"{root}/c2/fock_{i}_881.npy") for i in range(START, STOP)]
kover = [1/kmesh[0]*np.load(f"{root}/c2/over_{i}_881.npy") for i in range(START, STOP)]

for ifr in range(len(frames)):
    for ik, k in enumerate(kfock[ifr]):
        kfock[ifr][ik] = fix_orbital_order(k, frames[ifr], orbitals[ORBS]) #### TODO <<
        kover[ifr][ik] = fix_orbital_order(kover[ifr][ik], frames[ifr], orbitals[ORBS]) #### TODO <<

dataset = QMDataset(frames = frames, kmesh = kmesh, 
                               fock_kspace = [kfock[ifr][:] for ifr in range(len(frames))], 
                               overlap_kspace = [kover[ifr][:] for ifr in range(len(frames))],
                               device = device, orbs = orbitals[ORBS], orbs_name = ORBS)

#============================================================================================================================================================
# TARGETS
target_blocks, target_coupled_blocks = get_targets(dataset, device = device)

#============================================================================================================================================================
# FEATURES
try:
    print("Loading hfeat...", end = ' ', flush = True)
    hfeat = load(f'{workdir}/hfeat.npz', use_numpy = False).to(arrays = 'torch')
    print("Done.", flush = True)
except Exception as e:
    print(e, flush = True)
    try:
        print("Loading rhonui...", end = ' ', flush = True)
        rhonui = load(f'{workdir}/rhonui.npz', use_numpy = False).to(arrays = 'torch')
        print("Done.", flush = True)
        print("Loading rhoij...", end = ' ', flush = True)
        rhoij = load(f'{workdir}/rhoij.npz', use_numpy = False).to(arrays = 'torch')
        print("Done.", flush = True)
    except Exception as e:
        print(e, flush = True)
        print("Computing rhoij and rhonui...", end = ' ', flush = True)
        rhonui, rhoij = compute_pairfeatures(hypers_pair, hypers_atom, dataset, return_rho0ij = False, both_centers = both_centers, all_pairs = all_pairs, LCUT = LCUT, device = device)
        print("Done.", flush = True)
        print("Saving rhoij and rhonui...", end = ' ', flush = True)
        rhonui.save(f'{workdir}/rhonui')
        rhoij.save(f'{workdir}/rhoij')
        print("Done.", flush = True)
    print("Computing hfeat...", end = ' ', flush = True)
    print("Done.", flush = True)
    hfeat = twocenter_features_periodic_NH(single_center = rhonui, pair = rhoij, all_pairs = True)
    print("Saving hfeat...", end = ' ', flush = True)
    hfeat.save(f'{workdir}/hfeat.npz')
    print("Done.", flush = True)

hfeat_train = hfeat 
target_train = target_coupled_blocks

#============================================================================================================================================================
# TRAINING
target_kspace = dataset.fock_kspace
CG = ClebschGordanReal(lmax = LCUT, device = device)

model = LinearModelPeriodic(twocfeat = hfeat_train, 
                            target_blocks = target_train,
                            frames = dataset.structures, orbitals = dataset.basis, 
                            device = device,
                            nhidden = nhidden, 
                            nlayers = nlayers, 
                            train_kspace = True)

model = model.double()
losses = []
para = {}
grad = {}
for key in model.model:
    for module in model.model[key].children():
        for m in module.children():
            torch.nn.init.constant_(m.weight, 1.0)
pred0 = model()
norms = np.array([torch.norm(b.values).item() for k, b in pred0.items()])
norms /= norms.max()
model = LinearModelPeriodic(twocfeat = hfeat_train, 
                            target_blocks = target_train,
                            frames = dataset.structures, orbitals = dataset.basis, 
                            device = device,
                            nhidden = nhidden, 
                            nlayers = nlayers, 
                            train_kspace = True)

model = model.double()
losses = []
para = {}
grad = {}
optimizers = []
schedulers = []
for i, key in enumerate(model.model):
    optimizers.append(torch.optim.Adam(model.model[key].parameters(), lr = norms[i]*max_lr))
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], factor=scheduler_factor, patience=scheduler_patience, verbose=True))

kpts_train = [dataset.cells[ifr].get_scaled_kpts(dataset.cells[ifr].make_kpts(dataset.kmesh[ifr]))[:].tolist() for ifr in range(len(dataset.structures))]
nk = len(kpts_train[0])
target_kspace = [x[:nk] for x in dataset.fock_kspace]

kpts_ = torch.tensor(kpts_train).to(device)

import yappi
filename = 'callgrind.mlelec.prof'
yappi.set_clock_type('WALL')
yappi.start(builtins=True)
for epoch in range(nepoch):
    
    model.train(True)

    for ik, key in enumerate(model.model):
        optimizers[ik].zero_grad()
    
    epoch_loss = 0
    pred = model()
    n_predictions = sum([np.prod(b.values.shape) for _, b in pred.items()])
    loss = L2_kspace_loss(pred, 
                          target_kspace, 
                          dataset, 
                          cg = CG, 
                          kpts = kpts_)
    
    loss.backward(retain_graph = True)
    epoch_loss += loss
    losses.append(epoch_loss.item())

    for ik, key in enumerate(model.model):
        optimizers[ik].step()
        schedulers[ik].step(epoch_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:>7d}, train loss on all blocks {epoch_loss.item():>15.10f}, train loss per prediction {np.sqrt(epoch_loss.item())/n_predictions:>6.5e}", flush = True)

        for key in model.model:
            if key not in para:
                para[key] = []
                grad[key] = []
            for param in model.model[key].parameters():
                param
                grad[key].append(param.grad.norm().item())
                para[key].append(param.norm().item())


    if epoch % 1000 == 0 and epoch > 0:
        print('Saving', flush = True)
        np.save(f'{workdir}/para.npy', para)
        np.save(f'{workdir}/grad.npy', grad)
        np.savetxt(f'{workdir}/losses.dat', losses)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
            'loss': loss
            }, f'{workdir}/model.ckpt')

stats = yappi.get_func_stats()
stats.save(filename, type = 'callgrind')

