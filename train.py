#!/usr/bin/env python
# coding: utf-8

# Import modules

from pathlib import Path
from ase.io import read
from ase.visualize import view
import numpy as np 
import torch 
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.learn import Dataset, DataLoader, IndexedDataset
from metatensor.learn.data import group as mts_group, group_and_join as group_and_join_mts

from mlelec.data.dataset import QMDataset
from mlelec.features.acdc import pair_features, single_center_features, twocenter_features_periodic_NH, twocenter_hermitian_features
from mlelec.features.acdc import compute_features

from mlelec.utils.twocenter_utils import _to_coupled_basis, _to_uncoupled_basis, map_targetkeys_to_featkeys
from mlelec.utils.pbc_utils import (matrix_to_blocks, 
                                    blocks_to_matrix,
                                    kmatrix_to_blocks, 
                                    TMap_bloch_sums, 
                                    precompute_phase)
from mlelec.utils.symmetry import ClebschGordanReal
from mlelec.utils.target_utils import get_targets 

from mlelec.models.linear import LinearModelPeriodic
from mlelec.metrics import L2_loss_meanzero as L2_loss


device = 'cpu'
orbitals = {6: [[1,0,0],[2,0,0],[2,1,-1], [2,1,0],[2,1,1]]}
orbital_names = 'sto-3g'
START = 0
STOP = 20

cutoff = 8

# features hypers
max_radial  = 8
max_angular = 6
atomic_gaussian_width = 0.3
return_rho0ij = False
both_centers = False
all_pairs = False
sort_orbs = True
LCUT = 3

# Training 
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
nhidden = 128
nlayers = 2
nepochs = 20000
# nepochs_realH = 1000
save_every = 100
activation = 'SiLU'
train_bias = True


workdir = './'
featdir = f'{workdir}/features'
traindir = f'{workdir}/train'

Path(workdir).mkdir(exist_ok=True)
Path(featdir).mkdir(exist_ok=True)
Path(traindir).mkdir(exist_ok=True)

data_dir = "/exports/commonscratch/pegolo/cp2k_ham/allotropes"
data_prefix = ""

indices, kx, ky, kz = np.loadtxt(f'{data_dir}/kmesh.dat', unpack = True, dtype = np.int32)
kmesh = np.array([kx, ky, kz]).T

indices = indices[START:STOP]
kmesh = kmesh[START:STOP].tolist()
FT_norm = 2*np.prod(np.mean(kmesh, axis = 0))
frames = [read(f'{data_dir}/{i}/cell_{i}.xyz') for i in indices]
rfock = [np.load(f"{data_dir}/{data_prefix}{i}/realfock_{i}.npy", allow_pickle = True).item() for i in indices]
rover = [np.load(f"{data_dir}/{data_prefix}{i}/realoverlap_{i}.npy", allow_pickle = True).item() for i in indices]

for f in frames:
    f.wrap(center = (0,0,0), eps = 1e-30)
print('Creating dataset...', end = ' ', flush = True)
dataset = QMDataset(frames = frames, 
                               kmesh = kmesh, 
                               dimension = 3,
                               fock_realspace = rfock, 
                               overlap_realspace = rover, 
                               device = device, 
                               orbs = orbitals, 
                               orbs_name = orbital_names)


print('Done', flush = True)

try:
    target_coupled_blocks = mts.load(f'{featdir}/target_coupled_blocks')
    overlap_coupled_blocks = mts.load(f'{featdir}/overlap_coupled_blocks')
    target_blocks = mts.load(f'{featdir}/target_blocks')
    # k_target_blocks = mts.load(f'{featdir}/k_target_blocks') # Cannot save/load complex tensormaps
    print('Targets loaded from disk', flush = True)
except:
    print('Creating targets...', end = ' ', flush = True)
    target_blocks, target_coupled_blocks = get_targets(dataset, cutoff = cutoff, device = device, all_pairs = all_pairs, sort_orbs = sort_orbs)
    _, overlap_coupled_blocks = get_targets(dataset, cutoff = cutoff, device = device, target = 'overlap')
    
    mts.save(f'{featdir}/target_coupled_blocks', target_coupled_blocks)
    mts.save(f'{featdir}/overlap_coupled_blocks', overlap_coupled_blocks)
    mts.save(f'{featdir}/target_blocks', target_blocks)
    # mts.save(f'{featdir}/k_target_blocks', k_target_blocks, use_numpy = True) # Cannot save/load complex tensormaps

# k_target_blocks = kmatrix_to_blocks(dataset, cutoff = cutoff, all_pairs = all_pairs, sort_orbs = sort_orbs)
# k_target_coupled_blocks = _to_coupled_basis(k_target_blocks, skip_symmetry = False, device = device, translations= False)

# phase, indices, kpts_idx = precompute_phase(target_coupled_blocks, dataset, cutoff = cutoff)
# k_target_coupled_blocks = TMap_bloch_sums(target_coupled_blocks, phase, indices, kpts_idx, return_tensormap = True)

print('Done', flush = True)

# # Features

hypers_pair = {'cutoff': cutoff,
               'max_radial': max_radial,
               'max_angular': max_angular,
               'atomic_gaussian_width': atomic_gaussian_width,
               'center_atom_weight': 1,
               "radial_basis": {"Gto": {}},
               "cutoff_function": {"ShiftedCosine": {"width": 0.1}}}

hypers_atom = {'cutoff': 4,
               'max_radial': max_radial,
               'max_angular': max_angular,
               'atomic_gaussian_width': 0.3,
               'center_atom_weight': 1,
               "radial_basis": {"Gto": {}},
               "cutoff_function": {"ShiftedCosine": {"width": 0.1}}}

try:
    print('Loading features...', end = ' ', flush = True)
    features = mts.load(f'{featdir}/all_features')
except FileNotFoundError:
    print('File with all the features not found. Trying to load single features...', end = ' ', flush = True)
    features = [mts.load(f'{featdir}/hfeat_{i}') for i in indices]
    features = mts.rename_dimension(mts.remove_dimension(mts.permute_dimensions(features, axis = 'samples', dimensions_indexes = [6,0,1,2,3,4,5]), axis = 'samples', name = 'structure'), axis = 'samples', old = 'tensor', new = 'structure')
    print('Done', flush = True)
except:
    print('Not found! Computing features...', end = ' ', flush = True)
    features = compute_features(dataset, hypers_atom, hypers_pair=hypers_pair, lcut = LCUT)
    mts.save(f'{featdir}/all_features', features)
    print('Done', flush = True)


# # Train

model = LinearModelPeriodic(twocfeat = features, 
                            target_blocks = target_coupled_blocks,
                            frames = dataset.structures, 
                            orbitals = dataset.basis, 
                            device = device,
                            bias = train_bias,
                            nhidden = nhidden, 
                            nlayers = nlayers,
                            activation = activation,
                            apply_norm = True)
model = model.double()

optimizers = []
schedulers = []
for i, key in enumerate(model.model):
    optimizers.append(torch.optim.Adam(model.model[key].parameters(), lr = 5e-3))
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], factor = 0.5, patience = 1000, verbose=True))

split_by_axis = "samples"
split_by_dimension = "structure"

split_by_axis = "samples"
split_by_dimension = "structure"

grouped_labels = [mts.Labels(names = split_by_dimension, values = torch.tensor([A])) for A in mts.unique_metadata(target_coupled_blocks, 
                                                                                                                  axis = split_by_axis, 
                                                                                                                  names = split_by_dimension)]
split_target = mts.split(target_coupled_blocks, split_by_axis, grouped_labels)

grouped_labels = [mts.Labels(names = split_by_dimension, values = torch.tensor([A])) for A in mts.unique_metadata(overlap_coupled_blocks, 
                                                                                                                  axis = split_by_axis, 
                                                                                                                  names = split_by_dimension)]
split_overlaps = mts.split(overlap_coupled_blocks, split_by_axis, grouped_labels)

grouped_labels = [mts.Labels(names = split_by_dimension, values = torch.tensor([A])) for A in mts.unique_metadata(features, axis = split_by_axis, 
                                                                                                                  names = split_by_dimension)]
split_features = mts.split(features, split_by_axis, grouped_labels)

ml_data = IndexedDataset(descriptor = split_features, 
                         target = split_target, 
                         overlap = split_overlaps,
                         sample_id = [g.values.tolist()[0][0] for g in grouped_labels])
batch_size = 1

overnorms = [torch.stack([dataset.fock_realspace[ifr][T]*dataset.overlap_realspace[ifr][T] for T in dataset.fock_realspace[ifr]]).sum() for ifr in range(len(dataset))]

dataloader = DataLoader(ml_data, 
                        batch_size = batch_size, 
                        shuffle = False, 
                        collate_fn = lambda x: group_and_join_mts(x, join_kwargs = {'different_keys': 'union', 'remove_tensor_name': True}))

losses = []

print('Start training.', flush = True)
for epoch in range(nepochs):
    
    epoch_loss = 0

    lr = []
    for ib, batch in enumerate(dataloader):
        model.train(True)

        for ik, key in enumerate(model.model):
            optimizers[ik].zero_grad()
        
        pred = model.predict_batch(batch.descriptor, batch.target)
        
        # Compute the loss
        all_losses, epoch_loss = L2_loss(pred, batch.target, loss_per_block = True, norm = overnorms[batch.sample_id[0]], overlap = batch.overlap)
        
        # Total loss
        batch_loss = batch_loss.item()
        epoch_loss += batch_loss

        # Loop through submodels and backpropagate
        for ik, (loss, key) in enumerate(zip(all_losses, model.model)):
            loss.backward(retain_graph = False)
            torch.nn.utils.clip_grad_norm_(model.model[key].parameters(), 1)
            optimizers[ik].step()
            schedulers[ik].step(loss)
            lr.append(schedulers[ik].get_last_lr())

    losses.append(epoch_loss)
    if epoch % 1 == 0:
        print(f"Epoch {epoch:>7d}, train loss {epoch_loss:>15.10f}, avg lr = {np.mean(lr)}", flush = True)

    if epoch % save_every == 0 and epoch != 0:
        print('Saving model...', end = ' ', flush = True)
        
        mts.save(f'{workdir}/predictions_{epoch}', pred)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
            'scheduler_state_dict': [scheduler.state_dict() for scheduler in schedulers],
            'loss': epoch_loss
            }, f'{workdir}/model_{epoch}.ckpt')

        np.savetxt(f'{workdir}/losses.dat', losses)

        print('Done', flush = True)
