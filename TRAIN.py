#!/usr/bin/env python
# coding: utf-8

# Import modules
import sys
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

from mlelec.models.linear_integrated import LinearModelPeriodic
from mlelec.metrics import L2_loss_meanzero as L2_loss
from mlelec.utils.twocenter_utils import lowdin_orthogonalize


device = 'cpu'
orbitals = {6: [[1,0,0],[2,0,0],[2,1,-1], [2,1,0],[2,1,1]]}
orbital_names = 'sto-3g'
START = 0
STOP = int(sys.argv[1])

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
nhidden = 16
nlayers = 1
nepochs = 1001
# nepochs_realH = 1000
save_every = 50
activation = 'SiLU'
train_bias = True
patience = 10*int(STOP-START)
sched_factor = 0.8
lr = 0.003

workdir = './'
featdir = f'../../../features'
traindir = f'{workdir}/train_{START}_{STOP}'

Path(workdir).mkdir(exist_ok=True)
Path(featdir).mkdir(exist_ok=True)
Path(traindir).mkdir(exist_ok=True)

data_dir = "/exports/commonscratch/pegolo/cp2k_ham/small_cell/sto-3g"
data_prefix = "C2_174_881_"

######################################################################################################################################################
kmesh = [8,8,1]
indices = np.loadtxt('../../../random_indices.txt', dtype = int)[START:STOP]
frames = [read(f'{data_dir}/C2_174.extxyz', index = i) for i in indices]
rfock = [np.load(f"{data_dir}/{data_prefix}{i}/realfock_{i}.npy", allow_pickle = True).item() for i in indices]
rover = [np.load(f"{data_dir}/{data_prefix}{i}/realoverlap_{i}.npy", allow_pickle = True).item() for i in indices]

for f in frames:
    f.pbc=[True,True,False]
    f.wrap(center = (0,0,0), eps = 1e-60)
    f.pbc=True

print('Creating dataset...', end = ' ', flush = True)
dataset = QMDataset(frames = frames, 
                   kmesh = kmesh, 
                   dimension = 2,
                   fock_realspace = rfock, 
                   overlap_realspace = rover, 
                   device = device, 
                   orbs = orbitals, 
                   orbs_name = orbital_names)
######################################################################################################################################################
kmesh = [8,8,1]
indices = np.loadtxt('../../../random_indices.txt', dtype = int)[120:140]
frames = [read(f'{data_dir}/C2_174.extxyz', index = i) for i in indices]
rfock = [np.load(f"{data_dir}/{data_prefix}{i}/realfock_{i}.npy", allow_pickle = True).item() for i in indices]
rover = [np.load(f"{data_dir}/{data_prefix}{i}/realoverlap_{i}.npy", allow_pickle = True).item() for i in indices]

for f in frames:
    f.pbc=[True,True,False]
    f.wrap(center = (0,0,0), eps = 1e-60)
    f.pbc=True

print('Creating dataset...', end = ' ', flush = True)
testset = QMDataset(frames = frames, 
                   kmesh = kmesh, 
                   dimension = 2,
                   fock_realspace = rfock, 
                   overlap_realspace = rover, 
                   device = device, 
                   orbs = orbitals, 
                   orbs_name = orbital_names)
######################################################################################################################################################

print('Done', flush = True)

try:
    target_coupled_blocks = mts.load(f'{traindir}/target_coupled_blocks')
    # overlap_coupled_blocks = mts.load(f'{traindir}/overlap_coupled_blocks')
    # target_blocks = mts.load(f'{traindir}/target_blocks')
    print('Targets loaded from disk', flush = True)
except:
    print('Creating targets...', end = ' ', flush = True)
    _, target_coupled_blocks = get_targets(dataset, cutoff = cutoff, device = device, all_pairs = all_pairs, sort_orbs = sort_orbs)
    # _, overlap_coupled_blocks = get_targets(dataset, cutoff = cutoff, device = device, target = 'overlap')
    
    mts.save(f'{traindir}/target_coupled_blocks', target_coupled_blocks)
    # mts.save(f'{traindir}/overlap_coupled_blocks', overlap_coupled_blocks)
    # mts.save(f'{traindir}/target_blocks', target_blocks)

try:
    test_coupled_blocks = mts.load(f'{traindir}/test_coupled_blocks')
    # overlap_coupled_blocks = mts.load(f'{traindir}/overlap_coupled_blocks')
    # target_blocks = mts.load(f'{traindir}/target_blocks')
    print('Targets loaded from disk', flush = True)
except:
    print('Creating targets...', end = ' ', flush = True)
    _, test_coupled_blocks = get_targets(testset, cutoff = cutoff, device = device, all_pairs = all_pairs, sort_orbs = sort_orbs)
    # _, overlap_coupled_blocks = get_targets(dataset, cutoff = cutoff, device = device, target = 'overlap')
    
    mts.save(f'{traindir}/test_coupled_blocks', test_coupled_blocks)
    # mts.save(f'{traindir}/overlap_coupled_blocks', overlap_coupled_blocks)
    # mts.save(f'{traindir}/target_blocks', target_blocks)

print('Loaded', flush = True)

target_keynames = target_coupled_blocks.keys.names
keys = []
blocks= []
for k,b in target_coupled_blocks.items(): 
    li,lj, L = k['l_i'], k['l_j'], k['L']
    inversion_sigma = (-1) ** (li + lj + L)
    keys.append(torch.cat((k.values, torch.tensor([inversion_sigma]))))
    blocks.append(b.copy())
target_coupled_blocks = TensorMap( Labels(k.names+['inversion_sigma'], torch.stack(keys)), blocks)
                 
target_coupled_blocks_copy = target_coupled_blocks.copy()
target_coupled_blocks = target_coupled_blocks.keys_to_properties(['n_i', 'l_i',  'n_j','l_j'])

target_keynames = test_coupled_blocks.keys.names
keys = []
blocks= []
for k,b in test_coupled_blocks.items(): 
    li,lj, L = k['l_i'], k['l_j'], k['L']
    inversion_sigma = (-1) ** (li + lj + L)
    keys.append(torch.cat((k.values, torch.tensor([inversion_sigma]))))
    blocks.append(b.copy())
test_coupled_blocks = TensorMap( Labels(k.names+['inversion_sigma'], torch.stack(keys)), blocks)
                 
test_coupled_blocks_copy = test_coupled_blocks.copy()
test_coupled_blocks = test_coupled_blocks.keys_to_properties(['n_i', 'l_i',  'n_j','l_j'])

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

# Load features

#########################################################################################################################
features = []
for i in range(0, STOP, 20):
    print(i, i+20)
    features.append(mts.load(f'{featdir}/features_{i}_{i+20}_seed73'))
if len(features) == 1:
    features = features[0]
else:
    features = mts.join(features, axis = 'samples')
    blocks = []
    for k, b in features.items():
        sample_values = b.samples.values
        sample_values[:, 0] += sample_values[:, 6]*20
        samples = mts.Labels(b.samples.names[:-1], sample_values[:, :-1])
        blocks.append(mts.TensorBlock(
        values = b.values,
        samples = samples,
        properties = b.properties,
        components = b.components
        ))
    features = mts.TensorMap(features.keys, blocks)

test_features = []
for i in range(120, 140, 20):
    print(i, i+20)
    test_features.append(mts.load(f'{featdir}/features_{i}_{i+20}_seed73'))
if len(test_features) == 1:
    test_features = test_features[0]
else:
    test_features = mts.join(test_features, axis = 'samples')
    blocks = []
    for k, b in test_features.items():
        sample_values = b.samples.values
        sample_values[:, 0] += sample_values[:, 6]*20
        samples = mts.Labels(b.samples.names[:-1], sample_values[:, :-1])
        blocks.append(mts.TensorBlock(
        values = b.values,
        samples = samples,
        properties = b.properties,
        components = b.components
        ))
    test_features = mts.TensorMap(test_features.keys, blocks)

print('#loaded', flush = True) #loaded AF
#########################################################################################################################

# # Model

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
    optimizers.append(torch.optim.Adam(model.model[key].parameters(), lr = lr))
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[-1], factor = sched_factor, patience = patience))

split_by_axis = "samples"
split_by_dimension = "structure"

split_by_axis = "samples"
split_by_dimension = "structure"

grouped_labels = [mts.Labels(names = split_by_dimension, values = torch.tensor([A])) for A in mts.unique_metadata(target_coupled_blocks, 
                                                                                                                  axis = split_by_axis, 
                                                                                                                  names = split_by_dimension)]
split_target = mts.split(target_coupled_blocks, split_by_axis, grouped_labels)

# grouped_labels = [mts.Labels(names = split_by_dimension, values = torch.tensor([A])) for A in mts.unique_metadata(overlap_coupled_blocks, 
#                                                                                                                   axis = split_by_axis, 
#                                                                                                                   names = split_by_dimension)]
#split_overlaps = mts.split(overlap_coupled_blocks, split_by_axis, grouped_labels)

grouped_labels = [mts.Labels(names = split_by_dimension, values = torch.tensor([A])) for A in mts.unique_metadata(features, axis = split_by_axis, 
                                                                                                                  names = split_by_dimension)]
split_features = mts.split(features, split_by_axis, grouped_labels)

ml_data = IndexedDataset(descriptor = split_features, 
                         target = split_target, 
 #                        overlap = split_overlaps,
                         sample_id = [g.values.tolist()[0][0] for g in grouped_labels])
batch_size = 1

#overnorms = [torch.stack([dataset.fock_realspace[ifr][T]*dataset.overlap_realspace[ifr][T] for T in dataset.fock_realspace[ifr]]).sum() for ifr in range(len(dataset))]

dataloader = DataLoader(ml_data, 
                        batch_size = batch_size, 
                        shuffle = True, 
                        collate_fn = lambda x: group_and_join_mts(x, join_kwargs = {'different_keys': 'union', 'remove_tensor_name': True}))

losses = []

from mlelec.metrics import L2_loss

BEST = np.inf
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
        all_losses, batch_loss = L2_loss(pred, batch.target, loss_per_block = True) #, norm = overnorms[batch.sample_id[0]], overlap = batch.overlap)
        
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

        model.train(False)
        
        test_pred = model.predict(test_features, test_coupled_blocks)
        test_loss = L2_loss(test_pred, test_coupled_blocks).item()
        
        if test_loss < BEST:
            BEST = test_loss
            mts.save(f'{traindir}/predictions_{epoch}', model())
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                'scheduler_state_dict': [scheduler.state_dict() for scheduler in schedulers],
                'loss': epoch_loss,
                'val_loss': test_loss,
                'train_predictions': model(),
                'val_predictions': test_pred
                }, f'{traindir}/model_{epoch}.ckpt')
            
        model.train(True)