from ase.io import read
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np 
import torch 
torch.set_default_dtype(torch.float64)

import rascaline.torch

import metatensor.torch as mts

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.learn import IndexedDataset, DataLoader
from metatensor.learn.data import group as mts_group, group_and_join as group_and_join_mts

from mlelec.data.dataset import QMDataset, split_by_Aij_mts
from mlelec.utils.twocenter_utils import _to_coupled_basis
from mlelec.utils.pbc_utils import matrix_to_blocks, kmatrix_to_blocks, TMap_bloch_sums, precompute_phase, kblocks_to_matrix, kmatrix_to_blocks, blocks_to_matrix, matrix_to_blocks
from mlelec.utils.plot_utils import print_matrix, matrix_norm, block_matrix_norm, plot_block_errors
from mlelec.features.acdc import compute_features
from mlelec.utils.target_utils import get_targets
from mlelec.models.linear import LinearModelPeriodic
from mlelec.metrics import L2_loss, L2_loss_meanzero

device = 'cpu'

orbitals = {
    'sto-3g': {5: [[1,0,0],[2,0,0],[2,1,-1], [2,1,0],[2,1,1]], 
               6: [[1,0,0],[2,0,0],[2,1,-1], [2,1,0],[2,1,1]], 
               7: [[1,0,0],[2,0,0],[2,1,-1], [2,1,0],[2,1,1]]}, 
    
    'def2svp': {6: [[1,0,0],[2,0,0],[3,0,0],[2,1,1], [2,1,-1],[2,1,0], [3,1,1], [3,1,-1],[3,1,0], [3,2,-2], [3,2,-1],[3,2,0], [3,2,1],[3,2,2]]},
    'benzene': {6: [[2,0,0],[2,1,-1], [2,1,0],[2,1,1]], 1:[[1,0,0]]},
    'gthszvmolopt': {
        6: [[2, 0, 0], [2, 1, -1], [2, 1, 0], [2, 1, 1]],
        
        16: [[3,0,0], 
             [3,1,-1], [3,1,0], [3,1,1]],

        42: [[4,0,0], 
             [5,0,0], 
             [4,1,-1], [4,1,0], [4,1,1], 
             [4, 2, -2], [4, 2, -1], [4, 2, 0], [4, 2, 1], [4, 2, 2]]}
}

workdir = './'
START = 0 
STOP = 5
ORBS = 'sto-3g'
root = f'{workdir}/examples/data/periodic/deepH_graphene/wrap/'
data_dir = root
frames = read(f'{data_dir}/wrapped_deepH_graphene.xyz', slice(START, STOP))
rfock = [np.load(f"{data_dir}/realfock_{i}.npy", allow_pickle = True).item() for i in range(START, STOP)]
rover = [np.load(f"{data_dir}/realoverlap_{i}.npy", allow_pickle = True).item() for i in range(START, STOP)]
kmesh = [1,1,1]
dataset = QMDataset(frames = frames, 
                               kmesh = kmesh, 
                               dimension = 2,
                               fock_realspace = rfock, 
                               overlap_realspace = rover, 
                               device = device, 
                               orbs = orbitals[ORBS], 
                               orbs_name = 'sto-3g')

cutoff = 6
target_blocks, target_coupled_blocks = get_targets(dataset, cutoff = cutoff, device = device)
_, overlap_coupled_blocks = get_targets(dataset, cutoff = cutoff, device = device, target = 'overlap')

max_radial  = 6
max_angular = 4
atomic_gaussian_width = 0.3

hypers_pair = {'cutoff': cutoff,
               'max_radial': max_radial,
               'max_angular': max_angular,
               'atomic_gaussian_width': atomic_gaussian_width,
               'center_atom_weight': 1,
               "radial_basis": {"Gto": {}},
               "cutoff_function": {"ShiftedCosine": {"width": 7.85}}}


hypers_atom = {'cutoff': 4,
               'max_radial': max_radial,
               'max_angular': max_angular,
               'atomic_gaussian_width': 0.5,
               'center_atom_weight': 1,
               "radial_basis": {"Gto": {}},
               "cutoff_function": {"ShiftedCosine": {"width": 3.85}}}


return_rho0ij = False
both_centers = False
LCUT = 3

load_features = True
if load_features:
    try:
        features = mts.load(f'{root}/feat_{len(dataset)}')
        print('Loaded')
    except:
        print('Not found')
        features = compute_features(dataset, hypers_atom, hypers_pair=hypers_pair, lcut = LCUT)
        print('Computed')
        features.save(f'{root}/feat_{len(dataset)}')
        print('Saved')
else:
    features = compute_features(dataset, hypers_atom, hypers_pair=hypers_pair, lcut = LCUT)
    print('Computed')
    features.save(f'{root}/feat_{len(dataset)}')
    print('Saved')

from scipy.linalg import eigvalsh
eigvals = []
fff = dataset.bloch_sum(blocks_to_matrix(target_coupled_blocks, dataset))
sss = dataset.bloch_sum(blocks_to_matrix(overlap_coupled_blocks, dataset))
for A, (H, S) in enumerate(zip(fff, sss)):
    ee = []
    for ik in range(len(H)):
        ee.append(torch.from_numpy(eigvalsh(H[ik].numpy(), S[ik].numpy())))
    eigvals.append(torch.stack(ee))

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
                         overlap_kpoints = dataset.overlap_kspace,
                         eigenvalues = eigvals,
                         sample_id = [g.values.tolist()[0][0] for g in grouped_labels])


seed = 10
torch.manual_seed(seed)
np.random.seed(seed)

model = LinearModelPeriodic(twocfeat = features, 
                            target_blocks = target_coupled_blocks,
                            frames = dataset.structures, orbitals = dataset.basis, 
                            device = device,
                            bias = True,
                            nhidden = 16, 
                            nlayers = 1,
                            activation = 'SiLU',
                            apply_norm = True
                           )
model = model.double()

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3) #, betas = (0.8, 0.9)))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 100)

from metatensor.learn import DataLoader

batch_size = 1
dataloader = DataLoader(ml_data, batch_size = batch_size, shuffle = False, collate_fn = lambda x: group_and_join_mts(x, join_kwargs = {'different_keys': 'union', 'remove_tensor_name': True}))

from mlelec.metrics import Eigval_loss
loss_fn = Eigval_loss #L2_loss #_meanzero

from mlelec.utils.twocenter_utils import lowdin_orthogonalize

# %%timeit -n 1 -r 1

train_kspace = False
LOSS_LIST = []

lr = scheduler.get_last_lr()

nepoch = 50000
for epoch in range(nepoch):

    # Train against real space targets
    LOSS = 0
    # lr = []
    for ib, batch in enumerate(dataloader):
        
        model.train(True)
        
        # for ik, key in enumerate(model.model):
            # optimizers[ik].zero_grad()
        optimizer.zero_grad()
        
        pred = model.predict_batch(batch.descriptor, batch.target)
        HT = blocks_to_matrix(pred, dataset)
        HK = dataset.bloch_sum(HT)[batch.sample_id[0]]
        pred_eigvals = torch.stack([torch.linalg.eigvalsh(lowdin_orthogonalize(h, s)) for h, s in zip(HK, batch.overlap_kpoints)])

        loss = loss_fn(pred_eigvals, batch.eigenvalues)
       
        LOSS += loss.item()

        loss.backward() #retain_graph = False)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        scheduler.step(loss)
        if not np.isclose(scheduler.get_last_lr(), lr):
            lr = scheduler.get_last_lr()
            print(f'lr changed to {lr}')
  
    if epoch >= 0: #% 10 == 0:
        # print(f"Epoch {epoch:>7d}, train loss on all blocks {epoch_loss:>15.10f}, train loss per prediction {np.sqrt(epoch_loss)/n_predictions:>6.5e}")
        # print(f"Epoch {epoch:>7d}, train loss real {loss_real[-1]:>15.10f}") #, train loss k {loss_k[-1]:>15.10f}, train loss per prediction {np.sqrt(epoch_loss)/n_predictions:>6.5e}")
        LOSS_LIST.append(LOSS)
        print(f"Epoch {epoch:>7d}, train loss {LOSS:>15.10f}, avg lr = {np.mean(lr)}") #, train loss k {loss_k[-1]:>15.10f}, train loss per prediction {np.sqrt(epoch_loss)/n_predictions:>6.5e}")

