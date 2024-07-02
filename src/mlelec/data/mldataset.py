from typing import Dict, List, Optional, Union, Tuple
import warnings
import copy
from collections import defaultdict
from pathlib import Path
import itertools

import numpy as np

from ase.io import read

import torch
from torch.utils.data import Dataset
import torch.utils.data as data

import metatensor.torch as mts
from metatensor.torch import TensorMap, Labels, TensorBlock
from metatensor.learn import IndexedDataset

from mlelec.targets import ModelTargets
from mlelec.utils.target_utils import get_targets
from mlelec.utils.twocenter_utils import map_targetkeys_to_featkeys
from mlelec.utils.pbc_utils import unique_Aij_block
from mlelec.data.dataset import QMDataset
from mlelec.features.acdc import compute_features

import xitorch
from xitorch.linalg import symeig

class MLDataset():
    '''
    Goes from DFT data stored in QMDataset to a torch-compatible dataset ready for machine learning.
    '''
    def __init__(
        self,
        qmdata: QMDataset,
        device: str = "cpu",
        model_type: Optional[str] = "acdc",
        item_names: Optional[Union[str, List]] = 'fock_blocks',
        sort_orbs: Optional[bool] = True,
        all_pairs: Optional[bool] = False,
        skip_symmetry: Optional[bool] = False,
        orbitals_to_properties: Optional[bool] = True,
        cutoff: Optional[Union[int, float]] = None,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        features: Optional[torch.ScriptObject] = None,
        training_strategy: Optional[str] = "two_center",
        hypers_atom: Optional[Dict] = None,
        hypers_pair: Optional[Dict] = None,
        lcut: Optional[int] = None,
        **kwargs,
    ):
    
        self.device = device
        self.nstructs = len(qmdata.structures)
        self.rng = None
        self.kwargs = kwargs
        self.cutoff = cutoff
        self.qmdata = qmdata #copy.deepcopy(qmdata) # Is deepcopy required?
        self.structures = self.qmdata.structures
        self.natoms_list = [len(frame) for frame in self.structures]
        self.species = set([tuple(f.numbers) for f in self.structures])
        self.sort_orbs = sort_orbs
        self.all_pairs = all_pairs
        self.skip_symmetry = skip_symmetry
        self.orbitals_to_properties = orbitals_to_properties
        self.model_type = model_type  # flag to know if we are using acdc features or want to cycle through positions
        training_strategy = _flattenname(training_strategy)
        # if self.model_type == "acdc":
            # self.features = features

        self._compute_model_metadata()
        if lcut < max(self.model_metadata.keys['L']):
            lcut = max(self.model_metadata.keys['L'])

        # Set features
        self._set_features(features, training_strategy, hypers_atom, hypers_pair, lcut)
        
        if shuffle:
            self._shuffle(shuffle_seed)
        else:
            self.indices = torch.arange(self.nstructs)
        
        

        # Initialize items
        if isinstance(item_names, str):
            item_names = [item_names]
        if model_type =='acdc':
            item_names.append('features')

        self.item_names = [_flattenname(t) for t in item_names]
        orig_item_names = {_flattenname(t): t for t in item_names}
        
        # Set targets

        items = {}
        for name in self.item_names:

            if name not in self._implemented_items():
                warnings.warn(f"Target {name} is not implemented! Skipping it")
                continue
            
            if name == 'fockblocks':
                items[name] = self.compute_coupled_blocks()

            elif name == 'fockrealspace':
                if self.qmdata._ismolecule:
                    l = self.compute_tensors(self.qmdata.fock_realspace)

                else:
                    items[name] = []
                    l = self.compute_tensors([torch.stack(list(h.values())) for h in self.qmdata.fock_realspace])
                    items[name] = [{T: H[iT] for iT, T in enumerate(self.qmdata.fock_realspace[ifr])} for ifr, H in enumerate(l)]

            elif name == 'overlaprealspace':
                if self.qmdata._ismolecule:
                    l = self.compute_tensors(self.qmdata.overlap_realspace)

                else:
                    items[name] = []
                    l = self.compute_tensors([torch.stack(list(h.values())) for h in self.qmdata.overlap_realspace])
                    items[name] = [{T: H[iT] for iT, T in enumerate(self.qmdata.overlap_realspace[ifr])} for ifr, H in enumerate(l)]

            elif name == 'fockkspace':
                assert not self.qmdata._ismolecule, "k-space Hamiltonian not available for molecules."
                items[name] = self.compute_tensors(self.qmdata.fock_kspace)

            elif name == 'overlapkspace':
                assert not self.qmdata._ismolecule, "k-space overlap not available for molecules."
                items[name] = self.compute_tensors(self.qmdata.overlap_kspace)
            
            elif name == 'eigenvalues':
                if 'atomresolveddensity' not in self.item_names:
                    items[name] = self.compute_eigenvalues(return_eigenvectors = False)

            elif name == 'atomresolveddensity':
                if 'eigenvalues' in self.item_names:
                    T, e  = self.compute_atom_resolved_density(return_eigenvalues = True)
                    items['eigenvalues'] = e

                else:
                    T  = self.compute_atom_resolved_density(return_eigenvalues = False)
                items[name] = T

            elif name == 'features':
                assert self.features is not None, "Features not set, call _set_features() first"
                
            else:
                raise ValueError(f"This looks like a bug! {name} is in MLDataset._implemented_items but it is not properly handled in the loop.")
                
        # Define dictionary of targets

        # sets the first target as the primary target - # FIXME
        # self.target_class = ModelTargets(self.qmdata.item_names[0])
        # self.target = self.target_class.instantiate(
        #     next(iter(self.qmdata.target.values())),
        #     frames=self.structures,
        #     orbitals=self.qmdata.aux_data.get("orbitals", None),
        #     device=device,
        #     **kwargs,
        # )

        

        # Train/validation/test fractions
        # Default is 70/20/10 split
        self.train_frac = kwargs.get("train_frac", 0.7)
        self.val_frac = kwargs.get("val_frac", 0.2)
        self.test_frac = kwargs.get("test_frac", 0.1)
        self._split_indices(self.train_frac, self.val_frac, self.test_frac)

        try:
            assert np.isclose(self.train_frac + self.val_frac + self.test_frac, 1, rtol = 1e-6, atol = 1e-5)
        except AssertionError:
            self.test_frac = 1 - (self.train_frac + self.val_frac)
            assert self.test_frac > 0
            warnings.warn(f'Selected `test_frac` incorrect. Changed to {self.test_frac}')


        self.items = {}
        for k in items:
            item = items[k]
            if isinstance(item, torch.ScriptObject):
                if item._type().name() == 'TensorMap':
                    self.items[k], self.grouped_labels = self._split_by_structure(item)
            elif isinstance(item, list) or isinstance(item, torch.Tensor):
                self.items[k] = item

        self.items['features'], self.grouped_labels = self._split_by_structure(self.features)

        try:
            assert self.grouped_labels is not None
        except:
            self.grouped_labels = [Labels(names = 'structure', 
                                          values = A.reshape(-1, 1)) for A in self.indices]
            
        # Initialize mentatensor.learn.IndexedDataset
        _dict = {orig_item_names[k]: [self.items[k][A] for A in self.train_idx] for k in self.items}
        _group_lbl = [self.grouped_labels[A] for A in self.train_idx]
        self.train_dataset = IndexedDataset(sample_id = _group_lbl, **_dict)
        # self.train_dataset = IndexedDataset.from_dict(_d, sample_id = _g)

        _dict = {orig_item_names[k]: [self.items[k][A] for A in self.val_idx] for k in self.items}
        _group_lbl = [self.grouped_labels[A] for A in self.val_idx]
        self.val_dataset = IndexedDataset(sample_id = _group_lbl, **_dict)
        # self.val_dataset = IndexedDataset.from_dict(_d, sample_id = _g)

        _dict = {orig_item_names[k]: [self.items[k][A] for A in self.test_idx] for k in self.items}
        _group_lbl = [self.grouped_labels[A] for A in self.test_idx]
        self.test_dataset = IndexedDataset(sample_id = _group_lbl, **_dict)
        # self.test_dataset = IndexedDataset.from_dict(_d, sample_id = _g)


    def _shuffle(self, random_seed: int = None):
        '''
        Shuffle structure indices
        '''

        if random_seed is None:
            self.rng = torch.default_generator
        else:
            self.rng = torch.Generator().manual_seed(random_seed)

        self.indices = torch.randperm(
            self.nstructs, generator=self.rng
        ).to(self.device)

        # update self.structures to reflect shuffling
        # self.structures_original = self.structures.copy()
        # self.structures = [self.structures_original[i] for i in self.indices]
        # self.target.shuffle(self.indices)
        # self.qmdata.shuffle(self.indices)

    def _get_subset(self, y: TensorMap, indices: torch.tensor):
        '''
        Get the subset of the TensorMap whose samples have "structure" in `indices`
        '''
        # indices = indices.cpu().numpy()
        
        assert isinstance(y, TensorMap)
        # for k, b in y.items():
        #     b = b.values.to(device=self.device)
        return mts.slice(y, axis = "samples", labels = Labels(names = ["structure"], values = indices.reshape(-1, 1)))
    
    def _split_by_structure(self, blocks: TensorMap, split_by_axis: Optional[str] = "samples", split_by_dimension: Optional[str] = "structure") -> TensorMap:
        grouped_labels = [mts.Labels(names = split_by_dimension, 
                                     values = A.reshape(-1, 1)) for A in mts.unique_metadata(blocks, 
                                                                              axis = split_by_axis, 
                                                                              names = split_by_dimension).values]
        return mts.split(blocks, split_by_axis, grouped_labels), grouped_labels

    def _split_indices(
        self,
        train_frac: float = None,
        val_frac: float = None,
        test_frac: Optional[float] = None,
    ):
        '''
        Train/validation/test splitting
        '''
        # TODO: handle this smarter

        # overwrite self train/val/test indices
        if train_frac is not None: self.train_frac = train_frac
        if val_frac is not None: self.val_frac = val_frac

        if test_frac is None:
            test_frac = 1 - (self.train_frac + self.val_frac)
            self.test_frac = test_frac
            assert self.test_frac > 0
        else:
            try:
                self.test_frac = test_frac
                assert np.isclose(self.train_frac + self.val_frac + self.test_frac, 1, rtol=1e-6, atol=1e-5), (self.train_frac + self.val_frac + self.test_frac, "Split fractions do not add up to 1")
            except:
                self.test_frac = 1 - (self.train_frac + self.val_frac)
                assert self.test_frac > 0

        splits = [int(np.rint(s*self.nstructs)) for s in [self.train_frac, self.val_frac, self.test_frac]]

        try:
            assert sum(splits) == self.nstructs
        except AssertionError:
            splits[-1] = self.nstructs - splits[0] - splits[1]

        self.train_idx, self.val_idx, self.test_idx = torch.split(self.indices, splits)

        # self.train_idx = self.indices[:int(self.train_frac * self.nstructs)].sort()[0]
        # self.val_idx = self.indices[int(self.train_frac * self.nstructs) : int((self.train_frac + self.val_frac) * self.nstructs)].sort()[0]
        # self.test_idx = self.indices[int((self.train_frac + self.val_frac) * self.nstructs) :].sort()[0]

        if self.test_frac > 0:
            assert (len(self.test_idx)> 0  # and len(self.val_idx) > 0 and len(self.train_idx) > 0
            ), "Split indices not generated properly"

        # self.target_train = self._get_subset(self.target.blocks, self.train_idx)
        # self.target_val = self._get_subset(self.target.blocks, self.val_idx)
        # self.target_test = self._get_subset(self.target.blocks, self.test_idx)

        self.train_frames = [self.structures[i] for i in self.train_idx]
        self.val_frames = [self.structures[i] for i in self.val_idx]
        self.test_frames = [self.structures[i] for i in self.test_idx]

    def _set_features(self, features, training_strategy, hypers_atom, hypers_pair, lcut):

        if features is None and self.model_type == "acdc":
            assert hypers_atom is not None, "`hypers_atom` must be present when `features` is not provided."
            assert lcut is not None, f"`lcut` must be present when `features` is not provided."
            
            if training_strategy == "twocenter":
                assert hypers_pair is not None, f"`hypers_pair` must be present when `features` is not provided and `training_strategy` is {training_strategy}."
                assert hypers_pair['cutoff'] == self.cutoff, f"`hypers_pair['cutoff']` must be equal to self.cutoff."

                self.features = compute_features(self.qmdata, 
                                                 hypers_atom, 
                                                 hypers_pair = hypers_pair, 
                                                 lcut = lcut, 
                                                 all_pairs = self.all_pairs, 
                                                 device = self.device,
                                                #  training_strategy = training_strategy, # TODO
                                                 **self.kwargs)
                                                 
            else:
                raise NotImplementedError(f"Training strategy {training_strategy} not implemented.")
        else:
            if training_strategy == 'twocenter':
                assert 'neighbors' in features.sample_names, f"Features must contain 'neighbors' label for {training_strategy}."
            elif training_strategy == 'one_center':
                assert 'neighbors' not in features.sample_names, f"Features must not contain 'neighbors' label for {training_strategy}."
            else:
                raise NotImplementedError(f"Training strategy {training_strategy} not implemented")
            self.features = features
        
        # if not hasattr(self, "train_idx"):
        #     warnings.warn("No train/val/test split found, deafult split used")
        #     self._split_indices(train_frac=0.7, val_frac=0.2, test_frac=0.1)

        # self.feature_names = features.keys.values
        # self.feat_train = self._get_subset(self.features, self.train_idx)
        # self.feat_val = self._get_subset(self.features, self.val_idx)
        # self.feat_test = self._get_subset(self.features, self.test_idx)

    def _set_model_return(self, model_return: str = "blocks"):
        ## Helper function to set output in __get_item__ for model training
        assert model_return in ["blocks", "tensor"], "model_target must be one of [blocks, tensor]"
        self.model_return = model_return

    def __len__(self):
        return self.nstructs

    # def __getitem__(self, idx):
    #     if not self.model_type == "acdc":
    #         return self.structures[idx], self.target.tensor[idx]
    #     else:
    #         assert self.features is not None, "Features not set, call _set_features() first"
    #         x = mts.slice(self.features, axis="samples", labels=Labels(names=["structure"], values = idx.reshape(-1, 1)))

    #         if self.model_return == "blocks":
    #             y = mts.slice(self.target.blocks, axis="samples", labels = Labels(names=["structure"], values = idx.reshape(-1, 1)))
    #         else:
    #             idx = [i.item() for i in idx]
    #             y = self.target.tensor[idx]
 
    #         return x, y, idx

    def collate_fn(self, batch):
        x = batch[0][0]
        y = batch[0][1]
        idx = batch[0][2]
        return {"input": x, "output": y, "idx": idx}
    
    def compute_tensors(self, tensors: Union[torch.Tensor, List]):
        
        out_tensors = []

        for ifr, tensor in enumerate(tensors):

            shape = tensor.shape
            leading_shape = shape[:-2]
            indices = itertools.product(*[range(dim) for dim in leading_shape])
            
            dtype = tensor.dtype
            
            out_tensor = torch.empty_like(tensor)

            frame = self.qmdata.structures[ifr]
            natm = len(frame)
            basis = [len(self.qmdata.basis[s]) for s in frame.numbers]
            basis = itertools.product(basis, basis)
            posit = itertools.product(range(natm), range(natm))
            mask = frame.get_all_distances(mic = True) <= self.cutoff
            mask_tensor = [torch.ones(size, dtype = dtype) if mask[p] else torch.zeros(size, dtype = dtype) for p, size in zip(posit, basis)]
            mask = torch.vstack([torch.hstack([mask_tensor[natm*i+j] for j in range(natm)]) for i in range(natm)])
    
            for index in indices:

                M = tensor[index].clone()
                out_tensor[index] = M*mask
            
            out_tensors.append(out_tensor)
        
        return out_tensors
    
    def compute_coupled_blocks(self, matrix = None, use_overlap = False):
        #TODO: move to a target class?
        if use_overlap:
            target = 'overlap'
        else:
            target = 'fock'

        blocks = get_targets(self.qmdata, 
                             cutoff = self.cutoff, 
                             target = target, 
                             all_pairs = self.all_pairs, 
                             sort_orbs = self.sort_orbs, 
                             skip_symmetry = self.skip_symmetry,
                             device = self.device, 
                             matrix = matrix,
                             orbitals_to_properties = self.orbitals_to_properties,
                             return_uncoupled = False)
        
        return blocks

    def compute_eigenvalues(self, return_eigenvectors = False):
        #TODO: move to a target class?

        if self.qmdata._ismolecule:
            As = self.qmdata.fock_realspace
            Ms = self.qmdata.overlap_realspace
        else:
            As = self.qmdata.fock_kspace
            Ms = self.qmdata.overlap_kspace

        eigenvalues_list = []
        if return_eigenvectors:
            eigenvectors_list = []

        # Iterate through structures
        for ifr, (A, M) in enumerate(zip(As, Ms)):
        
            shape = A.shape        
            # assert shape[-2] == shape[-1], "Matrices are not square!"
        
            # if M is not None:
            #     assert M.shape == shape, "Overlaps do not have the same shape as Matrices!"
            
            # Loop through all dimensions but the last two
            leading_shape = shape[:-2]
            indices = itertools.product(*[range(dim) for dim in leading_shape])

            # Preallocate output tensor for eigenvalues
            eigenvalues_shape = leading_shape + (shape[-1],)
            eigenvalues = torch.empty(eigenvalues_shape, dtype = torch.float64)

            # Preallocate output tensor for eigenvectors
            if return_eigenvectors:
                eigenvectors_shape = leading_shape + (shape[-1], shape[-1])
                eigenvectors = torch.empty(eigenvectors_shape, dtype = torch.complex64)

            # Iterate over all possible indices of the leading dimensions
            for index in indices:

                # Define xitorch LinarOperators
                Ax = xitorch.LinearOperator.m(A[index])
                Mx = xitorch.LinearOperator.m(M[index]) if M is not None else None

                # Compute eigenvalues and eigenvectors
                eigvals, eigvecs = symeig(Ax, M = Mx)

                # Store eigenvalues
                eigenvalues[index] = eigvals

                if return_eigenvectors:
                    # Store eigenvectors
                    eigenvectors[index] = eigvecs
            
            eigenvalues_list.append(eigenvalues)
            if return_eigenvectors:
                eigenvectors_list.append(eigenvectors)

        if return_eigenvectors:
            return eigenvalues_list, eigenvectors_list
        else:
            return eigenvalues_list
        
    def compute_atom_resolved_density(self, return_eigenvalues: Optional[bool] = True):
        #TODO: move to a target class?

        # Function to create nested lists
        def create_nested_list(shape):
            if len(shape) == 0:
                return None
            return [create_nested_list(shape[1:]) for _ in range(shape[0])]
        
        # Function to set value in nested list by indices
        def set_nested_list_value(nested_list, indices, value):
            for idx in indices[:-1]:
                nested_list = nested_list[idx]
            nested_list[indices[-1]] = value

        if self.qmdata._ismolecule:
            A = self.qmdata.fock_realspace
            M = self.qmdata.overlap_realspace
        else:
            A = self.qmdata.fock_kspace
            M = self.qmdata.overlap_kspace

        use_overlap = False        
        if M is not None:
            use_overlap = True

        eigenvalues, eigenvectors = self.compute_eigenvalues(A, M = M, return_eigenvectors = True)
        
        frames = self.qmdata.structures
        basis = self.qmdata.basis

        leading_shape = eigenvectors.shape[:-2]
        indices = itertools.product(*[range(dim) for dim in leading_shape])

        # Preallocate nested lists to store the results
        T = create_nested_list(leading_shape)
        ifr = -1
        for index in indices:

            # Eigenvectors matrix
            C = eigenvectors[index]

            if ifr != frames[indices[0]]:
                ifr = indices[0]
                frame = frames[ifr]
                natm = len(frame)
                nelec = sum(frame.numbers)
                occ = torch.tensor([2.0+0.0j if i <= nelec//2 else 0.0+0.0j for i in range(C.shape[0])])
                split_idx = [len(basis[s]) for s in frame.numbers]

            # Compute the one-particle density matrix
            rho = torch.einsum('n,in,jn', occ, C, C.conj())

            if use_overlap:
                # Compute the population matrix
                sqrt_S = _sqrtm_newton_schulz(M[index])
                P = sqrt_S @ rho @ sqrt_S
            else:
                # Use rho
                P = rho

            # Compute the matrix elements of the atom resolved density matrix
            blocks = [block for slice_ in torch.split(P, split_idx, dim = 0) for block in torch.split(slice_, split_idx, dim = 1)]
            T_ = torch.norm(blocks, dim = (1, 2)).reshape(natm, natm)

            set_nested_list_value(T, index, T_)
        
        if return_eigenvalues:
            return T, eigenvalues
        else:
            return T
        
    def _compute_model_metadata(self):

        qmdata = self.qmdata
        
        if not self.orbitals_to_properties:
            raise NotImplementedError("Only orbitals to properties implemented.")
        
        key_names = ['block_type', 'species_i', 'species_j', 'L', 'inversion_sigma']
        property_names = ['n_i', 'l_i', 'n_j', 'l_j']
        property_values = {}
        
        species = list(qmdata.basis.keys())
        for s1, s2 in itertools.product(species, repeat = 2):
            same_species = s1 == s2
            if same_species:
                block_types = [-1,0,1]
            else: 
                block_types = [2]
            nl1 = np.unique([nlm[:2] for nlm in qmdata.basis[s1]], axis = 0)
            nl2 = np.unique([nlm[:2] for nlm in qmdata.basis[s2]], axis = 0)

            for (n1, l1), (n2, l2) in zip(nl1, nl2):
                
                for L in range(abs(l1-l2), l1+l2+1):
                    sigma = (-1)**(l1+l2+L)
                    
                    for block_type in block_types:
                        key = block_type, s1, s2, L, sigma
                        
                        if key not in property_values:
                            property_values[key] = []
                        
                        property_values[key].append([n1,l1,n2,l2])

        blocks = []
        keys = []
        dummy_label = Labels(['dummy'], torch.tensor([[0]], device = qmdata.device))
        for k in property_values:
            keys.append(k)
            blocks.append(
                TensorBlock(
                    samples = dummy_label,
                    properties = Labels(property_names, torch.tensor(property_values[k], device = qmdata.device)),
                    components = [dummy_label],
                    values = torch.zeros((1, 1, len(property_values[k])), dtype = torch.float32, device = qmdata.device)
                )
            )

        self.model_metadata = mts.sort(TensorMap(Labels(key_names, torch.tensor(keys, device = qmdata.device)), blocks))

                


        
    def _implemented_items(self):
        return [
            'fockblocks', 
            'overlapblocks', 
            'fockrealspace',
            'fockkspace',
            'overlaprealspace',
            'overlapkspace',
            'eigenvalues',
            'atomresolveddensity', 
            'features'
            ]

def _flattenname(string):
    return ''.join(''.join(string.split('_')).split(' ')).lower()

def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    expected_num_dims = 2
    if matrix.dim() != expected_num_dims:
        raise ValueError(f'Input dimension equals {matrix.dim()}, expected {expected_num_dims}')

    if num_iters <= 0:
        raise ValueError(f'Number of iteration equals {num_iters}, expected greater than 0')

    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, requires_grad=False).to(matrix)
    Z = torch.eye(dim, dim, requires_grad=False).to(matrix)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1).to(matrix)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.]).to(error), atol=1e-5):
            break

    return s_matrix, error