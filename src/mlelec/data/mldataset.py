from typing import Dict, List, Optional, Union, Tuple, NamedTuple
from collections import namedtuple
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
        device: str = None,
        model_type: Optional[str] = "acdc",
        item_names: Optional[Union[str, List]] = 'fock_blocks',
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        # TODO: to kwargs?
        features: Optional[torch.ScriptObject] = None,
        training_strategy: Optional[str] = "two_center",
        hypers_atom: Optional[Dict] = None,
        hypers_pair: Optional[Dict] = None,
        lcut: Optional[int] = None,
        cutoff: Optional[Union[int, float]] = None,
        sort_orbs: Optional[bool] = True,
        all_pairs: Optional[bool] = False,
        skip_symmetry: Optional[bool] = False,
        orbitals_to_properties: Optional[bool] = True,
        **kwargs,
    ):
    
        self.qmdata = qmdata #copy.deepcopy(qmdata) # Is deepcopy required?
        self.device = self.qmdata.device if device is None else device
        self.nstructs = len(self.qmdata.structures)
        self.rng = None
        self.kwargs = kwargs
        self.cutoff = cutoff
        self.structures = self.qmdata.structures
        self.natoms_list = [len(frame) for frame in self.structures]
        self.species = set([tuple(f.numbers) for f in self.structures])
        self.sort_orbs = sort_orbs
        self.all_pairs = all_pairs
        self.skip_symmetry = skip_symmetry
        self.orbitals_to_properties = orbitals_to_properties
        self.model_type = model_type  # flag to know if we are using acdc features or want to cycle through positions
        training_strategy = _flattenname(training_strategy)

        self._compute_model_metadata()
        if lcut < max(self.model_metadata.keys['L']):
            lcut = max(self.model_metadata.keys['L'])

        # Set features
        self._set_features(features, training_strategy, hypers_atom, hypers_pair, lcut, kwargs.get('compute_features', True))
        print('Features set')
        
        # Initialize items
        if isinstance(item_names, str):
            item_names = [item_names]
        if self.model_type =='acdc':
            item_names.append('features')

        # Set items
        self._set_items(item_names)
        print('Items set')

        # TODO: target classes?
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

        # Shuffle indices
        if shuffle:
            self._shuffle(shuffle_seed)
        else:
            self.indices = torch.arange(self.nstructs)
        
        # Train/validation/test fractions
        train_frac = kwargs.get("train_frac", None)
        val_frac = kwargs.get("val_frac", None)
        test_frac = kwargs.get("test_frac", None)

        if train_frac is not None:
            self._split_indices(train_frac, val_frac, test_frac)
            self._split_items(self.train_frac, self.val_frac, self.test_frac)

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
        val_frac: Optional[float] = None,
        test_frac: Optional[float] = None,
    ):
        '''
        Train/validation/test splitting
        '''

        fractions = [train_frac, val_frac, test_frac]
        defined_fractions = [f for f in fractions if f is not None]
        
        if not defined_fractions:
            # If all fractions are None, use default split
            self.train_frac = 0.7
            self.val_frac = 0.2
            self.test_frac = 0.1
        elif len(defined_fractions) == 3:
            # If all fractions are defined, check if they sum to 1
            if abs(sum(defined_fractions) - 1.0) > 1e-6:
                raise ValueError("The sum of train, validation, and test fractions must be 1.")
            self.train_frac, self.val_frac, self.test_frac = fractions
        else:
            # Handle cases where some fractions are defined
            remaining = 1.0 - sum(defined_fractions)
            undefined_count = fractions.count(None)
            
            if remaining < 0:
                raise ValueError("The sum of defined fractions must be less than or equal to 1.")
            
            # Assign defined fractions and distribute remaining among undefined
            self.train_frac = train_frac if train_frac is not None else (remaining / undefined_count if undefined_count > 0 else 0)
            self.val_frac = val_frac if val_frac is not None else (remaining / undefined_count if undefined_count > 0 else 0)
            self.test_frac = test_frac if test_frac is not None else (remaining / undefined_count if undefined_count > 0 else 0)

        # Ensure all fractions are set and sum to 1
        assert all(0 <= f <= 1 for f in [self.train_frac, self.val_frac, self.test_frac]), \
            "All fractions must be between 0 and 1."
        assert abs(sum([self.train_frac, self.val_frac, self.test_frac]) - 1.0) < 1e-6, \
            "The sum of all fractions must be 1."

        splits = [int(np.rint(s*self.nstructs)) for s in [self.train_frac, self.val_frac, self.test_frac]]
        
        try:
            assert sum(splits) == self.nstructs
        except AssertionError:
            splits[np.argmax(splits)] += self.nstructs - sum(splits)

        self.train_idx, self.val_idx, self.test_idx = torch.split(self.indices, splits)

        if self.test_frac > 0:
            assert (len(self.test_idx)> 0  # and len(self.val_idx) > 0 and len(self.train_idx) > 0
            ), "Split indices not generated properly"

        self.train_frames = [self.structures[i] for i in self.train_idx]
        self.val_frames = [self.structures[i] for i in self.val_idx]
        self.test_frames = [self.structures[i] for i in self.test_idx]

    def _set_features(self, features, training_strategy, hypers_atom, hypers_pair, lcut, compute_features):

        if not compute_features:
            self.features = None
            warnings.warn('Features not computed nor set.')
        elif features is None and self.model_type == "acdc":
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
                assert 'neighbor' in features.sample_names, f"Features must contain 'neighbor' label for {training_strategy}."
            elif training_strategy == 'one_center':
                assert 'neighbor' not in features.sample_names, f"Features must not contain 'neighbor' label for {training_strategy}."
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

    def _set_items(self, item_names):
        
        _item_names = [_flattenname(t) for t in item_names]
        self.item_names = {_flattenname(t): t for t in item_names}
        
        # Set targets

        items = {}
        for flat_name in _item_names:

            name = self.item_names[flat_name]

            if flat_name not in self._implemented_items():
                warnings.warn(f"Target {name} is not implemented! Skipping it")
                continue
            
            if flat_name == 'fockblocks':
                items[name] = self.compute_coupled_blocks()

            elif flat_name == 'fockrealspace':
                if self.qmdata._ismolecule:
                    l = self.compute_tensors(self.qmdata.fock_realspace)

                else:
                    items[name] = []
                    l = self.compute_tensors([torch.stack(list(h.values())) for h in self.qmdata.fock_realspace])
                    items[name] = [{T: H[iT] for iT, T in enumerate(self.qmdata.fock_realspace[ifr])} for ifr, H in enumerate(l)]

            elif flat_name == 'overlaprealspace':
                if self.qmdata._ismolecule:
                    l = self.compute_tensors(self.qmdata.overlap_realspace)

                else:
                    items[name] = []
                    l = self.compute_tensors([torch.stack(list(h.values())) for h in self.qmdata.overlap_realspace])
                    items[name] = [{T: H[iT] for iT, T in enumerate(self.qmdata.overlap_realspace[ifr])} for ifr, H in enumerate(l)]

            elif flat_name == 'fockkspace':
                assert not self.qmdata._ismolecule, "k-space Hamiltonian not available for molecules."
                items[name] = self.compute_tensors(self.qmdata.fock_kspace)

            elif flat_name == 'overlapkspace':
                assert not self.qmdata._ismolecule, "k-space overlap not available for molecules."
                items[name] = self.compute_tensors(self.qmdata.overlap_kspace)
            
            elif flat_name == 'eigenvalues':
                if 'atomresolveddensity' not in _item_names:
                    items[name] = self.compute_eigenvalues(return_eigenvectors = False)

            elif flat_name == 'atomresolveddensity':
                if 'eigenvalues' in _item_names:
                    T, e  = self.compute_atom_resolved_density(return_eigenvalues = True)
                    items['eigenvalues'] = e

                else:
                    T  = self.compute_atom_resolved_density(return_eigenvalues = False)
                items[name] = T

            elif flat_name == 'features':
                assert self.features is not None, "Features not set, call _set_features() first"
                items[name] = self.features
                
            else:
                raise ValueError(f"This looks like a bug! {flat_name} is in MLDataset._implemented_items but it is not properly handled in the loop.")
            
        self.items = items

    def _split_items(self, train_frac, val_frac, test_frac):
        try:
            assert np.isclose(train_frac + val_frac + test_frac, 1, rtol = 1e-6, atol = 1e-5)
        except AssertionError:
            test_frac = 1 - (train_frac + val_frac)
            assert test_frac > 0
            warnings.warn(f'Selected `test_frac` incorrect. Changed to {test_frac}')

        split_items = {}
        for k, item in self.items.items():
            
            if isinstance(item, torch.ScriptObject):
                if item._type().name() == 'TensorMap':
                    split_items[k], grouped_labels = self._split_by_structure(item)
            elif isinstance(item, list) or isinstance(item, torch.Tensor):
                split_items[k] = item

        try:
            assert grouped_labels is not None
        except:
            grouped_labels = [Labels(names = 'structure', 
                                     values = A.reshape(-1, 1)) for A in self.indices]
            
        # Initialize mentatensor.learn.IndexedDataset
        _dict = {k: [split_items[k][A] for A in self.train_idx] for k in split_items}
        _group_lbl = [grouped_labels[A].values.tolist()[0][0] for A in self.train_idx]
        self.train_dataset = IndexedDataset(sample_id = _group_lbl, **_dict)

        _dict = {k: [split_items[k][A] for A in self.val_idx] for k in split_items}
        _group_lbl = [grouped_labels[A].values.tolist()[0][0] for A in self.val_idx]
        self.val_dataset = IndexedDataset(sample_id = _group_lbl, **_dict)

        _dict = {k: [split_items[k][A] for A in self.test_idx] for k in split_items}
        _group_lbl = [grouped_labels[A].values.tolist()[0][0] for A in self.test_idx]
        self.test_dataset = IndexedDataset(sample_id = _group_lbl, **_dict)


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
            mask_tensor = [torch.ones(size, dtype = dtype, device = self.device) if mask[p] else torch.zeros(size, dtype = dtype, device = self.device) for p, size in zip(posit, basis)]
            mask = torch.vstack([torch.hstack([mask_tensor[natm*i+j] for j in range(natm)]) for i in range(natm)])
    
            for index in indices:

                M = tensor[index].clone().to(device = self.device)
                out_tensor[index] = M*mask
            
            out_tensors.append(out_tensor)

            mask_tensor = None
            mask = None
        
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
                eigenvectors = torch.empty(eigenvectors_shape, dtype = torch.complex128)

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
            Ms = self.qmdata.overlap_realspace
        else:
            Ms = self.qmdata.overlap_kspace

        use_overlap = False        
        if Ms is not None:
            use_overlap = True

        eigenvalues, eigenvectors = self.compute_eigenvalues(return_eigenvectors = True)
        
        frames = self.qmdata.structures
        basis = self.qmdata.basis

        T = []

        for ifr, (eval, evec, frame, M) in enumerate(zip(eigenvalues, eigenvectors, frames, Ms)):

            leading_shape = evec.shape[:-2]
            indices = itertools.product(*[range(dim) for dim in leading_shape])

            natm = len(frame)
            nelec = sum(frame.numbers)
            split_idx = [len(basis[s]) for s in frame.numbers]

            T__ = create_nested_list(leading_shape)
            
            for index in indices:

                # Eigenvectors matrix
                C = evec[index]

                # Occupations
                occ = torch.tensor([2.0+0.0j if i <= nelec//2 else 0.0+0.0j for i in range(C.shape[0])], dtype = torch.complex128)

                # Compute the one-particle density matrix
                rho = torch.einsum('n,in,jn', occ, C, C.conj())

                if use_overlap:
                    # Compute the population matrix
                    sqrt_S, _ = _sqrtm_newton_schulz(M[index])

                    P = sqrt_S @ rho @ sqrt_S
                else:
                    # Use rho
                    P = rho

                # Compute the matrix elements of the atom resolved density matrix
                blocks = [block for slice_ in torch.split(P, split_idx, dim = 0) for block in torch.split(slice_, split_idx, dim = 1)]
                T_ = torch.tensor([torch.norm(b) for b in blocks], device = self.device).reshape(natm, natm)

                set_nested_list_value(T__, index, T_)
            
            T.append(torch.stack(T__))
        
        if return_eigenvalues:
            return T, eigenvalues
        else:
            return T
        
    def _compute_model_metadata(self ):

        qmdata = self.qmdata
        species_pair = np.unique([comb for frame in qmdata.structures for comb in itertools.combinations_with_replacement(np.unique(frame.numbers), 2)], axis = 0)

        # key_names = ['block_type', 'species_i', 'species_j', 'L', 'inversion_sigma']
        # property_names = ['n_i', 'l_i', 'n_j', 'l_j', 'dummy']
        # property_values = {}
        key_names = ['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j', 'L']
        if self.orbitals_to_properties:
            key_names += ['inversion_sigma']
        keys = []

        for s1, s2 in species_pair:
            same_species = s1 == s2

            nl1 = np.unique([nlm[:2] for nlm in qmdata.basis[s1]], axis = 0)
            nl2 = np.unique([nlm[:2] for nlm in qmdata.basis[s2]], axis = 0)

            if same_species:
                block_types = [-1,0,1]
                orbital_list = [(a, b) for a, b in itertools.product(nl1.tolist(), nl2.tolist()) if a <= b]
            else: 
                if s1 > s2:
                    continue
                block_types = [2]
                orbital_list = itertools.product(nl1, nl2)
            
            for block_type in block_types:
                for (n1, l1), (n2, l2) in orbital_list:
                    for L in range(abs(l1-l2), l1+l2+1):
                        sigma = (-1)**(l1+l2+L)

                        if s1 == s2 and n1 == n2 and l1 == l2:
                            if ((sigma == -1 and block_type in (0, 1)) or (sigma == 1 and block_type == -1)) and not self.skip_symmetry:
                                continue

                        if self.orbitals_to_properties:
                            key = block_type, s1, n1, l1, s2, n2, l2, L, sigma
                        else:
                            key = block_type, s1, n1, l1, s2, n2, l2, L

                        keys.append(key)
                        # if key not in property_values:
                        #     property_values[key] = []
                        # property_values[key].append([n1,l1,n2,l2,0])

        blocks = []
        # keys = []
        dummy_label = Labels(['dummy'], torch.tensor([[0]], device = qmdata.device))
        for k in keys:
            # keys.append(k)
            blocks.append(
                TensorBlock(
                    samples = dummy_label,
                    properties = dummy_label, #Labels(property_names, torch.tensor(property_values[k], device = qmdata.device)),
                    components = [dummy_label],
                    values = torch.zeros((1, 1, 1), dtype = torch.float32, device = qmdata.device) #torch.zeros((1, 1, len(property_values[k])), dtype = torch.float32, device = qmdata.device)
                )
            )

        self.model_metadata = mts.sort(TensorMap(Labels(key_names, torch.tensor(keys, device = qmdata.device)), blocks))
        if self.orbitals_to_properties:
            self.model_metadata = self.model_metadata.keys_to_properties(['n_i', 'l_i', 'n_j', 'l_j'])


    def group_and_join(self,
        batch: List[NamedTuple],
        fields_to_join: Optional[List[str]] = None,
        join_kwargs: Optional[dict] = {'different_keys': 'union', 'remove_tensor_name': True},
    ) -> NamedTuple:

        data = []
        names = batch[0]._fields
        if fields_to_join is None:
            fields_to_join = names
        if join_kwargs is None:
            join_kwargs = {}
        for name, field in zip(names, list(zip(*batch))):
            if name == "sample_id":  # special case, keep as is
                data.append(field)
                continue

            if name in fields_to_join:  # Join tensors if requested
                if isinstance(field[0], torch.ScriptObject) and field[0]._has_method(
                    "keys_to_properties"
                ):  # inferred metatensor.torch.TensorMap type
                    data.append(mts.join(field, axis="samples", **join_kwargs))
                elif isinstance(field[0], torch.Tensor):  # torch.Tensor type
                    try:
                        data.append(torch.stack(field))
                    except RuntimeError:
                        data.append(field)
                else:
                    data.append(field)

            else:  # otherwise just keep as a list
                data.append(field)

        return namedtuple("Batch", names)(*data)        
    
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