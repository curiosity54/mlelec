import itertools
import warnings
from collections import defaultdict, namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.learn import IndexedDataset
from metatensor.torch import Labels, TensorBlock, TensorMap

from mlelec.data.derived_properties import (
    compute_atom_resolved_density,
    compute_dipoles,
    compute_eigenvalues,
)
from mlelec.data.qmdataset import QMDataset
from mlelec.features.acdc import compute_features
from mlelec.utils.target_utils import get_targets
from mlelec.utils.twocenter_utils import fix_orbital_order


class Items(NamedTuple):
    fock_blocks: Optional[TensorMap] = None
    overlap_blocks: Optional[TensorMap] = None
    fock_realspace: Optional[List[TensorMap]] = None
    overlap_realspace: Optional[List[TensorMap]] = None
    fock_kspace: Optional[TensorMap] = None
    overlap_kspace: Optional[TensorMap] = None
    eigenvalues: Optional[List[torch.Tensor]] = None
    eigenvectors: Optional[List[torch.Tensor]] = None
    atom_resolved_density: Optional[List[torch.Tensor]] = None
    density_matrix: Optional[List[torch.Tensor]] = None
    features: Optional[torch.ScriptObject] = None
    dipoles: Optional[List[torch.Tensor]] = None


class MLDataset:
    """
    Converts DFT data stored in QMDataset to a torch-compatible dataset ready for
    machine learning.
    """

    def __init__(
        self,
        qmdata: QMDataset,
        device: Optional[str] = None,
        model_type: Optional[str] = "acdc",
        item_names: Optional[Union[str, List[str]]] = "fock_blocks",
        shuffle: Optional[bool] = False,
        shuffle_seed: Optional[int] = None,
        features: Optional[Union[str, torch.ScriptObject]] = None,
        training_strategy: Optional[str] = "two_center",
        hypers_atom: Optional[Dict] = None,
        hypers_pair: Optional[Dict] = None,
        lcut: Optional[int] = None,
        cutoff: Optional[Union[int, float]] = None,
        sort_orbs: Optional[bool] = True,
        all_pairs: Optional[bool] = False,
        skip_symmetry: Optional[bool] = False,
        orbitals_to_properties: Optional[bool] = True,
        train_frac: Optional[float] = 0.7,
        val_frac: Optional[float] = 0.2,
        test_frac: Optional[float] = 0.1,
        model_basis: Optional[Dict] = None,
        model_basis_name: Optional[str] = None,
        fix_p_orbital_order: Optional[bool] = False,
        apply_condon_shortley: Optional[bool] = False,
        aux_overlap_realspace: Optional[Union[List, torch.Tensor]] = None,
        aux_overlap_kspace: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self._qmdata = qmdata
        self.device = device or self.qmdata.device
        self._cutoff = cutoff if cutoff is not None else hypers_pair["cutoff"]
        self.rng = None
        self.kwargs = kwargs
        self._model_type = model_type
        self._training_strategy = _flattenname(training_strategy)
        self._sort_orbs = sort_orbs
        self._all_pairs = all_pairs
        self._skip_symmetry = skip_symmetry
        self._orbitals_to_properties = orbitals_to_properties
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed
        self._model_basis = model_basis or qmdata.basis
        self._model_basis_name = model_basis_name or qmdata.basis_name
        self.fix_p_orbital_order = fix_p_orbital_order
        self._orbital_order_fixed = False
        self.apply_condon_shortley = apply_condon_shortley
        self.aux_overlap_realspace = aux_overlap_realspace
        self.aux_overlap_kspace = aux_overlap_kspace

        self._compute_model_metadata()

        if lcut is not None and lcut < max(self.model_metadata.keys["L"]):
            lcut = max(self.model_metadata.keys["L"])

        if isinstance(features, str):
            features = self.load_features(features, self.device)
        self._set_features(features, hypers_atom, hypers_pair, lcut)

        self.item_names = [item_names] if isinstance(item_names, str) else item_names
        if self.model_type == "acdc" and kwargs.get("calc_features", True):
            self.item_names.append("features")

        self._initialize_tensors()

        self._set_items(self.item_names)

        self.indices = torch.arange(self.nstructs)
        self._update_datasets()

    def __repr__(self):
        return (
            f"MLDataset(\n"
            f"  qmdata: {self.qmdata},\n"
            f"  device: {self.device},\n"
            f"  model_type: {self.model_type},\n"
            f"  item_names: {self.item_names},\n"
            f"  shuffle: {self._shuffle},\n"
            f"  shuffle_seed: {self._shuffle_seed},\n"
            f"  features: {self.features is not None},\n"
            f"  items: {self.items is not None},\n"
            f"  training_strategy: {self.training_strategy},\n"
            f"  cutoff: {self.cutoff},\n"
            f"  sort_orbs: {self.sort_orbs},\n"
            f"  all_pairs: {self.all_pairs},\n"
            f"  skip_symmetry: {self.skip_symmetry},\n"
            f"  orbitals_to_properties: {self.orbitals_to_properties},\n"
            f"  train_frac: {self.train_frac},\n"
            f"  val_frac: {self.val_frac},\n"
            f"  test_frac: {self.test_frac},\n"
            f"  fix_p_orbital_order: {self.fix_p_orbital_order},\n"
            f"  apply_condon_shortley: {self.apply_condon_shortley}\n"
            f")"
        )

    @property
    def qmdata(self):
        return self._qmdata

    @property
    def nstructs(self):
        return len(self.qmdata.structures)

    @property
    def structures(self):
        return self.qmdata.structures

    @property
    def natoms_list(self):
        return [len(frame) for frame in self.structures]

    @property
    def species(self):
        return {tuple(frame.numbers) for frame in self.structures}

    @property
    def sort_orbs(self):
        return self._sort_orbs

    @property
    def all_pairs(self):
        return self._all_pairs

    @property
    def skip_symmetry(self):
        return self._skip_symmetry

    @property
    def orbitals_to_properties(self):
        return self._orbitals_to_properties

    @property
    def model_type(self):
        return self._model_type

    @property
    def training_strategy(self):
        return self._training_strategy

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value
        self._update_items_on_cutoff_change()

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    @property
    def items(self):
        return self._items

    @items.setter
    def items(self, items):
        self._items = items

    @property
    def model_basis(self):
        return self._model_basis

    @model_basis.setter
    def model_basis(self, basis):
        self._model_basis = basis
        self._compute_model_metadata()

    @property
    def model_basis_name(self):
        return self._model_basis_name

    def update_splits(
        self,
        train_frac=None,
        val_frac=None,
        test_frac=None,
        shuffle=None,
        shuffle_seed=None,
    ):
        """
        Update the fractions and shuffling parameters for splitting the dataset.
        """

        if shuffle is not None:
            self._shuffle = shuffle
        if shuffle_seed is not None:
            self._shuffle_seed = shuffle_seed

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
                raise ValueError(
                    "The sum of train, validation, and test fractions must be 1."
                )
            self.train_frac, self.val_frac, self.test_frac = fractions
        else:
            # Handle cases where some fractions are defined
            remaining = 1.0 - sum(defined_fractions)
            undefined_count = fractions.count(None)

            if remaining < 0:
                raise ValueError(
                    "The sum of defined fractions must be less than or equal to 1."
                )

            # Assign defined fractions and distribute remaining among undefined
            self.train_frac = (
                train_frac
                if train_frac is not None
                else (remaining / undefined_count if undefined_count > 0 else 0)
            )
            self.val_frac = (
                val_frac
                if val_frac is not None
                else (remaining / undefined_count if undefined_count > 0 else 0)
            )
            self.test_frac = (
                test_frac
                if test_frac is not None
                else (remaining / undefined_count if undefined_count > 0 else 0)
            )

        # Ensure all fractions are set and sum to 1
        assert all(
            0 <= f <= 1 for f in [self.train_frac, self.val_frac, self.test_frac]
        ), "All fractions must be between 0 and 1."
        assert (
            abs(sum([self.train_frac, self.val_frac, self.test_frac]) - 1.0) < 1e-6
        ), "The sum of all fractions must be 1."

        self._update_datasets()

    def _shuffle_indices(self):
        """
        Shuffle structure indices
        """
        if self._shuffle:
            if self._shuffle_seed is None:
                self.rng = torch.default_generator
            else:
                self.rng = torch.Generator().manual_seed(self._shuffle_seed)
            self.indices = torch.randperm(self.nstructs, generator=self.rng).to(
                self.device
            )
        else:
            self.indices = torch.arange(self.nstructs)

    def _get_subset(self, y: TensorMap, indices: torch.tensor):
        """
        Get the subset of the TensorMap whose samples have "structure" in `indices`
        """
        assert isinstance(y, TensorMap)
        return mts.slice(
            y,
            axis="samples",
            labels=Labels(names=["structure"], values=indices.reshape(-1, 1)),
        )

    def _split_indices(self):
        """
        Train/validation/test splitting
        """
        if self.nstructs == 1:
            largest_fraction = max(self.train_frac, self.val_frac, self.test_frac)
            if largest_fraction == self.train_frac:
                self.train_idx, self.val_idx, self.test_idx = (
                    torch.tensor([0], device=self.device),
                    torch.tensor([], device=self.device),
                    torch.tensor([], device=self.device),
                )
            elif largest_fraction == self.val_frac:
                self.train_idx, self.val_idx, self.test_idx = (
                    torch.tensor([], device=self.device),
                    torch.tensor([0], device=self.device),
                    torch.tensor([], device=self.device),
                )
            else:
                self.train_idx, self.val_idx, self.test_idx = (
                    torch.tensor([], device=self.device),
                    torch.tensor([], device=self.device),
                    torch.tensor([0], device=self.device),
                )
        else:
            splits = [
                int(np.rint(s * self.nstructs))
                for s in [self.train_frac, self.val_frac, self.test_frac]
            ]
            if sum(splits) != self.nstructs:
                splits[np.argmax(splits)] += self.nstructs - sum(splits)
            self.train_idx, self.val_idx, self.test_idx = torch.split(
                self.indices, splits
            )

        self.train_frames = [self.structures[i] for i in self.train_idx]
        self.val_frames = [self.structures[i] for i in self.val_idx]
        self.test_frames = [self.structures[i] for i in self.test_idx]

    def _set_features(self, features, hypers_atom, hypers_pair, lcut):
        if features is None and self.model_type == "acdc":
            assert (
                hypers_atom is not None
            ), "`hypers_atom` must be present when `features` is not provided."
            assert (
                lcut is not None
            ), "`lcut` must be present when `features` is not provided."

            if self.training_strategy == "twocenter":
                assert hypers_pair is not None, (
                    "`hypers_pair` must be present when `features` is not provided "
                    f"and `training_strategy` is {self.training_strategy}."
                )
                assert (
                    hypers_pair["cutoff"] == self.cutoff
                ), "`hypers_pair['cutoff']` must be equal to self.cutoff."

                self.features = compute_features(
                    self.qmdata,
                    hypers_atom,
                    hypers_pair=hypers_pair,
                    lcut=lcut,
                    all_pairs=self.all_pairs,
                    device=self.device,
                    **self.kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Training strategy {self.training_strategy} not implemented."
                )
        else:
            self.features = features

    @staticmethod
    def load_features(file_path: str, device: Optional[str] = None):
        """
        Load features from a file.

        Args:
            file_path (str): Path to the file containing the features.
            device (str, optional): Device to use for the features.

        Returns:
            torch.ScriptObject: Loaded features.
        """

        features = mts.load(file_path)
        if device is not None:
            features = features.to(device=device)

        return features

    def _set_items(self, item_names):
        _item_names = [_flattenname(t) for t in item_names]
        item_names_dict = {_flattenname(t): t for t in item_names}

        items_dict = {}

        for flat_name in _item_names:
            name = item_names_dict[flat_name]

            if flat_name not in self._implemented_items():
                warnings.warn(f"Target {name} is not implemented! Skipping it")
                continue

            if flat_name == "fockblocks":
                items_dict[name] = self.compute_coupled_blocks()

            elif flat_name == "overlapblocks":
                items_dict[name] = self.compute_coupled_blocks(use_overlap=True)

            elif flat_name in ["fockrealspace", "overlaprealspace"]:
                items_dict[name] = self._compute_real_space_tensors(flat_name)

            elif flat_name in ["fockkspace", "overlapkspace"]:
                items_dict[name] = self._compute_k_space_tensors(flat_name)

            elif flat_name == "eigenvalues":
                if "atomresolveddensity" not in _item_names:
                    _eval, _evec = self.compute_eigenvalues(return_eigenvectors=True)
                    items_dict[name] = _eval
                    items_dict["eigenvectors"] = _evec

            elif flat_name == "atomresolveddensity":
                if "eigenvalues" in _item_names:
                    T, rho, e, _evec = self.compute_atom_resolved_density(
                        return_eigenvalues=True,
                        return_rho=True,
                        return_eigenvectors=True,
                        use_overlaps=True,
                    )
                    items_dict["eigenvalues"] = e
                    items_dict["density_matrix"] = rho
                    items_dict["eigenvectors"] = _evec
                else:
                    T = self.compute_atom_resolved_density(
                        return_eigenvalues=False, use_overlaps=True
                    )
                items_dict[name] = T

            elif flat_name == "dipoles":
                dipoles = self.compute_dipoles()
                items_dict["dipoles"] = dipoles

            elif flat_name == "features":
                assert (
                    self.features is not None
                ), "Features not set, call _set_features() first"
                items_dict[name] = self.features

            else:
                raise ValueError(
                    (
                        f"This looks like a bug! {flat_name} is in "
                        "MLDataset.implemented_items but it is not "
                        "properly handled in the loop."
                    )
                )

        self.items = Items(**items_dict)

    def _initialize_tensors(self):
        """
        Initialize working copies of tensors with fixed orbital order
        and Condon-Shortley phase if necessary.
        """
        frames = self.qmdata.structures

        def apply_fixes(tensor, frame, basis):
            if self.fix_p_orbital_order:
                tensor = fix_orbital_order(tensor.clone(), frame, basis)
            if self.apply_condon_shortley:
                cs = np.concatenate(
                    [
                        (-1)
                        ** (
                            (np.array(basis[n])[:, 2] > 0)
                            * (np.abs(np.array(basis[n])[:, 2]))
                        )
                        for n in frame.numbers
                    ]
                ).flatten()[:, np.newaxis]
                cs = cs @ cs.T
                tensor = tensor * cs
            return tensor

        if self.qmdata.is_molecule:
            self.fock_realspace = (
                [
                    apply_fixes(T, frames[i], self.qmdata.basis)
                    for i, T in enumerate(self.qmdata.fock_realspace)
                ]
                if self.qmdata.fock_realspace is not None
                else None
            )
            self.overlap_realspace = (
                [
                    apply_fixes(T, frames[i], self.qmdata.basis)
                    for i, T in enumerate(self.qmdata.overlap_realspace)
                ]
                if self.qmdata.overlap_realspace is not None
                else None
            )
            self.aux_overlap_realspace = (
                [
                    apply_fixes(T, frames[i], self.model_basis)
                    for i, T in enumerate(self.aux_overlap_realspace)
                ]
                if self.aux_overlap_realspace is not None
                else None
            )

        else:
            self.fock_realspace = (
                [
                    dict(
                        (k, apply_fixes(v, frames[i], self.qmdata.basis))
                        for k, v in d.items()
                    )
                    for i, d in enumerate(self.qmdata.fock_realspace)
                ]
                if self.qmdata.fock_realspace is not None
                else None
            )
            self.overlap_realspace = (
                [
                    dict(
                        (k, apply_fixes(v, frames[i], self.qmdata.basis))
                        for k, v in d.items()
                    )
                    for i, d in enumerate(self.qmdata.overlap_realspace)
                ]
                if self.qmdata.overlap_realspace is not None
                else None
            )
            self.aux_overlap_realspace = (
                [
                    dict(
                        (k, apply_fixes(v, frames[i], self.model_basis))
                        for k, v in d.items()
                    )
                    for i, d in enumerate(self.aux_overlap_realspace)
                ]
                if self.aux_overlap_realspace is not None
                else None
            )

            self.fock_kspace = (
                [
                    torch.stack(
                        list(
                            map(
                                lambda k: apply_fixes(k, frames[i], self.qmdata.basis),
                                d,
                            )
                        )
                    )
                    for i, d in enumerate(self.qmdata.fock_kspace)
                ]
                if self.qmdata.fock_kspace is not None
                else None
            )
            self.overlap_kspace = (
                [
                    torch.stack(
                        list(
                            map(
                                lambda k: apply_fixes(k, frames[i], self.qmdata.basis),
                                d,
                            )
                        )
                    )
                    for i, d in enumerate(self.qmdata.overlap_kspace)
                ]
                if self.qmdata.overlap_kspace is not None
                else None
            )
            self.aux_overlap_kspace = (
                [
                    torch.stack(
                        list(
                            map(
                                lambda k: apply_fixes(k, frames[i], self.model_basis), d
                            )
                        )
                    )
                    for i, d in enumerate(self.aux_overlap_kspace)
                ]
                if self.aux_overlap_kspace is not None
                else None
            )

        if not self.fix_p_orbital_order and not self.apply_condon_shortley:
            self.fock_realspace = self.qmdata.fock_realspace
            self.overlap_realspace = self.qmdata.overlap_realspace
            self.fock_kspace = self.qmdata.fock_kspace
            self.overlap_kspace = self.qmdata.overlap_kspace

    def _compute_real_space_tensors(self, name):
        basis = self.qmdata.basis
        if name == "fockrealspace":
            tensor_list = self.fock_realspace
        elif name == "overlaprealspace":  # overlaprealspace
            if self.aux_overlap_realspace is None:
                tensor_list = self.overlap_realspace
            else:
                tensor_list = self.aux_overlap_realspace
                basis = self.model_basis

        if self.qmdata.is_molecule:
            return self.compute_tensors(tensor_list, basis)
        else:
            tensor_stack = self.compute_tensors(
                [torch.stack(list(h.values())) for h in tensor_list], basis
            )
            return [
                {T: H[iT] for iT, T in enumerate(tensor_list[ifr])}
                for ifr, H in enumerate(tensor_stack)
            ]

    def _compute_k_space_tensors(self, name):
        assert (
            not self.qmdata.is_molecule
        ), f"k-space {name} not available for molecules."
        basis = self.qmdata.basis
        if name == "fockkspace":
            tensor_list = self.fock_kspace
        elif name == "overlapkspace":
            if self.aux_overlap_kspace is None:
                tensor_list = self.overlap_kspace
            else:
                tensor_list = self.aux_overlap_kspace
                basis = self.model_basis

        return self.compute_tensors(tensor_list, basis)

    def _update_items_on_cutoff_change(self):
        """
        Update items affected by compute_tensors and compute_coupled_blocks
        when cutoff is changed.
        """
        item_names = [
            name
            for name in self.item_names
            if _flattenname(name)
            in {
                "fockblocks",
                "overlapblocks",
                "fockrealspace",
                "overlaprealspace",
                "fockkspace",
                "overlapkspace",
            }
        ]
        self._set_items(item_names)

    def _update_datasets(self):
        """
        Update datasets based on current train_frac, val_frac, test_frac,
        and shuffle properties.
        """
        self._shuffle_indices()
        self._split_indices()
        self._split_items()

    def _split_items(self):
        split_items = {}
        for k, item in self.items._asdict().items():
            if (
                isinstance(item, torch.ScriptObject)
                and item._type().name() == "TensorMap"
            ):
                split_items[k], grouped_labels = self._split_by_structure(item)
            elif isinstance(item, list) or isinstance(item, torch.Tensor):
                split_items[k] = item

        grouped_labels = [
            Labels(names="structure", values=A.reshape(-1, 1)) for A in self.indices
        ]

        self.train_dataset = self._create_indexed_dataset(
            split_items, self.train_idx, grouped_labels
        )
        self.val_dataset = self._create_indexed_dataset(
            split_items, self.val_idx, grouped_labels
        )
        self.test_dataset = self._create_indexed_dataset(
            split_items, self.test_idx, grouped_labels
        )

    def _split_by_structure(self, blocks: TensorMap) -> TensorMap:
        grouped_labels = [
            mts.Labels(names="structure", values=A.reshape(-1, 1))
            for A in mts.unique_metadata(
                blocks, axis="samples", names="structure"
            ).values
        ]
        return mts.split(blocks, "samples", grouped_labels), grouped_labels

    def _create_indexed_dataset(self, split_items, indices, grouped_labels):
        _dict = {k: [split_items[k][A] for A in indices] for k in split_items}
        # _group_lbl = [grouped_labels[A].values.tolist()[0][0] for A in indices]
        return IndexedDataset(sample_id=indices.tolist(), **_dict)

    def _compute_model_metadata(self):
        basis = self.model_basis

        species_pair = np.unique(
            [
                comb
                for frame in self.structures
                for comb in itertools.combinations_with_replacement(
                    np.unique(frame.numbers), 2
                )
            ],
            axis=0,
        )
        max_count = defaultdict(lambda: 0)
        for species_counts in [
            np.unique(frame.numbers, return_counts=True) for frame in self.structures
        ]:
            for s, c in zip(*species_counts):
                max_count[s] = c if c > max_count[s] else max_count[s]

        key_names = [
            "block_type",
            "species_i",
            "n_i",
            "l_i",
            "species_j",
            "n_j",
            "l_j",
            "L",
        ]
        if self.orbitals_to_properties:
            key_names += ["inversion_sigma"]
        keys = []

        for s1, s2 in species_pair:
            same_species = s1 == s2

            nl1 = np.unique([nlm[:2] for nlm in basis[s1]], axis=0)
            nl2 = np.unique([nlm[:2] for nlm in basis[s2]], axis=0)

            if same_species:
                block_types = [0] if max_count[s1] == 1 else [-1, 0, 1]
                orbital_list = [
                    (a, b)
                    for a, b in itertools.product(nl1.tolist(), nl2.tolist())
                    if a <= b
                ]
            else:
                if s1 > s2:
                    continue
                block_types = [2]
                orbital_list = itertools.product(nl1, nl2)

            for block_type in block_types:
                for (n1, l1), (n2, l2) in orbital_list:
                    for L in range(abs(l1 - l2), l1 + l2 + 1):
                        sigma = (-1) ** (l1 + l2 + L)

                        if s1 == s2 and n1 == n2 and l1 == l2:
                            if (
                                (sigma == -1 and block_type in (0, 1))
                                or (sigma == 1 and block_type == -1)
                            ) and not self.skip_symmetry:
                                continue

                        if self.orbitals_to_properties:
                            key = block_type, s1, n1, l1, s2, n2, l2, L, sigma
                        else:
                            key = block_type, s1, n1, l1, s2, n2, l2, L

                        keys.append(key)

        blocks = []
        dummy_label = Labels(["dummy"], torch.tensor([[0]], device=self.device))
        for k in keys:
            blocks.append(
                TensorBlock(
                    samples=dummy_label,
                    properties=dummy_label,
                    components=[dummy_label],
                    values=torch.zeros(
                        (1, 1, 1), dtype=torch.float32, device=self.device
                    ),
                )
            )

        self.model_metadata = mts.sort(
            TensorMap(Labels(key_names, torch.tensor(keys, device=self.device)), blocks)
        )
        if self.orbitals_to_properties:
            self.model_metadata = mts.sort(
                self.model_metadata.keys_to_properties(["n_i", "l_i", "n_j", "l_j"])
            )

    def compute_tensors(self, tensors: Union[torch.Tensor, List], basis_dict: dict):
        out_tensors = []

        for ifr, tensor in enumerate(tensors):
            shape = tensor.shape
            leading_shape = shape[:-2]
            indices = itertools.product(*[range(dim) for dim in leading_shape])
            dtype = tensor.dtype
            out_tensor = torch.empty_like(tensor)

            frame = self.qmdata.structures[ifr]
            natm = len(frame)
            basis = [len(basis_dict[s]) for s in frame.numbers]
            basis = itertools.product(basis, basis)
            posit = itertools.product(range(natm), range(natm))
            mask = frame.get_all_distances(mic=True) <= self.cutoff
            mask_tensor = [
                (
                    torch.ones(size, dtype=dtype, device=self.device)
                    if mask[p]
                    else torch.zeros(size, dtype=dtype, device=self.device)
                )
                for p, size in zip(posit, basis)
            ]
            mask = torch.vstack(
                [
                    torch.hstack([mask_tensor[natm * i + j] for j in range(natm)])
                    for i in range(natm)
                ]
            )

            for index in indices:
                M = tensor[index].clone().to(device=self.device)
                out_tensor[index] = M * mask

            out_tensors.append(out_tensor)

        return out_tensors

    def compute_coupled_blocks(self, matrix=None, use_overlap=False):
        target = "overlap" if use_overlap else "fock"
        return get_targets(
            self.qmdata,
            cutoff=self.cutoff,
            target=target,
            all_pairs=self.all_pairs,
            sort_orbs=self.sort_orbs,
            skip_symmetry=self.skip_symmetry,
            device=self.device,
            matrix=matrix,
            orbitals_to_properties=self.orbitals_to_properties,
            return_uncoupled=False,
        )

    def compute_eigenvalues(self, return_eigenvectors=False):
        if self.qmdata.is_molecule:
            As = self.qmdata.fock_realspace
            Ms = self.qmdata.overlap_realspace
        else:
            As = self.qmdata.fock_kspace
            Ms = self.qmdata.overlap_kspace

        return compute_eigenvalues(As, Ms, return_eigenvectors)

    def compute_dipoles(self):
        assert (
            self.qmdata.is_molecule
        ), "PySCF dipoles are ill-defined for periodic systems"
        return compute_dipoles(
            self.qmdata.fock_realspace,
            self.qmdata.overlap_realspace,
            mols=self.qmdata.mols,
            unfix=self.fix_p_orbital_order,
            frames=self.structures,
            basis=self.qmdata.basis,
            basis_name=self.qmdata.basis_name,
            requires_grad=False,
        )

    def compute_atom_resolved_density(
        self,
        return_rho=False,
        return_eigenvalues=True,
        return_eigenvectors=False,
        use_overlaps=False,
    ):
        eigenvalues, eigenvectors = self.compute_eigenvalues(return_eigenvectors=True)
        frames = self.qmdata.structures
        basis = self.qmdata.basis
        ncore = self.qmdata.ncore
        if use_overlaps:
            S = (
                self.qmdata.overlap_realspace
                if self.qmdata.is_molecule
                else self.qmdata.overlap_kspace
            )
        else:
            S = None

        ard, rhos = compute_atom_resolved_density(
            eigenvectors, frames, basis, ncore, overlaps=S
        )

        to_return = [ard]
        if return_rho:
            to_return.append(rhos)
        if return_eigenvalues:
            to_return.append(eigenvalues)
        if return_eigenvectors:
            to_return.append(eigenvectors)

        return tuple(to_return)

    def group_and_join(
        self,
        batch: List[NamedTuple],
        fields_to_join: Optional[List[str]] = None,
        join_kwargs: Optional[dict] = {
            "different_keys": "union",
            "remove_tensor_name": True,
        },
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
            "fockblocks",
            "overlapblocks",
            "fockrealspace",
            "fockkspace",
            "overlaprealspace",
            "overlapkspace",
            "eigenvalues",
            "atomresolveddensity",
            "density_matrix",
            "features",
            "dipoles",
        ]


def _flattenname(string):
    return "".join("".join(string.split("_")).split(" ")).lower()


def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(
    matrix: torch.Tensor, num_iters: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError(
            f"Input dimension equals {matrix.dim()}, expected {expected_num_dims}"
        )

    if num_iters <= 0:
        raise ValueError(
            f"Number of iteration equals {num_iters}, expected greater than 0"
        )

    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p="fro")
    Y = matrix.div(norm_of_matrix)
    Id = torch.eye(dim, dim, requires_grad=False).to(matrix)
    Z = torch.eye(dim, dim, requires_grad=False).to(matrix)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1).to(matrix)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * Id - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.0]).to(error), atol=1e-5):
            break

    return s_matrix, error
