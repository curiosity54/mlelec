# Dataset class  - to load and pass around structures, targets and
# required auxillary data wherever necessary

from typing import List, Dict
import torch
import ase
from metatensor import TensorMap


class Dataset:
    def __init__(
        self,
        structures: List[ase.Atoms],
        target: torch.tensor,
        aux_data: Dict = None,
    ):
        # aux_data could be overlaps for H-learning, Lattice translations etc.
        self.structures = structures
        self.target = target
        self.nstructs = len(structures)
        self.natoms_list = [len(frame) for frame in structures]
        self.species = set([f.number for f in self.structures])

        if aux_data is not None:
            self.aux_data = {}
            for k in aux_data:
                self.aux_data[k] = aux_data[k]

        self.features = None
        self.feat_keys = None

    def set_features(self, feat: TensorMap):
        self.features = feat
        self.feat_keys = self.features.keys()
