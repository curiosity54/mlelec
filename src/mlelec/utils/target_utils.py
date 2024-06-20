from typing import Optional, Union

import torch

import metatensor.torch as mts
from metatensor.torch import TensorMap, Labels

from mlelec.data.dataset import QMDataset
from mlelec.utils.pbc_utils import matrix_to_blocks
from mlelec.utils.twocenter_utils import _to_coupled_basis

def get_targets(dataset: QMDataset,
                cutoff: Optional[Union[int,float,None]] = None, 
                target: Optional[str] = 'fock', 
                all_pairs: Optional[bool] = False, 
                sort_orbs: Optional[bool] = True,
                skip_symmetry: Optional[bool] = False,
                device: Optional[str] = "cpu", 
                matrix = None,
                orbitals_to_properties = False
                ):
    
    blocks = matrix_to_blocks(dataset, device = device, cutoff = cutoff, all_pairs = all_pairs, target = target, sort_orbs = sort_orbs, matrix = matrix)
    coupled_blocks = _to_coupled_basis(blocks, skip_symmetry = skip_symmetry, device = device, translations = True)

    blocks = mts.sort(blocks)
    coupled_blocks = mts.sort(coupled_blocks)

    if orbitals_to_properties:
        keys = []
        tblocks= []
        for k, b in coupled_blocks.items(): 
            li, lj, L = k['l_i'], k['l_j'], k['L']
            inversion_sigma = (-1) ** (li + lj + L)
            keys.append(torch.cat((k.values, torch.tensor([inversion_sigma]))))
            tblocks.append(b.copy())
        coupled_blocks = TensorMap(Labels(k.names+['inversion_sigma'], torch.stack(keys)), tblocks)
        coupled_blocks = coupled_blocks.keys_to_properties(['n_i', 'l_i',  'n_j','l_j'])
    
    return blocks, coupled_blocks