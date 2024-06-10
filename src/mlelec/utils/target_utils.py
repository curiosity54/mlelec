from typing import Optional, Union

import metatensor.torch as mts

from mlelec.data.dataset import PySCFPeriodicDataset
from mlelec.utils.pbc_utils import matrix_to_blocks
from mlelec.utils.twocenter_utils import _to_coupled_basis

from mlelec.utils.metatensor_utils import sort_hack

def get_targets(dataset: PySCFPeriodicDataset,
                cutoff: Optional[Union[int,float,None]] = None, 
                target: Optional[str] = 'fock', 
                all_pairs: Optional[bool] = False, 
                sort_orbs: Optional[bool] = True,
                skip_symmetry: Optional[bool] = False,
                device: Optional[str] = "cpu", 
                ):
    
    blocks = matrix_to_blocks(dataset, device = device, cutoff = cutoff, all_pairs = all_pairs, target = target, sort_orbs = sort_orbs)
    coupled_blocks = _to_coupled_basis(blocks, skip_symmetry = skip_symmetry, device = device, translations = True)

    # TODO: when the sort bug is restored, uncomment the next two lines
    # blocks = mts.sort(blocks)
    # coupled_blocks = mts.sort(coupled_blocks)

    # TODO: for now, HACK
    blocks = sort_hack(blocks).to(device)
    coupled_blocks = sort_hack(coupled_blocks).to(device)

    # for k, b in blocks.items():

    # blocks = blocks.to(arrays='numpy')
    # blocks = mts.sort(blocks)
    # blocks = blocks.to(arrays='torch')
    
    # coupled_blocks = coupled_blocks.to(arrays='numpy')
    # coupled_blocks = coupled_blocks.to(arrays='torch')
    
    return blocks, coupled_blocks