# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed
from typing import List, Optional, Union, Tuple, Dict
import torch
import numpy as np
import warnings
import ase

import rascaline.torch
from rascaline.torch import SphericalExpansion
from rascaline.torch import SphericalExpansionByPair as PairExpansion

import metatensor.torch as mts
from metatensor.torch import TensorMap, TensorBlock, Labels
import metatensor # FIXME: remove once the sort bug is solved 

from mlelec.features.acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    _pca,
    relabel_keys,
    fix_gij,
    drop_blocks_L,
)
from mlelec.targets import SingleCenter, TwoCenter
from mlelec.data.dataset import MLDataset

# TODO: use rascaline.clebsch_gordan.combine_single_center_to_nu when support for multiple centers is added

use_native = True  # True for rascaline

def single_center_features(
    frames, hypers, order_nu, lcut=None, cg=None, device="cpu", **kwargs
):  
    print(device, 'single center features')
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(rascaline.torch.systems_to_torch(frames), use_native_system = use_native)
    rho1i = acdc_standardize_keys(rhoi)
    rho1i = rho1i.keys_to_properties(["species_neighbor"])
    # print(rhoi[0].samples)
    
    # rho1i = _standardize(rho1i)
    if order_nu == 1:
        # return upto lcut? # TODO: change this maybe
        return drop_blocks_L(rho1i, lcut)
        # return rho1i
    if lcut is None:
        lcut = 10
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax = L, device = device)
    rho_prev = rho1i
    # compute nu order feature recursively
    for _ in range(order_nu - 2):
        # rho_x = cg_increment(
        rho_x = cg_combine(
            rho_prev,
            rho1i,
            clebsch_gordan=cg,
            lcut=lcut,
            other_keys_match=["species_center"],
            device=device,
        )
        rho_prev = _pca(
            rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None)
        )

    # rho_x = cg_increment(
    rho_x = cg_combine(
        rho_prev,
        rho1i,
        clebsch_gordan=cg,
        lcut=lcut,
        other_keys_match=["species_center"],
        feature_names=kwargs.get("feature_names", None),
        device=device,
    )
    if kwargs.get("pca_final", False):
        warnings.warn("PCA final features")
        rho_x = _pca(rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None))
    return rho_x


def pair_features(
    frames: List[ase.Atoms],
    hypers: Dict,
    hypers_pair: Dict = None,
    cg=None,
    rhonu_i: TensorMap = None,
    order_nu: Union[
        List[int], int
    ] = None,  # List - useful when combining nu on i and nu' on j
    all_pairs: bool = False,
    both_centers: bool = False,
    lcut: int = 3,
    device="cpu",
    kmesh=None,
    return_rho0ij=False,
    backend='torch',
    overwrite_cutoff = False,
    **kwargs,
):
    print(device, 'pair features')
    """
    hypers: dictionary of hyperparameters for the pair expansion as in Rascaline
    cg: object of utils:symmetry:ClebschGordanReal
    rhonu_i: TensorMap of single center features
    order_nu: int or list of int, order of the spherical expansion
    """
    
    if not isinstance(frames, list):
        frames = [frames]

    if lcut is None:
        lcut = 10

    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L, device=device)

    if hypers_pair is None:
        hypers_pair = hypers

    if all_pairs:    
        repframes = [f.repeat(kmesh[ifr]) for ifr, f in enumerate(frames)]
        min_cutoff = np.max([np.max(f.get_all_distances(mic = False)) for f in repframes])
        if hypers_pair["cutoff"] < min_cutoff:
            if overwrite_cutoff:
                hypers_pair["cutoff"] = np.ceil(min_cutoff)
                warnings.warn(f"Overwriting hyperparameter 'cutoff' to new value {hypers_pair['cutoff']} for all pair feature.")
            else:
                warnings.warn(f"The selected cutoff is less than the maximum distance as repeated for kmesh ({np.ceil(min_cutoff)}) among atoms in the system!")

    calculator = PairExpansion(**hypers_pair)
    rho0_ij = calculator.compute(rascaline.torch.systems_to_torch(frames), use_native_system = use_native)
    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)
    

    if return_rho0ij:
        return rho0_ij

    blocks = []
    for key, block in rho0_ij.items():
        same_species = key['species_center'] == key['species_neighbor']
        sample_labels = []
        value_indices = []

        # negative_list = []
        for isample, sample in enumerate(block.samples):
            ifr, i, j, x, y, z = sample.values[:6] # [sample[lab] for lab in ['structure', 'center', 'neighbor', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c']]
            same_atoms = i == j
            is_central_cell = x == 0 and y == 0 and z == 0
            
            if False: #[i, j, x, y, z] in negative_list: # <<<<<<<< THIS MAKES HALF THE SAMPLES IN FEATURES than in targets ##FIXME pls 
                continue
            else:
                value_indices.append(isample)
                sample_labels.append([ifr, i, j, x, y, z, 1])
        
                # Look for negative translation for |bt|=1
                if not ((same_atoms and is_central_cell)):
                    if not same_species:
                        continue
                    
                    sample_labels.append([ifr, j, i, x, y, z, -1])
                    # negative_list.append([j, i, -x, -y, -z])

                    # neg_label = Labels(["structure", "center", "neighbor", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                                    #    values = torch.tensor([ifr, j, i, -x, -y, -z]).reshape(1, -1))[0]
                    neg_label = torch.tensor([ifr, j, i, -x, -y, -z])
                    mappedidx = block.samples.position(neg_label)
                    
                    assert isinstance(mappedidx, int), (mappedidx, neg_label, key)
                    value_indices.append(mappedidx)
        

        sample_labels = torch.tensor(sample_labels)
        
        # FIXME: hack to sort the block while waiting for the metatensor.torch.sort to be fixed

        torch_block = TensorBlock(
                values = block.values[value_indices],
                samples = Labels(
                    block.samples.names + ['sign'],
                    sample_labels,
                ),
                components = block.components,
                properties = block.properties,
            )
        
        from mlelec.utils.metatensor_utils import sort_block_hack
        blocks.append(sort_block_hack(torch_block))
        
        # blocks.append(
        #     mts.sort_block(TensorBlock(
        #         values = block.values[value_indices],
        #         samples = Labels(
        #             block.samples.names + ['sign'],
        #             sample_labels,
        #         ),
        #         components = block.components,
        #         properties = block.properties,
        #     ), axes = 'samples')
        # )

    rho0_ij = TensorMap(keys = rho0_ij.keys, blocks = blocks)
    
    if isinstance(order_nu, list):
        assert (
            len(order_nu) == 2
        ), "specify order_nu as [nu_i, nu_j] for correlation orders for i and j respectively"
        order_nu_i, order_nu_j = order_nu
    else:
        assert isinstance(order_nu, int), f"order_nu = {order_nu}. Specify order_nu as int or list of 2 ints"
        order_nu_i = order_nu

###------ removed only for BENZENE TEST -------------
    # # remove periodic boundary conditions if not present
    # if not (frames[0].pbc.any()):
    #     for _ in ["cell_shift_a", "cell_shift_b", "cell_shift_c"]:
    #         rho0_ij = operations.remove_dimension(rho0_ij, axis="samples", name=_)
#################################################

    if rhonu_i is None:
        lmax = hypers["max_angular"]
        rhonu_i = single_center_features(
            frames, order_nu=order_nu_i, hypers=hypers, lcut=lmax, cg=cg, device = device, kwargs=kwargs
        )
    rhonu_ij = cg_combine(
        rhonu_i,
        rho0_ij,
        clebsch_gordan=cg,
        other_keys_match=["species_center"],
        lcut=lcut,
        feature_names=kwargs.get("feature_names", None),
        device=device,
    )
    if not both_centers:
        return rhonu_ij

    else:
        if "order_nu_j" not in locals():
            warnings.warn("nu_j not defined, using nu_i for nu_j as well")
            order_nu_j = order_nu_i
        if order_nu_j != order_nu_i:
            rhonup_j = single_center_features(
                frames,
                order_nu=order_nu_j,
                hypers=hypers,
                lcut=lcut,
                cg=cg,
                device=device,
                kwargs=kwargs,
            )
        else:
            rhonup_j = rhonu_i.copy()

        rhoj = relabel_keys(rhonup_j, "species_neighbor")

        # build rhoj x gij
        rhonu_nupij = cg_combine(
            rhoj,
            rhonu_ij,
            # rhoj,
            lcut=lcut,
            other_keys_match=["species_neighbor"],
            clebsch_gordan=cg,
            mp=True,  # for combining with neighbor
            feature_names=kwargs.get("feature_names", None),
            device=device,
        )

        return rhonu_nupij

def twocenter_features_periodic_NH(
    single_center: TensorMap, pair: TensorMap, all_pairs = False, device = 'cpu'
) -> TensorMap:
    from collections import defaultdict

    keys = []
    blocks = []
    if "cell_shift_a" not in pair.keys.names:
        assert "cell_shift_b" not in pair.keys.names
        assert "cell_shift_c" not in pair.keys.names

    for k, b in single_center.items():
        keys.append(tuple(k) + (k["species_center"], 0,))
        # `Try to handle the case of no computed features
        if len(list(b.samples.values)) == 0:
            samples_array = b.samples
        else:
            samples_array = b.samples.values
            samples_array = torch.hstack([samples_array, samples_array[:, -1:]])
        blocks.append(
            TensorBlock(
                samples = Labels(
                    names = b.samples.names + ["neighbor", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                    values = torch.nn.functional.pad(samples_array, (0, 3, 0, 0)),
                ),
                components = b.components,
                properties = b.properties,
                values = b.values,
            ).to(device = device)
        )

    for k, b in pair.items():
        if all_pairs:
            diff_species= k["species_center"] != k["species_neighbor"]
        else: 
            diff_species = k["species_center"] < k["species_neighbor"]

        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            atom_i = b.samples["center"]
            atom_j = b.samples["neighbor"]
            Tx = b.samples["cell_shift_a"]
            Ty = b.samples["cell_shift_b"]
            Tz = b.samples["cell_shift_c"]
            cell_is_zero = ((Tx == 0) & (Ty == 0) & (Tz == 0))
            positive_sign = b.samples["sign"] == 1

            if all_pairs:
                different_atoms = (atom_i != atom_j)
                avoid_double_counting_atoms = True
            else:
                different_atoms = (atom_i < atom_j)
                avoid_double_counting_atoms = atom_i <= atom_j

            idx_ij = torch.where(positive_sign & ((cell_is_zero & different_atoms) | (~cell_is_zero & avoid_double_counting_atoms)))[0]

            if len(idx_ij) == 0:
                continue

            samplecopy = b.samples.values[:, :]
            block_values = b.values

            f_ijT = {1: defaultdict(lambda: torch.zeros(block_values.shape[1:], device = device)), 
                     -1: defaultdict(lambda: torch.zeros(block_values.shape[1:], device = device))}

            for idx, AijTs in enumerate(samplecopy.tolist()):
                A, i, j, Tx, Ty, Tz, sign = AijTs

                bv = block_values[idx]
                if sign == 1:   
                    f_ijT[1][A, i, j, Tx, Ty, Tz] += bv
                    f_ijT[-1][A, i, j, Tx, Ty, Tz] += bv
                else:
                    f_ijT[1][A, j, i, Tx, Ty, Tz] += bv
                    f_ijT[-1][A, j, i, Tx, Ty, Tz] -= bv

            # for I in f_ijT[1]:
                # print(f_ijT[1][I].norm().item())

            samplelist = samplecopy[idx_ij][:,:-1]
            values_plus1 = []
            values_minus1 = []
            [(values_plus1.append(f_ijT[1][tuple(AijT)]), values_minus1.append(f_ijT[-1][tuple(AijT)]))  for AijT in samplelist.tolist()]

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            
            blocks.append(
                TensorBlock(
                    samples = Labels(
                        names = b.samples.names[:-1],
                        values = samplelist,
                    ),
                    components = b.components,
                    properties = b.properties,
                    values = torch.stack(values_plus1),
                )
            )
            blocks.append(
                TensorBlock(
                    samples = Labels(
                        names = b.samples.names[:-1],
                        values = samplelist,
                    ),
                    components = b.components,
                    properties = b.properties,
                    values = torch.stack(values_minus1),
                )
            )
        
        elif diff_species:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(TensorBlock(
                values = b.values, 
                components = b.components,
                properties = b.properties,
                samples = Labels(b.samples.names[:-1], b.samples.values[:,:-1])).to(device = device))

    return TensorMap(
        keys = Labels(
            names = pair.keys.names + ["block_type"],
            values = torch.tensor(keys),
        ).to(device = device),
        blocks = blocks,
    )


# retain only positive shifts in the end
def twocenter_hermitian_features(
    single_center: TensorMap,
    pair: TensorMap,
) -> TensorMap:
    # actually special class of features for Hermitian (rank2 tensor)
    keys = []
    blocks = []
    if single_center is not None:

        for k, b in single_center.items():
            keys.append(
                tuple(k)
                + (
                    k["species_center"],
                    0,
                )
            )
            # `Try to handle the case of no computed features
            if len(list(b.samples.values)) == 0:
                samples_array = b.samples
            else:
                samples_array = b.samples.values
                samples_array = torch.hstack([samples_array, samples_array[:, -1:]])
            blocks.append(
                TensorBlock(
                    samples = Labels(
                        names = b.samples.names + ["neighbor"],
                        values = samples_array,
                    ),
                    components = b.components,
                    properties = b.properties,
                    values = b.values,
                )
            )

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            
            idx_up = torch.where(b.samples["center"] < b.samples["neighbor"])[0]
            if len(idx_up) == 0:
                continue
            
            idx_lo = torch.where(b.samples["center"] > b.samples["neighbor"])[0]

            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            smp_up, smp_lo = 0, 0
            for smp_up in range(len(idx_up)):
                # ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                ij = b.samples.view(["center", "neighbor"]).values[idx_up[smp_up]]
                for smp_lo in range(smp_up, len(idx_lo)):
                    ij_lo = b.samples.view(["neighbor", "center"]).values[idx_lo[smp_lo]]
                    # ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]

                    if (b.samples["structure"][idx_up[smp_up]] != b.samples["structure"][idx_lo[smp_lo]]):
                        raise ValueError(f"Could not find matching ji term for sample {b.samples[idx_up[smp_up]]}")
                    
                    if tuple(ij) == tuple(ij_lo):
                        idx_lo[smp_up], idx_lo[smp_lo] = idx_lo[smp_lo], idx_lo[smp_up]
                        break

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))

            blocks.append(
                TensorBlock(
                    samples = Labels(names = b.samples.names, values = b.samples.values[idx_up]),
                    components = b.components,
                    properties = b.properties,
                    values = (b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                )
            )

            blocks.append(
                TensorBlock(
                    samples = Labels(
                        names = b.samples.names,
                        values = b.samples.values[idx_up],
                    ),
                    components = b.components,
                    properties = b.properties,
                    values = (b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                )
            )

        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(b.clone())

    keys = np.pad(keys, ((0, 0), (0, 3)))
    return TensorMap(
        keys = Labels(names = pair.keys.names + ["block_type"] + ["cell_shift_a", "cell_shift_b", "cell_shift_c"],
                      values = torch.tensor(keys, dtype = torch.int32)),
        blocks = blocks,
    )

def compute_features_for_target(dataset: MLDataset, device=None, **kwargs):
    hypers = kwargs.get("hypers", None)
    # if dataset.molecule_data.pbc:
    if hypers is None:
        print("Computing features with default hypers")
        hypers = {
            "cutoff": 4.0,
            "max_radial": 6,
            "max_angular": 3,
            "atomic_gaussian_width": 0.3,
            "center_atom_weight": 1,
            "radial_basis": {"Gto": {}},
            "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
        }
    single = single_center_features(
        dataset.structures,
        hypers,
        order_nu=kwargs.get("order_nu", 2),
        lcut=hypers["max_angular"],
        device=device,
    )
    if isinstance(dataset.target, SingleCenter):
        features = single
    elif isinstance(dataset.target, TwoCenter) or issubclass(dataset.target, TwoCenter):
        pairs = pair_features(
            dataset.structures,
            hypers,
            order_nu=kwargs.get("order_nu_pair", 1),
            lcut=hypers["max_angular"],
            feature_names=single[0].properties.names,
            device=device,
            both_centers=kwargs.get("both_centers", False),
        )
        features = twocenter_hermitian_features(single, pairs)
    else:
        raise ValueError(f"Target type {type(dataset.target)} not supported")
    return features

from mlelec.data.dataset import PySCFPeriodicDataset
def compute_features(dataset: PySCFPeriodicDataset,
                     hypers_atom: dict,
                     lcut: int,
                     hypers_pair: Optional[dict] = None,
                     both_centers: Optional[bool] = False,
                     all_pairs: Optional[bool] = False, 
                     device: Optional[str] = 'cpu',
                     **kwargs):
    
    # TODO: is this the right place/format for this function?
    import time
    if hypers_pair is None:
        hypers_pair = hypers_atom
    return_rho0ij = kwargs.get("return_rho0ij", False)
    
    # now = time.time()
    rhoij = pair_features(dataset.structures, hypers_atom, hypers_pair, order_nu = 1, all_pairs = all_pairs, both_centers = both_centers,
                          kmesh = dataset.kmesh, device = device, lcut = lcut, return_rho0ij = return_rho0ij)  
    # print(f'pair in {now-time.time()}')
    if both_centers and not return_rho0ij:
        NU = 3
    else:
        NU = 2
    rhonui = single_center_features(dataset.structures, hypers_atom, order_nu = NU, lcut = lcut, device = device, feature_names = rhoij.property_names)

    # return rhonui, rhoij
    # print(f'atom in {now-time.time()}')
    hfeat = twocenter_features_periodic_NH(single_center = rhonui, pair = rhoij, all_pairs = all_pairs, device = device)
    # print(f'symm in {now-time.time()}')
    return hfeat


# def twocenter_features_periodic_NH_OLD(
#     single_center: TensorMap, pair: TensorMap, all_pairs = False, device= None
# ) -> TensorMap:
#     print('i am here ')

#     keys = []
#     blocks = []
#     if "cell_shift_a" not in pair.keys.names:
#         assert "cell_shift_b" not in pair.keys.names
#         assert "cell_shift_c" not in pair.keys.names
#         # return twocenter_hermitian_features(single_center, pair)

#     for k, b in single_center.items():
#         keys.append(tuple(k) + (k["species_center"], 0,))
#         # `Try to handle the case of no computed features
#         if len(list(b.samples.values)) == 0:
#             samples_array = b.samples
#         else:
#             samples_array = np.asarray(b.samples.values)
#             samples_array = np.hstack([samples_array, samples_array[:, -1:]])
#         blocks.append(
#             TensorBlock(
#                 samples=Labels(
#                     names=b.samples.names
#                     + [
#                         "neighbor",
#                         "cell_shift_a",
#                         "cell_shift_b",
#                         "cell_shift_c"
#                     ],
#                     values=np.pad(samples_array, ((0, 0), (0, 3))),
#                 ),
#                 components=b.components,
#                 properties=b.properties,
#                 values=b.values,
#             )
#         )

#     # TODO: why two loops?
#     # PAIRS SHOULD NOT CONTRIBUTE to BLOCK TYPE 0
#     # for (k, b) in pair.items():
#     #     if k["species_center"] == k["species_neighbor"]:  # self translared pairs
            
#         #     idx = np.where(
#         #           (b.samples["center"] == b.samples["neighbor"])
#         #         & (b.samples["cell_shift_a"] == 0)
#         #         & (b.samples["cell_shift_b"] == 0)
#         #         & (b.samples["cell_shift_c"] == 0)
#         #     )[0]
#         #     print(len(idx))
#         #     if len(idx) != 0:
#         #         # SHOULD BE ZERO
#         #         raise ValueError("btype0 should be zero for pair", b.samples.values[idx])
#         # else: 
#         #     print("off-site, different species")


#     for k, b in pair.items():
#         if all_pairs:
#             diff_species= k["species_center"] != k["species_neighbor"]
#         else: 
#             diff_species = k["species_center"] < k["species_neighbor"]

#         if k["species_center"] == k["species_neighbor"]:
#             # off-site, same species
#             atom_i = b.samples["center"]
#             atom_j = b.samples["neighbor"]
#             Tx = b.samples["cell_shift_a"]
#             Ty = b.samples["cell_shift_b"]
#             Tz = b.samples["cell_shift_c"]
#             cell_is_zero = ((Tx == 0) & (Ty == 0) & (Tz == 0))
#             positive_sign = b.samples["sign"] == 1

#             if all_pairs:
#                 different_atoms = (atom_i != atom_j)
#                 avoid_double_counting_atoms = True
#             else:
#                 different_atoms = (atom_i < atom_j)
#                 avoid_double_counting_atoms = atom_i <= atom_j
#             idx_ij = np.where(positive_sign & ( (cell_is_zero & different_atoms) | (~cell_is_zero & avoid_double_counting_atoms)))[0]

#             if len(idx_ij) == 0:
#                 continue

#             # if len(np.where(b.samples["center"] > b.samples["neighbor"])[0]) == 0:
#             #     print(
#             #         "Corresponding swapped pair not found",
#             #         np.array(b.samples.values)[idx_ij],
#             #     )

#             # we need to find the "ji" position that matches each "ij" sample.
#             # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
#             idx_ji = []
#             samplecopy = np.array(b.samples.values[:, :])

#             # for smp_up in range(len(idx_up)):
#             for idx in idx_ij:
#                 structure, i, j, Tx, Ty, Tz, sign = b.samples.values[idx]

#                 if i == j == Tx == Ty == Tz == 0:
#                     continue
               
#                 # Sample to symmetrize over
#                 ji_entry = np.array([structure, j, i, Tx, Ty, Tz, -1])

#                 # Find the index of the corresponding sample index in the block
#                 where_ji = np.argwhere(np.all(samplecopy == ji_entry, axis = 1))

#                 # Ensure that there is only one matching sample
#                 assert where_ji.shape == (1, 1), (where_ji.shape, where_ji, ji_entry)
#                 idx_ji.append(where_ji[0, 0])
#             keys.append(tuple(k) + (1,))
#             keys.append(tuple(k) + (-1,))
            
#             blocks.append(
#                 TensorBlock(
#                     samples=Labels(
#                         names=b.samples.names[:-1],
#                         values=np.asarray(
#                             b.samples.values[idx_ij][:, :-1]  # We don't keep the "sign" sample dimension since it's always +1
#                         ),
#                     ),
#                     components=b.components,
#                     properties=b.properties,
#                     values=(b.values[idx_ij] + b.values[idx_ji]),
#                 )
#             )
#             blocks.append(
#                 TensorBlock(
#                     samples=Labels(
#                         names=b.samples.names[:-1],
#                         values=np.asarray(
#                             b.samples.values[idx_ij][:, :-1]  # We don't keep the "sign" sample dimension since it's always +1
#                         ),
#                     ),
#                     components=b.components,
#                     properties=b.properties,
#                     values=(b.values[idx_ij] - b.values[idx_ji]),
#                 )
#             )
        
#         elif diff_species:
#             # off-site, different species
#             keys.append(tuple(k) + (2,))
#             # blocks.append(b.copy())
#             blocks.append(TensorBlock(
#                 values = b.values, 
#                 components = b.components,
#                 properties = b.properties,
#                 samples = Labels(b.samples.names[:-1], np.asarray(b.samples.values)[:,:-1])
#                 )
#                 )


#     return TensorMap(
#         keys=Labels(
#             names=pair.keys.names + ["block_type"],
#             values=np.asarray(keys),
#         ),
#         blocks=blocks,
#     )
