# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed

from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion
import ase

from metatensor import TensorMap, TensorBlock, Labels, sort_block
import metatensor.operations as operations
import torch
import numpy as np
import warnings
from mlelec.utils.metatensor_utils import labels_where
from mlelec.features.acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    _pca,
    relabel_keys,
    fix_gij,
    drop_blocks_L,
    block_to_mic_translation,
    _standardize,
)
from typing import List, Optional, Union, Tuple
import tqdm

# TODO: use rascaline.clebsch_gordan.combine_single_center_to_nu when support for multiple centers is added

use_native = True  # True for rascaline


def single_center_features(
    frames, hypers, order_nu, lcut=None, cg=None, device="cpu", **kwargs
):
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(frames, use_native_system=use_native)
    rhoi = rhoi.keys_to_properties(["species_neighbor"])
    # print(rhoi[0].samples)
    rho1i = acdc_standardize_keys(rhoi)
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
        cg = ClebschGordanReal(lmax=L, device=device)
    rho_prev = rho1i
    # compute nu order feature recursively
    for _ in range(order_nu - 2):
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

    rho_x = cg_increment(
        rho_prev,
        rho1i,
        clebsch_gordan=cg,
        lcut=lcut,
        other_keys_match=["species_center"],
        feature_names=kwargs.get("feature_names", None),
    )
    if kwargs.get("pca_final", False):
        warnings.warn("PCA final features")
        rho_x = _pca(rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None))
    return rho_x


def pair_features(
    frames: List[ase.Atoms],
    hypers: dict,
    hypers_pair: dict = None,
    cg=None,
    rhonu_i: TensorMap = None,
    order_nu: Union[
        List[int], int
    ] = None,  # List currently not supported - useful when combining nu on i and nu' on j
    all_pairs: bool = False,
    both_centers: bool = False,
    lcut: int = 3,
    device="cpu",
    kmesh=None,
    T_dict=None,
    counter=None,
    mic=False,
    return_rho0ij=False,
    **kwargs,
):
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
        # cg = ClebschGordanReal(lmax=lcut)

    if hypers_pair is None:
        hypers_pair = hypers

    if all_pairs:    
        repframes = [f.repeat(kmesh[ifr]) for ifr, f in enumerate(frames)]
        hypers_pair["cutoff"] = np.ceil(np.max([np.max(f.get_all_distances(mic = mic)) for f in repframes]))
        warnings.warn(f"Overwriting hyperparameter 'cutoff' to new value {hypers_pair['cutoff']} for all pair feature.")

    calculator = PairExpansion(**hypers_pair)
    rho0_ij = calculator.compute(frames, use_native_system = use_native)

    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)

    if not mic and return_rho0ij:
        return rho0_ij


    print(hypers_pair)
    # ----- MIC mapping ------
    if mic:
        from mlelec.utils.pbc_utils import scidx_from_unitcell, scidx_to_mic_translation

        # generate all the translations and assign values based on mic map
        # assert kmesh is not None, "kmesh must be specified for MIC mapping"
        # we should add the missing samples
        warnings.warn(f"Using kmesh {kmesh} for MIC mapping")
        blocks = []
        for key, block in rho0_ij.items():
            block_species_i = key['species_center']
            block_species_j = key['species_neighbor']
            all_frames = np.unique(block.samples.values[:, 0])
            value_indices = []
            fixed_sample = []


            for ifr in all_frames:
                
                for T_dummy in T_dict[ifr]:
                    for T in T_dict[ifr][T_dummy]:
                        pairs = np.where(counter[ifr][T])

                        for i, j in zip(*pairs):

                            if not all_pairs:
                                ij_distance = np.linalg.norm(frames[ifr].cell.array.T @ np.array(T) + frames[ifr].positions[j] - frames[ifr].positions[i])
                                if ij_distance > hypers_pair['cutoff']:
                                    continue

                            ai, aj = frames[ifr].numbers[i], frames[ifr].numbers[j]
                        
                            if ai != block_species_i or aj != block_species_j:
                                continue
                        
                            x, y, z = T
                            mic_label = Labels(
                                [
                                    "structure",
                                    "center",
                                    "neighbor",
                                    "cell_shift_a",
                                    "cell_shift_b",
                                    "cell_shift_c",
                                ],
                                values=np.asarray([ifr, i, j, x, y, z]).reshape(1, -1),
                            )[0]
                            # print(mic_label)
                            mappedidx = block.samples.position(mic_label)

                            assert isinstance(mappedidx, int), (mappedidx, mic_label, key,)

                            value_indices.append(mappedidx)
                            fixed_sample.append([ifr, i, j, T_dummy[0], T_dummy[1], T_dummy[2], 1])  # Here we still label features with the original T

                            mic_label = Labels(
                                [
                                    "structure",
                                    "center",
                                    "neighbor",
                                    "cell_shift_a",
                                    "cell_shift_b",
                                    "cell_shift_c",
                                ],
                                values=np.asarray([ifr, j, i, -x, -y, -z]).reshape(1, -1),
                            )[0]
                            mappedidx = block.samples.position(mic_label)
                            if ai!=aj:
                                continue
                            assert isinstance(mappedidx, int), (mappedidx, mic_label, key)

                            value_indices.append(mappedidx)
                            fixed_sample.append([ifr, j, i, T_dummy[0], T_dummy[1], T_dummy[2], -1])

            fixed_sample = np.asarray(fixed_sample)
            
            blocks.append(
                sort_block(TensorBlock(
                    values=block.values[value_indices],
                    samples=Labels(
                        block.samples.names + ['sign'],
                        fixed_sample,
                    ),
                    components=block.components,
                    properties=block.properties,
                ), axes = 'samples')
            )

        rho0_ij = TensorMap(keys=rho0_ij.keys, blocks=blocks)

    else:

        blocks = []
        for key, block in rho0_ij.items():
            print(block.values.shape,'1')
            block_species_i = key['species_center']
            block_species_j = key['species_neighbor']
            all_frames = np.unique(block.samples.values[:, 0])
            sample_labels = []
            value_indices = []

            negative_list = []
            for isample, sample in enumerate(block.samples):
                ifr = sample['structure']
                i = sample['center']
                j = sample['neighbor']
                x = sample['cell_shift_a']
                y = sample['cell_shift_b']
                z = sample['cell_shift_c']

                if False: #[i, j, x, y, z] in negative_list: # <<<<<<<< THIS MAKES HALF THE SAMPLES IN FEATURES than in targets ##FIXME pls 
                    continue
                else:
                    value_indices.append(isample)
                    sample_labels.append([ifr, i, j, x, y, z, 1])
                    if not (j==i and x==0 and y==0 and z==0):
                        sample_labels.append([ifr, j, i, x, y, z, -1])
                        negative_list.append([j, i, -x, -y, -z])

                        neg_label = Labels(["structure","center","neighbor","cell_shift_a","cell_shift_b","cell_shift_c",],
                                                values = np.asarray([ifr, j, i, -x, -y, -z]).reshape(1, -1))[0]
                        mappedidx = block.samples.position(neg_label)
                        assert isinstance(mappedidx, int), (mappedidx, neg_label, key)
                        value_indices.append(mappedidx)
            

            sample_labels = np.asarray(sample_labels)
            
            blocks.append(
                sort_block(TensorBlock(
                    values = block.values[value_indices],
                    samples = Labels(
                        block.samples.names + ['sign'],
                        sample_labels,
                    ),
                    components = block.components,
                    properties = block.properties,
                ), axes = 'samples')
            )
            print(blocks[-1].values.shape, '2')
        rho0_ij = TensorMap(keys = rho0_ij.keys, blocks = blocks)
    
    if mic and return_rho0ij:
        return rho0_ij 
    
    if isinstance(order_nu, list):
        assert (
            len(order_nu) == 2
        ), "specify order_nu as [nu_i, nu_j] for correlation orders for i and j respectively"
        order_nu_i, order_nu_j = order_nu
    else:
        assert isinstance(order_nu, int), "specify order_nu as int or list of 2 ints"
        order_nu_i = order_nu

    if not (frames[0].pbc.any()):
        for _ in ["cell_shift_a", "cell_shift_b", "cell_shift_c"]:
            rho0_ij = operations.remove_dimension(rho0_ij, axis="samples", name=_)

    # must compute rhoi as sum of rho_0ij
    if rhonu_i is None:
        rhonu_i = single_center_features(
            frames, order_nu=order_nu_i, hypers=hypers, lcut=lcut, cg=cg, kwargs=kwargs
        )
        # rhonu_i = _standardize(rhonu_i)
    # if not both_centers:
    rhonu_ij = cg_combine(
        rhonu_i,
        rho0_ij,
        clebsch_gordan=cg,
        other_keys_match=["species_center"],
        lcut=lcut,
        feature_names=kwargs.get("feature_names", None),
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
        )
        # combine with rhoi
        # rhonu_nupij = cg_combine(
        #     rhonu_i,
        #     rhonuij,
        #     lcut=lcut,
        #     other_keys_match=["species_center"],
        #     clebsch_gordan=cg,
        #     feature_names=kwargs.get("feature_names", None),
        # )

        return rhonu_nupij


# TODO: MP feature
# elements = np.unique(frames[0].numbers)#np.unique(np.hstack([f.numbers for f in frames]))
# rhoMPi = contract_rho_ij(rhonu_nuijp, elements, rho(NU=nu+nu'+1)i.property_names)
# print("MPi computed")

# rhoMPij = cg_increment(rhoMPi, rho0_ij, lcut=lcut, other_keys_match=["species_center"], clebsch_gordan=cg)


def twocenter_features_periodic_NH(
    single_center: TensorMap, pair: TensorMap, all_pairs = False
) -> TensorMap:

    keys = []
    blocks = []

    for k, b in single_center.items():
        keys.append(tuple(k) + (k["species_center"], 0,))
        # `Try to handle the case of no computed features
        if len(list(b.samples.values)) == 0:
            samples_array = b.samples
        else:
            samples_array = np.asarray(b.samples.values)
            samples_array = np.hstack([samples_array, samples_array[:, -1:]])
        blocks.append(
            TensorBlock(
                samples=Labels(
                    names=b.samples.names
                    + [
                        "neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c"
                    ],
                    values=np.pad(samples_array, ((0, 0), (0, 3))),
                ),
                components=b.components,
                properties=b.properties,
                values=b.values,
            )
        )

    # TODO: why two loops?
    # PAIRS SHOULD NOT CONTRIBUTE to BLOCK TYPE 0
    for (k, b) in pair.items():
        if k["species_center"] == k["species_neighbor"]:  # self translared pairs
            
            idx = np.where(
                (b.samples["center"] == b.samples["neighbor"])
                & (b.samples["cell_shift_a"] != 0)
                & (b.samples["cell_shift_b"] != 0)
                & (b.samples["cell_shift_c"] != 0)
            )[0]

            if len(idx) != 0:
                # SHOULD BE ZERO
                raise ValueError("btype0 should be zero for pair")
        else: 
            print("off-site, different species")


    for k, b in pair.items():
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
            idx_ij = np.where(positive_sign & ( (cell_is_zero & different_atoms) | (~cell_is_zero & avoid_double_counting_atoms)))[0]

            if len(idx_ij) == 0:
                continue

            if len(np.where(b.samples["center"] > b.samples["neighbor"])[0]) == 0:
                print(
                    "Corresponding swapped pair not found",
                    np.array(b.samples.values)[idx_ij],
                )

            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            idx_ji = []
            samplecopy = np.array(b.samples.values[:, :])

            # for smp_up in range(len(idx_up)):
            for idx in idx_ij:
                # Sample values except the MIC cell shifts
                structure, i, j, Tx, Ty, Tz, sign = b.samples.values[idx]

                if i == j == Tx == Ty == Tz == 0:
                    continue
               
                # Sample to symmetrize over
                ji_entry = np.array([structure, j, i, Tx, Ty, Tz, -1])

                # Find the index of the corresponding sample index in the block
                where_ji = np.argwhere(np.all(samplecopy == ji_entry, axis = 1))

                # Ensure that there is only one matching sample
                assert where_ji.shape == (1, 1), (where_ji.shape, where_ji, ji_entry)
                idx_ji.append(where_ji[0, 0])
            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names[:-1],
                        values=np.asarray(
                            b.samples.values[idx_ij][:, :-1]  # We don't keep the "sign" sample dimension since it's always +1
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_ij] + b.values[idx_ji]),
                )
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names[:-1],
                        values=np.asarray(
                            b.samples.values[idx_ij][:, :-1]  # We don't keep the "sign" sample dimension since it's always +1
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_ij] - b.values[idx_ji]),
                )
            )
        elif k["species_center"] != k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            # blocks.append(b.copy())
            blocks.append(TensorBlock(
                values = b.values, 
                components = b.components,
                properties = b.properties,
                samples = Labels(b.samples.names[:-1], np.asarray(b.samples.values)[:,:-1])
                )
                )


    return TensorMap(
        keys=Labels(
            names=pair.keys.names + ["block_type"],
            values=np.asarray(keys),
        ),
        blocks=blocks,
    )


# retain only positive shifts in the end
def twocenter_hermitian_features(
    single_center: TensorMap,
    pair: TensorMap,
) -> TensorMap:
    # keep this function only for molecules - hanldle 000 shift using hermitian PBC - MIC mapping #FIXME
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
                samples_array = np.asarray(b.samples.values)
                samples_array = np.hstack([samples_array, samples_array[:, -1:]])
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names + ["neighbor"],
                        values=samples_array,
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=b.values,
                )
            )

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_up = np.where(b.samples["center"] < b.samples["neighbor"])[0]
            if len(idx_up) == 0:
                continue
            idx_lo = np.where(b.samples["center"] > b.samples["neighbor"])[0]

            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            smp_up, smp_lo = 0, 0
            for smp_up in range(len(idx_up)):
                # ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                ij = b.samples.view(["center", "neighbor"]).values[idx_up[smp_up]]
                for smp_lo in range(smp_up, len(idx_lo)):
                    ij_lo = b.samples.view(["neighbor", "center"]).values[
                        idx_lo[smp_lo]
                    ]
                    # ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]
                    if (
                        b.samples["structure"][idx_up[smp_up]]
                        != b.samples["structure"][idx_lo[smp_lo]]
                    ):
                        raise ValueError(
                            f"Could not find matching ji term for sample {b.samples[idx_up[smp_up]]}"
                        )
                    if tuple(ij) == tuple(ij_lo):
                        idx_lo[smp_up], idx_lo[smp_lo] = idx_lo[smp_lo], idx_lo[smp_up]
                        break

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))

            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx_up]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                )
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx_up]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                )
            )
        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(b.copy())
    keys = np.pad(keys, ((0, 0), (0, 3)))
    return TensorMap(
        keys=Labels(
            names=pair.keys.names
            + ["block_type"]
            + ["cell_shift_a", "cell_shift_b", "cell_shift_c"],
            # + ["cell_shift_a_MIC", "cell_shift_b_MIC", "cell_shift_c_MIC"],
            values=np.asarray(keys, dtype=np.int32),
        ),
        blocks=blocks,
    )

def twocenter_hermitian_features_periodic(
    single_center: TensorMap,
    pair: TensorMap,
    shift: Optional[Tuple[int, int, int]] = None,
    antisymmetric: bool = False,
):
    if shift == [0, 0, 0]:
        return twocenter_hermitian_features(single_center, pair)

    keys = []
    blocks = []

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:  # self translared pairs
            idx = np.where(b.samples["center"] == b.samples["neighbor"])[0]
            if len(idx) == 0:
                continue
            keys.append(tuple(k) + (0,))
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(b.samples.values[idx]),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=b.values[idx],
                )
            )
        else:
            raise NotImplementedError  # Handle periodic case for different species

    for k, b in pair.items():
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_up = np.where(b.samples["center"] < b.samples["neighbor"])[0]
            if len(idx_up) == 0:
                continue
            idx_lo = np.where(b.samples["center"] > b.samples["neighbor"])[0]
            if len(idx_lo) == 0:
                print(
                    "Corresponding swapped pair not found",
                    np.array(b.samples.values)[idx_up],
                )
            # else:
            # print(np.array(b.samples.values)[idx_up], "corresponf to", np.array(b.samples.values)[idx_lo])
            # we need to find the "ji" position that matches each "ij" sample.
            # we exploit the fact that the samples are sorted by structure to do a "local" rearrangement
            smp_up, smp_lo = 0, 0
            for smp_up in range(len(idx_up)):
                # ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                ij = b.samples.view(["center", "neighbor"]).values[idx_up[smp_up]]
                for smp_lo in range(smp_up, len(idx_lo)):
                    ij_lo = b.samples.view(["neighbor", "center"]).values[
                        idx_lo[smp_lo]
                    ]
                    # ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]
                    if (
                        b.samples["structure"][idx_up[smp_up]]
                        != b.samples["structure"][idx_lo[smp_lo]]
                    ):
                        raise ValueError(
                            f"Could not find matching ji term for sample {b.samples[idx_up[smp_up]]}"
                        )
                    if tuple(ij) == tuple(ij_lo):
                        idx_lo[smp_up], idx_lo[smp_lo] = idx_lo[smp_lo], idx_lo[smp_up]
                        break

            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            # print(k.values, b.values.shape, idx_up.shape, idx_lo.shape)
            if not antisymmetric:
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                    )
                )
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                    )
                )
            else:
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                    )
                )
                blocks.append(
                    TensorBlock(
                        samples=Labels(
                            names=b.samples.names,
                            values=np.asarray(b.samples.values[idx_up]),
                        ),
                        components=b.components,
                        properties=b.properties,
                        values=(b.values[idx_up] + b.values[idx_lo]) / np.sqrt(2),
                    )
                )
        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(b.copy())

    # kkeys = [list(k) for k in keys]
    # print([len(k) for k in keys])
    # print(keys[2], keys[3], )
    # print(np.asarray(kkeys).shape   )
    # print(Labels(
    #         names=pair.keys.names + ["block_type"],
    #         values=np.asarray(keys),
    #     ),)
    return TensorMap(
        keys=Labels(
            names=pair.keys.names + ["block_type"],
            values=np.asarray(keys),
        ),
        blocks=blocks,
    )


from mlelec.targets import SingleCenter, TwoCenter
from mlelec.data.dataset import MLDataset


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
