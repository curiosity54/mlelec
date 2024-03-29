# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed

from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion
import ase

# TODO: support SphericalExpansion, PairExpansion calculators from outside of rascaline

from metatensor import TensorMap, TensorBlock, Labels
import metatensor.operations as operations
import torch
import numpy as np
import warnings

from mlelec.features.acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    _pca,
    relabel_key_contract,
    fix_gij,
)
from typing import List, Optional, Union, Tuple

# TODO: use rascaline.clebsch_gordan.combine_single_center_to_nu when support for multiple centers is added


def single_center_features(frames, hypers, order_nu, lcut=None, cg=None, **kwargs):
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(frames)
    rhoi = rhoi.keys_to_properties(["species_neighbor"])
    # print(rhoi[0].samples)
    rho1i = acdc_standardize_keys(rhoi)

    if order_nu == 1:
        return rho1i
    if lcut is None:
        lcut = 10
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal
        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L)
    rho_prev = rho1i
    # compute nu order feature recursively
    for _ in range(order_nu - 2):
        rho_x = cg_combine(
            rho_prev,
            rho1i,
            clebsch_gordan=cg,
            lcut=lcut,
            other_keys_match=["species_center"],
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
    )
    if kwargs.get("pca_final", False):
        warnings.warn("PCA final features")
        rho_x = _pca(
            rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None)
        )
    return rho_x


def pair_features(
    frames: List[ase.Atoms],
    hypers: dict,
    cg=None,
    rhonu_i: TensorMap = None,
    order_nu: Union[List[int], int] = None,
    all_pairs: bool = False,
    both_centers: bool = False,
    lcut: int = 3,
    max_shift = None,
    **kwargs,
):
    if not isinstance(frames, list):
        frames = [frames]
    if lcut is None:    
        lcut = 10
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal
        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L)
        # cg = ClebschGordanReal(lmax=lcut)

    calculator = PairExpansion(**hypers)
    rho0_ij = calculator.compute(frames)

    if all_pairs:
        hypers_allpairs = hypers.copy()
        if max_shift is None and hypers["cutoff"] < np.max(
        [np.max(f.get_all_distances()) for f in frames]
    ):
            hypers_allpairs["cutoff"] = np.ceil(
                np.max([np.max(f.get_all_distances()) for f in frames])
            )
            nmax = int(hypers_allpairs["max_radial"]/ hypers["cutoff"] * hypers_allpairs["cutoff"])
            hypers_allpairs["max_radial"] = nmax
        elif max_shift is not None:
            repframes = [f.repeat(max_shift) for f in frames]
            hypers_allpairs["cutoff"] = np.ceil(
                np.max([np.max(f.get_all_distances()) for f in repframes])
            )
        
            warnings.warn(
                f"Using cutoff {hypers_allpairs['cutoff']} for all pairs feature"
            )
        else:
            warnings.warn(
                f"Using unchanged hypers for all pairs feature"
            )
        calculator_allpairs = PairExpansion(**hypers_allpairs)
        
        rho0_ij = calculator_allpairs.compute(frames)

    # rho0_ij = acdc_standardize_keys(rho0_ij)
    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)

    if not (frames[0].pbc.any()):
        for _ in ["cell_shift_a", "cell_shift_b", "cell_shift_c"]:
            rho0_ij = operations.remove_dimension(rho0_ij, axis="samples", name=_)

    if rhonu_i is None:
        rhonu_i = single_center_features(
            frames, order_nu=order_nu, hypers=hypers, lcut=lcut, cg=cg, kwargs=kwargs
        )
    if not both_centers:
        rhonu_ij = cg_combine(
            rhonu_i,
            rho0_ij,
            clebsch_gordan=cg,
            other_keys_match=["species_center"],
            lcut=lcut,
            feature_names=kwargs.get("feature_names", None),
        )
        return rhonu_ij

    else:
        # build the feature with atom-centered density on both centers
        # rho_ij = rho_i x gij x rho_j
        rhonu_ip = relabel_key_contract(rhonu_i)
        # gji = relabel_key_contract(gij)
        rho0_ji = relabel_key_contract(rho0_ij)

        rhonu_ijp = cg_increment(
            rhonu_ip,
            rho0_ji,
            lcut=lcut,
            other_keys_match=["species_contract"],
            clebsch_gordan=cg,
            mp=True,
        )

        rhonu_nuijp = cg_combine(
            rhonu_i,
            rhonu_ijp,
            lcut=lcut,
            other_keys_match=["species_center"],
            clebsch_gordan=cg,
        )
        return rhonu_nuijp


# TODO: MP feature
# elements = np.unique(frames[0].numbers)#np.unique(np.hstack([f.numbers for f in frames]))
# rhoMPi = contract_rho_ij(rhonu_nuijp, elements, rho(NU=nu+nu'+1)i.property_names)
# print("MPi computed")

# rhoMPij = cg_increment(rhoMPi, rho0_ij, lcut=lcut, other_keys_match=["species_center"], clebsch_gordan=cg)


def twocenter_hermitian_features(
    single_center: TensorMap, pair: TensorMap, 
) -> TensorMap:
    # actually special class of features for Hermitian (rank2 tensor)
    keys = []
    blocks = []
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

    return TensorMap(
        keys=Labels(
            names=pair.keys.names + ["block_type"],
            values=np.asarray(keys, dtype=np.int32),
        ),
        blocks=blocks,
    )


def twocenter_features(single_center: TensorMap, pair: TensorMap) -> TensorMap:
    # no hermitian symmetry
    keys = []
    blocks = []
    for k, b in single_center.items():
        keys.append(
            tuple(k)
            + (
                k["species_center"],
                0,
            )
        )
    pass


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


def compute_features_for_target(dataset: MLDataset, **kwargs):
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
        dataset.structures, hypers, order_nu=2, lcut=hypers["max_angular"]
    )
    if isinstance(dataset.target, SingleCenter):
        features = single
    elif isinstance(dataset.target, TwoCenter):
        pairs = pair_features(
            dataset.structures,
            hypers,
            order_nu=1,
            lcut=hypers["max_angular"],
            feature_names=single[0].properties.names,
        )
        features = twocenter_hermitian_features(single, pairs)
    else:
        raise ValueError(f"Target type {type(dataset.target)} not supported")
    return features
