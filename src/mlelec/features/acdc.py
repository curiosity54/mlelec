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

from mlelec.features.acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    _pca,
    relabel_key_contract,
)
from typing import List, Optional, Union

# TODO: use rascaline.clebsch_gordan.combine_single_center_to_nu when support for multiple centers is added


def single_center_features(frames, hypers, order_nu, lcut=None, cg=None, **kwargs):
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(frames)
    rhoi = rhoi.keys_to_properties(["species_neighbor"])
    rho1i = acdc_standardize_keys(rhoi)

    if order_nu == 1:
        return rho1i
    if lcut is None:
        lcut = 10
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal

        cg = ClebschGordanReal(lmax=lcut)
    rho_prev = rho1i
    # compute nu order feature recursively
    for _ in range(order_nu - 2):
        rho_x = cg_increment(
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
    **kwargs,
):
    if cg is None:
        from mlelec.utils.symmetry import ClebschGordanReal

        cg = ClebschGordanReal(lmax=lcut)

    calculator = PairExpansion(**hypers)
    rho0_ij = calculator.compute(frames)
    if not (frames[0].pbc.any()):
        for _ in ["cell_shift_a", "cell_shift_b", "cell_shift_c"]:
            rho0_ij = operations.remove_dimension(rho0_ij, axis="samples", name=_)

    if all_pairs and hypers["interaction_cutoff"] < np.max(
        [np.max(f.get_all_distances()) for f in frames]
    ):
        hypers_allpairs = hypers.copy()
        hypers_allpairs["interaction_cutoff"] = np.ceil(
            np.max([np.max(f.get_all_distances()) for f in frames])
        )
        calculator_allpairs = PairExpansion(hypers_allpairs)
        rho0_ij = calculator_allpairs.compute(frames)

    # rho0_ij = acdc_standardize_keys(rho0_ij)
    import metatensor

    blocks = []
    for k, b in rho0_ij.items():
        bl = metatensor.sort_block(b)
        # print(b.samples, bl.samples)
        blocks.append(bl)
    rho0_ij = TensorMap(rho0_ij.keys, blocks)

    rho0_ij = acdc_standardize_keys(rho0_ij)
    if rhonu_i is None:
        rhonu_i = single_center_features(
            frames, order_nu=order_nu, hypers=hypers, lcut=lcut, cg=cg, kwargs=kwargs
        )
    print(rhonu_i.keys, rho0_ij.keys)
    if not both_centers:
        rhonu_ij = cg_combine(
            rhonu_i,
            rho0_ij,
            clebsch_gordan=cg,
            other_keys_match=["species_center"],
            lcut=lcut,
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
    single_center: TensorMap, pair: TensorMap
) -> TensorMap:
    # actually special class of features for Hermitian (rank2 tensor)
    keys = []
    blocks = []
    # central blocks
    for k, b in single_center:
        keys.append(
            tuple(k)
            + (
                k["species_center"],
                0,
            )
        )
        # ===
        # `Try to handle the case of no computed features
        if len(b.samples.tolist()) == 0:
            samples_array = b.samples
        else:
            samples_array = np.vstack(b.samples.tolist())
            samples_array = np.hstack([samples_array, samples_array[:, -1:]]).astype(
                np.int32
            )
        # ===
        # samples_array = np.vstack(b.samples.tolist())
        blocks.append(
            TensorBlock(
                samples=Labels(
                    names=b.samples.names + ("neighbor",),
                    values=samples_array,
                    # values=np.asarray(
                    #     np.hstack([samples_array, samples_array[:, -1:]]),
                    #     dtype=np.int32,
                    # ),
                ),
                components=b.components,
                properties=b.properties,
                values=b.values,
            )
        )

    for k, b in pair:
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
                ij = b.samples[idx_up[smp_up]][["center", "neighbor"]]
                for smp_lo in range(smp_up, len(idx_lo)):
                    ij_lo = b.samples[idx_lo[smp_lo]][["neighbor", "center"]]
                    if (
                        b.samples[idx_up[smp_up]]["structure"]
                        != b.samples[idx_lo[smp_lo]]["structure"]  # noqa: W503
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
                        values=np.asarray(b.samples[idx_up].tolist(), dtype=np.int32),
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
                        values=np.asarray(b.samples[idx_up].tolist(), dtype=np.int32),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=(b.values[idx_up] - b.values[idx_lo]) / np.sqrt(2),
                )
            )
        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(
                TensorBlock(
                    samples=b.samples,
                    components=b.components,
                    properties=b.properties,
                    values=b.values.copy(),
                )
            )

    return TensorMap(
        keys=Labels(
            names=pair.keys.names + ("block_type",),
            values=np.asarray(keys, dtype=np.int32),
        ),
        blocks=blocks,
    )


def twocenter_features():
    # no hermitian symmetry
    pass
