# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed

from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion
import ase

from metatensor import TensorMap, TensorBlock, Labels
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
    max_shift=None,
    device="cpu",
    kmesh=None,
    desired_shifts=None,
    counter=None,
    mic=False,
    return_rho0ij=False,
    where_Ts_allowed=None,
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
    calculator = PairExpansion(**hypers_pair)
    rho0_ij = calculator.compute(frames, use_native_system=use_native)
    factor = 1

    if all_pairs:
        if mic:
            factor = 1  # we only need half the cell - so we use half the cutoff
        hypers_allpairs = hypers_pair.copy()
        if max_shift is None and hypers["cutoff"] < np.max(
            [np.max(f.get_all_distances(mic=True)) / factor for f in frames]
        ):
            hypers_allpairs["cutoff"] = np.ceil(
                np.max([np.max(f.get_all_distances(mic=mic)) / factor for f in frames])
            )

        elif max_shift is not None:
            repframes = [f.repeat(max_shift) for f in frames]
            hypers_allpairs["cutoff"] = np.ceil(
                np.max(
                    [np.max(f.get_all_distances(mic=mic)) / factor for f in repframes]
                )
            )
            # FIXME - miic = mic
            warnings.warn(
                f"Using cutoff {hypers_allpairs['cutoff']} for all pairs feature"
            )
        else:
            warnings.warn(f"Using unchanged hypers for all pairs feature")
        print("hypers_pair", hypers_allpairs)
        calculator_allpairs = PairExpansion(**hypers_allpairs)

        rho0_ij = calculator_allpairs.compute(frames, use_native_system=use_native)

    # rho0_ij = acdc_standardize_keys(rho0_ij)
    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)
    if not mic and return_rho0ij:
        return rho0_ij
    # ----- MIC mapping ------
    if mic:
        from mlelec.utils.pbc_utils import scidx_from_unitcell, scidx_to_mic_translation

        # generate all the translations and assign values based on mic map
        assert kmesh is not None, "kmesh must be specified for MIC mapping"
        # we should add the missing samples
        warnings.warn(f"Using kmesh {kmesh} for MIC mapping")
        blocks = []
        blocks_minus = []

        for key, block in rho0_ij.items():

            all_frIJ = np.unique(block.samples.values[:, :3], axis=0)
            value_indices = []
            value_indices_minus = []
            fixed_sample = []
            fixed_sample_minus = []
            for ifr, i, j in all_frIJ:
                # Should we instead loop over the keys of H_T_fix- No
                # because we need to associate feats labelled with T to T[iT+1] where T_is not allowd
                T_keys = list(counter.keys())
                for iT, T in enumerate(T_keys):
                    if i == j and list(T) == [0, 0, 0]:
                        continue
                    T_is_allowed = where_Ts_allowed[T]
                    if not T_is_allowed:
                        print("Skipping", T)
                        continue

                    if counter[T][i, j] != 0:
                        x, y, z = T
                    else:
                        assert counter[T_keys[iT + 1]][i, j] != 0, (
                            T_keys[iT + 1],
                            i,
                            j,
                            counter[T_keys[iT + 1]],
                        )
                        x, y, z = T_keys[iT + 1]

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
                    mappedidx = block.samples.position(mic_label)

                    assert isinstance(mappedidx, int), (mappedidx, mic_label)
                    value_indices.append(mappedidx)
                    fixed_sample.append(
                        [
                            ifr,
                            i,
                            j,
                            T[0],  # Here we still label features with the original T
                            T[1],
                            T[2],
                        ]
                    )
                    # print("1 adding", ifr, i, j, T[0], T[1], T[2], x, y, z)
                    if [x, y, z] != [0, 0, 0]:
                        mic_label = Labels(
                            [
                                "structure",
                                "center",
                                "neighbor",
                                "cell_shift_a",
                                "cell_shift_b",
                                "cell_shift_c",
                            ],
                            values=np.asarray(
                                [
                                    ifr,
                                    j,
                                    i,
                                    -x,
                                    -y,
                                    -z,
                                ]
                            ).reshape(1, -1),
                        )[0]
                        mappedidx = block.samples.position(mic_label)

                        assert isinstance(mappedidx, int), (mappedidx, mic_label)
                        value_indices.append(mappedidx)
                        fixed_sample.append(
                            [
                                ifr,
                                j,
                                i,
                                -T[0],
                                -T[1],
                                -T[2],
                            ]
                        )
            # print(len(value_indices), len(fixed_sample))
            value_indices, value_pos = np.unique(value_indices, return_index=True)
            # print(np.unique(np.asarray(fixed_sample)[value_pos], axis=0).shape)
            fixed_sample = np.asarray(fixed_sample)[value_pos]
            
            blocks.append(
                TensorBlock(
                    values=block.values[value_indices],
                    samples=Labels(
                        block.samples.names,
                        fixed_sample,
                    ),
                    components=block.components,
                    properties=block.properties,
                )
            )

        rho0_ij = TensorMap(keys=rho0_ij.keys, blocks=blocks)
        # rho0_ij_minus = TensorMap(keys=rho0_ij.keys, blocks=blocks_minus)
    
    if mic and return_rho0ij:
        return rho0_ij#, rho0_ij_minus   
    
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
    single_center: TensorMap, pair: TensorMap, shift=None
) -> TensorMap:
    # no hermitian symmetry
    # if shift == [0, 0, 0]:
    #     return twocenter_hermitian_features(single_center, pair)

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
                    names=b.samples.names
                    + [
                        "neighbor",
                        "cell_shift_a",
                        "cell_shift_b",
                        "cell_shift_c",
                        # "cell_shift_a_MIC",
                        # "cell_shift_b_MIC",
                        # "cell_shift_c_MIC",
                    ],
                    values=np.pad(samples_array, ((0, 0), (0, 3))),
                ),
                components=b.components,
                properties=b.properties,
                values=b.values,
            )
        )

    # PAIRS SHOULD NOT CONTRIBUTE to BLOCK TYPE 0
    for (k, b) in pair.items():
        if k["species_center"] == k["species_neighbor"]:  # self translared pairs
            # idx = []
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
            raise NotImplementedError  # Handle periodic case for different species

    for k, b in pair.items():
        positive_shifts_idx = []
        if k["species_center"] == k["species_neighbor"]:
            # off-site, same species
            idx_ij = np.where(
                (b.samples["center"] <= b.samples["neighbor"])
                # & (b.samples["cell_shift_a"] >= 0)
                # & (b.samples["cell_shift_b"] >= 0)
                # & (b.samples["cell_shift_c"] >= 0)
            )[0]

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
            samplecopy = np.array(b.samples.values[:, :6])

            # for smp_up in range(len(idx_up)):
            for idx in idx_ij:
                # Sample values except the MIC cell shifts
                structure, i, j, Tx, Ty, Tz = b.samples.values[idx]
                # Invert i, j, and translation vector to define new sample
                ji_entry = np.array([structure, j, i, -Tx, -Ty, -Tz])

                # Find the index of the corresponding sample index in the block
                where_ji = np.argwhere(np.all(samplecopy == ji_entry, axis=1))

                # Ensure that there is only one matching sample
                assert where_ji.shape == (1, 1), where_ji.shape
                idx_ji.append(where_ji[0, 0])
            keys.append(tuple(k) + (1,))
            keys.append(tuple(k) + (-1,))
            # print(k.values, b.values.shape, idx_up.shape, idx_lo.shape)

            # print("SHAPE", np.shape(idx_up), b.samples.values.shape)
            # print("--------------------------------------")
            # print("l", k["spherical_harmonics_l"])
            # print(list(np.array(b.samples.values[idx_ij])[:3]))
            # print(list(np.array(b.values[idx_ij])[:3]))
            # print(list(np.array(b.samples.values[idx_ji])[:3]))
            # print(list(np.array(b.values[idx_ji])[:3]))
            # normdiff = torch.norm(b.values[idx_ij] - b.values[idx_ji], dim=(1, 2))
            # wherediff = torch.where(normdiff != 0)
            # assert torch.norm(b.values[idx_ij] - b.values[idx_ji]) == 0, (
            #     b.samples.values[idx_ij][wherediff],
            #     b.samples.values[idx_ji][wherediff],
            #     torch.norm(b.values[idx_ij][wherediff] + b.values[idx_ji][wherediff]),
            # )
            # print("\n\n")
            sample_label_ij = Labels(
                b.samples.names,
                # np.array([[0, 0, 1, 1, 4, 0]),
                # np.array([[0, 0, 1, 7, 1, 0]]),
                # np.array([[0, 0, 0, 4, 0, 0]]),
                # np.array([[0, 0, 0, 1, 0, 0]]),
                np.array([[0, 0, 1, 4, 0, 0]]),
            )
            sample_label_ji = Labels(
                b.samples.names,
                # np.array([[0, 1, 0, -1, -4, 0, 0, 0, 0]]),
                # np.array([[0, 1, 0, -7, -1, 0, 0, 0, 0]]),
                # np.array([[0, 0, 0, -4, 0, 0, 0, 0, 0]]),
                # np.array([[0, 0, 0, -1, 0, 0, 0, 0, 0]]),
                np.array([[0, 1, 0, -4, 0, 0]]),
            )
            pos_ij = np.where(
                np.all(b.samples.values[:, :6] == sample_label_ij.values[:, :6], axis=1)
            )[0]

            # pos_ij = b.samples.position(sample_label_ij[0])
            pos_ji = np.where(
                np.all(b.samples.values[:, :6] == sample_label_ji.values[:, :6], axis=1)
            )[0]

            print("l", k["spherical_harmonics_l"])
            print("k", k.values)
            print(
                "values",
                torch.norm(
                    b.values[pos_ij]
                    - (-1) ** k["spherical_harmonics_l"] * b.values[pos_ji]
                ),
                torch.norm(b.values[pos_ij] - b.values[pos_ji]),
                torch.norm(b.values[pos_ij] + b.values[pos_ji]),
                # b.values[pos_ij],
                # b.values[pos_ji],
                torch.norm(b.values[pos_ij]),
                torch.norm(b.values[pos_ji]),
                # pos_ij,
                b.samples.values[pos_ij],
                # pos_ji,
                b.samples.values[pos_ji],
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(
                            b.samples.values[idx_ij]  # [positive_shifts_idx]
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                    # values=b.values[idx_up],
                    values=(b.values[idx_ij] + b.values[idx_ji]),
                    # / 2,  # idx_lo
                )
            )
            blocks.append(
                TensorBlock(
                    samples=Labels(
                        names=b.samples.names,
                        values=np.asarray(
                            b.samples.values[idx_ij]  # [positive_shifts_idx]
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                    # values=b.values[idx_ji],  ##[idx_ji],
                    values=(b.values[idx_ij] - b.values[idx_ji]),
                )
            )

        elif k["species_center"] < k["species_neighbor"]:
            # off-site, different species
            keys.append(tuple(k) + (2,))
            blocks.append(b.copy())

    return TensorMap(
        keys=Labels(
            names=pair.keys.names + ["block_type"],
            values=np.asarray(keys),
        ),
        blocks=blocks,
    )


# retain only positive shifts in the end


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
