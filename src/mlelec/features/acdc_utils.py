import numpy as np
from equistore import Labels, TensorBlock, TensorMap
from itertools import product
from .acdc_mini import acdc_standardize_keys, cg_increment, cg_combine, _remove_suffix
from mlelec.utils.symmetry import ClebschGordanReal
from rascaline import SphericalExpansion, SphericalExpansionByPair


def flatten(x):
    # works for tuples of the form ((a,b,c), d) to (a,b,c,d)
    if isinstance(x, tuple):
        return tuple(x[0]) + (x[1],)
    elif isinstance(x, list):
        # list of tuples
        flat_list_tuples = []
        for aa in x:
            flat_list_tuples.append(flatten(aa))
        return flat_list_tuples


def cg_combine(
    x_a,
    x_b,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
    filter_sigma=[-1, 1],
    other_keys_match=None,
    sorted_l="auto",
    mp=False,
):
    """
    modified cg_combine from acdc_mini.py to add the MP contraction, that contracts over NOT the center but the neighbor yielding |rho_j> |g_ij>, can be merged
    """

    # determines the cutoff in the new features
    lmax_a = max(x_a.keys["spherical_harmonics_l"])
    lmax_b = max(x_b.keys["spherical_harmonics_l"])
    if lcut is None:
        lcut = lmax_a + lmax_b

    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(lcut)

    if sorted_l == "auto":
        sorted_l = True
        if "neighbor" in x_b.sample_names or "neighbor" in x_a.sample_names:
            # similar only when combining two rho1i's (not rho1i with gij or |r_ij> with |r_ik>)
            sorted_l = False

    other_keys_a = tuple(
        name
        for name in x_a.keys.names
        if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"]
    )
    other_keys_b = tuple(
        name
        for name in x_b.keys.names
        if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"]
    )
    if mp:
        if other_keys_match is None:
            OTHER_KEYS = [k + "_a" for k in other_keys_a] + [
                k + "_b" for k in other_keys_b
            ]
        else:
            OTHER_KEYS = (
                [
                    k + ("_a" if k in other_keys_b else "")
                    for k in other_keys_a
                    if k not in other_keys_match
                ]
                + [
                    k + ("_b" if k in other_keys_a else "")
                    for k in other_keys_b
                    if k not in other_keys_match
                ]
                + other_keys_match
            )
    else:
        if other_keys_match is None:
            OTHER_KEYS = [k + "_a" for k in other_keys_a] + [
                k + "_b" for k in other_keys_b
            ]
        else:
            OTHER_KEYS = (
                other_keys_match
                + [
                    k + ("_a" if k in other_keys_b else "")
                    for k in other_keys_a
                    if k not in other_keys_match
                ]
                + [
                    k + ("_b" if k in other_keys_a else "")
                    for k in other_keys_b
                    if k not in other_keys_match
                ]
            )

    if x_a.block(0).has_gradient("positions"):
        grad_components = x_a.block(0).gradient("positions").components
    else:
        grad_components = None

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        NU = x_a.keys[0]["order_nu"] + x_b.keys[0]["order_nu"]
        feature_names = (
            tuple(n + "_a" for n in x_a.property_names)
            + ("k_" + str(NU),)
            + tuple(n + "_b" for n in x_b.property_names)
            + ("l_" + str(NU),)
        )

    X_idx = {}
    X_blocks = {}
    X_samples = {}
    X_grad_samples = {}
    X_grads = {}

    for index_a, block_a in x_a:
        lam_a = index_a["spherical_harmonics_l"]
        sigma_a = index_a["inversion_sigma"]
        order_a = index_a["order_nu"]
        properties_a = (
            block_a.properties
        )  # pre-extract this block as accessing a c property has a non-zero cost
        samples_a = block_a.samples
        for index_b, block_b in x_b:
            lam_b = index_b["spherical_harmonics_l"]
            sigma_b = index_b["inversion_sigma"]
            order_b = index_b["order_nu"]
            properties_b = block_b.properties
            samples_b = block_b.samples
            samples_final = samples_b
            b_slice = list(range(len(samples_b)))
            if sorted_l and lam_b < lam_a:
                continue

            if other_keys_match is None:
                OTHERS = tuple(index_a[name] for name in other_keys_a) + tuple(
                    index_b[name] for name in other_keys_b
                )
            else:
                OTHERS = tuple(
                    index_a[k] for k in other_keys_match if index_a[k] == index_b[k]
                )
                if len(OTHERS) < len(other_keys_match):
                    continue
                # adds non-matching keys to build outer product
                if mp:
                    OTHERS = (
                        tuple(
                            index_a[k]
                            for k in other_keys_a
                            if k not in other_keys_match
                        )
                        + OTHERS
                    )
                    OTHERS = (
                        tuple(
                            index_b[k]
                            for k in other_keys_b
                            if k not in other_keys_match
                        )
                        + OTHERS
                    )
                else:
                    OTHERS = OTHERS + tuple(
                        index_a[k] for k in other_keys_a if k not in other_keys_match
                    )
                    OTHERS = OTHERS + tuple(
                        index_b[k] for k in other_keys_b if k not in other_keys_match
                    )

            if mp:
                if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                    center_slice = []
                    smp_a, smp_b = 0, 0
                    while smp_b < samples_b.shape[0]:
                        # print(index_b, samples_b[smp_b][["structure", "center", "neighbor"]], index_a, samples_a[smp_a])
                        idx = [
                            idx
                            for idx, tup in enumerate(samples_a)
                            if tup[0] == samples_b[smp_b]["structure"]
                            and tup[1] == samples_b[smp_b]["neighbor"]
                        ][0]
                        center_slice.append(idx)
                        smp_b += 1
                    center_slice = np.asarray(center_slice)
                #                     print(index_a, index_b, center_slice,  block_a.samples, block_b.samples)
                else:
                    center_slice = slice(None)
            else:
                if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                    neighbor_slice = []
                    smp_a, smp_b = 0, 0
                    while smp_b < samples_b.shape[0]:
                        if (
                            samples_b[smp_b][["structure", "center"]]
                            != samples_a[smp_a]
                        ):
                            if smp_a + 1 < samples_a.shape[0]:
                                smp_a += 1
                        neighbor_slice.append(smp_a)
                        smp_b += 1
                    neighbor_slice = np.asarray(neighbor_slice)
                    print(
                        index_a,
                        index_b,
                        neighbor_slice,
                        block_a.samples[neighbor_slice],
                        block_b.samples,
                    )

                elif "neighbor" in samples_b.names and "neighbor" in samples_a.names:
                    # taking tensor products of gij and gik
                    neighbor_slice = []
                    b_slice = []
                    samples_final = []
                    smp_a, smp_b = 0, 0
                    """
                    while smp_b < samples_b.shape[0]:
                        idx= [idx for idx, tup in enumerate(samples_a) if tup[0] ==samples_b[smp_b]["structure"] and tup[1] == samples_b[smp_b]["center"]]
                        neighbor_slice.extend(idx)
                        b_slice.extend([smp_b]*len(idx))
                        samples_final.extend(flatten(list(product([samples_b[smp_b]],block_a.samples.asarray()[idx][:,-1]))))
                        smp_b+=1
                    """
                    sc_b = (-1, -1)
                    while smp_b < samples_b.shape[0]:
                        # checks if structure index needs updating
                        if (
                            samples_b[smp_b]["center"] != sc_b[1]
                            or samples_b[smp_b]["structure"] != sc_b[0]
                        ):
                            # checks if structure index needs updating
                            sc_b = samples_b[smp_b][["structure", "center"]]
                            idx = np.where(
                                (
                                    samples_b[smp_b]["structure"]
                                    == samples_a["structure"]
                                )
                                & (samples_b[smp_b]["center"] == samples_a["center"])
                            )[0]

                            smp_a_idx = samples_a["neighbor"][idx].view(np.int32)
                            if "pair_id" in block_a.samples.names:
                                smp_a_idx = (
                                    samples_a["pair_id"][idx].view(np.int32),
                                    samples_a["neighbor"][idx].view(np.int32),
                                )
                                smp_a_idx = [
                                    (
                                        smp_a_idx[0][i].view(np.int32),
                                        smp_a_idx[1][i].view(np.int32),
                                    )
                                    for i in range(len(smp_a_idx[0]))
                                ]
                                # smp_a_idx = Labels(["pair_id", "neighbor"], np.array(x, dtype=np.int32))
                        neighbor_slice.extend(idx)
                        b_slice.extend([smp_b] * len(idx))
                        # samples_final.extend(flatten(list(product([samples_b[smp_b]],smp_a_idx))))
                        # samples_final.extend(np.hstack([[tuple(samples_b[smp_b-1])]*8, smp_a_idx[:,np.newaxis] ]) )
                        # print(smp_a_idx, samples_b[smp_b])
                        if "pair_id" in block_a.samples.names:
                            samples_final.extend(
                                [tuple(samples_b[smp_b]) + idx for idx in smp_a_idx]
                            )
                            # print(samples_final)
                        else:
                            samples_final.extend(
                                [tuple(samples_b[smp_b]) + (idx,) for idx in smp_a_idx]
                            )
                        smp_b += 1
                    neighbor_slice = np.asarray(neighbor_slice)
                    if (
                        "pair_id" in block_a.samples.names
                        and "pair_id" in block_b.samples.names
                    ):
                        # print(np.asarray(samples_final).shape)
                        samples_final = Labels(
                            [
                                "structure",
                                "pair_id1",
                                "center",
                                "neighbor_1",
                                "pair_id2",
                                "neighbor_2",
                            ],
                            np.asarray(samples_final, dtype=np.int32),
                        )
                    else:
                        samples_final = Labels(
                            ["structure", "center", "neighbor_1", "neighbor_2"],
                            np.asarray(samples_final, dtype=np.int32),
                        )
                elif "neighbor_1" in samples_b.names:
                    # combining three center feature with rho_{i i1 i2}
                    neighbor_slice = []
                    b_slice = []
                    smp_a, smp_b = 0, 0
                    """
                    while smp_b < samples_b.shape[0]:
                        idx= [idx for idx, tup in enumerate(samples_a) if tup[0] ==samples_b[smp_b]["structure"] and tup[1] == samples_b[smp_b]["center"]]
                        neighbor_slice.extend(idx)
                        b_slice.extend([smp_b]*len(idx))
                        smp_b+=1
                    """
                    sc_b = (-1, -1)
                    while smp_b < samples_b.shape[0]:
                        # checks if structure index needs updating
                        if (
                            samples_b[smp_b]["center"] != sc_b[1]
                            or samples_b[smp_b]["structure"] != sc_b[0]
                        ):
                            # checks if structure index needs updating
                            sc_b = samples_b[smp_b][["structure", "center"]]
                            idx = np.where(
                                (
                                    samples_b[smp_b]["structure"]
                                    == samples_a["structure"]
                                )
                                & (samples_b[smp_b]["center"] == samples_a["center"])
                            )[0]
                        neighbor_slice.extend(idx)
                        b_slice.extend([smp_b] * len(idx))
                        smp_b += 1
                    neighbor_slice = np.asarray(neighbor_slice)
                #                     print(samples_b[b_slice], samples_a[neighbor_slice])

                else:
                    neighbor_slice = slice(None)

            # determines the properties that are in the select list
            sel_feats = []
            sel_idx = []
            sel_feats = (
                np.indices((len(properties_a), len(properties_b))).reshape(2, -1).T
            )

            prop_ids_a = []
            prop_ids_b = []
            for n_a, f_a in enumerate(properties_a):
                prop_ids_a.append(tuple(f_a) + (lam_a,))
            for n_b, f_b in enumerate(properties_b):
                prop_ids_b.append(tuple(f_b) + (lam_b,))
            prop_ids_a = np.asarray(prop_ids_a)
            prop_ids_b = np.asarray(prop_ids_b)
            sel_idx = np.hstack(
                [prop_ids_a[sel_feats[:, 0]], prop_ids_b[sel_feats[:, 1]]]
            )  # creating a tensor product
            if len(sel_feats) == 0:
                continue
            # loops over all permissible output blocks. note that blocks will
            # be filled from different la, lb
            for L in range(np.abs(lam_a - lam_b), 1 + min(lam_a + lam_b, lcut)):
                # determines parity of the block
                S = sigma_a * sigma_b * (-1) ** (lam_a + lam_b + L)
                if not S in filter_sigma:
                    continue
                NU = order_a + order_b
                KEY = (
                    NU,
                    S,
                    L,
                ) + OTHERS
                if not KEY in X_idx:
                    X_idx[KEY] = []
                    X_blocks[KEY] = []
                    X_samples[KEY] = samples_final
                    if grad_components is not None:
                        X_grads[KEY] = []
                        X_grad_samples[KEY] = block_b.gradient("positions").samples

                # builds all products in one go
                if mp:
                    if isinstance(center_slice, slice) or len(center_slice):
                        one_shot_blocks = clebsch_gordan.combine_einsum(
                            block_a.values[center_slice][:, :, sel_feats[:, 0]],
                            block_b.values[:, :, sel_feats[:, 1]],
                            L,
                            combination_string="iq,iq->iq",
                        )

                        if grad_components is not None:
                            raise ValueError("grads not implemented with MP")
                    else:
                        one_shot_blocks = []

                else:
                    if isinstance(neighbor_slice, slice) or len(neighbor_slice):
                        one_shot_blocks = clebsch_gordan.combine_einsum(
                            block_a.values[neighbor_slice][:, :, sel_feats[:, 0]],
                            block_b.values[b_slice][:, :, sel_feats[:, 1]],
                            L,
                            combination_string="iq,iq->iq",
                        )

                        if grad_components is not None:
                            grad_a = block_a.gradient("positions")
                            grad_b = block_b.gradient("positions")
                            grad_a_data = np.swapaxes(grad_a.data, 1, 2)
                            grad_b_data = np.swapaxes(grad_b.data, 1, 2)
                            one_shot_grads = clebsch_gordan.combine_einsum(
                                block_a.values[grad_a.samples["sample"]][
                                    neighbor_slice, :, sel_feats[:, 0]
                                ],
                                grad_b_data[b_slice][..., sel_feats[:, 1]],
                                L=L,
                                combination_string="iq,iaq->iaq",
                            ) + clebsch_gordan.combine_einsum(
                                block_b.values[grad_b.samples["sample"]][b_slice][
                                    :, :, sel_feats[:, 1]
                                ],
                                grad_a_data[neighbor_slice, ..., sel_feats[:, 0]],
                                L=L,
                                combination_string="iq,iaq->iaq",
                            )
                    else:
                        one_shot_blocks = []

                # now loop over the selected features to build the blocks

                X_idx[KEY].append(sel_idx)
                if len(one_shot_blocks):
                    X_blocks[KEY].append(one_shot_blocks)
                if grad_components is not None:
                    X_grads[KEY].append(one_shot_grads)

    # turns data into sparse storage format (and dumps any empty block in the process)
    nz_idx = []
    nz_blk = []
    for KEY in X_blocks:
        L = KEY[2]
        # create blocks
        if len(X_blocks[KEY]) == 0:
            continue  # skips empty blocks
        nz_idx.append(KEY)
        #         print(KEY, X_samples[KEY], len(X_blocks[KEY]) , X_blocks[KEY][0])
        block_data = np.concatenate(X_blocks[KEY], axis=-1)
        sph_components = Labels(
            ["spherical_harmonics_m"],
            np.asarray(range(-L, L + 1), dtype=np.int32).reshape(-1, 1),
        )
        newblock = TensorBlock(
            values=block_data,
            samples=X_samples[KEY],
            components=[sph_components],
            properties=Labels(
                feature_names, np.asarray(np.vstack(X_idx[KEY]), dtype=np.int32)
            ),
        )

        nz_blk.append(newblock)
    X = TensorMap(
        Labels(
            ["order_nu", "inversion_sigma", "spherical_harmonics_l"] + OTHER_KEYS,
            np.asarray(nz_idx, dtype=np.int32),
        ),
        nz_blk,
    )
    return X


def cg_increment(
    x_nu,
    x_1,
    clebsch_gordan=None,
    lcut=None,
    filter_sigma=[-1, 1],
    other_keys_match=None,
    mp=False,
):
    """Specialized version of the CG product to perform iterations with nu=1 features"""

    nu = x_nu.keys["order_nu"][0]

    feature_roots = _remove_suffix(x_1.block(0).properties.names)

    if nu == 1:
        feature_names = (
            tuple(root + "_1" for root in feature_roots)
            + ("l_1",)
            + tuple(root + "_2" for root in feature_roots)
            + ("l_2",)
        )
    else:
        feature_names = (
            tuple(x_nu.block(0).properties.names)
            + ("k_" + str(nu + 1),)
            + tuple(root + "_" + str(nu + 1) for root in feature_roots)
            + ("l_" + str(nu + 1),)
        )

    return cg_combine(
        x_nu,
        x_1,
        feature_names=feature_names,
        clebsch_gordan=clebsch_gordan,
        lcut=lcut,
        filter_sigma=filter_sigma,
        other_keys_match=other_keys_match,
        mp=mp,
    )
