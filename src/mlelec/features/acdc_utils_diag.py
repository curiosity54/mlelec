import numpy as np
from metatensor import Labels, TensorBlock, TensorMap
from itertools import product
import re
import torch
from metatensor import sort_block
from typing import List, Optional, Union
import metatensor
import metatensor.operations as operations

from mlelec.utils.symmetry import ClebschGordanReal
from tqdm.notebook import tqdm  # <<< change this

def fix_gij(rho0_ij):
    """
    - Add self pairs
    - Sort samples
    - Add species_neighbor to properties
    """
    blocks = []
    for key, block in rho0_ij.items():
        try:
            neigh_species = key["second_atom_type"]  # add species_neighbor to  properties
        except:
            try:
                neigh_species = key["species_atom_2"] # OLD - rascaline
            except:
                raise ValueError("Key does not contain 'second_atom_type' or 'species_atom_2'")
        bprops = np.concatenate(
            (
                np.array([[neigh_species]] * len(block.properties.values)),
                block.properties.values,
            ),
            axis=1,
        )
        properties = Labels(["species_neighbor_1"] + block.properties.names, bprops)
        try:
            L = key["o3_lambda"]
            lname = "o3_lambda"
        except:
            try:
                L = key["spherical_harmonics_l"]
                lname = "spherical_harmonics_l"
            except:
                raise ValueError("Key does not contain 'o3_lambda' or 'spherical_harmonics_l'")
        if L != 0:
            val = list(key.values)
            val[key.names.index(lname)] = 0
            key_copy = Labels(key.names, np.asarray(val).reshape(1, -1))

            bsamples = rho0_ij.block(key_copy).samples
            bvalues = np.zeros(
                (bsamples.values.shape[0], block.values.shape[1], block.values.shape[2])
            )
            tot = [list(l) for l in np.asarray(bsamples.values)]
            bsam = np.asarray(block.samples.values)
            idx = [tot.index(list(b)) for b in bsam]

            bvalues[idx] = block.values
            block = TensorBlock(
                values=bvalues,
                samples=bsamples,
                components=block.components,
                properties=properties,
            )
        else:
            block = TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=properties,
            )
        block = metatensor.sort_block(block)  # sort block samples
        blocks.append(block)
    return TensorMap(rho0_ij.keys, blocks)

def _remove_suffix(names, new_suffix=""):
    suffix = re.compile("_[0-9]?$")
    rname = []
    for name in names:
        match = suffix.search(name)
        if match is None:
            rname.append(name + new_suffix)
        else:
            rname.append(name[: match.start()] + new_suffix)
    return rname

def acdc_standardize_keys(descriptor, drop_pair_id=True):
    """Standardize the naming scheme of density expansion coefficient blocks (nu=1)"""

    key_names = np.array(descriptor.keys.names, dtype='<U22')
    if not "spherical_harmonics_l" in key_names:
        try:
            key_names[np.where(key_names == "o3_lambda")[0]] = "spherical_harmonics_l"
        except:
            raise ValueError(
            "Descriptor missing spherical harmonics channel key `spherical_harmonics_l` or `o3_lambda`" 
        )
    try:

        if "center_type" in key_names:
            key_names[np.where(key_names == "center_type")[0]] = "species_center"
        elif "first_atom_type" in key_names: # rascaline decided to name these keys differently if computing ACDC or pair term
            key_names[np.where(key_names == "first_atom_type")[0]] = "species_center"
    except: #check old rascaline
        if "species_atom_1" in key_names:
            key_names[np.where(key_names == "species_atom_1")[0]] = "species_center"
        else:
            raise ValueError("Descriptor missing species center key `species_center` or `center_type`")

    try:
        if "neighbor_type" in key_names:
            key_names[np.where(key_names == "neighbor_type")[0]] = "species_neighbor"
        elif "second_atom_type" in key_names:
            key_names[np.where(key_names == "second_atom_type")[0]] = "species_neighbor"
    except: #check old rascaline
        if "species_atom_2" in key_names:
            key_names[np.where(key_names == "species_atom_2")[0]] = "species_neighbor"
        else:
            raise ValueError("Descriptor missing species neighbor key `species_neighbor` or `neighbor_type`")
    
    if not "inversion_sigma" in key_names:
        if "o3_sigma" not in key_names:
            key_names = np.asarray(["inversion_sigma"]+ list(key_names), dtype='<U22')

        else:
            key_names[np.where(key_names == "o3_sigma")[0]] = "inversion_sigma"


    if not "order_nu" in key_names:
        key_names = np.asarray(["order_nu"] + list(key_names), dtype='<U22')

    
    key_names = tuple(key_names)
    component_names = ["spherical_harmonics_m" if b == "o3_mu" else b for b in descriptor.component_names]
    
    blocks = []
    keys = []
    for key, block in descriptor.items():
        key = tuple(key)
        if "o3_sigma" not in key_names:
            key = (1,) + key
        if "order_nu" not in key_names:
            key = (1,) + key
        keys.append(key)
        property_names = _remove_suffix(block.properties.names, "_1")
        new_components = [Labels(component_names[i], c.values) for i,c in enumerate(block.components)]

        sample_names = [
           "center" if b == "first_atom" or b=="atom"  else("neighbor" if b=="second_atom" else("structure" if b=="system" else b))
            for b in block.samples.names
        ]
        # converts pair_id to shifted neighbor numbers
        # if "pair_id" in sample_names and drop_pair_id:
        #     new_samples = block.samples.values.copy().reshape(-1, len(sample_names))
        #     # dtype=np.int32
        #     icent = np.where(np.asarray(sample_names) == "center")[0]
        #     ineigh = np.where(np.asarray(sample_names) == "neighbor")[0]
        #     ipid = np.where(np.asarray(sample_names) == "pair_id")[0]
        #     new_samples[:, ineigh] += new_samples[:, ipid] * new_samples[:, icent].max()
        #     new_samples = Labels(
        #         [n for n in sample_names if n != "pair_id"],
        #         new_samples[
        #             :,
        #             [
        #                 i
        #                 for i in range(len(block.samples.names))
        #                 if block.samples.names[i] != "pair_id"
        #             ],
        #         ],
        #     )
        # else:
        new_samples = Labels(
            sample_names,
            np.asarray(block.samples.values).reshape(-1, len(sample_names)),
        )
        # convert values to TORCH TENSOR <<<<
        blocks.append(
            TensorBlock(
                values=torch.tensor(block.values),
                # values=np.asarray(block.values),
                samples=new_samples,
                components=new_components,
                properties=Labels(
                    property_names,
                    np.asarray(block.properties.values).reshape(
                        -1, len(property_names)
                    ),
                ),
            )
        )

    return TensorMap(
        keys=Labels(names=key_names, values=np.asarray(keys)),
        blocks=blocks,
    )


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

def sample_atom_atom(block_a, block_b):
    return slice(None)

def sample_atom_pair(block_a, block_b, is_mp = False):

    samples_a = block_a.samples
    samples_b = block_b.samples

    if is_mp:
        center_slice = []
        smp_a, smp_b = 0, 0
        while smp_b < samples_b.values.shape[0]:
            # print(index_b, samples_b[smp_b][["structure", "center", "neighbor"]], index_a, samples_a[smp_a])
            idx = [idx for idx, tup in enumerate(samples_a) if tup[0] == samples_b[smp_b]["structure"] and tup[1] == samples_b[smp_b]["neighbor"]][0]
            center_slice.append(idx)
            smp_b += 1
        center_slice = np.asarray(center_slice)
        return center_slice
    
    else:
        neighbor_slice = []
        smp_a, smp_b = 0, 0
        
        while smp_b < samples_b.values.shape[0]:
            if (samples_b[smp_b]["structure"], samples_b[smp_b]["center"],) != (samples_a[smp_a]["structure"], samples_a[smp_a]["center"]):
                if smp_a + 1 < samples_a.values.shape[0]:
                    smp_a += 1
            neighbor_slice.append(smp_a)
            smp_b += 1
        neighbor_slice = np.asarray(neighbor_slice)
        return neighbor_slice    

# Serious TODO: Cleanup please FIXME
def cg_combine(
    x_a,
    x_b,
    feature_names=None,
    clebsch_gordan=None,
    lcut=None,
    filter_sigma=[-1, 1],
    other_keys_match=None,
    mp=False,
    device=None,
    diag_tensor=False, 
):
    """
    modified cg_combine from acdc_mini.py to add the MP contraction, that contracts over NOT the center but the neighbor yielding |rho_j> |g_ij>, can be merged
    """
    # determines the cutoff in the new features
    lmax_a = max(x_a.keys["spherical_harmonics_l"])
    lmax_b = max(x_b.keys["spherical_harmonics_l"])
    if lcut is None:
        lcut = lmax_a + lmax_b + 1

    if clebsch_gordan is None:
        clebsch_gordan = ClebschGordanReal(max(lcut, lmax_a, lmax_b) + 1, device=device)

    other_keys_a = tuple(name for name in x_a.keys.names if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"])
    other_keys_b = tuple(name for name in x_b.keys.names if name not in ["spherical_harmonics_l", "order_nu", "inversion_sigma"])

    if mp:
        if other_keys_match is None:
            OTHER_KEYS = [k + "_a" for k in other_keys_a] + [k + "_b" for k in other_keys_b]
        else:
            OTHER_KEYS = ([k + ("_a" if k in other_keys_b else "") for k in other_keys_a if k not in other_keys_match] + \
                          [k + ("_b" if k in other_keys_a else "") for k in other_keys_b if k not in other_keys_match] + \
                           other_keys_match)
            
    else:
        if other_keys_match is None:
            OTHER_KEYS = [k + "_a" for k in other_keys_a] + [k + "_b" for k in other_keys_b]
        else:
            OTHER_KEYS = (other_keys_match + [k + ("_a" if k in other_keys_b else "") for k in other_keys_a if k not in other_keys_match] + \
                                             [k + ("_b" if k in other_keys_a else "") for k in other_keys_b if k not in other_keys_match])

    # automatic generation of the output features names
    # "x1 x2 x3 ; x1 x2 -> x1_a x2_a x3_a k_nu x1_b x2_b l_nu"
    if feature_names is None:
        NU = x_a.keys[0]["order_nu"] + x_b.keys[0]["order_nu"]
        feature_names = ['species_neighbor_1', 'n_1']
        for i in range(2, NU+1):
            if diag_tensor:
                feature_names.extend([f'k_{i}', f'l_{i}'])
            else:
            ## full tensor product of properties
                feature_names.extend([f'k_{i}', f'species_neighbor_{i}', f'n_{i}', f'l_{i}'])


    X_idx = {}
    X_blocks = {}
    X_samples = {}

    l_sigma_nu = ["spherical_harmonics_l", "inversion_sigma", "order_nu"]

    for index_a, block_a in x_a.items():
        block_a = sort_block(block_a, axes="samples")
        lam_a, sigma_a, order_a = [index_a[label] for label in l_sigma_nu]
        properties_a = block_a.properties  # pre-extract this block as accessing a c property has a non-zero cost
        samples_a = block_a.samples

        for index_b, block_b in x_b.items():
            block_b = sort_block(block_b, axes="samples")
            lam_b, sigma_b, order_b = [index_b[label] for label in l_sigma_nu]
            properties_b = block_b.properties
            samples_b = block_b.samples

            samples_final = samples_b
            
            b_slice = list(range(len(samples_b)))

            if other_keys_match is None:
                OTHERS = tuple(index_a[name] for name in other_keys_a) + tuple(index_b[name] for name in other_keys_b)
            else:
                OTHERS = tuple(index_a[k] for k in other_keys_match if index_a[k] == index_b[k])

                if len(OTHERS) < len(other_keys_match):
                    continue
                # adds non-matching keys to build outer product
                if mp:
                    OTHERS = (tuple(index_a[k] for k in other_keys_a if k not in other_keys_match) + OTHERS)
                    OTHERS = (tuple(index_b[k] for k in other_keys_b if k not in other_keys_match) + OTHERS)

                else:
                    OTHERS = OTHERS + tuple(index_a[k] for k in other_keys_a if k not in other_keys_match)
                    OTHERS = OTHERS + tuple(index_b[k] for k in other_keys_b if k not in other_keys_match)

            if mp:
                # combine rho_j and g_ji (i.e. combine dentiy on neighbor j with the g_ji vector where i is the central atom)
                if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                    center_slice = sample_atom_pair(block_a, block_b, is_mp = True)
                else:
                    center_slice = sample_atom_atom(block_a, block_b)

            else:
                # combine rho_i and g_ij (i.e. combine dentiy on central atom i with the g_ji vector where i is the central atom)
                if "neighbor" in samples_b.names and "neighbor" not in samples_a.names:
                    neighbor_slice = sample_atom_pair(block_a, block_b)
                    
                else:
                    neighbor_slice = sample_atom_atom(block_a, block_b)

            # determines the properties that are in the select list
            sel_feats = []
            sel_idx = []
            if diag_tensor:
                idx_nu1 = np.where((properties_b.values[:,:2] == properties_a.values[:,:2][:,None]).all(-1))[1]
                sel_feats = np.array(list(zip(range(len(properties_a.values)), idx_nu1)))
                print(sel_feats.shape, 'sel_feats')
            else:
                sel_feats = np.indices((len(properties_a), len(properties_b))).reshape(2, -1).T
            
            prop_ids_a = []
            prop_ids_b = []

            for f_a in properties_a:
                prop_ids_a.append(tuple(f_a) + (lam_a,))
            for f_b in properties_b:
                prop_ids_b.append(tuple(f_b) + (lam_b,))
            
            prop_ids_a = np.asarray(prop_ids_a)
            prop_ids_b = np.asarray(prop_ids_b)
            
            sel_idx = np.hstack([prop_ids_a[sel_feats[:, 0]][:,:], prop_ids_b[sel_feats[:, 1]][:,-1:]])  # creating a tensor product
            # print(sel_idx ,"JJJJJJ"  )
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
                KEY = (NU, S, L,) + OTHERS

                if not KEY in X_idx:
                    X_idx[KEY] = []
                    X_blocks[KEY] = []
                    X_samples[KEY] = samples_final

                # builds all products in one go
                if mp:
                    if isinstance(center_slice, slice) or len(center_slice):
                        one_shot_blocks = clebsch_gordan.combine(block_a.values[center_slice][:, :, sel_feats[:, 0]],
                                                                 block_b.values[:, :, sel_feats[:, 1]],
                                                                 L
                                                                 )

                    else:
                        one_shot_blocks = []

                else:
                    if isinstance(neighbor_slice, slice) or len(neighbor_slice):
                        one_shot_blocks = clebsch_gordan.combine(block_a.values[neighbor_slice][:, :, sel_feats[:, 0]],
                                                                 block_b.values[b_slice][:, :, sel_feats[:, 1]],
                                                                 L
                                                                 )

                    else:
                        one_shot_blocks = []

                X_idx[KEY].append(sel_idx)
                if len(one_shot_blocks):
                    X_blocks[KEY].append(one_shot_blocks)

    # turns data into sparse storage format (and dumps any empty block in the process)
    nz_idx = []
    nz_blk = []
    for KEY in X_blocks:
        L = KEY[2]
        # create blocks
        if len(X_blocks[KEY]) == 0:
            continue  # skips empty blocks
        nz_idx.append(KEY)
        block_data = torch.cat(X_blocks[KEY], dim=-1)
        sph_components = Labels(["spherical_harmonics_m"], np.asarray(range(-L, L + 1)).reshape(-1, 1))
        newblock = TensorBlock(
            values=block_data,
            samples=X_samples[KEY],
            components=[sph_components],
            properties=Labels(feature_names, np.asarray(np.vstack(X_idx[KEY]))))

        nz_blk.append(newblock)

    X = TensorMap(Labels(["order_nu", "inversion_sigma", "spherical_harmonics_l"] + OTHER_KEYS, np.asarray(nz_idx)), nz_blk)
    return X

def cg_increment(
    x_nu,
    x_1,
    clebsch_gordan=None,
    lcut=None,
    filter_sigma=[-1, 1],
    other_keys_match=None,
    mp=False,
    feature_names=None,
    device=None,
    diag_tensor= True,
):
    """Specialized version of the CG product to perform iterations with nu=1 features"""

    nu = x_nu.keys["order_nu"][0]

    feature_roots = _remove_suffix(x_1.property_names)
    len(feature_roots)
    if feature_names is None:
        if not diag_tensor:
        # feature_names = tuple(f"k_{i}" for i in range(2, nu+1)) + (f"l_{nu}")
            feature_names = (tuple(x_nu.property_names)+
                            ("k_" + str(nu + 1),)
                            + tuple(root + "_" + str(nu + 1) for root in feature_roots)
                            + ("l_" + str(nu + 1),)
                            )
        else:
            if nu==1: 
                feature_names = (tuple(x_nu.property_names)+
                            ("k_" + str(nu + 1),)+ ("l_" + str(nu + 1),)
                            )
            else:
                print(nu, _remove_suffix(x_nu.property_names)[:2] , _remove_suffix(x_1.property_names))
                assert _remove_suffix(x_nu.property_names)[:2] == _remove_suffix(x_1.property_names)

                # diag tensor prod such that species_neighbor_{nu+1}, n_{nu +1} = species_neighbor_1, n_1
                feature_names = (tuple(x_nu.property_names)+
                            ("k_" + str(nu + 1),)+  ("l_" + str(nu + 1),)
                            )
            print(feature_names, 'cg_inc')

    return cg_combine(
        x_nu,
        x_1,
        feature_names=feature_names,
        clebsch_gordan=clebsch_gordan,
        lcut=lcut,
        filter_sigma=filter_sigma,
        other_keys_match=other_keys_match,
        mp=mp,
        diag_tensor = diag_tensor, 
        device=device,
    )

def relabel_keys(tensormap, key_name: str = None):
    # TODO: support key_name to be a dictionary of {key_name: new_name}
    """Relabel the key to contract with other_keys_match, for ACDC - 'species_center' gets renamed to 'key_name'
    while for N-center ACDC 'species_neighbor' gets renamed to 'key_name'"""
    if key_name is None:
        key_name = "species_contract"
    new_tensor_blocks = []
    new_tensor_keys = []
    for k, b in tensormap.items():
        key = tuple(k)
        block = b.copy()
        new_tensor_blocks.append(block)
        new_tensor_keys.append(key)
    if "species_neighbor" in tensormap.keys.names:
        # Relabel neighbor species as species_contract to be the channel to contract |rho_j> |g_ij>
        new_tensor_keys = Labels(
            (
                "order_nu",
                "inversion_sigma",
                "spherical_harmonics_l",
                "species_center",
                key_name,
            ),
            np.asarray(new_tensor_keys),
        )
    else:
        # Relabel center species as species_contract to be the channel to contract |rho_j>
        new_tensor_keys = Labels(
            (
                "order_nu",
                "inversion_sigma",
                "spherical_harmonics_l",
                key_name,
            ),
            np.asarray(new_tensor_keys),
        )

    new_tensormap = TensorMap(new_tensor_keys, new_tensor_blocks)
    return new_tensormap

def contract_rho_ij(rhoijp, elements, property_names=None):
    """contract the doubly decorated pair feature rhoijp = |rho_{ij}^{[nu, nu']}> to return  |rho_{i}^{[nu <- nu']}>"""
    rhoMPi_keys = list(set([tuple(list(x)[:-1]) for x in rhoijp.keys]))
    rhoMPi_blocks = []
    if property_names == None:
        property_names = rhoijp.property_names

    for key in rhoMPi_keys:
        contract_blocks = []
        contract_properties = []
        contract_samples = (
            []
        )  # rho1i.block(rho1i.blocks_matching(species_center=key[-1])[0]).samples #samples for corres key

        for ele in elements:
            blockidx = rhoijp.blocks_matching(species_contract=ele)
            sel_blocks = [
                rhoijp.block(i)
                for i in blockidx
                if key == tuple(list(rhoijp.keys[i])[:-1])
            ]
            if not len(sel_blocks):
                #                 print(key, ele, "skipped")
                continue
            assert (
                len(sel_blocks) == 1
            )  # sel_blocks is the corresponding rho11 block with the same key and species_contract = ele
            block = sel_blocks[0]
            filter_idx = list(zip(block.samples["structure"], block.samples["center"]))
            #             #len(block.samples)==len(filter_idx)
            struct, center = np.unique(block.samples["structure"]), np.unique(
                block.samples["center"]
            )
            possible_block_samples = list(product(struct, center))

            block_samples = []
            ij_samples = []
            block_values = []

            for isample, sample in enumerate(possible_block_samples):
                sample_idx = [
                    idx
                    for idx, tup in enumerate(filter_idx)
                    if tup[0] == sample[0] and tup[1] == sample[1]
                ]
                if len(sample_idx) == 0:
                    continue
                #             #print(key, ele, sample, block.samples[sample_idx])
                block_samples.append(sample)
                ij_samples.append(block.samples[sample_idx])
                block_values.append(
                    block.values[sample_idx].sum(axis=0)
                )  # sum j belonging to ele,
                # block_values has as many entries as samples satisfying (key, ele) so in general we have a ragged list
                # of contract_blocks

            contract_blocks.append(block_values)
            contract_samples.append(block_samples)
            contract_properties.append(block.properties.asarray())

        all_block_samples = sorted(list(set().union(*contract_samples)))
        #         print('nsamples',len(all_block_samples) )
        all_block_values = np.zeros(
            (
                (len(all_block_samples),)
                + block.values.shape[1:]
                + (len(contract_blocks),)
            )
        )
        for ib, bb in enumerate(contract_samples):
            nzidx = [
                i for i in range(len(all_block_samples)) if all_block_samples[i] in bb
            ]
            #             print(elements[ib],key, bb, all_block_samples)
            all_block_values[nzidx, :, :, ib] = contract_blocks[ib]

        new_block = TensorBlock(
            values=all_block_values.reshape(
                all_block_values.shape[0], all_block_values.shape[1], -1
            ),
            samples=Labels(
                ["structure", "center"], np.asarray(all_block_samples, np.int32)
            ),
            components=block.components,
            properties=Labels(
                list(property_names),
                np.asarray(np.vstack(contract_properties), np.int32),
            ),
        )

        rhoMPi_blocks.append(new_block)
    rhoMPi = TensorMap(
        Labels(
            ["order_nu", "inversion_sigma", "spherical_harmonics_l", "species_center"],
            np.asarray(rhoMPi_keys),
        ),
        rhoMPi_blocks,
    )

    return rhoMPi

def compute_rhoi_pca(
    rhoi,
    npca: Optional[Union[float, List[float]]] = None,
    slice_samples: Optional[int] = None,
):
    """computes PCA contraction with combined elemental and radial channels.
    returns the contraction matrices
    """
    if isinstance(npca, list):
        assert len(npca) == len(rhoi)
    else:
        npca = [npca] * len(rhoi)
    pca_vh_all = []
    s_sph_all = []
    pca_blocks = []
    for idx, (key, block) in enumerate(rhoi.items()):
        # nu, sigma, l, spi = key.values
        if slice_samples is not None:
            # FIXME - doesnt work for cuda tensors
            block = operations.slice_block(
                block,
                axis="samples",
                labels=Labels(
                    block.samples.names, block.samples.values[::slice_samples]
                ),
            )

        nsamples = len(block.samples)
        ncomps = len(block.components[0])
        xl = block.values.reshape((len(block.samples) * len(block.components[0]), -1))
        # print(xl)
        # standardize features here <<<<<<<
        std = torch.std(xl, axis=0)
        std[np.isclose(std, 0)] = 1
        assert ~np.isclose(torch.min(torch.abs(std)), 0), "STD is zero!"
        xl = xl / std[None, :]
        u, s, vh = torch.linalg.svd(xl, full_matrices=False)
        eigs = torch.pow(s, 2) / (xl.shape[0] - 1)
        eigs_ratio = eigs / torch.sum(eigs)
        explained_var = torch.cumsum(eigs_ratio, dim=0)
        # print(explained_var)
        if npca[idx] is None:
            npc = vh.shape[1]

        elif 0 < npca[idx] < 1:
            try:
                npc = (explained_var > npca[idx]).nonzero()[1, 0]
            except:
                npc = min(xl.shape[0], xl.shape[1])
        # allow absolute number of features to retain
        elif npca[idx] < min(xl.shape[0], xl.shape[1]):
            npc = npca[idx]

        else:
            npc = min(xl.shape[0], xl.shape[1])
        retained = torch.arange(npc)

        s_sph_all.append(s[retained])
        pca_vh_all.append(vh[retained].T)
        # print("singular values", s[retained]/s[0])
        # print(vh[retained].T.shape, len(block.properties))
        pblock = TensorBlock(
            values=vh[retained].T,
            components=[],
            samples=Labels(
                ["pca"],
                np.asarray(
                    [i for i in range(len(block.properties))]
                ).reshape(-1, 1),
            ),
            properties=Labels(
                ["pca"],
                np.asarray(
                    [i for i in range(vh[retained].T.shape[1])]
                ).reshape(-1, 1),
            ),
        )
        pca_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, pca_blocks)
    return pca_tmap, pca_vh_all, s_sph_all


def get_pca_tmap(rhoi, pca_vh_all):
    assert len(rhoi.keys) == len(pca_vh_all)
    pca_blocks = []
    for idx, (key, block) in enumerate(rhoi):
        vt = pca_vh_all[idx]
        # print(vt.shape)
        pblock = TensorBlock(
            values=vt,
            components=[],
            samples=Labels(
                ["pca"],
                np.asarray(
                    [i for i in range(len(block.properties))]
                ).reshape(-1, 1),
            ),
            properties=Labels(
                ["pca"],
                np.asarray([i for i in range(vt.shape[-1])]).reshape(
                    -1, 1
                ),
            ),
        )
        pca_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, pca_blocks)
    return pca_tmap


def apply_pca(rhoi, pca_tmap):
    new_blocks = []
    for idx, (key, block) in enumerate(rhoi.items()):
        # nu, sigma, l, spi = key
        sigma = key["inversion_sigma"]
        l = key["spherical_harmonics_l"]
        xl = block.values.reshape((len(block.samples) * len(block.components[0]), -1))
        vt = pca_tmap.block(
            key
        ).values  # spherical_harmonics_l=l, inversion_sigma=sigma).values
        xl_pca = (xl @ vt).reshape((len(block.samples), len(block.components[0]), -1))
        #         print(xl_pca.shape)
        pblock = TensorBlock(
            values=xl_pca,
            components=block.components,
            samples=block.samples,
            properties=Labels(
                ["pca"],
                np.asarray(
                    [i for i in range(xl_pca.shape[-1])]
                ).reshape(-1, 1),
            ),
        )
        new_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, new_blocks)
    return pca_tmap


def _pca(feat, npca: Union[float, None] = 0.95, slice_samples: Optional[int] = None):
    """
    feat: TensorMap
    """
    if npca is None:
        return feat
    feat_projection, feat_vh_blocks, feat_eva_blocks = compute_rhoi_pca(
        feat, npca=npca, slice_samples=slice_samples
    )
    feat_pca = apply_pca(feat, feat_projection)
    return feat_pca


def drop_blocks_L(tmap, lcut):
    ls_drop = np.arange(lcut + 1, max(tmap.keys["spherical_harmonics_l"]) + 1)
    mask = np.isin(tmap.keys["spherical_harmonics_l"], ls_drop)
    keys = Labels(tmap.keys.names, tmap.keys.values[mask])
    return operations.drop_blocks(tmap, keys)
