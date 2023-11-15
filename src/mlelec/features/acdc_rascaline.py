# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed

from rascaline import SphericalExpansion
from rascaline import SphericalExpansionByPair as PairExpansion
from metatensor import TensorMap, TensorBlock, Labels
import metatensor.operations as operations
import torch
import numpy as np

from mlelec.features.acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    _pca,
)
from typing import List, Optional, Union
import scipy 

# TODO: use rascaline.clebsch_gordan.combine_single_center_to_nu when support for multiple centers is added

def compute_rhoi_pca(rhoi, npca: Optional[Union[float, List[float]]]=None, ):
    """ computes PCA contraction with combined elemental and radial channels.
    returns the contraction matrices
    """
    if isinstance(npca, list): 
        assert len(npca) == len(rhoi)
    else: 
        npca = [npca]*len(rhoi)

    pca_vh_all = []
    s_sph_all = []
    pca_blocks = []
    for idx, (key, block) in enumerate(rhoi):
        nu, sigma, l, spi = key
        nsamples = len(block.samples)
        ncomps = len(block.components[0])
        xl = block.values.reshape((len(block.samples)*len(block.components[0]),-1))
        u,s, vh = torch.linalg.svd(xl, full_matrices=False)
        eigs= torch.pow(s, 2) / (xl.shape[0] - 1)
        eigs_ratio = eigs / torch.sum(eigs)
        explained_var = torch.cumsum(eigs_ratio, dim=0)

        if npca[idx] is None:
            npc = vh.shape[1]

        elif 0 < npca[idx] < 1:
            npc = (explained_var > npca[idx]).nonzero()[1, 0]
        # allow absolute number of features to retain 
        elif  npca[idx] < min(xl.shape[0], xl.shape[1]):
            npc = npca[idx]
        
        else: 
            npc = min(xl.shape[0], xl.shape[1])
        retained = torch.arange(npc)

        s_sph_all.append(s[retained])
        pca_vh_all.append(vh[retained].T)
        print("singular values", s[retained]/s[0])
        pblock = TensorBlock( values = vh[retained].T ,
                                 components = [],
                                 samples = Labels(["pca"], np.asarray([i for i in range(len(block.properties))], dtype=np.int32).reshape(-1,1)),
                                 properties = Labels(["pca"], np.asarray([i for i in range(vh[-npca[idx]:][::-1].T.shape[-1])], dtype=np.int32).reshape(-1,1))
                                )
        pca_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, pca_blocks)
    return pca_tmap, pca_vh_all, s_sph_all

def get_pca_tmap(rhoi, pca_vh_all):
    assert len(rhoi.keys) == len(pca_vh_all)
    pca_blocks = []
    for idx, (key, block) in enumerate(rhoi):
        vt = pca_vh_all[idx]
        print(vt.shape)
        pblock = TensorBlock( values = vt ,
                                 components = [], 
                                 samples = Labels(["pca"], np.asarray([i for i in range(len(block.properties))], dtype=np.int32).reshape(-1,1)),
                                 properties = Labels(["pca"], np.asarray([i for i in range(vt.shape[-1])], dtype=np.int32).reshape(-1,1))
                                )   
        pca_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, pca_blocks)
    return pca_tmap

def apply_pca(rhoi, pca_tmap):
    new_blocks = []
    for idx, (key, block) in enumerate(rhoi):
        nu, sigma, l, spi = key
        xl = block.values.reshape((len(block.samples)*len(block.components[0]),-1))
        vt = pca_tmap.block(spherical_harmonics_l = l, inversion_sigma = sigma).values
        xl_pca = (xl@vt).reshape((len(block.samples),len(block.components[0]),-1))
#         print(xl_pca.shape)
        pblock = TensorBlock( values = xl_pca,
                                 components = block.components,
                                 samples = block.samples,
                                 properties = Labels(["pca"], np.asarray([i for i in range(xl_pca.shape[-1])], dtype=np.int32).reshape(-1,1))
                                )
        new_blocks.append(pblock)
    pca_tmap = TensorMap(rhoi.keys, new_blocks)
    return pca_tmap

def _pca(feat):
    """
    feat: TensorMap
    """
    nsamples = 
    feat_forpca = operations.slice(
        feat,
        samples=Labels(
            ["structure"], np.array(list(range(0, , 5)), np.int32).reshape(-1, 1)
        ),
    )

    feat_projection, feat_vh_blocks, feat_eva_blocks = compute_rhoi_pca(
        feat_forpca, npca=[25, 25, 40, 25, 15, 20, 22]
    )
    # print("feat eva", feat_eva_blocks)
    feat_pca = apply_pca(feat, feat_projection)
    return feat_pca


def single_center_features(frames, order_nu, hypers, lcut=None, cg=None):
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
    for _ in range(order_nu - 1):
        rho_x = cg_increment(
            rho_prev,
            rho1i,
            clebsch_gordan=cg,
            lcut=lcut,
            other_keys_match=["species_center"],
        )
        rho_prev = _pca(rho_x)

    rho_x = cg_increment(
        rho_prev,
        rho1i,
        clebsch_gordan=cg,
        lcut=lcut,
        other_keys_match=["species_center"],
    )

    return rho_x


def pair_feature(frames, hypers, cg, rhonui=None):
    calculator = PairExpansion(**hypers)
    rhoij = calculator.compute(frames)
    rhoij = acdc_standardize_keys(rhoij)
    if rhonui is None:
        rhonui = single_center_features(order_nu, hypers)
    rhoij_nu = cg_combine(
        rhonui, rhoij, clebsch_gordan=cg, other_keys_match=["species_center"], lcut=0
    )


def twocenter_features():
    # actually special class of features for Hamiltonian (rank2 ) tensor
    pass



