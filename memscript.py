from metatensor import Labels, TensorBlock, TensorMap
import numpy as np 
import torch
dummy_prop = Labels(['dummy'], np.array([[0]]))
def TMap_bloch_sums(target_blocks, phase, indices):

    aaaa=np.random.randn(10000)
    _Hk = {}
    _Hk0 = {}
    for k, b in target_blocks.items():

        # LabelValues to tuple
        kl = tuple(k.values.tolist())

        # Block type
        bt = kl[0]

        # define dummy key pointing to block type 1 when block type is zero
        if bt == 0:
            _kl = (1, *kl[1:])
        else:
            _kl = kl

        if _kl not in _Hk:
            _Hk[_kl] = {}

        # Loop through the unique (ifr, i, j) triplets
        b_values = b.values.to(next(iter(next(iter(phase.values())).values())))
        for I, (ifr, i, j) in enumerate(phase[kl]):
            
            idx = indices[kl][ifr,i,j]
            values = b_values[idx]
            vshape = values.shape
            pshape = phase[kl][ifr, i, j].shape

            # equivalent to torch.einsum('Tmnv,kT->kmnv', values.to(phase[kl][ifr, i, j]), phase[kl][ifr, i, j]), but faster
            contraction = (phase[kl][ifr, i, j]@values.reshape(vshape[0], -1)).reshape(pshape[0], *vshape[1:])

            if bt != 0:
                # block type not zero: create dictionary element
                if (ifr, i, j) in _Hk[_kl]:
                    _Hk[_kl][ifr, i, j] += contraction
                else:
                    _Hk[_kl][ifr, i, j] = contraction
            else:
                # block type zero
                if (ifr, i, j) in _Hk[_kl]:
                    # if the corresponding bt = +1 element exists, sum to it the bt=0 contribution
                    _Hk[_kl][ifr, i, j] += contraction*np.sqrt(2)
                else:
                    # The corresponding bt = +1 element does not exist. Create the dictionary element
                    _Hk[_kl][ifr, i, j] = contraction*np.sqrt(2)
                    
    # Now store in a tensormap
    _k_target_blocks = []
    keys = []
    for kl in _Hk:

        same_orbitals = kl[2] == kl[5] and kl[3] == kl[6]

        values = []
        samples = []
        
        for ifr, i, j in sorted(_Hk[kl]):
            
            # skip when same orbitals, atoms, and block type == -1
            # print(kl[0], '|', kl[2], kl[3], kl[5], kl[6],'|',ifr,i,j)
            # if not (same_orbitals and (i == j) and (kl[0] == -1)):
                # if kl[0] == -1:
                #     print(kl[2], kl[5], kl[3], kl[6],ifr,i,j)
                # Fill values and samples
                values.append(_Hk[kl][ifr, i, j])
                samples.extend([[ifr, i, j] + [ik] for ik in range(_Hk[kl][ifr, i, j].shape[0])])
            
        values = torch.concatenate(values)
        _, n_mi, n_mj, _ = values.shape
        samples = Labels(['structure', 'center', 'neighbor', 'kpoint'], np.array(samples))
        components = [Labels(['m_i'], np.arange(-n_mi//2+1, n_mi//2+1).reshape(-1,1)), Labels(['m_j'], np.arange(-n_mj//2+1, n_mj//2+1).reshape(-1, 1))]
        
        _k_target_blocks.append(
            TensorBlock(
                samples = samples,
                components = components,
                properties = dummy_prop,
                values = values
            )
        )
        
        keys.append(list(kl))

    _k_target_blocks = TensorMap(Labels(['block_type', 'species_i', 'n_i', 'l_i', 'species_j', 'n_j', 'l_j'], np.array(keys)), _k_target_blocks)

    return _k_target_blocks

# def calltbl(target_blocks, phase, indices):
#     from mlelec.utils.pbc_utils import TMap_bloch_sums
#     TMap_bloch_sums(target_blocks, phase, indices)
