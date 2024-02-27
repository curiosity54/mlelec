
import numpy as np
import scipy.sparse

import scipy.io as io
from read_elsi import read_elsi_to_csc

import sys,os

from ase.io import read
from glob import glob

from mlelec.utils.twocenter_utils_fhiaims import _matrix_to_blocks, _to_matrix, _to_coupled_basis, _to_uncoupled_basis
from mlelec.utils.twocenter_utils_fhiaims import to_coupled_blocks, to_uncoupled_matrices#,change_rotation_direction
from mlelec.utils.twocenter_utils_fhiaims import symmetrize_matrices, check_rotation

def read_upper_half_realham(foldername):
    ''' Function that reads the realhamiltonian file as output by (the manipulated) FHI-aims. 
        It returns the full (#cells, #BF, #BF) real space Hamiltonian along with the number of non-zero elements in the Hamiltonian.
    '''

    nbf, ncell=get_nbf_ncell(foldername)

    ham=np.zeros((int(ncell)-1, int(nbf), int(nbf))) 
    with open('{}/realspaceHam.txt'.format(foldername)) as f:
        lines=f.readlines()
    for line in lines[1:]:
        icell, ibf2, ibf1, idx_real, spin, ham_ij =list(filter(lambda x: x != '', line.split('\n')[0].split(' ')))
        ham[int(icell)-1, int(ibf2)-1, int(ibf1)-1]=float(ham_ij)

    return ham, idx_real #idx_real is the number of non 0 elements in the real space hamiltonian


def get_nbf_ncell(foldername):
    ''' Function that reads the number of basis functions and number of translated cells from the aims.out file in the provided folder with the name foldername
    '''
    with open('{}/aims.out'.format(foldername)) as f:
        lines=f.readlines()
    for line in lines[1:]:
        if line.startswith('  | Maximum number of basis functions            :'):#       78
            m=[i for i in line.split(' ') if i!='']
            nbf=m[-1]
        if line.startswith('  | Number of super-cells in hamiltonian [n_cells_in_hamiltonian]:'):#         104
            m=[i for i in line.split('\n')[0].split(' ') if i!='']
            ncell=m[-1] 
    return nbf, ncell

def read_upper_half_realovl(foldername):
    ''' Function that reads the overlap-matrix file as output by (the manipulated) FHI-aims. 
        It returns the (#BF, #BF) overlap matrix.
    '''

    nbf, ncell=get_nbf_ncell(foldername)

    ovl=np.zeros((int(ncell)-1,int(nbf), int(nbf)))
#    print(ham.shape)
    with open('{}/overlap_complete.txt'.format(foldername)) as f:
        lines=f.readlines()

    for line in lines[1:]:
        icell, ibf2, ibf1, idx_real, ovl_ij =list(filter(lambda x: x != '', line.split('\n')[0].split(' ')))
#        print(icell, ibf2, ibf1, idx_real, spin, ham_ij)
        ovl[int(icell)-1, int(ibf2)-1, int(ibf1)-1]=float(ovl_ij)

#    sparse_ham = scipy.sparse.coo_array(ham)

    return ovl #idx_real is the number of non 0 elements in the real space hamiltonian

def read_ovl(foldername):
    ''' Function that reads the overlap-matrix file as output by (the manipulated) FHI-aims. 
        It returns the (#BF, #BF) overlap matrix.
    '''
    nbf, ncell=get_nbf_ncell(foldername)
    ovl=np.zeros((int(nbf), int(nbf)))
#    print(ham.shape)
    for line in lines[:]:
        ibf2, ibf1, ovl_ij =list(filter(lambda x: x != '', line.split('\n')[0].split(' ')))
        ovl[int(ibf2)-1, int(ibf1)-1]=float(ovl_ij)
        ovl[int(ibf1)-1, int(ibf2)-1]=float(ovl_ij)

    return ovl #idx_real is the number of non 0 elements in the real space hamiltonian

def read_kphase(foldername):
    ''' Function that reads a k phase file and returns the kphase of each of the translated (and original) cell.
    '''
    with open(foldername) as f:
        lines=f.readlines()

    icell =int( list(filter(lambda x: x != '', lines[-1].split(' ')))[0] )
    kpoint =int( list(filter(lambda x: x != '', lines[-1].split(' ')))[1] )

    kphase=np.zeros((icell),dtype=np.complex_)
    try:
        w=int( list(filter(lambda x: x != '', lines[0].split(' ')))[0] )
        start=0
    except:
        start=1

    for n, line in enumerate(lines[start:]):
        val=line.split('(')[-1].split(')')[0].split(',')
        icell =int( list(filter(lambda x: x != '', line.split(' ')))[0] )
        com=float(val[0])+float(val[1])*1j
        kphase[icell-1]=com
    return kphase#,weights

def read_out(lines, cmplx=False, startline=0):
    ''' Function that reads an .mtx file (produced from the aims output .csc) of the Hamiltonian at a specific k.
    '''
    nrbf =int( list(filter(lambda x: x != '', lines[-1].split(' ')))[0] )

    if not cmplx:
        ham=np.empty((nrbf,nrbf))
        for line in lines[startline:]:
            n,m,h=list(filter(lambda x: x != '', line.split('\n')[0].split(' ')))
            n=int(n)
            m=int(m)
            h=float(h)
            ham[n-1,m-1]=h
            ham[m-1,n-1]=h
    else:
        ham=np.empty((nrbf,nrbf),dtype=np.complex_)
        for line in lines[3:]:
            n,m,h,hi=list(filter(lambda x: x != '', line.split('\n')[0].split(' ')))
            n=int(n)
            m=int(m)
            h=float(h)
            h=float(hi)
            ham[n-1,m-1]=h+hi *1j
            ham[m-1,n-1]=h+hi *1j

    return ham

def get_cellidx(foldername):
    ''' Function that reads the cell shifts of the translated cells from aims.out and saves them in a file called cell_idx.npy
    '''
    with open(foldername) as f:
        lines=f.readlines()
    
    
    for line in lines:
        if '  | include' in line:#  | include super-cells idx: [i_cell_1]           1 newindex [i_cell_new]           1
            n_cells=int(line.split('\n')[0].split(' ')[-1])

    cell_idx=np.empty((n_cells,3),dtype=np.int64)
    
    n=0
    for line in lines:
        if ' cell idx' in line:# cell idx ham           0           0           0
            z=[i for i in line.split('\n')[0].split(' ') if i!='']
            cell_idx[n,0]=int(z[-3])
            cell_idx[n,1]=int(z[-2])
            cell_idx[n,2]=int(z[-1])
    
            n+=1
    
    return cell_idx

def get_all_cellidx(foldernames, suffix=''):
    cell_i_list=[]
    for foldername in foldernames[:]:
        cell_idx=get_cellidx('{}/aims.out'.format(foldername))
        cell_i_list.append(cell_idx)
        #print('Nr of cells found:',len(cell_idx))
    
    np.savez('cell_idx_list{}'.format(suffix), *cell_i_list)
    return cell_i_list 


def read_realhams(foldernames, suffix=''):
    idxup=[]
    idxlo=[]
    cell_locs=[]

    for m,foldername in enumerate(foldernames[:]):
        iup=[]
        ilo=[]
        cell_locs.append( get_cellidx('{}/aims.out'.format(foldername)))
        for n,shift in enumerate(cell_locs[-1][:]):
            #print('s',shift)
            iup.append(n)
            #idxlo.append(cell_i_list.index([-shift[0],-shift[1],-shift[2]]))
            lo=[i for i,element in enumerate(cell_locs[-1][:]) if element[0]==-shift[0] and element[1]==-shift[1] and element[2]==-shift[2]]
            #print(lo)
            ilo.append(lo[0])
        idxup.append(iup)
        idxlo.append(ilo)

    #real space hamiltonian
    rh_list=[]

    for foldername in foldernames[:]:
        rham, rham_nonzero=read_upper_half_realham('{}'.format(foldername)) #real space Hamiltonian

        rh_list.append(rham)

    for m,h in enumerate(rh_list):
        for n in range(len(idxup[m])):
            if idxup[m][n]<len(h) and idxlo[m][n]<len(h):
                for i in range(h[0].shape[0]):
                    for j in range(0,i):
                        rh_list[m][idxlo[m][n]][i][j]=h[idxup[m][n]][j][i]
    np.savez('realspaceH_nonsym{}'.format(suffix), *rh_list)
    return rh_list

def read_realovls(foldernames, suffix=''):
    idxup=[]
    idxlo=[]
    cell_locs=[]

    for m,foldername in enumerate(foldernames[:]):
        iup=[]
        ilo=[]
        cell_locs.append( get_cellidx('{}/aims.out'.format(foldername)))
        for n,shift in enumerate(cell_locs[-1][:]):
            #print('s',shift)
            iup.append(n)
            lo=[i for i,element in enumerate(cell_locs[-1][:]) if element[0]==-shift[0] and element[1]==-shift[1] and element[2]==-shift[2]]
            #print(lo)
            ilo.append(lo[0])
        idxup.append(iup)
        idxlo.append(ilo)

    #real space hamiltonian
    rh_list=[]

    for foldername in foldernames[:]:
        rham=read_upper_half_realovl('{}'.format(foldername)) #real space Hamiltonian

        rh_list.append(rham)

    for m,h in enumerate(rh_list):
        for n in range(len(idxup[m])):
            if idxup[m][n]<len(h) and idxlo[m][n]<len(h):
                for i in range(h[0].shape[0]):
                    for j in range(0,i):
                        rh_list[m][idxlo[m][n]][i][j]=h[idxup[m][n]][j][i]
    np.savez('realspaceOvl_nonsym{}'.format(suffix), *rh_list)
    return rh_list

def get_ovls(foldernames):
    ''' Function that reads in th realspace overlaps in the given folders provided as a list of foldernames.
        This function requires the FHI-aims calculation to be run with the keyword 'output h_s_matrices' in the control.in file.
    '''
    #ovl
    ovl_list=[]
    for foldername in foldernames[:]:
        ovl=read_ovl('{}/overlap-matrix.out'.format(foldername)) 
        ovl_list.append(ovl)
    #print(ovl_list)
    np.savez('overlap', *ovl_list)
    return ovl_list


def get_kphase(foldernames, suffix=''):
    '''Function that reads the kphases from the provided folder. Uses aims.out to get the number of kpoints in calculation.
    '''
    klist=[]
    for foldername in foldernames[:]:

        nr_kpt=get_nr_kpts(foldername+'/aims.out')

        kphase_list=[]
        for kpt in range(1,nr_kpt+1):
            kphase=read_kphase('{}/kphase_kpt_{:06d}_scf_000000.txt'.format(foldername, kpt))
            kphase_list.append(kphase)
        klist.append(kphase_list)
    np.savez('kphases{}'.format(suffix), *klist)
    return klist

def get_nr_kpts(filename):
    '''Function that reads the number of kpoints from the provided aims.out file
    '''
    with open(filename) as f:
        lines=f.readlines()     
    for line in lines:
        if line.startswith('  | Number of k-points'):
            m=[i for i in line.split('\n')[0].split(' ') if i !='']
            nr_kpt=int(m[-1])
    return nr_kpt


def get_kpts_and_weights(foldernames,suffix=''): 
    '''Function that reads the kpoints and their weights from the aims.out file in the provided folder. 
       FHI-aims keyword 'output full' has to be set in order for this to get printed in aims.out.
    '''
    wlist=[]
    kpoints=[]
    for foldername in foldernames[:]:
        wl=[]
        kp=[]

        with open('{}/aims.out'.format(foldername)) as f:
            lines=f.readlines()
        for line in lines:
            if line.startswith('  | k-point:'):
#                print(line)
                m=[i for i in line.split('\n')[0].split(' ') if i !='']
#                print(m[9])
                wl.append(float(m[9]))
                kp.append([float(i) for i in m[4:7]])
#        print(kp)

        wlist.append(wl)
        kpoints.append(kp)

    np.savez('kweights{}'.format(suffix), *wlist)
    np.savez('kpoints{}'.format(suffix), *kpoints)
    return kpoints, wlist

def read_Hks(foldernames,suffix=''):
#
    #H_k

    sys.stdout = open(os.devnull, 'w')
    h_list=[]
    for foldername in foldernames[:]:
        hk_list=[]
        nr_kpt=get_nr_kpts(foldername+'/aims.out')
        for kpt in range(1,nr_kpt+1):

            hk=read_elsi_to_csc('{}/H_spin_01_kpt_{:06d}.csc'.format(foldername,kpt) ).toarray()
            hk_list.append(hk)
        h_list.append(hk_list)
#        print(len(hk_list))
    sys.stdout = sys.__stdout__

#    print('hlist',h_list)
    np.savez('H_k{}'.format(suffix), *h_list)
    return h_list

def get_phaseshift(k, cell_locs):
    '''Get the kphase according to the kpoint (k) and the relevant cell shifts (cell_locs) for FHI-aims.
       This function returns the kphase with an error of about 10**-5 in comparison to FHI-aims, I (HT) could not figure out why. Precision of kphase in FHI-aims is COMPLEX*16, here we deal with integer kpoints and integer cell_locs. I guess the rounding in the kpoints is the error. One could try to compute the kpoint positions from the k grid to fix this problem.
    '''
    return np.exp(2*np.pi*1j*np.dot(cell_locs,np.array(k).T)) #np.asarray(kplist)

def get_translation_dict(cell_i_list, rh2,maxshift=3):
    '''Change the list of realspace Hs to dictionaries for the individual shifts with the shift values as keys
    '''
    translated_matrices=[]
    for ifr in range(len(cell_i_list)): 
        ts={}
        for icell in range(len(cell_i_list[ifr])):
#            print(cell_i_list[ifr])
            if icell<len(rh2[ifr]):
                if not (abs(tuple(cell_i_list[ifr][icell])[0])>maxshift or abs(tuple(cell_i_list[ifr][icell])[1])>maxshift or abs(tuple(cell_i_list[ifr][icell])[2])>maxshift):
                    ts[tuple(cell_i_list[ifr][icell])]=np.array(rh2[ifr][icell])
        translated_matrices.append(ts)

    translated_matrices = {key: np.asarray([dictionary[key] for dictionary in translated_matrices]) for key in translated_matrices[0].keys()}
    return translated_matrices

def Hk_to_rH(H_k,kpoints,weights, cell):#, kphase):
    '''Function to get the realspace H from H_k for provided kpoints, weights and cell shifts

    '''
    newham=np.zeros((len(H_k),len(cell[0]),len(H_k[0][0]),len(H_k[0][0][0])), dtype=np.complex128)
    
    nrkpt=len(H_k[0])#.shape[0]
    
    for ifr in range(len(kpoints)):
        k=get_phaseshift(kpoints[ifr],cell[ifr]).conj()
        for icell in range(len(cell[ifr])): #fock
            for kpt in range(nrkpt):
                newham[ifr][icell]+=H_k[ifr][kpt]*k[icell][kpt]*weights[ifr][kpt]

    return np.real(newham)#, np.real(newham2)

def rH_to_Hk(fock, kpoint, cells, kphase=None):
    '''Function to get the k-dependent H from the realspace H for provided kpoints and cell shifts
       Difference here has to do with the calculation of the kphase, TODO: find out how to exactly get the kphase values from FHI-aims

    '''
    newham=np.zeros((fock.shape[1],fock.shape[2]), dtype=np.complex_)
    

    try:
        w=kphase[0]
        k=kphase#.conj()
        if k.shape[0]!=fock.shape[0]:
            print('Make sure the kphase is provided in shape [iframes, ikpt, icells]')


    except:
        k=get_phaseshift(kpoint,cells)#.conj()



    #k2=get_phaseshift(kpoint,cells)
    #print('wanted',k2.shape)
    #print('is',k.shape)
    for icell in range(len(fock)):
        newham+=fock[icell]*k[icell]#kphase[icell]
    return newham

def rH_to_Hks(fock, kpoints,cells, kphase=None):
    '''Function to get the k-dependent H from the realspace H for provided kpoints and cell shifts
       Difference here has to do with the calculation of the kphase, TODO: find out how to exactly get the kphase values from FHI-aims

    '''
    newham=np.zeros((len(kpoints),fock.shape[1],fock.shape[2]), dtype=np.complex_)
    
    nrkpts=len(kpoints)
    for kpt in range(nrkpts):

        try:
            k=kphase[kpt]
        except:
            k=None

        newham[kpt]=rH_to_Hk(fock,kpoints[kpt],cells, kphase=k)
    return newham


def rHs_to_Hks(fock, kpoints,cells,kphases=None):
    '''Function to get the k-dependent H from the realspace H for provided kpoints and cell shifts
       Difference here has to do with the calculation of the kphase, TODO: find out how to exactly get the kphase values from FHI-aims
       Providing a 
    '''


    hams=[]
    for ifr in range(len(fock)):

        try:
            k=kphases[ifr]
            #print('ifr',k.shape)
        except:
            k=None

        newham=rH_to_Hks(fock[ifr], kpoints[ifr],cells[ifr], kphase=k)
        hams.append(newham.copy())    
    return hams

def check_shift_hermicity(translated_matrices,normval=10**-10):
    for T in translated_matrices.keys():
        mT = tuple(-np.asarray(T))
        if not mT in translated_matrices.keys():
            raise ValueError("key not found")
    
        norm = np.linalg.norm(translated_matrices[T] - np.transpose(translated_matrices[mT], axes=(0,2,1)))
        if norm> normval:
            print(T, norm)


if __name__=='__main__':
    '''If executed, this script collects all the necessary data to machine learn the Hamiltonian matrix from the FHI-folders starting with 'struc' and writes them as npz files.
    '''


    foldernames=glob('struc*')
    
#    foldernames=['structure_{:05d}'.format(i) for i in range(3)]#174)]
    frames=[read('{}/geometry.in'.format(i)) for i in foldernames]

    print(foldernames)
    suffix=''

    hlist=[]
    for folder in foldernames:
        with open('{}/hamiltonian.out'.format(folder)) as f:
            lines=f.readlines()
        h=read_out(lines, cmplx=False, startline=0)
        print(h.shape)
        hlist.append(h)
    np.savez('hamiltonian{}'.format(suffix), *hlist)

    hlist=[]
    for folder in foldernames:
        with open('{}/overlap-matrix.out'.format(folder)) as f:
            lines=f.readlines()
        h=read_out(lines, cmplx=False, startline=0)
        #print(h)
        hlist.append(h)
    np.savez('overlap-matrix{}'.format(suffix), *hlist)
    o=np.load('overlap-matrix.npz', allow_pickle=True)
    ovl=[o['arr_{}'.format(j)] for j in range(len(o))]
    #print(ovl)

#getting the cell shifts, kphase and realspace H from aims requires a modified FHI-aims version, which needs the modifications pushed in github
    cell_i_list=get_all_cellidx(foldernames,suffix=suffix)
    rh2=read_realhams(foldernames,suffix=suffix)
    klist= get_kphase(foldernames, suffix=suffix)
    ovls= read_realovls(foldernames, suffix=suffix)


#These  values can be read from the original FHI-aims with setting 'output output_level  full' for printing the kpoints and weights in aims.out and 'output h_s_matrices'

    kpoints,wlist= get_kpts_and_weights(foldernames, suffix=suffix) 
    
    hk_list=read_Hks(foldernames, suffix=suffix)


#
##Going from Hk to reakspace H and back
#    newham= Hk_to_rH(hk_list,kpoints,wlist,cell_i_list)#,klist)
##    print(newham.shape)
#
##checking the quality of the realspace H obtrained from H(k)
##    for icell in range(len(rh2[0])):
##        print(icell,np.linalg.norm(newham[0][icell]-rh2[0][icell]),np.linalg.norm(newham2[0][icell]-rh2[0][icell]))
##        print(icell,np.linalg.norm(newham[1][icell]-rh2[1][icell]),np.linalg.norm(newham2[1][icell]-rh2[1][icell]))
##        print(icell,np.linalg.norm(newham[2][icell]-rh2[2][icell]),np.linalg.norm(newham2[2][icell]-rh2[2][icell]))
#    newHk=rH_to_Hk(rh2, kpoints,klist,cell_i_list)
##checking the quality of the H(k) obtrained from the realspace H
##    struc=0
##    for kpt in range(len(hk_list[0][:10])):
##        print(kpt,np.linalg.norm(hk_list[struc][kpt]-newHk[struc][kpt]))
#
##other useful modifications of the Hamiltonian
#    orbs = {6: [[1,0,0],[2,0,0],[2,1,1], [2,1,-1],[2,1,0]]}
#    translated_matrices=get_translation_dict(cell_i_list, rh2,maxshift=3)
#    check_shift_hermicity(translated_matrices,normval=10**-10) #prints only if shifts are not hermitian to its translated image -T -> H(T)=H(-T).T
#
##symmetrize realspace H of aims
#    print('Symmetrizing the Hamiltonian, as FHI-aims provides not completely symmteric values')
#    p=to_coupled_blocks(translated_matrices,frames, orbs)
#    B=symmetrize_matrices(p)
#    fock=to_uncoupled_matrices(B,frames,orbs)
#
##change rotation direction
#    print('Check rotation of the Hamiltonian and adjust to active rotation.')
#    rotations=[np.array([ 1.5078, -0.0463, -0.8321]),np.array([ 0.1443, -0.1218,  0.4463])]
#    print('Checking for passive rotation')
#    check_rotation(B,rotations, 'passive',precision=10**-4)
#    print('Changing rotation direction')
#    H_inverse=change_rotation_direction(B)
#    print('Checking for active rotation')
#    check_rotation(H_inverse,rotations, 'active',precision=10**-4)
#    fock_inv=to_uncoupled_matrices(H_inverse,frames,orbs)
#
#    g=change_rotation_direction(H_inverse)
#    fock2=to_uncoupled_matrices(g,frames,orbs)
#
##check precision of going to blocks and back for symmetrized fock
##    for shift in translated_matrices.keys():
##        print(np.linalg.norm(fock[shift].cpu().numpy()-fock2[shift].cpu().numpy()))


