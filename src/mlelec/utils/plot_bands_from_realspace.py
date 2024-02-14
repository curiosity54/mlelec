

import numpy as np
import scipy
from mlelec.utils.utils_fhiaims import rH_to_Hk,get_phaseshift
import matplotlib.pyplot as plt

def get_reciprocal_lattice_vector(latvec):
    ''' Function that computes the reciprocal lattice vectors from the real lattice vectors latvec
    '''
    rlatvec = []
    volume = (np.dot(latvec[0,:],np.cross(latvec[1,:],latvec[2,:])))
    rlatvec.append(np.array(2*np.pi*np.cross(latvec[1,:],latvec[2,:])/volume))
    rlatvec.append(np.array(2*np.pi*np.cross(latvec[2,:],latvec[0,:])/volume))
    rlatvec.append(np.array(2*np.pi*np.cross(latvec[0,:],latvec[1,:])/volume))
    rlatvec = np.asarray(rlatvec)
    return rlatvec


def get_ev(ham, ovl, Hartree_to_eV=True):
    ''' Function that calculates the eigenvalues for the provided Hamiltonian ham and overlap ovl. Converts Hartree energy to eV if desired (default=True).
    '''
    eigenval=scipy.linalg.eigvals(ham,ovl)#,UPLO='U')

    if Hartree_to_eV:
        hartreetoev=scipy.constants.physical_constants['hartree-electron volt relationship'][0]#27.211407952665

        evinev=(eigenval)*hartreetoev
        e_shift=np.array(sorted(evinev))
        return e_shift
    else:
        e_shift=np.array(sorted(eigenval))
        return e_shift

def get_bandenergies(rfock,rovl,cells,klist,npoints, Hartree_to_eV=True):
    '''Function that returns the bandenergies of the provided realspace fock, realspace overlap, cell shifts, kpoint list and number of kpoints per band.
    '''
    rev=[]
    for band in range(int(len(klist)/npoints)):
        evlist=[]
        for kpt in range(npoints):
            #kphase=get_phaseshift(klist[kpt+band*npoints], cells)
    
            s_k=rH_to_Hk(rovl, klist[kpt+band*npoints],  cells)        
            h_k=rH_to_Hk(rfock, klist[kpt+band*npoints],  cells)
            ev = get_ev(h_k, s_k, Hartree_to_eV=Hartree_to_eV)
            evlist.append(ev)
        rev.extend(evlist)   
    return rev


def interpolate_kpoints(symmpoints, npoints):
    '''Function that interpolates the kpoints between the provided symmetry points. Computes npoints between the kpoints
    '''
    klist=[]

    for n in range(len(symmpoints)-1):
        k=np.linspace(symmpoints[n],symmpoints[n+1], npoints)
        for i in range(npoints):
            #klist.append(list(k[i]))
            klist.append(k[i])
    return klist

def get_xposition_of_symmpoints(symmpoints, cell):
    '''Function that locates the symmetry points on the x-axis for plotting, needs the cell to compute the reciprocal lattice vectors which are used to compute the desired band lengths on the plot
    '''
    rlatvec=get_reciprocal_lattice_vector(cell)
    lengths=[np.linalg.norm(np.dot(rlatvec,symmpoints[i+1]) - np.dot(rlatvec,symmpoints[i])) for i in range(len(symmpoints)-1)]
    symmpoints_x=np.cumsum([0]+lengths)
    return symmpoints_x



def plot_bandstructure(frame,fock,ovl,cells, symmpoints, symmpoint_names,npoints=50,energyshift=0, ymin=-30, ymax=30, Hartree_to_eV=True):
    ''' Function that plots the desired band structure given a structure (frame, its cell vector is used), Hamiltonian matrix (Fock), Overlap matrix (ovl), the symmetry points between which the bandstructure should be plotted, and the names of the symmetry_points
    '''
    symmpoints_x=get_xposition_of_symmpoints(symmpoints, frame.cell)
    
    klist=interpolate_kpoints(symmpoints, npoints)
    xlist=interpolate_kpoints(symmpoints_x, npoints)
    rev=get_bandenergies(fock,ovl,cells,klist,npoints, Hartree_to_eV=Hartree_to_eV)

    plt.rcParams['lines.linewidth'] = 1
    ax_bands = plt.subplot(1,1,1)
    
    ax_bands.plot(xlist,np.array(rev)+energyshift, '-b')
   #ax_bands.plot(xlist,np.array(rev)+9, 'rx-')
    #print(rev)
    
    labels=[(symmpoints_x[i],symmpoint_names[i]) for i in range(len(symmpoints_x))]
    
    tickx = []
    tickl = []
    for xpos,l in labels:
        ax_bands.axvline(xpos,color='k',linestyle=":")
        tickx += [ xpos ]
        if len(l)>1:
            if l=="Gamma":
               l = "$\\"+l+"$"
        tickl += [ l ]
    for x, l in zip(tickx, tickl):
        print("| %8.3f %s" % (x, repr(l)))
    
    ax_bands.set_xlim(labels[0][0],labels[-1][0])
    ax_bands.set_xticks(tickx)
    ax_bands.set_xticklabels(tickl)
    ax_bands.set_ylim(ymin,ymax)


def plot_multiple_bandstructures(frames,focks,ovls,cell_shifts,symmpoints, symmpoint_names,npoints=50,energyshift=0, ymin=-30,ymax=30, lattice_from_first_frame=True):
    '''Similar to plot_bandstructure, but provides the possibility to plot multiple band structures in the same plot. For this, a list of frames, Hamiltonians, overlaps and cell shifts is expected.
    '''
#note while this works for different cell vectors, it does not necessarily make sense to compare them, as the lengths between the symmetry points change
    
    #plt.rcParams['lines.linewidth'] = 1
    ax_bands = plt.subplot(1,1,1)
     
    colors = plt.cm.jet(np.linspace(0,1,len(frames)+2))
    klist=interpolate_kpoints(symmpoints, npoints)

    symmpoints_x=get_xposition_of_symmpoints(symmpoints, frames[0].cell)
    xlist=interpolate_kpoints(symmpoints_x, npoints)

    for ifr in range(len(frames)):
    
        if (ifr>0 and not lattice_from_first_frame):    
            symmpoints_x=get_xposition_of_symmpoints(symmpoints, frames[ifr].cell)
            xlist=interpolate_kpoints(symmpoints_x, npoints)
                
        rev=get_bandenergies(focks[ifr],ovls[ifr],cell_shifts[ifr],klist, npoints)
        ax_bands.plot(xlist,np.array(rev)+energyshift, color=colors[ifr])
    
    labels=[(symmpoints_x[i],symmpoint_names[i]) for i in range(len(symmpoints_x))]
    
    tickx = []
    tickl = []
    for xpos,l in labels:
        ax_bands.axvline(xpos,color='k',linestyle=":")
        tickx += [ xpos ]
        if len(l)>1:
            if l=="Gamma":
               l = "$\\"+l+"$"
        tickl += [ l ]
    for x, l in zip(tickx, tickl):
        print("| %8.3f %s" % (x, repr(l)))
    
    ax_bands.set_xlim(labels[0][0],labels[-1][0])
    ax_bands.set_xticks(tickx)
    ax_bands.set_xticklabels(tickl)
    ax_bands.set_ylim(ymin,ymax)






