import numpy as np
from numpy.typing import NDArray
import astropy.units as u
import matplotlib.pyplot as plt
from ReadFile import Read
from Lab7 import RotateFrame
from CenterOfMass2 import CenterOfMass
from Snapshots import snapshotsCompile
import os

def majorAnnuli(
        x:NDArray[np.float64],
        y:NDArray[np.float64],
        Nmin:int,
        Nmax:int,
        delta:float = 0.15
) -> list[None|tuple[int,int]]:
    """
    This function finds the mass of each overlapping annulus

    Arguments:
        x (kpc): adjusted sorted x positions of the stars
        y (kpc): adjusted sorted y positions of the stars
        Nmin: The minimum number of stars per annulus
        Nmax: The maximum number of stars per annulus
        delta: A parameter that determines where to divide the annulus
               r[last-1]/r[first] < 10**delta
    Returns:
        binIndices: contains the minimum and maximum radius of each annulus
    """

    binIndices:list[None|tuple[int,int]] = []
    r:NDArray[np.float64] = np.linalg.norm(np.vstack((x,y)),axis=0)

    i:int = 0
    N:int = r.size

    while i < N:
        first:int = i
        last:int = np.min((N,i+Nmin))

        while last < N and (last-first) < Nmax and (r[last-1]/r[first] < 10**delta):
            last += 1
        binIndices.append((first,last))
        i = last
    return binIndices

def overlapAnnuli(
        binIndices:list[None|tuple[int,int]],
        r:NDArray[np.float64]
)->list[None|tuple[int,int]]:
    """
    This function finds the mass of each overlapping annulus
    
    Arguments:
        binIndices: contains the minimum and maximum radius of each annulus
        r (kpc): sorted radii of the stars
    Returns:
        overlapIndices: binIndices with the added overlap regions between them.
    """

    overlapIndices:list[None|tuple[int,int]] = binIndices.copy()
    for j in range(len(binIndices)-1):
        small1:int
        small2:int
        large1:int
        large2:int
        small1,small2 = binIndices[j]
        large1,large2 = binIndices[j+1]

        med1:int = int(np.mean((small1,small2))+0.5)
        med2:int = int(np.mean((large1,large2))+0.5)

        overlapIndices.insert(2*j+1,(med1,med2))
    return overlapIndices
    
def fourier(
        MProf:NDArray[np.float64],
        mModes:NDArray[np.float64]
) -> tuple[float,NDArray[np.float64],NDArray[np.float64]]:
    """
    This function runs a simple Fourier analysis using a uniform (tophat) window.

    Arguments:
        MProf (10^10 Msun): the mass profile of the annuli
        mModes: all modes necessary for analysis
        mMax: largest mode for the Fourier analysis
    
    Returns:
        Sigma (10^10 Msun): the mean value of the profile
        mAmp: the amplitudes of mode m
        mPhase: the phases of mode m
    """
    
    n:int = MProf.size
    Sigma:float = np.mean(MProf)
    mAmp:NDArray[np.float64] = np.zeros(mModes.size)
    mPhase:NDArray[np.float64] = np.zeros(mModes.size)
    theta:NDArray[np.float64] = np.linspace(0,2*np.pi,n,endpoint=False)
    for i,m in enumerate(mModes):
        cn:float = np.sum(MProf*np.exp(-1j*m*theta))
        mAmp[i] = np.abs(cn)
        mPhase[i] = np.angle(cn)
    return Sigma,mAmp,mPhase

def barStrength(
        x:NDArray[np.float64],
        y:NDArray[np.float64],
        m:NDArray[np.float64],
        annuli:list[None|tuple[int,int]],
        nphi:int,
        mMax:int
) -> tuple[NDArray[np.float64],float]:
    """
    This function finds the mass of each overlapping annulus and performs a Fourier
    analysis to compute the bar strength as the different in the root-mean-squared
    of the even and odd modes

    Arguments:
        x (kpc): adjusted sorted x positions of the stars
        y (kpc): adjusted sorted y positions of the stars
        m (10^10 Msun): sorted masses of the stars
        annuli: contains the minimum and maximum radius of each annulus
                This should be taken from the overlapping annuli
        nphi: number of desired azimuthal bins
        mMax: largest mode for the Fourier analysis
    
    Returns:
        strengthsArr: an array of S values for each overlapping bin.
        galaxyStrength: the maximum S value (used to quantify the overall bar strength).
    """
    r = np.linalg.norm(np.vstack((x,y)),axis=0)
    phi = np.mod(np.arctan2(y,x),2*np.pi)
    
    mModes:NDArray[np.float64] = np.arange(1,mMax+1)
    strengthsList:list[None|float] = []
    for (first, last) in annuli:
        i = np.where((first<r) & (r<last))
        mAnnuli:NDArray[np.float64] = m[i]
        phiAnnuli:NDArray[np.float64] = phi[i]

        MProf:NDArray[np.float64]
        MProf,_ = np.histogram(phiAnnuli,bins=nphi,range=(0,2*np.pi),weights=mAnnuli)

        Sigma:float
        mAmp:NDArray[np.float64]
        mPhase:NDArray[np.float64]
        Sigma, mAmp, mPhase = fourier(MProf, mModes)
     
        oddAmp:NDArray[np.float64] = mAmp[0::2]
        evenAmp:NDArray[np.float64] = mAmp[1::2]
        # annulusStrength:float = np.sqrt(np.sum(evenAmp**2)) - np.sqrt(np.sum(oddAmp**2))
        annulusStrength:float = np.sqrt(np.sum(evenAmp**2))
        strengthsList.append(annulusStrength)
    strengthsArr:NDArray[np.float64] = np.array(strengthsList)
    galaxyStrength:float = np.max(strengthsArr)
    return strengthsArr,galaxyStrength

def plotGalaxyStrengthPerSnapshot(strengths:NDArray[np.float64],galaxy:str) -> None:
    fig,ax = plt.subplots()
    ax.plot(strengths[:,0],strengths[:,1],c='r')
    ax.set(title=f"{galaxy} Strength per snapshot",xlabel='Snapshot',ylabel='Bar Strength')
    fig.savefig(f"{galaxy}_bar_strength.pdf")

def main() -> None:
    foci:tuple[int,...] = (275,410,470) # 3.929 Gyr, # 5.857 Gyr, # 6.714 Gyr

    first:int = 0
    last:int = 801

    # generate majors from the halfway points between foci
    fociArr:NDArray[np.int_] = np.sort(np.array(foci))
    halfways:NDArray[np.int_] = (fociArr[:-1]+fociArr[1:])//2
    majors:list[int] = [first,*tuple(halfways),last]

    snapshots:tuple[int,...] = snapshotsCompile(4,foci=foci)

    # Parameters
    Nmin:int = 2000 # minimum number of particles per major annulus
    Nmax:int = 10000 # maximum number of particles per major annulus
    nphi:int = 360 # angle iterations to cover a circle
    mMax:int = nphi//2 # Nyquist frequency for Fourier analysis
    galaxies = ('MW','M31')
    strengths:NDArray[np.float64] = np.zeros((len(snapshots),2))
    for galaxy in galaxies:
        for i,snapshot in enumerate(snapshots):
            ScriptDir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(ScriptDir,galaxy,f"{galaxy}_{snapshot:03d}.txt")
            time:u.Quantity
            total:int
            data:NDArray[np.float64]
            time,total,data = Read(filename)

            com:CenterOfMass2.CenterOfMass = CenterOfMass(filename,2)
            comMass:NDArray[np.float64] = com.m

            COMP:u.Quantity = com.COM_P(0.1)
            COMV:u.Quantity = com.COM_V(COMP[0],COMP[1],COMP[2])

            # Determine positions of disk particles relative to COM
            xcom:NDArray[np.float64] = com.x - COMP[0].value
            ycom:NDArray[np.float64] = com.y - COMP[1].value
            zcom:NDArray[np.float64] = com.z - COMP[2].value
            rcom:NDArray[np.float64] = np.vstack((xcom,ycom,zcom)).T

            vxcom:NDArray[np.float64] = com.vx - COMV[0].value
            vycom:NDArray[np.float64] = com.vy - COMV[1].value
            vzcom:NDArray[np.float64] = com.vz - COMV[2].value
            vcom:NDArray[np.float64] = np.vstack((vxcom,vycom,vzcom)).T

            rComps,vComps = RotateFrame(rcom,vcom)
            rUnsorted:NDArray[np.float64] = np.linalg.norm(np.vstack((rComps[0],rComps[1],rComps[2])),axis=0)
            ri:int = np.argsort(rUnsorted)
            r:NDArray[np.float64] = rUnsorted[ri]

            mass:NDArray[np.float64] = comMass[ri]
            x:NDArray[np.float64] = rComps[0][ri]
            y:NDArray[np.float64] = rComps[1][ri]
            z:NDArray[np.float64] = rComps[2][ri]

            # NOTE: The velocity is currently unused
            vx:NDArray[np.float64] = vComps[0][ri]
            vy:NDArray[np.float64] = vComps[1][ri]
            vz:NDArray[np.float64] = vComps[2][ri]
            v:NDArray[np.float64] = np.linalg.norm(np.vstack((vx,vy,vz)),axis=0)


            majorAnnuliIndices:list[None|tuple[int,int]] = majorAnnuli(x,y,Nmin,Nmax)
            overlappingAnnuliIndices:list[None|tuple[int,int]] = overlapAnnuli(majorAnnuliIndices,r)
            strengthsArr,galaxyStrength = barStrength(x,y,mass,overlappingAnnuliIndices,nphi,mMax)

            strengths[i] = np.array([snapshot,galaxyStrength])
            if majors and snapshot > min(majors):
                # plotBarStrengthPerAnnulus(strengthsArr,overlappingAnnuliIndices,r,snapshot,galaxy)
                majors.pop(0)
                print(snapshot)
        plotGalaxyStrengthPerSnapshot(strengths,galaxy)

if __name__ == '__main__':
    main()
