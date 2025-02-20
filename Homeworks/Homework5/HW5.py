import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
from ReadFile import Read
from CenterOfMass import CenterOfMass as COM

class MassProfile:
# Class to determine the mass profile and rotation curve of a given galaxy at a given snapshot
    def __init__(self,galaxy:str,snap:int) -> None:
        '''
        This class accepts a galaxy and snapshot value to access a corresponding
        txt file and uses the data to generate a mass profile and rotation curve

        Inputs:
            galaxy: The name of a galaxy as it appears in the desired txt file.
            snap (Myr): The specific instance in time of a galaxy
        '''
        # fixes all snapshots to 3 digits and assembles the filename
        ilbl = '000' + str(snap)
        ilbl = ilbl[-3:]
        self.filename=f"{galaxy}_{ilbl}.txt"

        # reads and saves all data using Read from ReadFile
        self.time, self.total, self.data = Read(self.filename)
        self.ptype = self.data['type']
        self.m = self.data['m']
        self.x = self.data['x']
        self.y = self.data['y']
        self.z = self.data['z']
        self.vx = self.data['vx']
        self.vy = self.data['vy']
        self.vz = self.data['vz']

        self.gname = galaxy # saves the name of the galaxy

    def MassEnclosed(self,ptype:int,r:np.ndarray[np.float64]) -> u.Quantity:
        '''
        This method finds the total masses of a certain type of particle
        enclosed in a given array of radii

        Inputs:
            pType: The desired paricle type where
                1 - Dark Matter Halo
                2 - Disk Stars
                3 - Bulge Stars

            r (kpc): An array of radii

        Outputs:
            m_enc (Msun): An array of masses enclosed in the values in r
        '''
        # COM is a Class from CenterOfMass that finds the center of mass of a
        # galaxy per a given particle type
        com = COM(self.filename,2) # ptype = 2 (disk)
        com_p = com.COM_P(0.1) # delta = 0.1

        # filtering for a specific particle
        pIndices = np.where(ptype == self.ptype)
        x,y,z,m = self.x[pIndices],self.y[pIndices],self.z[pIndices],self.m[pIndices]
        m_enc = np.zeros(r.size) # allocates space for the enclosed mass array

        for i,r_i in enumerate(r): # iterate through radii
            xcom = x*u.kpc-com_p[0] # find x with respect to the center of mass
            ycom = y*u.kpc-com_p[1] # find y with respect to the center of mass
            zcom = z*u.kpc-com_p[2] # find z with respect to the center of mass
            R = np.linalg.norm(np.vstack((xcom,ycom,zcom)), axis=0) # total distance to the
                                                                    # center of mass position
            rIndices = np.where(R < r_i*u.kpc) # removes particles with radii larger than the
                                               # iteration radius r_i
            m_enc[i] = np.sum(self.m[rIndices]) # sums all the masses of a given particle type
                                                # enclosed in r_i
        m_enc = m_enc*10**10*u.Msun # Apply units before returning
        return m_enc

    def MassEnclosedTotal(self,r:np.ndarray[np.float64]) -> u.Quantity:
        '''
        This method sums the values of m_enc found in self.MassEnclosed
        for each particle type

        Inputs:
            r (kpc): An array of radii

        Outputs:
            m_tot (Msun): The total array of masses.
        '''
        m_tot = 0 # Initial total mass
        n = 3 if self.gname == 'M33' else 4 # M33 doesn't have bulge particles
        m_tot += sum(self.MassEnclosed(i,r) for i in range(1,n)) # total mass enclosed in r
        return m_tot

    def HernquistMass(self,r:float,a:float,Mhalo:u.Quantity) -> u.Quantity:
        '''
        This method calculates the Hernquist mass profile of a given radius

        Inputs:
            r (kpc): a radius value
            a (kpc): the scale radius. Determined by eye
            Mhalo (kpc): the halo mass enclosed in 30 kpc
        Outputs:
            M (Msun): The hernquist mass
        '''
        M = Mhalo*r**2/(a+r)**2 # Hernquist mass profile equation
        return M

    def CircularVelocity(self,ptype:int,r:np.ndarray[np.float64]) -> u.Quantity:
        '''
        This method finds the circular velocity using the mass enclosed from
        self.MassEnclosed and r. 

        Inputs:
            pType: The desired paricle type where
                1 - Dark Matter Halo
                2 - Disk Stars
                3 - Bulge Stars

            r (kpc): An array of radii

        Outputs:
            v (km/s): The circular velocity from a given radius
        '''
        M = self.MassEnclosed(ptype,r)
        v = np.sqrt(G*M/(r*u.kpc)) # Derived from conservation of energy
        return v

    def CircularVelocityTotal(self,r:np.ndarray[np.float64]) -> u.Quantity:
        '''
        This method finds the circular velocity using the total mass enclosed
        from self.MassEnclosedTotal and r. 

        Inputs:
            r (kpc): An array of radii

        Outputs:
            v (km/s): The circular velocity from a given radius
        '''
        M = self.MassEnclosedTotal(r)
        v = np.sqrt(G*M/(r*u.kpc))
        return v

    def HernquistVCirc(self,r:float,a:float,Mhalo:u.Quantity) -> u.Quantity:
        '''
        This method finds the circular velocity following the Hernquist profile.

        Inputs:
            r (kpc): a radius value
            a (kpc): the scale radius. Determined by eye
            Mhalo (kpc): the halo mass enclosed in 30 kpc
        Outputs:
            v (km/s): the circular velocity from the hernquist mass
        '''

        M = self.HernquistMass(r,a,Mhalo)
        v = np.sqrt(G*M/(r*u.kpc))
        return v

def main() -> None:
    global G
    G = G.to(u.kpc*u.km**2/u.s**2/u.Msun) # gravitational constant in Msun*kpc*km^2*s^-2
    r = np.arange(0.1,30.1,0.1) # radius values from 0.1-30 (delta: 0.1)

    # Mass Profiles
    MW = MassProfile('MW',0)
    M31 = MassProfile('M31',0)
    M33 = MassProfile('M33',0)

    aMW,aM31,aM33 = 18,16,11 # Scale radii
    MWhalo = MW.MassEnclosed(1,np.array([30])) # MW halo mass enclosed in 30 kpc
    M31halo = M31.MassEnclosed(1,np.array([30])) # M31 halo mass enclosed in 30 kpc
    M33halo = M33.MassEnclosed(1,np.array([30])) # M33 halo mass enclosed in 30 kpc
    
    
    ### QUESTION 8: Milky Way Mass Profile
    
    figMW,axMW = plt.subplots()
    colors = ['b','r','g']
    ptypes = ['halo','disk','bulge']
    # plots mass of a given particle enclosed in r
    [axMW.plot(r, MW.MassEnclosed(i+1, r),c=colors[i], label=ptypes[i]) for i in range(3)]
    # plots total mass enclosed in r
    axMW.plot(r,MW.MassEnclosedTotal(r),c='k',label='total',linestyle='dashed')
    # plots the Hernquist mass profile
    axMW.plot(r,MW.HernquistMass(r,aMW,MWhalo),c='orange',label='Hernquist',linestyle='dashdot')
    axMW.set(
        title=r'$\text{MW Mass Profile}\,\,(a=%i\,\text{kpc})$'.replace(r'%i',str(a)),
        xlabel=r"$\text{R}_\text{com}\,[\text{kpc}]$",
        ylabel=r"$\text{M}_\text{enc}\,[\text{M}_\odot]$"
    )
    axMW.legend()
    axMW.semilogy()
    figMW.savefig('MW_Mass_Profile.png')

    ### QUESTION 8: Andromeda Mass Profile
    
    figM31,axM31 = plt.subplots()
    colors = ['b','r','g']
    ptypes = ['halo','disk','bulge']
    # plots mass of a given particle enclosed in r
    [axM31.plot(r, M31.MassEnclosed(i+1, r),c=colors[i], label=ptypes[i]) for i in range(3)]
    # plots total mass enclosed in r
    axM31.plot(r,M31.MassEnclosedTotal(r),c='k',label='total',linestyle='dashed')
    # plots the Hernquist mass profile
    axM31.plot(r,M31.HernquistMass(r,aM31,M31halo),c='orange',label='Hernquist',linestyle='dashdot')
    axM31.set(
        title=r'$\text{M31 Mass Profile}\,\,(a=%i\,\text{kpc})$'.replace(r'%i',str(a)),
        xlabel=r"$\text{R}_\text{com}\,[\text{kpc}]$",
        ylabel=r"$\text{M}_\text{enc}\,[\text{M}_\odot]$"
    )
    axM31.legend()
    axM31.semilogy()
    figM31.savefig('M31_Mass_Profile.png')

    ### QUESTION 8: Triangulum Mass Profile
    
    figM33,axM33 = plt.subplots()
    colors = ['b','r','g']
    ptypes = ['halo','disk','bulge']
    # plots mass of a given particle enclosed in r
    [axM33.plot(r, M33.MassEnclosed(i+1, r),c=colors[i], label=ptypes[i]) for i in range(2)]
    # plots total mass enclosed in r
    axM33.plot(r,M33.MassEnclosedTotal(r),c='k',label='total',linestyle='dashed')
    # plots the Hernquist mass profile
    axM33.plot(r,M33.HernquistMass(r,aM33,M33halo),c='orange',label='Hernquist',linestyle='dashdot')
    axM33.set(
        title=r'$\text{M33 Mass Profile}\,\,(a=%i\,\text{kpc})$'.replace(r'%i',str(a)),
        xlabel=r"$\text{R}_\text{com}\,[\text{kpc}]$",
        ylabel=r"$\text{M}_\text{enc}\,[\text{M}_\odot]$"
    )
    axM33.legend()
    axM33.semilogy()
    figM33.savefig('M33_Mass_Profile.png')

    ### QUESTION 9: Milky Way Rotation Curve
    
    figMWv,axMWv = plt.subplots()
    colors = ['b','r','g']
    ptypes = ['halo','disk','bulge']
    # plots the rotation curve given the mass of a given particle enclosed in r
    [axMWv.plot(r, MW.CircularVelocity(i+1, r),c=colors[i], label=ptypes[i]) for i in range(3)]
    # plots the rotation curve given the total mass enclosed in r
    axMWv.plot(r,MW.CircularVelocityTotal(r),c='k',label='total',linestyle='dashed')
    # plots the rotation curve given the Hernquist mass profile
    axMWv.plot(r,MW.HernquistVCirc(r,aMW,MWhalo),c='orange',label='Hernquist',linestyle='dashdot')
    axMWv.set(
        title=r'$\text{MW Rotation Curve}\,\,(a=%i\,\text{kpc})$'.replace(r'%i',str(a)),
        xlabel=r"$\text{v}_\text{rot}\,[\text{km/s}]$",
        ylabel=r"$\text{M}_\text{enc}\,[\text{M}_\odot]$"
    )
    axMWv.legend()
    figMWv.savefig('MW_Rotation_Curve.png')

     ### QUESTION 9: Andromeda Rotation Curve
    
    figM31v,axM31v = plt.subplots()
    colors = ['b','r','g']
    ptypes = ['halo','disk','bulge']
    # plots the rotation curve given the mass of a given particle enclosed in r
    [axM31v.plot(r, M31.CircularVelocity(i+1, r),c=colors[i], label=ptypes[i]) for i in range(3)]
    # plots the rotation curve given the total mass enclosed in r
    axM31v.plot(r,M31.CircularVelocityTotal(r),c='k',label='total',linestyle='dashed')
    # plots the rotation curve given the Hernquist mass profile
    axM31v.plot(r,M31.HernquistVCirc(r,aM31,M31halo),c='orange',label='Hernquist',linestyle='dashdot')
    axM31v.set(
        title=r'$\text{M31 Rotation Curve}\,\,(a=%i\,\text{kpc})$'.replace(r'%i',str(a)),
        xlabel=r"$\text{v}_\text{rot}\,[\text{km/s}]$",
        ylabel=r"$\text{M}_\text{enc}\,[\text{M}_\odot]$"
    )
    axM31v.legend()
    figM31v.savefig('M31_Rotation_Curve.png')

     ### QUESTION 9: Triangulum Rotation Curve
    
    figM33v,axM33v = plt.subplots()
    colors = ['b','r','g']
    ptypes = ['halo','disk','bulge']
    # plots the rotation curve given the mass of a given particle enclosed in r
    [axM33v.plot(r, M33.CircularVelocity(i+1, r),c=colors[i], label=ptypes[i]) for i in range(2)]
    # plots the rotation curve given the total mass enclosed in r
    axM33v.plot(r,M33.CircularVelocityTotal(r),c='k',label='total',linestyle='dashed')
    # plots the rotation curve given the Hernquist mass profile
    axM33v.plot(r,M33.HernquistVCirc(r,aM33,M33halo),c='orange',label='Hernquist',linestyle='dashdot')
    axM33v.set(
        title=r'$\text{M33 Rotation Curve}\,\,(a=%i\,\text{kpc})$'.replace(r'%i',str(a)),
        xlabel=r"$\text{v}_\text{rot}\,[\text{km/s}]$",
        ylabel=r"$\text{M}_\text{enc}\,[\text{M}_\odot]$"
    )
    axM33v.legend()
    figM33v.savefig('M33_Rotation_Curve.png')

if __name__ == '__main__':
    main()
