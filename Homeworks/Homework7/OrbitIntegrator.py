# Christian Burt
# ASTR 400B
# Homework 7
# 28 March 2025

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from numpy.typing import NDArray

from CenterOfMass2 import CenterOfMass as COM
from GalaxyMass import ComponentMass as CompMass
from OrbitCOM import rvRel

G:float = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value # gravitational constant in Msun*kpc*km^2*s^-2


class M33AnalyticOrbit:
    """ This class calculates the analytical orbit of M33 around M31 assuming point masses """
    
    def __init__(self,filename:str) -> None: # **** add inputs
        """
        This method initializes the M33AnalyticOrbit class

        Inputs:
            filename - the name of the output txt file
        """

        self.filename = filename # stores filename as class variable
        
        M33com:CenterOfMass2.CenterOfMass = COM('M33_000.txt')

        delta = 0.1 # defines when to stop the shrinking-sphere method.
        M33p:NDArray[np.float64] = M33com.COM_P(delta).value # The COM position of M33
        M33v:NDArray[np.float64] = M33com.COM_V(*(M33p*u.kpc)).to(u.kpc/u.Gyr).value # The COM velocity of M33
        
        M31com:CenterOfMass2.CenterOfMass = COM('M31_000.txt')

        M31p:NDArray[np.float64] = M31com.COM_P(delta).value # The COM position of M31
        M31v:NDArray[np.float64] = M31com.COM_V(*(M31p*u.kpc)).to(u.kpc/u.Gyr).value # The COM velocity of M33
        
        self.r:NDArray[np.float64] = M33p-M31p # relative position between the COMs of M33 and M31
        self.v:NDArray[np.float64] = M33v-M31v # relative velocity between the COMs of M33 and M31
        
        # Scale length of the disk
        self.rdisk:float = 5 # kpc

        # The total mass of the disk particles in M31 calculated using the ComponentMass from GalaxyMass.py
        self.Mdisk:float = CompMass('M31_000.txt').value*1e12 # Msun
        
        # Scale length of the bulge
        self.rbulge:float = 1 # kpc

        # The total mass of the bulge particles in M31 calculated using the ComponentMass from GalaxyMass.py
        self.Mbulge:float = CompMass('M31_000.txt',3).value*1e12 # Msun
        
        # Scale length of the Halo taken from aM31 in the main function of MassProfile.py
        self.rhalo = 59 # kpc

        # The total mass of the halo particles in M31 calculated using the ComponentMass from GalaxyMass.py
        self.Mhalo:float = CompMass('M31_000.txt',1).value*1e12 # Msun
     
    
    
    def HernquistAccel(self,M:float,r_a:float,r:NDArray[np.float64]) -> NDArray[np.float64]:
        """
        This method calculates the gravitational acceleration induced by a Hernquist Profile

        Inputs:
            M (Msun): The total mass of the specified galactic particles (e.g. self.Mhalo)
            r_a (kpc): The scale length of the specified galactic particles (e.g. self.rhalo)
            r (kpc): The relative position vector between the COMs of M33 and M31 

        Outputs:
            acc (kpc/Gyr^2): The gravitational acceleration induced by a Hernquist Profile
        """
        
        r_mag:np.float64 = np.linalg.norm(r) # the magnitude of r
        acc:NDArray[np.float64] = -G*M/(r_mag*(r_a+r_mag)**2)*r # The formula for the Hernquist Acceleration
        
        return acc
    
    def MiyamotoNagaiAccel(self,M:float,r_d:float,r:NDArray[np.float64]) -> NDArray[np.float64]:
        """
        This method calculates the gravitational acceleration induced by a Miyamoto-Nagai Profile

        Inputs:
            M (Msun): The total mass of the specified galactic particles 
            r_d (kpc): The scale length of the disk particles
            r (kpc): The relative position vector between the COMs of M33 and M31 

        Outputs:
            acc (kpc/Gyr^2): The gravitational acceleration induced by a Miyamoto-Nagai Profile
        """

        z_d:float = r_d/5.0 # a scale length for the z axis
        x,y,z = r # This is (in my opinion), more readable for the definition of R and B

        # Specifically defined quantities
        R:float = np.linalg.norm((x,y))
        B:float = r_d+np.linalg.norm((z,z_d))

        acc:NDArray[np.float64] = -G*M/(np.linalg.norm((R,B)))**3
        acc *= np.array([1,1,B/np.linalg.norm((z,z_d))]) # The formula for the Miyamoto-Nagai Acceleration

        return acc
    
    def M31Accel(self,r:NDArray[np.float64]) -> NDArray[np.float64]: # input should include the position vector, r
        """
        This method adds the gravitational acceleration values on different particle types
        induced by a Hernquist and Miyamoto-Nagai Profile

        Inputs:
            r (kpc): The relative position vector between the COMs of M33 and M31 

        Outputs:
            acc (kpc/Gyr^2): The gravitational acceleration induced by a Miyamoto-Nagai Profile
        """


        accel:NDArray[np.float64] = np.zeros(3)
        accel += self.HernquistAccel(self.Mhalo,self.rhalo,r) # Halo particles
        accel += self.MiyamotoNagaiAccel(self.Mdisk,self.rdisk,r) # Disk particles
        accel += self.HernquistAccel(self.Mbulge,self.rbulge,r) # Bulge particles
        
            
        return accel # Total acceleration
    
    def LeapFrog(
            self,
            dt:float,
            r:NDArray[np.float64],
            v:NDArray[np.float64]
    ) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
        """ 
        This method calculates a timestep of particle interaction using the Leapfrog integration technique.
        Inputs:
            r (kpc): The relative position vector between the COMs of M33 and M31 
            v (kpc/Gyr): The relative position vector between the COMs of M33 and M31 

        Outputs:
            rnew (kpc): The advancement of r by one timestep
            vnew (kpc/Gyr): The advancement of r by one timestepThe relative position vector between the COMs of M33 and M31 
        """

        # predict the 3d position vector at the next half timestep
        rhalf:NDArray[np.float64] = r+v*dt/2
        
        # predict the 3d velocity vector at the next full timestep using the acceleration field at rhalf 
        vnew:NDArray[np.float64] = v+self.M31Accel(rhalf)*dt 
        
        # repeat the first step using the new velocity 
        rnew:NDArray[np.float64] = rhalf+vnew*dt/2
        
        return rnew,vnew 
    
    def OrbitIntegration(self,t0:float,dt:float,tmax:float) -> None:
        """
        This method runs self.LeapFrog from t0 to tmax.
        Inputs:
            t0 (Gyr): The lower temporal bound of the integrator 
            dt (Gyr): The timestep interval of the leapfrog integrator
            tmax (Gyr): The upper temporal bound of the integrator 
        """
        if os.path.exists(self.filename): # An if statement that allows me to run this function
                                          # without it reading already existing files.
            return


        t:float = t0 # time
        
        orbit:NDArray[np.float64] = np.zeros((int(tmax/dt)+2,7)) # stores t,x,y,z,vx,vy,vz for each timestep
        orbit[0] = t0, *self.r, *self.v # initial values
        i = 0 # an iterator to track the row of orbit
        
        while (t < tmax):  # the leapfrog integrator

            # increment
            i += 1
            t += dt
           
            self.r,self.v = self.LeapFrog(dt,self.r,self.v) # One timestep of the leapfrog integrator 
            orbit[i] = t,*self.r,*self.v # save the value in the ith row of orbit
            
        # write orbit to self.filename
        np.savetxt(self.filename, orbit, fmt = "%11.3f"*7, comments='#', 
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                   .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        
def main() -> None:
    filename = 'M33AnalyticOrbit.txt' # txt file to store orbital parameters
    M33 = M33AnalyticOrbit(filename)

    t0,dt,tmax = 0,0.1,10 # start,timestep,end of the integrator
    M33.OrbitIntegration(t0,dt,tmax)

    figp,axp = plt.subplots() # position plot
    figv,axv = plt.subplots() # velocity plot
    
    # Obtaining data from the txt file
    data:NDArray[object] = np.genfromtxt(filename,dtype=None,names=True)
    t:NDArray[np.float64] = data['t']

    x:NDArray[np.float64] = data['x']
    y:NDArray[np.float64] = data['y']
    z:NDArray[np.float64] = data['z']
    r:NDArray[np.float64] = np.linalg.norm(np.vstack((x,y,z)),axis=0)

    vx:NDArray[np.float64] = data['vx']
    vy:NDArray[np.float64] = data['vy']
    vz:NDArray[np.float64] = data['vz']
    vr:NDArray[np.float64] = np.linalg.norm(np.vstack((vx,vy,vz)),axis=0)

    vr = (vr*u.kpc/u.Gyr).to(u.km/u.s).value

    # Obtaining data from the N-body simulator. See OrbitCOM.py
    M31file:str = f"orbit_M31.txt"
    M33file:str = f"orbit_M33.txt"

    M31data:NDArray[object] = np.genfromtxt(M31file,dtype=None,names=True)
    M33data:NDArray[object] = np.genfromtxt(M33file,dtype=None,names=True)

    M33pM31:NDArray[np.float64]
    M33vM31:NDArray[np.float64]
    M33pM31,M33vM31 = rM33pM31, M33vM31 = [r.value for r in rvRel(M33data, M31data)]

    tOrbitCOM:NDArray[np.float64] = M31data['t']


    # Plot the position calculated from the orbit integrator and compare it to the N-body simulation
    axp.plot(t,r,c='r',label = 'Leapfrog Integration')
    axp.plot(tOrbitCOM,M33pM31,c='c',ls='-.',label='van der Marel+12 \nsimulation')
    axp.set(title='Relative distance between M31 and M33',xlabel='time [Gyr]',ylabel = 'separation [kpc]')
    axp.legend()

    # Plot the velocity calculated from the orbit integrator and compare it to the N-body simulation
    axv.plot(t,vr,c='r',label = 'Leapfrog Integration')
    axv.plot(tOrbitCOM,M33vM31,c='c',ls='-.',label='van der Marel+12 \nsimulation')
    axv.set(title='Relative speed between M31 and M33',xlabel='time [Gyr]',ylabel = 'speed [km/s]')
    axv.legend()

    # Save the figures
    figp.savefig('M33-M31-rel-pos.png')
    figv.savefig('M33-M31-rel-vel.png')

if __name__ == '__main__':
    main()
