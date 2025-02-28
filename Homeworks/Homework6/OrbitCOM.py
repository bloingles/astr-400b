# Homework 6 Template
# G. Besla & R. Li

# import modules
import numpy as np
from numpy.typing import NDArray
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
import matplotlib
import os

from ReadFile import Read
from CenterOfMass2 import CenterOfMass

def OrbitCOM(galaxy:str,start:int,end:int,n:int) -> None:
    """This function that loops over all the desired snapshots to compute the COM pos and vel as a function of time and saves the value to a txt file.
    Inputs:
        galaxy: The name of a galaxy as it appears in the desired txt file
        start: The first snapshot needed for analysis
        end: The last snapshot needed for analysis
        n: The interval between each snapshot
    """
    
    # compose the filename for output
    fileout:str = f"orbit_{galaxy}.txt"

    if os.path.exists(fileout): # An if statement that allows me to run this function
                                # without it reading already existing files.
        return
    
    #  set tolerance and VolDec for calculating COM_P in CenterOfMass
    # for M33 that is stripped more, use different values for VolDec

    delta:float = 0.1
    volDec:float = 4.0 if galaxy == 'M33' else 2.0
    
    # generate the snapshot id sequence 
    # it is always a good idea to also check if the input is eligible (not required)
    if type(start) is type(end) is type(n) is int:
        end += n - (end-start) % n # This ensures the last value is less than
                                   # OR EQUAL TO the value of end 
        snap_ids:NDArray[np.int_] = np.arange(start,end,n)
    else:
        raise TypeError("Ensure start, end, and n are of type int.")
    
    # initialize the array for orbital info: t, x, y, z, vx, vy, vz of COM
    orbit:NDArray[np.float64] = np.zeros((snap_ids.size,7))
    
    for i,snap_id in enumerate(snap_ids): # loop over files
        # compose the data filename (be careful about the folder)
        filename:str = f"{galaxy}_{snap_id%1000:03d}.txt"
        filepath:str = os.path.join(galaxy,filename)

        # Initialize an instance of CenterOfMass class, using disk particles
        COM:CenterOfMass = CenterOfMass(filepath,2)

        # Store the COM pos and vel. Remember that now COM_P required VolDec
        COM_p:u.Quantity = COM.COM_P(0.1,volDec)
        COM_v:u.Quantity = COM.COM_V(COM_p[0],COM_p[1],COM_p[2])
    
        # store the time, pos, vel in ith element of the orbit array,  without units (.value) 
        # note that you can store 
        # a[i] = var1, *tuple(array1)
        orbit[i] = COM.time.value/1e3,*COM_p.value,*COM_v.value
        
        # print snap_id to see the progress
        print(snap_id)
        
    # write the data to a file
    # we do this because we don't want to have to repeat this process 
    # this code should only have to be called once per galaxy.
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                      .format('t','x','y','z','vx','vy','vz'))

# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  
def rvRel(
    datarray1: NDArray[object], 
    datarray2: NDArray[object]
) -> tuple[u.Quantity,u.Quantity]:
    """This function computes the magnitude of the difference between two vectors.
    
    It returns both the relative position and relative velocity for two 
    galaxies over the entire orbit.
    Inputs:
        datarray1: The array generated from the txt file of the COM positions and
                       velocities of one galaxy
        datarray2: The corresponding array for the galaxy for comparison

    Outputs:
        rRel: An array containing the magnitude of displacement between
                  the galaxies' COM positions in kpc
        vRel: An array containing the magnitude of velocity between
                 the galaxies' COM velocities in km/s
    """
    # Distance
    rRel:u.Quantity = np.linalg.norm(
        np.vstack((
            datarray1['x'] - datarray2['x'], 
            datarray1['y'] - datarray2['y'], 
            datarray1['z'] - datarray2['z']
        )), 
        axis=0
    )*u.kpc

    # Speed
    vRel:u.Quantity = np.linalg.norm(
        np.vstack((
            datarray1['vx'] - datarray2['vx'], 
            datarray1['vy'] - datarray2['vy'], 
            datarray1['vz'] - datarray2['vz']
        )), 
        axis=0
    )*u.km/u.s
    return rRel,vRel


def main() -> None:

    # Recover the orbits and generate the COM files for each galaxy
    # read in 800 snapshots in intervals of n=5
    # Note: This might take a little while - test your code with a smaller number of snapshots first!
    
    # READ BEFORE RUNNING: As I have written the function, its can ONLY generate new functions,
    #                      it CANNOT replace existing files. If you would like to generate new
    #                      files, delete or rename the existing files
    
    OrbitCOM('MW',0,800,5)
    OrbitCOM('M31',0,800,5)
    OrbitCOM('M33',0,800,5)

    # I will be honest, I don't know why we don't just output this from the function
    MWfile:str = f"orbit_MW.txt"
    M31file:str = f"orbit_M31.txt"
    M33file:str = f"orbit_M33.txt"

    # Read in the data files for the orbits of each galaxy that you just created
    # headers:  t, x, y, z, vx, vy, vz
    # using np.genfromtxt
    MWdata:NDArray[object] = np.genfromtxt(MWfile,dtype=None,names=True)
    M31data:NDArray[object] = np.genfromtxt(M31file,dtype=None,names=True)
    M33data:NDArray[object] = np.genfromtxt(M33file,dtype=None,names=True)
    # Stores the remaining data in a numpy array




    # Determine the magnitude of the relative position and velocities 

    # of MW and M31
    MWpM31:u.Quantity
    MWvM31:u.Quantity
    MWpM31,MWvM31 = rvRel(MWdata,M31data)

    # of M33 and M31
    M33pM31:u.Quantity
    M33vM31:u.Quantity
    M33pM31,M33vM31 = rvRel(M33data,M31data)

    # Plot the Orbit of the galaxies 
    #################################
    time = M31data['t']

    figp,axp = plt.subplots()
    axp.plot(time,MWpM31,color='r',label='MW-M31')
    axp.plot(time,M33pM31,color='c',linestyle='-.',label='M33-M31')
    axp.set(xlabel='Time [Gyr]',ylabel='Separation [kpc]')
    axp.legend()
    figp.savefig('LG_separation.png')

    # Plot the orbital velocities of the galaxies 
    #################################
    figv,axv = plt.subplots()
    axv.plot(time,MWvM31,color='r',label='MW-M31')
    axv.plot(time,M33vM31,color='c',linestyle='-.',label='M33-M31')
    axv.set(xlabel='Time [Gyr]',ylabel='Speed [km/s]')
    axv.legend()
    figv.savefig('LG_speed.png')

    # plt.show()
    # I used this to find specific values using the zoom-in tool

if __name__ == '__main__':
    main()
