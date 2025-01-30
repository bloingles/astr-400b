import numpy as np
import astropy.units as u
from ReadFile import Read

def ParticleInfo(filename:str,partType:int,partNumber:int) -> tuple:
    """
        This function analyzes data from a specified file processed in ReadFile.py to extract and return parameters describing the data

    Inputs:
        filename: The txt file in which the particle data is stored (e.g. MW_000.txt)
        partType: The desired paricle type where
            1 - Dark Matter
            2 - Disk Stars
            3 - Bulge Stars
        partNumber: The desired index of the particle of a particular type (e.g. the sixth Disk Star has partType = 2 and partNumber = 6)

    Outputs:
        time (Myr): The time at a given SnapNumber (taken directly from ReadFile.py)
        mass (solMass): The mass of the specified particle
        distance (kpc): The distance of the specified particle from Earth (found as the sum in quadrature of the x,y,z components of the distances
        velocity (km/s): The velocity of the specified particle from Earth (found as the sum in quadrature of the x,y,z components of the velocities
    """

    # Raises an exception if the partType argument is invalid
    if not type(partType) is int:
        raise TypeError("partType requires an integer")
    if partType < 1 or partType > 3 or not isinstance(partType,int):
        raise Exception("partType out of bounds")

    time,count,data = Read(filename) # Reads data from ReadFile.py
    index = np.where(data['type']==partType) # Considers parameters for a specified particle type
    massList = 10**10*data['m'][index]*u.solMass # A list of all masses for a specified particle type
    distanceList = np.sqrt(data['x'][index]**2+data['y'][index]**2+data['z'][index]**2)*u.kpc # A list of all distances for a specified particle type
    velocityList = np.sqrt(data['vx'][index]**2+data['vy'][index]**2+data['vz'][index]**2)*u.km/u.s # A list of all velocities for a specified particle type


    # Raises an exception if the partNumber argument is invalid
    if not type(partNumber) is int:
        raise TypeError("partNumber requires an integer")
    if partNumber < 1 or partNumber > massList.size:
        raise Exception("partNumber out of bounds")

    mass = np.around(massList[partNumber-1],3) # The mass of a specified particle
    distance = np.around(distanceList[partNumber-1],3) # The distance of a specified particle
    velocity = np.around(velocityList[partNumber-1],3) # The velocity of a specified particle

    return time,mass,distance,velocity

def main() -> None:
    filename = 'MW_000.txt'
    partType = 2 # 1 - Dark Matter; 2 - Disk Stars; 3 - Bulge Stars
    partNumber = 100
    time,mass,distance,velocity = ParticleInfo(filename,partType,partNumber)

    lightyears = np.around(distance.to(u.lyr),3) # Converts the distance from kpc to lyr
    
    print(f"\nParticle {partNumber} of type {int(partType)} at time = {time} has the following properties:")
    print(f"Mass = {mass}")
    print(f"Distance = {distance} = {lightyears}")
    print(f"Velocity = {velocity}\n")
    
if __name__ == '__main__':
    main()
