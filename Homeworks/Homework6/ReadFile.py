import numpy as np
import astropy.units as u

def Read(filename:str) -> tuple:
    '''
        This function reads a text file, extracts the time and total particle count, and organizees the remaining data into a numpy array

    Inputs:
        filename: The txt file in which the particle data is stored (e.g. MW_000.txt)

    Outputs:
        time (Myr): The time at a given SnapNumber
        count: An integer denoting the total number of particles in the text file
        data: A numpy array generated from the text file containing information regarding the type, mass, distance, and velocity of each particle
    '''
    with open(filename,'r') as f: # Opens the text file.
        # NOTE: I use this "with open..." structure because it automatically closes files, avoiding any potential errors regarding file cleanup
        line1 = f.readline() # Reads the first line (the time) of the text file
        label,value = line1.split() # Splits the line into its label and value
        time = float(value)*u.Myr # Converts the value from a string to a float and adds the appropriate astropy units

        line2 = f.readline() # Reads the second line (the count) of the text file
        label,value = line2.split() # Splits the line into its label and value
        count = int(value) # Converts the value from a string to an integer

    data = np.genfromtxt(filename,dtype=None,names=True,skip_header=3) # Stores the remaining data in a numpy array

    return time,count,data
