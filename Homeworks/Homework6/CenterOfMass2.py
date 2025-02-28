
# Homework 4
# Center of Mass Position and Velocity
# Solutions: G.Besla, R. Li, H. Foote


# remember this is just a template, you don't need to follow every step
# if you have your own method to solve the homework, it is totally fine



# import modules
import numpy as np
from numpy.typing import NDArray
import astropy.units as u
import astropy.table as tbl
from ReadFile import Read

class CenterOfMass:
# Class to define COM position and velocity properties of a given galaxy 
# and simulation snapshot

    def __init__(self, filename:str, ptype:int) -> None:
        ''' Class to calculate the 6-D phase-space position of a galaxy's center of mass using
        a specified particle type. 
            
            PARAMETERS
            ----------
            filename : `str`
                snapshot file
            ptype : `int; 1, 2, or 3`
                particle type to use for COM calculations
        '''
     
        # read data in the given file using Read
        self.time: NDArray[np.float64]
        self.total: NDArray[np.float64]
        self.data: NDArray[np.void]
        self.time,self.total,self.data = Read(filename)

        # create an array to store indexes of particles of desired Ptype
        self.index:tuple[NDArray[np.int_],...] = np.where(self.data['type'] == ptype)

        # store the mass, positions, velocities of only the particles of the given type
        # the following only gives the example of storing the mass
        self.m:NDArray[np.float64] = self.data['m'][self.index]
        # write your own code to complete this for positions and velocities
        self.x:NDArray[np.float64] = self.data['x'][self.index]
        self.y:NDArray[np.float64] = self.data['y'][self.index]
        self.z:NDArray[np.float64] = self.data['z'][self.index]
        self.vx:NDArray[np.float64] = self.data['vx'][self.index]
        self.vy:NDArray[np.float64] = self.data['vy'][self.index]
        self.vz:NDArray[np.float64] = self.data['vz'][self.index]

    def COMdefine(
        self,
        a:float|NDArray[np.float64]|u.Quantity,
        b:float|NDArray[np.float64]|u.Quantity,
        c:float|NDArray[np.float64]|u.Quantity,
        m:float|NDArray[np.float64]|u.Quantity
    ) -> tuple[np.float64,...]:

        ''' Method to compute the COM of a generic vector quantity by direct weighted averaging.
        
        PARAMETERS
        ----------
        a : `float or np.ndarray of floats`
            first vector component
        b : `float or np.ndarray of floats`
            second vector component
        c : `float or np.ndarray of floats`
            third vector component
        m : `float or np.ndarray of floats`
            particle masses
        
        RETURNS
        -------
        a_com : `float`
            first component on the COM vector
        b_com : `float`
            second component on the COM vector
        c_com : `float`
            third component on the COM vector
        '''
        
        # write your own code to compute the generic COM

        # Converts floats to numpy arrays. Does not affect numpy array inputs
        m,a,b,c = np.asarray(m),np.asarray(a),np.asarray(b),np.asarray(c)
        
        #using Eq. 1 in the homework instructions
        # xcomponent Center of mass
        a_com:np.float64 = np.sum(m*a)/np.sum(m)
        # ycomponent Center of mass
        b_com:np.float64 = np.sum(m*b)/np.sum(m)
        # zcomponent Center of mass
        c_com:np.float64 = np.sum(m*c)/np.sum(m)
        
        # return the 3 components separately
        return a_com,b_com,c_com

    def COM_P(self,delta:float,volDec:float=2.0) -> u.Quantity:
        '''
        Method to compute the position of the center of mass of the galaxy 
        using the shrinking-sphere method.

        PARAMETERS
        ----------
        delta : `float, optional`
            error tolerance in kpc. Default is 0.1 kpc
        
        RETURNS
        ----------
        p_COM : `np.ndarray of astropy.Quantity'
            3-D position of the center of mass in kpc
        '''

        # Center of Mass Position
        ###########################

        # Try a first guess at the COM position by calling COMdefine
        x_COM:np.float64
        y_COM:np.float64
        z_COM:np.float64
        x_COM, y_COM, z_COM = self.COMdefine(self.x, self.y, self.z, self.m)
        # compute the magnitude of the COM position vector.
        # write your own code below
        r_COM:np.float64 = np.linalg.norm((x_COM,y_COM,z_COM))

        
        # iterative process to determine the center of mass

        # change reference frame to COM frame
        # compute the difference between particle coordinates
        # and the first guess at COM position
        # write your own code below
        x_new:NDArray[np.float64] = self.x-x_COM
        y_new:NDArray[np.float64] = self.y-y_COM
        z_new:NDArray[np.float64] = self.z-z_COM
        r_new:NDArray[np.float64] = np.linalg.norm(np.vstack((x_new,y_new,z_new)),axis=0)

        # find the max 3D distance of all particles from the guessed COM
        # will re-start at half that radius (reduced radius)
        r_max:np.float64 = np.max(r_new)/volDec
        
        # pick an initial value for the change in COM position
        # between the first guess above and the new one computed from half that volume
        # it should be larger than the input tolerance (delta) initially
        change:float = 1000.0

        # start iterative process to determine center of mass position
        # delta is the tolerance for the difference in the old COM and the new one.    
        
        while (change > delta):
            # select all particles within the reduced radius (starting from original x,y,z, m)
            # write your own code below (hints, use np.where)
            index2:tuple[NDArray[np.int_],...] = np.where(r_new <= r_max)
            x2:NDArray[np.float64] = self.x[index2]
            y2:NDArray[np.float64] = self.y[index2]
            z2:NDArray[np.float64] = self.z[index2]
            m2:NDArray[np.float64] = self.m[index2]

            # Refined COM position:
            # compute the center of mass position using
            # the particles in the reduced radius
            # write your own code below
            x_COM2:np.float64
            y_COM2:np.float64
            z_COM2:np.float64
            x_COM2, y_COM2, z_COM2 = self.COMdefine(x2,y2,z2,m2)
            # compute the new 3D COM position
            # write your own code below
            r_COM2:np.float64 = np.linalg.norm((x_COM2, y_COM2, z_COM2))

            # determine the difference between the previous center of mass position
            # and the new one.
            change = np.abs(r_COM-r_COM2)

            # reduce the volume by a factor of 2 again
            r_max /= volDec

            # Change the frame of reference to the newly computed COM.
            # subtract the new COM
            # write your own code below
            x_new = self.x-x_COM2
            y_new = self.y-y_COM2
            z_new = self.z-z_COM2
            r_new = np.linalg.norm(np.vstack((x_new,y_new,z_new)),axis=0) 

            # set the center of mass positions to the refined values
            x_COM,y_COM,z_COM,r_COM = x_COM2,y_COM2,z_COM2,r_COM2

        # set the correct units using astropy and round all values
        # and then return the COM positon vector
        # write your own code below

        p_COM_arr:NDArray[np.float64] = np.array([x_COM, y_COM, z_COM])
        p_COM:u.Quantity = np.round(p_COM_arr,2)*u.kpc
        return p_COM
    
        
    def COM_V(self,x_COM:u.Quantity,y_COM:u.Quantity,z_COM:u.Quantity) -> u.Quantity:
        ''' Method to compute the center of mass velocity based on the center of mass
        position.

        PARAMETERS
        ----------
        x_COM : 'astropy quantity'
            The x component of the center of mass in kpc
        y_COM : 'astropy quantity'
            The y component of the center of mass in kpc
        z_COM : 'astropy quantity'
            The z component of the center of mass in kpc
            
        RETURNS
        -------
        v_COM : `np.ndarray of astropy.Quantity'
            3-D velocity of the center of mass in km/s
        '''
        
        # the max distance from the center that we will use to determine 
        #the center of mass velocity                   
        rv_max:u.Quantity = 15.0*u.kpc

        # determine the position of all particles relative to the center
        #     of mass position (x_COM, y_COM, z_COM)
        # write your own code below
        # Note that x_COM, y_COM, z_COM are astropy quantities and you
        #     can only subtract one astropy quantity from another
        # So, when determining the relative positions, assign the appropriate units to self.x
        xV:u.Quantity = self.x*u.kpc-x_COM
        yV:u.Quantity = self.y*u.kpc-y_COM
        zV:u.Quantity = self.z*u.kpc-z_COM
        rV:u.Quantity = np.linalg.norm(np.vstack((xV,yV,zV)),axis=0)
        
        # determine the index for those particles within the max radius
        # write your own code below
        indexV:tuple[NDArray[np.int_],...] = np.where(rV < rv_max)
        
        # determine the velocity and mass of those particles within the mas radius
        # write your own code below
        vx_new:NDArray[np.float64] = self.vx[indexV]
        vy_new:NDArray[np.float64] = self.vy[indexV]
        vz_new:NDArray[np.float64] = self.vz[indexV]
        m_new:NDArray[np.float64] = self.m[indexV]
        
        # compute the center of mass velocity using those particles
        # write your own code below
        vx_COM:np.float64
        vy_COM:np.float64
        vz_COM:np.float64
        vx_COM,vy_COM,vz_COM = self.COMdefine(vx_new,vy_new,vz_new,m_new)
        
        # create an array to store the COM velocity
        # write your own code below
        v_COM_arr:NDArray[np.float64] = np.array([vx_COM, vy_COM, vz_COM])

        # return the COM vector
        # set the correct units usint astropy
        # round all values
        v_COM:u.Quantity = np.round(v_COM_arr,2)*u.km/u.s
        return v_COM

        
# ANSWERING QUESTIONS
#######################
  
def main() -> None:

    # Create a Center of mass object for the MW, M31 and M33
    # below is an example of using the class for MW
    MW_COM:CenterOfMass = CenterOfMass("MW_000.txt", 2)

    print('\nQUESTION 1\n')
    
    # MW:   store the position and velocity COM
    MW_COM_p:u.Quantity = MW_COM.COM_P(0.1)
    MW_COM_v:u.Quantity = MW_COM.COM_V(MW_COM_p[0], MW_COM_p[1], MW_COM_p[2])
    print(f"The Milky Way has a center-of-mass position vector of {MW_COM_p}")
    print(f"and a velocity vector of {MW_COM_v}")

    # now write your own code to answer questions

    # Now finding the position and velocity vectors for M31...
    M31_COM:CenterOfMass = CenterOfMass("M31_000.txt", 2)
    M31_COM_p:u.Quantity = M31_COM.COM_P(0.1)
    M31_COM_v:u.Quantity = M31_COM.COM_V(M31_COM_p[0], M31_COM_p[1], M31_COM_p[2])
    print(f"\nM31 has a center-of-mass position vector of {M31_COM_p}")
    print(f"and a velocity vector of {M31_COM_v}")

    # And M33
    M33_COM:CenterOfMass = CenterOfMass("M33_000.txt", 2)
    M33_COM_p:u.Quantity = M33_COM.COM_P(0.1)
    M33_COM_v:u.Quantity = M33_COM.COM_V(M33_COM_p[0], M33_COM_p[1], M33_COM_p[2])
    print(f"\nM33 has a center-of-mass position vector of {M33_COM_p}")
    print(f"and a velocity vector of {M33_COM_v}")

    print('\nQUESTION 2\n')
    
    delta_r_MW:u.Quantity = np.linalg.norm(MW_COM_p-M31_COM_p)
    print(f"The separation between MW and M31 is \n{delta_r_MW:.3f}")

    delta_v_MW:u.Quantity = np.linalg.norm(MW_COM_v-M31_COM_v)
    print(f"The relative velocity of MW with respect to M31 is \n{delta_v_MW:.3f}")

    print('\nQUESTION 3\n')
    
    delta_r_M33:u.Quantity = np.linalg.norm(M33_COM_p-M31_COM_p)
    print(f"The separation between M33 and M31 is \n{delta_r_M33:.3f}")

    delta_v_M33:u.Quantity = np.linalg.norm(M33_COM_v-M31_COM_v)
    print(f"The relative velocity of M33 with respect to M31 is \n{delta_v_M33:.3f}")

    print('\nQUESTION 4\n')

    print(
        '''Using an iterative process to find the center of mass will be critical when
        stars between the Milky Way and Andromeda start mixing. Gravitational interactions
        will greatly increase the distance of some stars from the center, creating outliers
        that will skew the data, and an iterative method will remove them from the dataset
        ''')
    
if __name__ == '__main__' : 
    main()
