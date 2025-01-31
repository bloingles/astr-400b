
# # In Class Lab 1
# Must be uploaded to your Github repository under a "Labs/Lab1" by midnight thursday

# ## Part A:  The Local Standard of Rest
# Proper motion of Sgr A* from Reid & Brunthaler 2004
# $\mu = 6.379$ mas/yr 
# 
# Peculiar motion of the sun, $v_\odot$ = 12.24 km/s  (Schonrich 2010)
# 
# 
# $v_{tan} = 4.74 \frac{\mu}{\rm mas/yr} \frac{R_o}{\rm kpc} = V_{LSR} + v_\odot$
# 
# 
# ### a)
# 
# Create a function called VLSR to compute the local standard of rest (V$_{LSR}$).
# 
# The function should take as input: the solar radius (R$_o$), the proper motion (mu)
# and the peculiar motion of the sun in the $v_\odot$ direction.
# 
# Compute V$_{LSR}$ using three different values R$_o$: 
# 1. Water Maser Distance for the Sun :  R$_o$ = 8.34 kpc   (Reid 2014 ApJ 783) 
# 2. GRAVITY Collaboration Distance for the Sun:  R$_o$ = 8.178 kpc   (Abuter+2019 A&A 625)
# 3. Value for Distance to Sun listed in Sparke & Gallagher : R$_o$ = 7.9 kpc 
# 



# Import Modules 
import numpy as np # import numpy
import astropy.units as u # import astropy units
from astropy import constants as const # import astropy constants

def VLSR(R_o,mu=6.379,v_pec=12.24*u.km/u.s):
    # (Quantity,int,Quantity,Quantity) -> Quantity
    """
       This function computes the velocity at the local standard of rest

    Inputs:
       R_o (kpc): Distance from Sagittarius A* to the Sun
       mu (mas/yr): Proper motion of Sagittarius A* (Reid,Brunthaler+04)
       v_sun (km/s): the peculiar motion of the Sun in the v direction (Schonrich+10)

    Outputs:
       V_LSR (km/s): The local standard of rest
    
    """
    V_LSR = 4.74*mu*(R_o/u.kpc)*u.km/u.s-v_pec #The formula to find V_LSR
    return V_LSR

# ### b)
# 
# compute the orbital period of the sun in Gyr using R$_o$ from the GRAVITY Collaboration (assume circular orbit)
# 
# Note that 1 km/s $\sim$ 1kpc/Gyr

def TorbSun(R_o,v_c):
    # (Quantity,Quantity) -> Quantity
    '''
        This is a function that computes the orbital period of the Sun in Gyr
    
    Inputs:
        R_o (kpc): Distance from Sagittarius A* to the Sun
        v_c (km/s): Velocity of the Sun in the "v" direction

    Outputs:
        T (Gyr): Orbital Period
    '''
    v_kpc_Gyr = v_c.to(u.kpc/u.Gyr) # Converts v_c from km/s to kpc/Gyr
    T = 2*np.pi*R_o/v_kpc_Gyr #Orbital Period
    return T

# ### c)
# 
# Compute the number of rotations about the GC over the age of the universe (13.8 Gyr)


### Done in main() ###



# ## Part B  Dark Matter Density Profiles
# 
# ### a)
# Try out Fitting Rotation Curves 
# [here](http://wittman.physics.ucdavis.edu/Animations/RotationCurve/GalacticRotation.html)

### Done ###

# ### b)
# 
# 
# In the Isothermal Sphere model, what is the mass enclosed within the solar radius (R$_o$) in units of M$_\odot$? 
# 
# Recall that for the Isothermal sphere :
# $\rho(r) = \frac{V_{LSR}^2}{4\pi G r^2}$
# 
# Where $G$ = 4.4985e-6 kpc$^3$/Gyr$^2$/M$_\odot$, r is in kpc and $V_{LSR}$ is in km/s
# 
# What about at 260 kpc (in units of  M$_\odot$) ? 


# Density profile - rho = V_LSR^2/(4*pi*G*R^2)
# Mass(r) = Integrate rho*4*pi*r^2 dr
#           Integrate V_LSR^2/(4*pi*G*r^2)*4*pi*r^2 dr
#           Integrate V_LSR^2/G dr
#           V_LSR^2/G * r

def massIso(r,v_LSR):
    # (Quantity,Quantity) -> Quantity
    '''
        this function computes the dark matter mass enclosed
        within a given distance, r, assuming an Isothermal
        Sphere Model
        M(r) = v_LSR^2/G * r
    
    Inputs:
        r (kpc): The distance from the Galactic Center
        v_LSR (km/s): The Local Standard of Rest velocity

    Outputs:
        M (M_sun): mass enclosed within r
    '''
    
    v_kpc_Gyr = v_LSR.to(u.kpc/u.Gyr) # Converts v_LSR from km/s to kpc/Gyr
    M = v_kpc_Gyr**2/G*r # Isothermal Sphere model equation

    return M
    
    

# ## c) 
# 
# The Leo I satellite is one of the fastest moving satellite galaxies we know. 
# 
# 
# It is moving with 3D velocity of magnitude: Vtot = 196 km/s at a distance of 260 kpc (Sohn 2013 ApJ 768)
# 
# If we assume that Leo I is moving at the escape speed:
# 
# $v_{esc}^2 = 2|\Phi| = 2 \int G \frac{\rho(r)}{r}dV $ 
# 
# and assuming the Milky Way is well modeled by a Hernquist Sphere with a scale radius of $a$= 30 kpc, what is the minimum mass of the Milky Way (in units of M$_\odot$  ?  
# 
# How does this compare to estimates of the mass assuming the Isothermal Sphere model at 260 kpc (from your answer above)

# Potential for a Hernquist Sphere - Phi = -G*M/(r+a)
# Escape Speed - v_esc^2 = 2*G*M/(r+a)
# Rearrange for M - M = v_esc^2*(r+a)/(2*G)

def massHernVesc(v_esc,r,a=30*u.kpc):
    # (Quantity,Quantity,Quantity) -> Quantity
    '''
        This function determines the total dark maatter mass needed given an espace speed, assuming a Hernquist profile
        M = v_esc^2*(r+a)/(4*G)

    Inputs:
        v_esc (km/s): Escape speed or speed of satellite
        r (kpc): Distance from the Galactic Center
        a (kpc): Hernquist scale length (Default = 30 kpc)

    Outputs:
        M (M_sun): Mass enclosed within r
    '''
    v_esc_kpc_Gyr = v_esc.to(u.kpc/u.Gyr) # Converts v_esc from km/s to kpc/Gyr
    M = v_esc_kpc_Gyr**2*(r+a)/(2*G) # Hernquist model equation
    return M

### MAIN FUNCTION ###

def main() -> None:
    # ### Part A
    # ### a)
    R_o_values = 8.34*u.kpc, 8.178*u.kpc,7.9*u.kpc #from (Reid+14), (Abuter+19), (Sparke & Gallagher Text) respectively
    print()

    v_LSR = [] # Creates a list to store all possible values of V_LSR given your source for R_o
    
    for i,R_o in enumerate(R_o_values):  # Loops through R_o values for all sources
        v_LSR.append(VLSR(R_o)) # Appends the individual V_LSR measurement to the V_LSR list
        print(f"Given R_o={R_o}, the velocity of the local standard of rest is {np.around(v_LSR[i],3)}")

    # ### b)
    print()
    v_pec = 12.24*u.km/u.s # Given quantity
    v_sun = v_LSR[1] + v_pec # V_LSR[1] corresponds to the Abuter measurement for V_LSR
    T_Abuter = TorbSun(R_o_values[1],v_sun) # Orbital Period of the Sun per the Abuter measurement for R_o
    print(f"The orbital period of the Sun is {np.around(T_Abuter,3)}")

    # ### c)
    print()
    AgeOfUniverse = 13.8*u.Gyr # Age of the Universe
    RevCount = AgeOfUniverse/T_Abuter # Count of revolutions the Sun would have made assuming a
                                      # circular constant orbit and that the Sun is as old as the universe
    print(f"If the Sun were in a circular orbit around the Milky Way and were as old as the universe, it would have completed {np.around(RevCount,3)} revolutions")

    global G
    G = const.G.to(u.kpc**3/u.Gyr**2/u.solMass) # Converts the gravitational constant G to units of M_sun^-1*kpc^3*Gyr^-2

    # ### Part B
    # ### b)
    print()
    mIsoSolar = massIso(R_o_values[1],v_LSR[1]) # Mass enclosed within the solar orbital radius per the isothermal sphere model
    print(f"In the Isothermal Sphere model, {mIsoSolar:.2e} is enclosed within the solar orbital radius")

    mIso260 = massIso(260*u.kpc,v_LSR[1]) # Mass enclosed within a radius of 260 kpc per the isothermal sphere model
    print(f"In the Isothermal Sphere model, {mIso260:.2e} is enclosed within a radius of 260 kpc")

    # ### c)
    
    v_leo = 196*u.km/u.s # Speed of Leo I Sohn et al.
    r = 260*u.kpc # A chosen radius

    print()
    MLeoI = massHernVesc(v_leo,r,a=30*u.kpc) # Mass enclosed within a radius of 260 kpc per the Hernquist model
    print(f"In the Hernquist model, {MLeoI:2e} is enclosed within a radius of 260 kpc")

if __name__ == '__main__':
    main()
