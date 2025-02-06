import numpy as np
import astropy.units as u
from astropy.table import Table
from ReadFile import Read

def ComponentMass(filename:str,partType:int) -> u.quantity.Quantity:
    """
        This function analyzes data from a specified file processed in ReadFile.py to extract and return parameters describing the data

    Inputs:
        filename: The txt file in which the particle data is stored (e.g. MW_000.txt)
        partType: The desired paricle type where
            1 - Dark Matter
            2 - Disk Stars
            3 - Bulge Stars

    Outputs:
        componentSum (solMass): The total mass of a specified particle type in a specified galaxy.
    """

    # Raises an exception if the partType argument is invalid
    if not type(partType) is int:
        raise TypeError("partType requires an integer")
    if partType < 1 or partType > 3 or not isinstance(partType,int):
        raise Exception("partType out of bounds")

    time,count,data = Read(filename) # Reads data from ReadFile.py
    index = np.where(data['type']==partType) # Considers parameters for a specified particle type
    masses = 0.01*data['m'][index]*TMsun # A list of all masses for a specified particle type

    componentSum = np.round(np.sum(masses),3) # Rounded sum of all masses of a specified particle type
    return componentSum

def formatLaTeX(filename:str,data:np.ndarray) -> None:
    """
        This function writes a LaTeX table fit to my particular specifications
        NOTE: This function was written with the help of ChatGPT, because writing a
        LaTeX table by hand felt tedious.

        I have to assume there are better ways of doing this, but I appreciated the
        appeal of having a reuseable function that writes an astropy Table in LaTeX
        for me that didn't require me to manually edit the table tex file. If I
        never use this function again or if I have to modify a future table, C'est
        la vie.

    Inputs:
        filename: The tex file in which the table is written
        data: The Table object created from a numpy array using the
              astropy.table.Table method

    Outputs:
        None
    """

    # Converts table to LaTeX format
    data.write(filename, format="ascii.latex", overwrite=True)
    
    # Writes a properly formatted LaTeX table
    with open(filename, "w") as f:
        num_cols = len(data.colnames)  # Automatically detects the number of columns

        f.write(r"""\begin{table}[h]
        \centering
        \renewcommand{\arraystretch}{1.3}
        \setlength{\tabcolsep}{5pt}
        \resizebox{\textwidth}{!}{ % Automatically resizes the table to fit it on the pdf
        \begin{tabular}{"""+"|c"*num_cols+"|}\n    \\hline\n")

        # Column headers
        f.write(" & ".join([f"\\textbf{{{col}}}" for col in data.colnames]) + r" \\" "\n    \\hline\n")

        # Table rows
        for row in zip(*[data[col] for col in data.colnames]):
            f.write(" & ".join(map(str, row))+r" \\" "\n    \\hline\n")

        f.write(r"""\end{tabular}
        }

        % ADD CAPTION (CHANGE IF COPY-PASTING)
        
        \caption{This table displays the total masses of the galactic
        halo, disk stars, and bulge stars in the Milky Way Galaxy
        (MW), Andromeda Galaxy (M31) and Triangulum Galaxy (M33). In
        addition, the total mass of each galaxy and the ratio of
        stellar mass to total mass, also known as the baryon
        fraction, is included in the fifth and sixth columns
        respectively. Note that the galactic halo is assumed to be
        entirely composed of dark matter. All values are in units
        $10^{12}\,M_\odot$.}
        
        \label{tab:galaxyMasses}
        \end{table}""")

def main() -> None:
    # Defines a "Tera-Solar Masses unit requested in the problem
    global TMsun
    TMsun = u.def_unit('TMsun',1e12*u.solMass)
    u.add_enabled_units(TMsun)

    filenames = ['MW_000.txt','M31_000.txt','M33_000.txt']
    massTable = np.zeros((3,6),dtype=object) # Creates a 3x6 2D array with all values set to 0.

    for i,j in np.ndindex(massTable.shape): # Iterates through 2D array
        if j == 0:
            massTable[i,j] = filenames[i][:-8] # List of galaxies
        elif j <= 3:
            massTable[i,j] = ComponentMass(filenames[i],j).value # Component masses
            # NOTE: Here, I strip the custom units that I went through the trouble of defining.
            #       I could have not bothered with the units at all. I wanted the units to
            #       remain an integral part of the value, and only remove it for the sake of
            #       table formatting. Perhaps I'm just stubborn.
        elif j == 4:
            massTable[i,j] = sum(massTable[i,1:4]) # Total mass
        else:
            massTable[i,j] = np.round(sum(massTable[i,2:4])/massTable[i,4],3) # Baryon fraction

    # Saves massList into an astropy Table with appropriate column names
    data = Table(massTable, names=(
        "Galaxy Name",r"Halo Mass $(10^{12}M_\odot)$",r"Disk Mass $(10^{12}M_\odot)$",
        r"Bulge Mass $(10^{12}M_\odot)$",r"Total $(10^{12}M_\odot)$",r"$f_\text{bar}$"))

    formatLaTeX('hw3-table.tex',data) # Writes an astropy Table into LaTeX

if __name__ == '__main__':
    main()
