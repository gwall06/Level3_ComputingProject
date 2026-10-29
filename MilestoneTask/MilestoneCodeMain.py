# External libraries
import numpy as np
import scipy.constants as consts

import matplotlib.pyplot as plt

# Other files
import MilestoneGraphCode as graphFxns

# Description

"""
Milestone program to calculate the total luminosity of the accretion disk of a black hole from specific parameters.

Creates three graphs showing the the luminosity contribution from the different frequencies and temperatures, and how the
distance from the centre of the black hole influences the temperature.

"""

# Astronomy-specific constants 

solarMass = 2*(10**30) # kg

# Plotting parameters

numAnnuli = 10000
constTempLines = 10

# Modelling parameters

# Mass of the black hole.
bhMass = 10*solarMass # kg

# Accretion rate of the black hole.
accretionRate = 10**15 # kg/s

# Maximum and minimum frequencies over which we consider.
minFreq = 10**14 # Hz
maxFreq = 10**19 # Hz

# Maximum and minimum radii over which we model.
minRad = 6      # Dimensionless, ratio of R to R_g
maxRad = 10**5  # Dimensionless, ratio of R to R_g

# Functions implemented from literature.

def findRg(mass):
    return ((consts.G * mass)/(consts.c**2))

# Rg
rgVal = findRg(bhMass)

# Uses the given formula to convert the central radius to a temperature.
def findTemp(rad, rg, innerRad, mass, massRate):
    return (((3*consts.G*mass*massRate) / (8*consts.pi*consts.Stefan_Boltzmann*(rad**3)*(rg**3))) * (1 - (innerRad/rad)**(0.5)))**(0.25)

# Uses the given formula to convert temperatures and frequencies into flux, returning a matrix.
def findFlux(temp, freq):
    return ((2*consts.pi*consts.Planck*(freq**3)) / ((consts.c**2)*(np.exp((consts.Planck*freq)/(consts.Boltzmann*temp)) - 1)))

# Integrates the flux for a constant frequency over each annulus, returning an array of luminosities per unit frequency.
# Integration is currently approximated with a sum of bands calculated at the central position in annulus, however
# more accurate methods can be implemented.
def integrateFlux(bounds, fluxMatrix, radii):
    return np.matmul((fluxMatrix), ((4*consts.pi*rgVal**2 * radii) * (np.diff(bounds))).reshape(numAnnuli, 1)).flatten()

# Integrates the luminosities per unit frequency over the frequency range to give a total luminosity, using the trapezium method
def integrateLum(luminosities, frequencies):

    return np.trapezoid(luminosities, frequencies)

# Main code space
def main():

    # Modelling arrays

    # Array of bounds of the annuli
    boundArr = np.logspace(np.log10(minRad), np.log10(maxRad), num=(numAnnuli+1), base=10)

    # Array of centres of band (centres in log space)
    radArr = 10**(np.diff(np.log10(boundArr)) + np.delete(np.log10(boundArr),-1))

    # Array of frequencies
    freqArr = np.logspace(np.log10(minFreq), np.log10(maxFreq), num=(numAnnuli), base=10)

    # Get the temperatures from formula
    tempArr = findTemp(radArr, rgVal, minRad, bhMass, accretionRate)

    # Create a matrix of flux for temperature and frequencies
    fluxMatrix = findFlux(tempArr.reshape(1,numAnnuli), freqArr.reshape(numAnnuli,1))

    lumArr = np.array(integrateFlux(boundArr, fluxMatrix, radArr))

    totalLum = integrateLum(lumArr, freqArr)
    print(f"Total luminosity: {totalLum} W")

    print("Plotting log(nu * L_nu) against log(nu)!")
    graphFxns.mainPlot(freqArr, lumArr)

    print(f"Plotting log(T) against log(r)!")
    graphFxns.tempRadPlot(radArr, tempArr)

    print(f"Plotting log(L_nu) against log(nu) with lines of constant temperature!")
    graphFxns.constTempPlot(constTempLines, fluxMatrix, freqArr, tempArr, boundArr, radArr, rgVal, lumArr)

if __name__ == "__main__":
    main()
