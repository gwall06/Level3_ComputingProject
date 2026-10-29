import numpy as np
import scipy.constants as consts

import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Code for the milestone graphs

# Plot of log(\nu * L_\nu) against log(\nu)
def mainPlot(freqArr, lumArr):

    plt.plot(np.log10(freqArr), np.log10((lumArr*freqArr)))
    plt.ylabel("$log{\\left(\\nu L_{\\nu} \\right)}$")
    plt.xlabel("$log\\left(\\nu \\right)$")
    plt.show()

# Plot of temperature against radius
def tempRadPlot(radArr, tempArr):

    plt.plot(np.log10(radArr), np.log10(tempArr))
    plt.xlabel("$log\\left(\\frac{R}{R_g}\\right)$")
    plt.ylabel("$log\\left(T\\right)$")
    plt.show()

# Plots log(flux) against log(freq) with lines of constant temperature. The constant temperature lines have a scaling factor
# applied to account for the contributions from unsampled temperatures.

def constTempPlot(numSamples, fluxMatrix, freqArr, tempArr, bounds, radii, rg, lumArr):

    # Generates the indexes of the periodic samples of temperature that we use.
    sampleArr = np.linspace(0, len(tempArr) - 1, numSamples, dtype=int)

    # We scale the constant temperature luminosities by this factor to account for the contribution from dropped temperatures.
    scalingFactor = len(tempArr) / numSamples

    colourMap = plt.colormaps['seismic']
    norm = Normalize(vmin = tempArr[sampleArr].min(), vmax = tempArr[sampleArr].max())

    fig, ax = plt.subplots()

    for i in range(numSamples):

        # Create an array of the integrands of the fluxes for each frequency.
        fluxIntegrand = fluxMatrix.T[sampleArr[i]] * (4*consts.pi*rg**2*radii[sampleArr[i]]) * (bounds[sampleArr[i]+1] - bounds[sampleArr[i]])
        lineColour = colourMap(norm(round(tempArr[i])))
        plt.plot(np.log10(freqArr), np.log10(fluxIntegrand * scalingFactor), color = lineColour)  

    # As an overlay, plot the luminosity against the flux.
    plt.plot(np.log10(freqArr), np.log10(lumArr), linestyle='--', color='black', label = "Total Luminosity")      
        
    plt.ylabel("$log\\left(L_{\\nu}\\right)$")
    plt.xlabel("$log\\left(\\nu \\right)$")

    plt.gca().set_ylim([0,15])

    # Plot a sidebar with the colours
    scalarMap = ScalarMappable(norm=norm, cmap=colourMap)
    scalarMap.set_array([])

    colourBar = fig.colorbar(scalarMap, ax=ax)
    colourBar.set_label("Temperature / K")

    plt.legend()

    plt.show()

