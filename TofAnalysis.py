from tkinter import *
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .functions import *

def getTofwerk2R(*elements, make_plot = False) :
    """Imports data exported from the TofPilot software of TofWerk2R, and
    creates 1) a pandas datasset ready for further use and 2) a plot of the
    given data
    
    Call by:
    dataset = import_tofwerk2R(waveforms,element1, element2,....)
    
    -"dataset" the desired name of the dataset.
    -"waveforms" is the number of waveforms used during the data acquisition.
    Necessary for the conversion to cps.
    -"element" is the desired element to be used. Use the symbol and mass
    without space, and in quotes, e.g. "Li6","C12". Use "All" if you just want
    the entire acquired dataset.
    
    
    Browse files, click and wait for the dataset to load.
    
    DO NOT close the tkinter window that appears, else the program will crush. 
    Minimize it until your work is done.
    """

    filepath = filedialog.askopenfilename(title='Choose file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc")))
    ncfile = f'{filepath[:-3]}.nc'
    ds = io(filepath,ncfile)
    waveforms = ds.attrs['NbrWaveforms']
    data = ds.Data.loc[:,list(elements)]
    
    output = pd.DataFrame()
    
    for el in elements:
        loc_s = data.loc[:, el].to_dataframe().drop('mass', axis=1).squeeze()
        output[f'{el}'] = loc_s * waveforms
            
    if (make_plot):
        fig = plt.figure(figsize =(15,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title("TRA")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Intensity (cps)")
    
        for element in elements:
            ax.plot(data['datapoints'].values,output[element], alpha = 0.8, linewidth = 0.5)
        ax.legend(output)
    
    return output


def elementalRatio(indata, element_numerator, element_denominator, make_plot = False):
    """Imports data exported from the TofPilot software of TofWerk2R, and
    calculates the elemental ratio of given analytes.
    
    Call by:
    output, numerator_mean, denominator_mean, mean_ratio =
    elemental_ratios(waveforms,"Numerator Analyte","Denominator Analyte")
    
    -"output, numerator_mean, denominator_mean, mean_ratio", insert desired
    names, but in this order.
    -"waveforms" is the number of waveforms used during the data acquisition.
    Necessary for the conversion to cps.
    -"Numerator/Denominator Analyte". Use the symbols and nominal masses
    without space, and in quotes, e.g. "Li6","C12".
    
    
    Browse files, click and wait for the dataset to load.
    
    DO NOT close the tkinter window that appears, else the program will crush. 
    Minimize it until your work is done.
    """
        
    numerator = indata[element_numerator]
    denominator = indata[element_denominator]

    Ratio = numerator/denominator

    output = pd.DataFrame()
    output[element_numerator] = numerator
    output[element_denominator] = denominator
    output['Ratio'] = Ratio
    output = output.dropna()
    
    numerator_mean = numerator.mean()
    denominator_mean = denominator.mean()
    mean_ratio = numerator_mean/denominator_mean #to avoid problems of dividing with 0, divides the mean values, not the datsets on a per point ratio
    print("Ratio =",mean_ratio)

    # Plotting
    if (make_plot):
        sns.set()
        fig = plt.figure(figsize =(5,10), dpi = 80)
        ax = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        plt.subplots_adjust(hspace=0.3)
    
        ax.set_title(element_numerator)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity (cps)")
    
        ax2.set_title(element_denominator)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Intensity (cps)")
    
        ax.plot(output[str(element_numerator)],
                linewidth = 0.7)
        ax2.plot(output[str(element_denominator)],
                linewidth = 0.7)

        plt.savefig("Elemental_Ratios_Elements.png", bbox_inches = 'tight')
        sns.reset_orig()
    
    return output, numerator_mean, denominator_mean, mean_ratio




def simultaneousEvents(datain,*elements,make_plots=False):
    """Imports data exported from the TofPilot software of TofWerk2R. Exports
    histograms of the chosen events for the selected elements.
    
    Call by:
    dataset = import_tofwerk2R(waveforms,element1, element2,....)
    
    -"dataset" the desired name of the dataset.
    -"waveforms" is the number of waveforms used during the data acquisition.
    Necessary for the conversion to cps.
    -"element" is the desired element to be used. Use the symbol and mass
    without space, and in quotes, e.g. "Li6","C12". Use "All" if you just want
    the entire acquired dataset.
    
    
    Browse files, click and wait for the dataset to load.
    
    DO NOT close the tkinter window that appears, else the program will crush. 
    Minimize it until your work is done.
    """
    

    output = pd.DataFrame()
    
    for el in elements:
        loc_s = datain[el]
        events = find_events(loc_s,100)
        if not (events.empty):
            output[el] = events
            
            
        output = output.dropna()
    
    # Plotting
    
    if (make_plots):
        number = 0   
        
        for element in elements:
            number = number+1
            fig = plt.figure(number,figsize =(5,5))
            ax = fig.add_subplot(1,1,1)
            ax.set_title(str(element))
            ax.set_xlabel("Intensity (cps)")
            ax.set_ylabel("Frequency")
            ax.hist(output[str(element)],
                    linewidth = 0.5,
                    edgecolor = 'white',bins=20)
            plt.savefig(element)
    return output



