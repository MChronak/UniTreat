from tkinter import *
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .functions import *
from scipy.stats import linregress
import scipy.stats as stats
import glob

def UniTreat(*elements, 
             TE_element = "197Au",
             TE_freq = False, TE_size = False, # Choose how to calculate the TE
             numb_conc = 0, std_conc = 0, # Tr_eff_freq + density and diameter
             flow_rate = 0, density = 0, diameter = 0, # Tr_eff_size, need the Xaxis
             datapoints_per_segment=100, threshold_model = "Poisson", 
             make_plot = False, plot_name = 'name', export_csv = False, csv_name = 'name'
            ):
    """
    -"elements": The desired elements to be explored. Use the mass and symbol/Formula without space, and in quotes, e.g. "6Li","12C".
    -"TE_element : The element to be used in the TE calculations. Preset: 197Au. Enter the same way as "elements".
    -"TE_freq" : True/False. True for the TE being calculated through the concentration of the particle standard. Preset: False
        (!) Will require:
            1) flow_rate: The flow rate the analysis of the sample was conducted with, in mL/min.
            2) "numb_conc": The number concentration of the sample in particles per ml, if known already. If not, ommit entry.
                                                OR 
                -"std_conc": The concentration of the particle std in mg/L. 
                -"density": The density of the particles, e.g., of the element in g/cm^3. 
                -"diameter": The diameter of the particles in nm.
                (!) If numb_conc is known, the other three can be ommited. 
                
    -"TE_size" : True/False. True for the TE being calculated through the concentration of the particle standard. Preset: False 
        (!) Will require:
            1) flow_rate
            2) density
            3) diameter
    
    -"threshold_model": choice between Poisson (Default), Gaussian3, Gaussian5, or a user-defined value. To ne used for the threshold determination process.
    -"datapoints_per_segment": number of datapoints to be used per segment when segmenting the dataset. Default value = 100
    -"make_plot": True/False for a.png file of the histogram of the standard. Default value=False
    -"plot_name":string, to name your exported image.
    -"export_csv": True/False to export your sample information in .csv. Default value = False
    -"csv_name":string, to name your exported csv file. 
    
    
    """
    # Separating the imported files 
    
    folder = filedialog.askdirectory(title='Choose folder to open')
    
    filenames = glob.glob(folder + "/*.h5")
    
    blank_strings = []
    part_std_strings = []
    cali_std_strings = []
    sample_strings = []
    
    for name in filenames:
        if "part_std" in name:
            part_std_strings.append(name)
        elif "blank" in name:
            blank_strings.append(name)
        elif "cali_std" in name:
            cali_std_strings.append(name)
        else:
            sample_strings.append(name)
    
    # Transport Efficiency Calculations 
    tr_eff = 0
    if (TE_freq):
        # Input data
        output = pd.DataFrame()
        te_filepath = part_std_strings[0]
        ds = io(te_filepath)
        waveforms = ds.attrs['NbrWaveforms']
        data = ds.Data.loc[:,TE_element].to_dataframe().drop('mass', axis=1).squeeze()
        output[f'{TE_element}'] = data * waveforms
        
        #Determining detected particle frequency 
        
        events = find_events(output[TE_element],datapoints_per_segment,threshold_model)
        analysis_time = 0.046*waveforms*len(output)/60000 # Time in minutes
        freq = events.count()/analysis_time # pulses per minute
        
        # Choice between having or not having the number concentration
        if numb_conc == 0:
            mass_per_part = ((np.pi*(diameter**3)*density)/6)*(10**6) # in fg
            numb_conc = (std_conc/mass_per_part)*10**(9) # particles per mL 
        else:
            pass
        
        tr_eff = freq/(flow_rate*numb_conc)
        
    if (TE_size):
        pass
            
        
        
        
        
        
        
        
        
        
        # Yo, what if I go with startswith for the different concentrations
        
        
        

        
        
        
        
        
        
        
        
        
        
    #ds = io(filenames[0])
    #waveforms = ds.attrs['NbrWaveforms']
    #data = ds.Data.loc[:,list(elements)]
    
    #output = pd.DataFrame()
    
    #for el in elements:
     #   loc_s = data.loc[:, el].to_dataframe().drop('mass', axis=1).squeeze()
      #  output[f'{el}'] = loc_s * waveforms
       #     
    #if (make_plot):
     #   fig = plt.figure(figsize =(15,5))
      #  ax = fig.add_subplot(1,1,1)
       # ax.set_title("TRA")
        #ax.set_xlabel("Time (sec)")
        #ax.set_ylabel("Intensity (cts)")
    
        #for element in elements:
         #   ax.plot(data['datapoints'].values,output[element], alpha = 0.8, linewidth = 0.5)
       # ax.legend(output)
        #plt.savefig(plot_name+".png", bbox_inches = 'tight')
    
   # if (export):
    #    output.to_csv(csv_name+'.csv')
    
    return filenames, blank_strings, part_std_strings, cali_std_strings, sample_strings, tr_eff
