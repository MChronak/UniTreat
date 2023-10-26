from tkinter import *
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .functions import *
from scipy.stats import linregress
import scipy.stats as stats


def mass_hist(element, slope = 0, ion_efficiency =1, mass_fraction = 1, density = 0, datapoints_per_segment=100, threshold_model = "Poisson", make_plot = False, plot_name = 'name', export_csv = False, csv_name = 'name'):
    """
    Soon to be docksting
    
    """
    
    # Ask for Blank
    # Claculate the blank for the the elements
    
    filepath = filedialog.askopenfilename(title='Choose blank file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
    ds = io(filepath)
    waveforms = ds.attrs['NbrWaveforms']
    data = ds.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
    I_blank = (data * waveforms).mean()
    
    
    # Ask for the file in question
    # Identify the events on all asked elements.
    filepath2 = filedialog.askopenfilename(title='Choose sample file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
    ds2 = io(filepath2)
    waveforms2 = ds2.attrs['NbrWaveforms']
    data2 = ds2.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
    element_signal = data2*waveforms2
    
    events_output = pd.DataFrame()
    
    events = find_events(element_signal,datapoints_per_segment,threshold_model)
    events_output[element] = events
    
    
    
    # Calculate mass per event
    
    mass_per_event = ((((events_output)/I_blank)/ion_efficiency)/slope)/mass_fraction
    print(mass_per_event)
    
    # Going got particle size
    if density == 0:
        output = mass_per_event
        if (make_plot):
            sns.set()
            fig = plt.figure(figsize =(5,5))
            ax = fig.add_subplot(1,1,1)
            ax.set_title(element)
            ax.set_xlabel("Mass per event (μg)") #Need to check units
            ax.set_ylabel("Frequency")
            ax.hist(mass_per_event,
                    linewidth = 0.5,
                    edgecolor = 'white',bins=20)
            
            plt.savefig(plot_name)
    
    else:   
        diameter = ((6*(mass_per_event))/(np.pi*density))**(1/3)
        output = diameter
        if (make_plot):
            sns.set()
            fig = plt.figure(figsize =(5,5))
            ax = fig.add_subplot(1,1,1)
            ax.set_title(element)
            ax.set_xlabel("Size (nm)") #Need to check units
            ax.set_ylabel("Frequency")
            ax.hist(diameter,
                    linewidth = 0.5,
                    edgecolor = 'white',bins=20)
            
            plt.savefig(plot_name)   
            
    if (export_csv):
        output.to_csv(csv_name+'.csv')  
        
    return output