import numpy as np
import pandas as pd
import math
import decimal
import copy
from statistics import mean
from statistics import stdev
from tkinter import *
from tkinter import filedialog
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn
from deconvolute import deconvolute
import netCDF4 as nc


def find_events (dataset,datapoints_per_segment):
    """Separates the single-moieties related events from the background in a given dataset, and taking into account potential backgeound drifting.
    
    The given *dataset* is split into however many segments of *datapoints_per_segment* length. 
    The 'deconvolute' funtion is then applied on each of the segments. 
    The results for every output value are gathered into their respective groups.
    
    input: dataset, desired value of segment length
    
    output:
    events - A dataset containing the datapoints identified as particles.
    """
    division = len(dataset.index)/datapoints_per_segment # Defining the number of segments
    seg_number = int(round(division+0.5,0)) # making sure it's round and integer
    split = np.array_split(dataset, seg_number) #splitting the dataset
      
    split_event_dataset= pd.Series([], dtype='float64')   # Setting the starting values to 0 or empty, to make sure we avoid mistakes 
    series = range(0,seg_number,1)
    
    for i in series:
        split_event_dataset = split_event_dataset
        dataset = split[(i)]
        background, events, dissolved, dissolved_std, event_num, event_mean, event_std, loop_count, threshold = deconvolute(dataset)
        split_event_dataset = split_event_dataset.append(events)
    events = split_event_dataset
    return events


def single_cell_PCA(*elements):
    """Imports data exported from the TofPilot software of TofWerk2R
    
    Call by:
    dataset = single_cell_PCA(waveforms,element1, element2,....)
    
    -"dataset" the desired name of the dataset.
    -"waveforms" is the number of waveforms used during the data acquisition. Necessary for the conversion to cps and accurate event calculation.
    -"element" is the desired element to be used. Use the symbol and mass without space, and in quotes, e.g. "Li6","C12". Use "All" if you just want the entire acquired dataset.
    
    
    Browse files, click and wait for the dataset to load.
    
    DO NOT close the tkinter window that appears, else the program will crush. 
    Minimize it until your work is done.
    
    output: 
    
    - Dataset containing all pulses that contained ALL of the asked elements
    - PCA relevant triple-plot (bottom to top): Scree plot, PCA plot, loading score plot
    """
    from get_data import io
    filepath = filedialog.askopenfilename(title='Choose file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc")))
    ncfile = f'{filepath[:-3]}.nc'
    ds = io(filepath,ncfile)
    waveforms = ds.attrs['NbrWaveforms']
    data = ds.Data.loc[:,list(elements)]
    
    output = pd.DataFrame()
    
    for el in elements:
        loc_s = data.loc[:,el].to_dataframe().drop('mass', axis=1).squeeze()
        events = find_events(loc_s,100)
        output[f"{el}"]= events
            
         
    output = output.dropna()
    #PCA plotting
    scaled_data = preprocessing.scale(output)

    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
    labels = ["PC"+str(x) for x in range(1,len(per_var)+1)]

        #Make the plot frame
    fig = plt.figure(figsize =(5,15))
    ax = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    plt.subplots_adjust( hspace=0.3)

        #Scree plot
    ax.set_title("Scree Plot")    
    ax.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Percentage of Explained Variance")

    ax.plot(
        range(1,len(per_var)+1),
        np.cumsum(per_var),
        c='red',
        label='Cumul. Explained Variance')
    ax.legend(loc='center right')

        #PCA plot

    pca_df = pd.DataFrame(pca_data, columns=labels)

    ax2.set_title("PCA")
    ax2.scatter(pca_df.iloc[:,0], pca_df.iloc[:,1])
    ax2.set_xlabel("PC1 - {0}%".format(per_var[0]))
    ax2.set_ylabel("PC2 - {0}%".format(per_var[1]))
    ax2.hlines(y=0, xmin=pca_df.iloc[:,0].min() +0.1*pca_df.iloc[:,0].min(), xmax=pca_df.iloc[:,0].max()+0.1*pca_df.iloc[:,0].max(),color='black', linestyle='-', linewidth=0.75)
    ax2.vlines(x=0, ymin=pca_df.iloc[:,1].min() +0.1*pca_df.iloc[:,1].min(), ymax=pca_df.iloc[:,1].max()+0.1*pca_df.iloc[:,1].max(),color='black', linestyle='-', linewidth=0.75)

    # Loading Scores
    pre_loading_scores = pca.components_

    loading_scores = preprocessing.normalize(pre_loading_scores)

    n_features = pca.n_features_
    feature_names = [str(element) for element in elements]
    pc_list = [f'PC{i}' for i in list(range(1, n_features + 1))]
    pc_loadings = dict(zip(pc_list, loading_scores))

    loadings_df = pd.DataFrame.from_dict(pc_loadings)
    loadings_df['feature_names'] = feature_names
    loadings_df = loadings_df.set_index('feature_names')

    xs = loading_scores[0]
    ys = loading_scores[1]

    for i, varnames in enumerate(feature_names):
        ax3.scatter(xs[i], ys[i], s=200)
        ax3.text(xs[i], ys[i], varnames)
        ax3.arrow(
            0, 0, # coordinates of arrow base
            xs[i], # length of the arrow along x
            ys[i], # length of the arrow along y
            color='r', 
            head_width=0.01
            )

    ax3.set_title("Loading Scores")
    ax3.set_xlabel("PC1 - {0}%".format(per_var[0]))
    ax3.set_ylabel("PC2 - {0}%".format(per_var[1]))

    circle1 = plt.Circle((0, 0), radius = 1, color = "red", fill = False)
    ax3.hlines(y=0, xmin=-1, xmax=1,color='black', linestyle='-', linewidth=0.75)
    ax3.vlines(x=0, ymin=-1, ymax=1,color='black', linestyle='-', linewidth=0.75)
    ax3.add_patch(circle1)

    plt.savefig("The brilliancy plot", bbox_inches = 'tight', dpi = 300)
    return output




