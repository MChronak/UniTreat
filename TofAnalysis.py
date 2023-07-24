from tkinter import *
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .functions import *

def getTofwerk2R(*elements, make_plot = False) :
    """Imports .h5 files from the TofPilot software of TofWerk2R, and
    creates 1) a pandas dataframe ready for further use and 2) a plot of the
    given data
    
    Call by:
    output= import_tofwerk2R(element1, element2,...., make_plot=True/False)

    
    -"output" the desired name of the dataset.
    -"element" is the desired element to be used. Use the mass and symbol/Formula 
    without space, and in quotes, e.g. "6Li","12C".
    -"make_plot" for a time/intensity overview plot. Default value=False  
    
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
        ax.set_ylabel("Intensity (cts)")
    
        for element in elements:
            ax.plot(data['datapoints'].values,output[element], alpha = 0.8, linewidth = 0.5)
        ax.legend(output)
    
    return output


def elementalRatio(indata, element_numerator, element_denominator, make_plot = False):
    """Calculates the ratio of two elements on a per-datapoint basis.
    
    Call by:
    output, numerator_mean, denominator_mean, mean_ratio =
    elemental_ratios(datain,"Numerator Analyte","Denominator Analyte")
    
    -"output, numerator_mean, denominator_mean, mean_ratio", insert desired
    names, in this order.
    -"datain": preferably a pandas dataframe of the two analytes
    -"Numerator/Denominator Analyte". Use the mass and symbol/Formula 
    without space, and in quotes, e.g. "6Li","12C".
    -"make_plot" for time/intensity overview plots of both elements. Default value=False  
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
        ax.set_ylabel("Intensity (cts)")
    
        ax2.set_title(element_denominator)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Intensity (cts)")
    
        ax.plot(output[str(element_numerator)],
                linewidth = 0.7)
        ax2.plot(output[str(element_denominator)],
                linewidth = 0.7)

        plt.savefig("Element Ratio.png", bbox_inches = 'tight')
        sns.reset_orig()
    
    return output, numerator_mean, denominator_mean, mean_ratio




def simultaneousEvents(datain,*elements,make_plots=False):
    """Identifies the pulses containing simultaneous events of the given elements.
    
    Call by:
    dataset = import_tofwerk2R(datain,element1, element2,....,make_plot = True/False)
    
    -"dataset" the desired name of the dataset.
    - datain: preferably a pandas dataframe containing the desired elements
    -"element" is the desired element to be used. Use the mass and symbol/Formula 
    without space, and in quotes, e.g. "6Li","12C".
    -"make_plot" for histograms of the chosen elements, only for the pulses containing all 
    of the elements of interest. Default value=False.  
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
            ax.set_xlabel("Intensity (cts)")
            ax.set_ylabel("Frequency")
            ax.hist(output[str(element)],
                    linewidth = 0.5,
                    edgecolor = 'white',bins=20)
            plt.savefig(element)
    return output



def ratiosPerCell(datain,element_numerator,element_denominator,make_plot=False):
    """Calculates the ratio of two elements on a per-identified event basis.
    
    Call by:
    dataset, mean_ratio = ratiosPerCell(datain,element_numerator,
    element_denominator,make_plot=False)
    
    -"dataset" the desired name of the dataset.
    - "mean_ratio" is the desired name for the mean ratio of the EVENTS of the
      dataset
    -"element_numerator/denominator" are the desired elements to be used as
      numerator and denominator respectively. Use the mass and symbol/Formula 
      without space, and in quotes, e.g. "6Li","12C".
    -"make_plot" for histograms of the two elements' identified events and 
     the histogram of their ratio. Default value=False
    """
    
    output = pd.DataFrame()
    
        
    numerator = datain[element_numerator]
    denominator = datain[element_denominator]
        
    Ratio = numerator/denominator

    output[element_numerator] = numerator
    output[element_denominator] = denominator
    output['Ratio'] = Ratio
    
    if (make_plot):
        # Plotting
        sns.set()
        fig = plt.figure(figsize =(5,15))
        ax = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)
        plt.subplots_adjust(hspace=0.3)

        ax.set_title(element_numerator)
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Intensity (cts)")

        ax2.set_title(element_denominator)
        ax2.set_ylabel("Frequency")
        ax2.set_xlabel("Intensity (cts)")

        ax3.set_title("Ratio")
        ax3.set_xlabel("Ratio")
        ax3.set_ylabel("Frequency")

        ax.hist(numerator,
                linewidth = 0.7)
        ax2.hist(denominator,
                linewidth = 0.7)
        ax3.hist(Ratio,
                 edgecolor = "white"
                )
        plt.savefig("Element Ratio per cell")
        sns.reset_orig()
    
    mean_ratio = Ratio.mean() #to avoid problems of dividing with 0
    return output, mean_ratio




def singleCellPCA(datain,*elements):
    """identifies pulses containing events for the given elements, 
    and applies PCA. 
    
    Call by:
    dataset = import_single_cell_PCA(datain,element1, element2,....)
    
    -"dataset" the desired name of the dataset of the identified events containing all 
    the asked elements.
    -"element" is the desired element to be used. Use the symbol and mass
    without space, and in quotes, e.g. "6Li","12C".
    
    output: 
    
    - Dataset containing all pulses that contained ALL of the asked elements
    - PCA relevant triple-plot (bottom to top): Scree plot, PCA plot, loading
      score plot
    """
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    import matplotlib.pyplot as plt

    output = pd.DataFrame()
    
    for el in elements:
        loc_s = datain[el]
        events = find_events(loc_s,100)
        if not (events.empty):
            output[el] = events
            
            
        output = output.dropna()
            
    #PCA plotting
    
    sns.set()
    scaled_data = preprocessing.scale(output)

    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    per_var = np.round(pca.explained_variance_ratio_*100, decimals =1)
    labels = ["PC"+str(x) for x in range(1,len(per_var)+1)]

        #Make the plot frame
    fig = plt.figure(figsize =(5,18))
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
    
    plt.savefig("PCA plot", bbox_inches = 'tight', dpi = 80)
    sns.reset_orig()
    return output




