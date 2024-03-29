from tkinter import *
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .functions import *
from scipy.stats import linregress
import scipy.stats as stats

def getTofwerk2R(*elements, make_plot = False, plot_name = 'name', export = False, csv_name = 'name') :
    """Imports .h5 files from the TofPilot software of TofWerk2R, and
    creates 1) a pandas dataframe ready for further use and 2) a plot of the
    given data
    
    Input:
    
    -"output" the desired name of the dataset.
    -"element" is the desired element to be used. Use the mass and symbol/Formula 
    without space, and in quotes, e.g. "6Li","12C".
    -"make_plot": True/False for a time/intensity overview plot. Default value=False 
    -"plot_name": string to name your plot file. Default value = name
    -"export": True/False to export your selected elements as a csv file. Default value = False
    -"csv_name":string, to name your exported dataset. 
    
    Browse files, click and wait for the dataset to load.
    
    DO NOT close the tkinter window that appears, else the program will crush. 
    Minimize it until your work is done.
    """

    filepath = filedialog.askopenfilename(title='Choose file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
    ds = io(filepath)
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
        plt.savefig(plot_name+".png", bbox_inches = 'tight')
    
    if (export):
        output.to_csv(csv_name+'.csv')
    
    return output


def elementalRatio(indata, element_numerator, element_denominator, make_plot = False, plot_name = 'name', export = False, csv_name = 'name'):
    """Calculates the ratio of two elements on a per-datapoint basis.
    
    Call by:
    output, numerator_mean, denominator_mean, mean_ratio =
    elemental_ratios(datain,"Numerator Analyte","Denominator Analyte")
    
    -"output, numerator_mean, denominator_mean, mean_ratio", insert desired
    names, in this order.
    -"datain": preferably a pandas dataframe of the two analytes
    -"Numerator/Denominator Analyte". Use the mass and symbol/Formula 
    without space, and in quotes, e.g. "6Li","12C".
    -"make_plot": True/False for a time/intensity overview plot of both elements. Default value=False 
    -"plot_name": string to name your plot file. Default value = name
    -"export": True/False to export your selected elements as a csv file. Default value = False
    -"csv_name":string, to name your exported dataset. 
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

        plt.savefig(plot_name+".png", bbox_inches = 'tight')
        sns.reset_orig()
        
    if (export):
        output.to_csv(csv_name+'.csv')
        
    return output, numerator_mean, denominator_mean, mean_ratio




def simultaneousEvents(datain,*elements,make_plots=False, datapoints_per_segment=100, threshold_model = 'Poisson', export = False, csv_name = 'name'):
    """Identifies the pulses containing simultaneous events of the given elements.
    
    Call by:
    dataset = import_tofwerk2R(datain,element1, element2,....,make_plot = True/False)
    
    -"dataset" the desired name of the dataset.
    - datain: preferably a pandas dataframe containing the desired elements
    -"element" is the desired element to be used. Use the mass and symbol/Formula 
    without space, and in quotes, e.g. "6Li","12C".
    -"make_plot": True/False for histograms of the chosen elements, only for the pulses containing all 
    of the elements of interest. Default value=False.
    -"export": True/False to export your selected elements as a csv file. Default value = False
    -"csv_name":string, to name your exported dataset. 
    -"datapoints_per_segment": number of datapoints to be used per segment when segmenting the dataset. Default value = 100
    -"threshold_model": choice between Poisson (Default), Gaussian3, Gaussian5, or a user-defined value. To ne used for the threshold determination process. 
    """
    

    output = pd.DataFrame()
    
    for el in elements:
        loc_s = datain[el]
        events = find_events(loc_s,datapoints_per_segment,threshold_model)
        if not (events.empty):
            output[el] = events
            
            
        output = output.dropna()
    
    # Plotting
    
    if (make_plots):
        number = 0   
        sns.set()
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
        sns.reset_orig()   
    if (export):
        output.to_csv(csv_name+'.csv')
        
    return output



def ratiosPerCell(datain,element_numerator,element_denominator,make_plot=False, plot_name = 'name', export = False, csv_name = 'name'):
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
    -"make_plot": True/False for histograms of the two elements' identified events and 
     the histogram of their ratio. Default value=False
    -"plot_name": string to name your plot file. Default value = name
    -"export": True/False to export your selected elements as a csv file. Default value = False
    -"csv_name":string, to name your exported dataset. 
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
        plt.savefig(plot_name+".png", bbox_inches='tight')
        sns.reset_orig()
        
    if (export):
        output.to_csv(csv_name+'.csv')
        
    mean_ratio = Ratio.mean() #to avoid problems of dividing with 0
    return output, mean_ratio




def singleCellPCA(datain,*elements,datapoints_per_segment=100,threshold_model='Poisson', plot_name = 'name', export = False, csv_name = 'name'):
    """identifies pulses containing events for the given elements, 
    and applies PCA. 
    
    Call by:
    dataset = import_single_cell_PCA(datain,element1, element2,....)
    
    -"dataset" the desired name of the dataset of the identified events containing all 
    the asked elements.
    -"datain": the dataset on which to apply PCA, preferably a pandas Dataframe. 
    -"element" is the desired element to be used. Use the symbol and mass
    without space, and in quotes, e.g. "6Li","12C".
    -"datapoints_per_segment": number of datapoints to be used per segment when segmenting the dataset. Default value = 100
    -"threshold_model": choice between Poisson (Default), Gaussian3, Gaussian5, or a user-defined value. To ne used for the threshold determination process. 
    -"plot_name": string to name your plot file. Default value = name
    -"export": True/False to export your selected elements as a csv file. Default value = False
    -"csv_name":string, to name your exported dataset. 
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
        events = find_events(loc_s,datapoints_per_segment,threshold_model)
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
    
    plt.savefig(plot_name+".png", bbox_inches = 'tight', dpi = 80)
    sns.reset_orig()
    
    if (export):
        output.to_csv(csv_name+'.csv')
        
    return output


def segmenting (dataset,datapoints_per_segment, threshold_model='Poisson'):
    """Separates the single-moieties related events from the background in a
       given dataset, and taking into account potential background drifting.
    
    The given *dataset* is split into however many segments of
    *datapoints_per_segment* length. 
    The 'deconvolute' funtion is then applied on each of the segments. 
    The results for every output value are gathered into their respective
    groups.
    
    -"datapoints_per_segment": number of datapoints to be used per segment when segmenting the dataset. Default value = 100
    -"threshold_model": choice between Poisson (Default), Gaussian3, Gaussian5, or a user-defined value. To ne used for the threshold determination process. 
    
    input: dataset, desired value of segment length
    
    output:
    background - A dataset containing the datapoints not identified as events.
    events - A dataset containing the datapoints identified as particles.
    loops - A list of the iterations each segment needed before it was deemed
            particle-free.
    threshold - A list of the threshold values of each individual segment.
    dissolved - The average of the background dataset.
    dissolved_std - The standard deviation of the background dataset.
    event_num - The number of events.
    event_mean - The average of the events dataset.
    event_std - The standard deviation of the events dataset.
    loop_mean - The average value of the loops list.
    loop_std - The standard deviation value of the loops list.
    threshold_mean - The average value of the threshold list.
    threshold_std - The standard deviation of the threshold list.
    """
    division = len(dataset.index)/datapoints_per_segment # Defining the number of segments
    seg_number = int(round(division+0.5,0)) # making sure it's round and integer
    split = np.array_split(dataset, seg_number) #splitting the dataset
    
    split_total_count=0  # Setting the starting values to 0 or empty, to make sure we avoid mistakes
    split_event_dataset= pd.Series([], dtype='float64')
    split_background_dataset= pd.Series([], dtype='float64')
    split_threshold_dataset= []
    split_loopcount_dataset = []
    
    for ds in split:
        background, events, dissolved, dissolved_std, event_num, event_mean, event_std, loop_count, threshold = deconvolute(ds, threshold_model)
        split_total_count += event_num
        split_event_dataset = pd.concat((split_event_dataset,events))
        split_background_dataset = pd.concat((split_background_dataset,background))
        split_loopcount_dataset.append(loop_count)
        split_threshold_dataset.append(threshold)
    background = split_background_dataset
    events = split_event_dataset
    loops = split_loopcount_dataset
    final_threshold = split_threshold_dataset
    dissolved = background.mean()
    dissolved_std = background.std()
    event_num = split_event_dataset.dropna().count()
    event_mean = events.mean()
    event_std = events.std()
    loop_mean = np.mean(loops)
    loop_std = np.std(loops)
    threshold_mean = np.mean(final_threshold)
    threshold_std = np.std(final_threshold)

    return background, events, loops, final_threshold, dissolved, dissolved_std, event_num, event_mean, event_std, loop_mean, loop_std, threshold_mean, threshold_std

def conc_cal_curve(title,element,*Xaxis,make_plot = False, export = False, csv_name = 'name'):
    """
    Makes a calibration curve drom the given files, for a given element.
    
    Input:
    - "title" : Has to be a string. Appears at the top of the chart and in the name of the saved file, should you choose to save it.
    - "element": string, the desired element to be used. Use the symbol and mass
    without space, and in quotes, e.g. "6Li","12C"..
    
    -Xaxis: Use the concentrations of the standards in ppb (ng/mL), as numbers. When run, the program will ask for the relevant files, one by one. 
    
    -"make_plot": True/False for a saved image of the calibration curve. Default value=False
    -"export": True/False to export your dataset values and linear regression values in csv. Default value = False
    -"csv_name":string, to name your exported csv file. 

    Output: 
    slope, intercept, r_value, p_value, stderr : The parameters of the resulting equation.
    Chart depicting the resulting calibration curve in .png form.
    """
    plot_dict = {}
    
    for value in Xaxis: # value is supposedly given in ug/ml (ppm) by the user
        output = pd.DataFrame()
        filepath = filedialog.askopenfilename(title='Choose file for the '+str(value)+' ppb standard',
                                          filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
        ds = io(filepath)
        waveforms = ds.attrs['NbrWaveforms']
        data = ds.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
        output[f'{element}'] = (data * waveforms * 1000)/(0.046*waveforms) # to convert to cps. parenthesis is the dwell time in milliseconds.
        mean_value = output.mean()
        plot_dict[value] = mean_value[0]
        
    fig = plt.figure(figsize =(5,5))
    fig.suptitle(title)
    ax = fig.add_subplot(1,1,1)
    ax.plot(*zip(*sorted(plot_dict.items())),'ob')
    
    
    Xplot = np.array(list(plot_dict.keys()))
    Yplot = np.array(list(plot_dict.values()))
    
    slope, intercept, r_value, p_value, stderr = stats.linregress(Xplot,Yplot)
        
    mn=min(Xaxis)
    mx=max(Xaxis)
    x1=np.linspace(mn,mx)
    y1=slope*x1+intercept
    
    ax.set_ylabel("Intensity (cps)", size=14) # Just axes names
    ax.set_xlabel("Concentration (ppb)", size=14)
    
    ax.plot(x1,y1,'--r')
    ax.text(1.05*min(Xplot), 0.95*max(Yplot), 'y = ' + '{:.2f}'.format(intercept) + ' + {:.2f}'.format(slope) + 'x', size=14)
    ax.text(1.05*min(Xplot), 0.9*max(Yplot), '$R^{2}$=' + '{:.4f}'.format((r_value)**2), size=14)
    if (make_plot):
        plt.savefig(title, bbox_inches = 'tight', dpi = 300) 
    
    if (export):
        output = pd.DataFrame(plot_dict.items(),
                              columns = ['Concentration','Intensity']
                             )
        output2 = pd.DataFrame([0])
        output2['Slope'] = slope
        output2['Intercept'] = intercept
        output2['R value'] = r_value
        output2['$R^{2}$ value'] = r_value**2
        output2['P value'] = p_value
        output2['Stderr'] = stderr
        
        output.to_csv(csv_name+'.csv')
        output2.to_csv(csv_name+'_LinRegres'+'.csv')
        
    return plot_dict, slope, intercept, r_value, p_value, stderr




def mass_cal_curve(title,element,*Xaxis, flow_rate = 0, tr_eff = 0, make_plot = False, export = False, csv_name = 'name'):
    """
    Makes a calibration curve drom the given files, for a given element.
    
    Input:
    - "title" : Has to be a string. Appears at the top of the chart and in the name of the saved file, should you choose to save it.
    - "element": string, the desired element to be used. Use the symbol and mass
    without space, and in quotes, e.g. "6Li","12C"..
    
    -Xaxis: Use the concentrations of the standards in ppm (ug/mL), as numbers. When run, the program will ask for the relevant files, one by one.
    
    -"flow_rate": The analysis flow rate in mL/ms.
    -"tr_eff": The transport efficiency value, NOT in percentage form (so 5% = 0.05 for example).
    -"make_plot": True/False for a saved image of the calibration curve. Default value=False
    -"export": True/False to export your dataset values and linear regression values in csv. Default value = False
    -"csv_name":string, to name your exported csv file. 

    Output: 
    slope, intercept, r_value, p_value, stderr : The parameters of the resulting equation.
    Chart depicting the resulting calibration curve in .png form.
    
    Mass per event calculation based on: 
    H. E. Pace, N. J. Rogers, C. Jarolimek, V. A. Coleman, C. P. Higgins and J. F. Ranville, Analytical Chemistry, 2011, 83, 9361-9369.
    """
    plot_dict = {}
    
    for value in Xaxis: # value is supposedly given in ug/ml (ppm) by the user
        output = pd.DataFrame()
        filepath = filedialog.askopenfilename(title='Choose file for the '+str(value)+' ppm standard',
                                          filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
        ds = io(filepath)
        waveforms = ds.attrs['NbrWaveforms']
        data = ds.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
        output[f'{element}'] = data * waveforms #counts .
        mean_value = output.mean()
        
        dwell_time = 0.046*waveforms #in ms .Tested with new datasets, still holds true
        
        mass_per_event = tr_eff*flow_rate*dwell_time*value
        
        plot_dict[mass_per_event] = mean_value[0]
        
    fig = plt.figure(figsize =(5,5))
    fig.suptitle(title)
    ax = fig.add_subplot(1,1,1)
    ax.plot(*zip(*sorted(plot_dict.items())),'ob') # makes two lists from the items of the dictionary, matching the masses with the mean values
    
    Xplot = np.array(list(plot_dict.keys()))
    Yplot = np.array(list(plot_dict.values()))
    
    slope, intercept, r_value, p_value, stderr = stats.linregress(Xplot,Yplot)
        
    mn=min(Xplot)
    mx=max(Xplot)
    x1=np.linspace(mn,mx)
    y1=slope*x1+intercept
    
    ax.set_ylabel("Intensity (cts/event)", size=14) # Just axes names
    ax.set_xlabel("Mass per event (μg)", size=14)
    
    ax.plot(x1,y1,'--r')
    ax.text(1.05*min(Xplot), 0.95*max(Yplot), 'y = ' + '{:.2f}'.format(intercept) + ' + {:.2f}'.format(slope) + 'x', size=14)
    ax.text(1.05*min(Xplot), 0.9*max(Yplot), '$R^{2}$=' + '{:.4f}'.format((r_value)**2), size=14)
    if (make_plot):
        plt.savefig(title, bbox_inches = 'tight', dpi = 300) 
    
    if (export):
        output = pd.DataFrame(plot_dict.items(),
                              columns = ['Mass per event','Intensity']
                             )
        output2 = pd.DataFrame([0])
        output2['Slope'] = slope
        output2['Intercept'] = intercept
        output2['R value'] = r_value
        output2['$R^{2}$ value'] = r_value**2
        output2['P value'] = p_value
        output2['Stderr'] = stderr
        
        output.to_csv(csv_name+'.csv')
        output2.to_csv(csv_name+'_LinRegres'+'.csv')
        
    return plot_dict, slope, intercept, r_value, p_value, stderr

def tr_eff_freq(element, flow_rate = 0, numb_conc = 0, std_conc = 0, density = 0, diameter = 0, datapoints_per_segment=100, threshold_model = "Poisson", make_plot = False, plot_name = 'name', export_csv = False, csv_name = 'name'):
    """
    Calculates the transport efficiency of the system based on a particle standard of known mass concentration.
    
    Input:
    - "element": The desired element to be used. Use the mass and symbol/Formula without space, and in quotes, e.g. "6Li","12C".
    -"flow_rate": The flow rate the analysis of the sample was conducted with, in mL/min
    -"numb_conc": The number concentration of the sample in particles per ml, if known already. If not, ommit entry. 
    (!) OMMIT ENTRY of the following in numb_conc is entered:
        -"std_conc": The concentration of the particle std in mg/L. 
        -"density": The density of the particles, e.g., of the element in g/cm^3. 
        -"diameter": The diameter of the particles in nm.
        
        
    - -"threshold_model": choice between Poisson (Default), Gaussian3, Gaussian5, or a user-defined value. To ne used for the threshold determination process.
    -"datapoints_per_segment": number of datapoints to be used per segment when segmenting the dataset. Default value = 100
    -"make_plot": True/False for a.png file of the histogram of the standard. Default value=False
    -"plot_name":string, to name your exported image.
    -"export_csv": True/False to export your sample information in .csv. Default value = False
    -"csv_name":string, to name your exported csv file. 

    Output: 
    - The trasport efficiency.
    - Optional Plot of the particle popuation of the std
    - Optional export of all relevant values
    
    Trasnport efficiency calculation via particle frequency based on: 
    H. E. Pace, N. J. Rogers, C. Jarolimek, V. A. Coleman, C. P. Higgins and J. F. Ranville, Analytical Chemistry, 2011, 83, 9361-9369.
    
    """
    # Input data
    output = pd.DataFrame()
    filepath = filedialog.askopenfilename(title='Choose particle standard file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
    ds = io(filepath)
    waveforms = ds.attrs['NbrWaveforms']
    data = ds.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
    output[f'{element}'] = data * waveforms #counts. 
    
    #Determining detected particle frequency 
    
    events = find_events(output[element],datapoints_per_segment,threshold_model)
    analysis_time = 0.046*waveforms*len(output)/60000 # Time in minutes
    freq = events.count()/analysis_time # pulses per minute 

    # Choice between having or not having the number concentration
    if numb_conc == 0:
        mass_per_part = (np.pi*(diameter**3)*(density/(10**21)))/6
        # in g. The 10**21 chamges the density from g/cm3 to g/nm3
        numb_conc = (std_conc/10**(6))/mass_per_part  
        # particles per mL. /10**6 turns mg/l to g/ml
                     
    else:
        pass
                     
    
    # Determining transport efficiency
    
    tr_eff = (freq)/(flow_rate*numb_conc)
    
    # Optional plotting
    
    if (make_plot):
        sns.set()
        fig = plt.figure(figsize =(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(element)
        ax.set_xlabel("Intensity (cts)")
        ax.set_ylabel("Frequency")
        ax.hist(events,
                linewidth = 0.5,
                edgecolor = 'white',bins=20)
        plt.savefig(plot_name)
        sns.reset_orig()

    
    # Optional Exporting
        
    if (export_csv):
        exported_dataset = pd.DataFrame([0])
        exported_dataset['No. of events'] = events.count()
        exported_dataset['Analysis Time (min)'] = analysis_time
        exported_dataset['Std nummber concentration'] = numb_conc
        exported_dataset['Std mass per particle'] = mass_per_part
        exported_dataset['Transport efficiency (%)'] = tr_eff*100
        
        events.to_csv(csv_name+'_events'+'.csv')
        exported_dataset.to_csv(csv_name+'_info'+'.csv')
    
    return tr_eff 


def tr_eff_size(element, *Xaxis, flow_rate = 0, density = 0, diameter = 0, datapoints_per_segment=100, threshold_model = "Poisson", make_plot = False, plot_name = 'name', export_csv = False, csv_name = 'name'):
    """
    Calculates the transport efficiency of the system based on a particle standard of known size.
    
    Input:
    - "element": The desired element to be used. Use the mass and symbol/Formula without space, and in quotes, e.g. "6Li","12C".
    -Xaxis: Use the concentrations of the standards in ppm (ug/mL), as numbers. When run, the program will ask for the relevant files, one by one. 
    (!) Make sure all recording are of the same length 

    -"flow_rate": The flow rate the analysis of the sample was conducted with, in mL/ms 
    -"density": The density of the particles, e.g., of the element in g/cm^3. 
    -"diameter": The diameter of the particles in nm.
        
    -"threshold_model": choice between Poisson (Default), Gaussian3, Gaussian5, or a user-defined value. To ne used for the threshold determination process.
    -"datapoints_per_segment": number of datapoints to be used per segment when segmenting the dataset. Default value = 100
    -"make_plot": True/False for a.png file of the histogram of the standard. Default value=False
    -"plot_name":string, to name your exported image.
    -"export_csv": True/False to export your sample information in .csv. Default value = False
    -"csv_name":string, to name your exported csv file. 

    Output: 
    - The trasport efficiency.
    - Optional Plot of the particle popuation of the std.
    - Optional export of all relevant values.
    
    Trasnport efficiency calculation via particle size based on: 
    H. E. Pace, N. J. Rogers, C. Jarolimek, V. A. Coleman, C. P. Higgins and J. F. Ranville, Analytical Chemistry, 2011, 83, 9361-9369.
    
    """
    
    output = pd.DataFrame()
    part_cal_dict = {}
    
    # Ask for Particle Calibration Blank
    filepath = filedialog.askopenfilename(title='Choose particle calibration blank file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
    ds = io(filepath)
    waveforms = ds.attrs['NbrWaveforms']
    data = ds.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
    output["Blank"] = data * waveforms # counts then
    blank = output["Blank"].mean() # also in counts
    
    part_cal_dict[0] = blank   
    
    # Calculating the expected mass per particle
    
    mass_per_part = (np.pi*(diameter**3)*(density/(10**21)))/6 #in g
    # checked the mass calculation, it is correct
    
    # Ask for Particle Calibration STD
    
    output2=pd.DataFrame()
        
    filepath = filedialog.askopenfilename(title='Choose particle calibration standard file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
    ds = io(filepath)
    waveforms = ds.attrs['NbrWaveforms']
    data = ds.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
    output2["Particle STD"] = data * waveforms # counts
    
    # find the particles
    events = find_events(output2["Particle STD"],datapoints_per_segment,threshold_model)
    # Checked the histogram, looks okay
    particle_mean = events.mean() # still in counts
    # Cheched events number and mean intensity, looks okay. 
    
    part_cal_dict[mass_per_part] = particle_mean # so, the mass_per part calculated by the given values is the key, and the counts for it is the value.
        
    Xparts = np.array(list(part_cal_dict.keys()))
    Yparts = np.array(list(part_cal_dict.values()))
    
    parts_slope, parts_intercept, parts_r_value, parts_p_value, parts_stderr = stats.linregress(Xparts,Yparts)

     # Merge the two datasets. This way to avoid losing data from the particle standard
    
    output = pd.concat([output,output2], axis=1)    
    
    # Ask for Liquid calibration STD
    plot_dict = {}
    for value in Xaxis: # value is supposedly given in ug/ml (ppm) by the user
        filepath = filedialog.askopenfilename(title='Choose file for the '+str(value)+' ppm liquid standard',
                                          filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
        ds = io(filepath)
        waveforms = ds.attrs['NbrWaveforms']
        data = ds.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
        output[f'{value}'] = data * waveforms #counts
        mean_value = (data*waveforms).mean() # mean intensity, Ydiss, still in counts
        dwell_time = 0.046*waveforms #in ms
        mass_per_dwell = flow_rate*dwell_time*value/(10**6) # Wdiss, turning ug to g to match the calculated mass units from before
        plot_dict[mass_per_dwell] = mean_value
    
    Xplot = np.array(list(plot_dict.keys()))
    Yplot = np.array(list(plot_dict.values()))
    
    slope, intercept, r_value, p_value, stderr = stats.linregress(Xplot,Yplot)
    
    
    # Determining transport efficiency
    
    tr_eff = slope/parts_slope
    
    # Optional plotting
    
    if (make_plot):
        sns.set()
        fig = plt.figure(figsize =(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(element)
        ax.set_xlabel("Intensity (cts)")
        ax.set_ylabel("Frequency")
        ax.hist(events,
                linewidth = 0.5,
                edgecolor = 'white',bins=20)
        plt.savefig(plot_name)
        sns.reset_orig()

    
    # Optional Exporting
        
    if (export_csv):
        exported_dataset = pd.DataFrame([0])
        exported_dataset['No. of events'] = events.count()
        exported_dataset['STD mean Intensity'] = events.mean()
        exported_dataset['STD mass per particle'] = mass_per_part
        exported_dataset['Transport efficiency (%)'] = tr_eff*100
        
        events.to_csv(csv_name+'_events'+'.csv')
        output.to_csv(csv_name+'_datasets'+'.csv')
        exported_dataset.to_csv(csv_name+'_info'+'.csv')
        
        #print(events)
    
    return tr_eff 


def mass_hist(element, slope = 0, ion_efficiency =1, mass_fraction = 1, density = 0, datapoints_per_segment=100, threshold_model = "Poisson", make_plot = False, plot_name = 'name', export_csv = False, csv_name = 'name', bins = 50):
    """
    Makes a mass or size distribution for the sample. The function will ask for the blank file, and then the sample file.  
    
    Input:
    - "element": The desired element to be used. Use the mass and symbol/Formula without space, and in quotes, e.g. "6Li","12C".
    - "slope": the slope of the Intensity/mass per event calibration curve.
    - "ion_efficiency" : the particle ionization efficiency. It's conventionally set to 1, but if the case is different, can be changed. 
    - "mass_fraction": the mass fraction of the analyzed element in the analyzed particles. Preset to 1, e.g., the particles are consedered to consist of only the analyzed element.
    - "density" : the density of the analyzed element, in g/cm^3. Use optionally to obtain size distribution instead of mass distribution.        
    -"threshold_model": choice between Poisson (Default), Gaussian3, Gaussian5, or a user-defined value. To ne used for the threshold determination process.
    -"datapoints_per_segment": number of datapoints to be used per segment when segmenting the dataset. Default value = 100
    -"make_plot": True/False for a.png file of the histogram of the standard. Default value=False
    -"plot_name":string, to name your exported image.
    -"export_csv": True/False to export your sample information in .csv. Default value = False
    -"csv_name":string, to name your exported csv file. 

    Output: 
    - Calculated Mass or size per event, based on whether density was given or not. 
    - Optional Plot of the particle popuation.
    - Optional export of the output to a csv file.
    
    Mass conversion calculations based on: 
    H. E. Pace, N. J. Rogers, C. Jarolimek, V. A. Coleman, C. P. Higgins and J. F. Ranville, Analytical Chemistry, 2011, 83, 9361-9369.
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
    I_blank = (data * waveforms).mean() # in counts
    
    
    # Ask for the file in question
    # Identify the events on all asked elements.
    filepath2 = filedialog.askopenfilename(title='Choose sample file to open',
                                         filetypes = (("HDF5 files","*.h5"),
                                                      ("netCDF files","*.nc"))
                                         )
    ds2 = io(filepath2)
    waveforms2 = ds2.attrs['NbrWaveforms']
    data2 = ds2.Data.loc[:,element].to_dataframe().drop('mass', axis=1).squeeze()
    element_signal = data2*waveforms2 #in counts
    
    events_output = pd.DataFrame()
    
    events = find_events(element_signal,datapoints_per_segment,threshold_model)
    events_output[element] = events
    #print(events.count())
    ##### Tested the intensity output, it is correct. Intensity in Counts is coming out, and it is fine. The conversion to size doesn't work, either cause of the slope being bad from before, or because the conversion itself doesn't work. 
    
    # Calculate mass per event
    
    mass_per_event = (((((events_output)-I_blank)/ion_efficiency)/slope)/mass_fraction)#*(10**9) for conversion from ug to fg for comparison with report. Done, it's the same. So mass is also okay now. 
    
    # Choice between particle size or mass
    if density == 0:
        output = mass_per_event
        Xlabel = "Mass per event (μg)"
    
    else:   
        diameter = ((6*(mass_per_event))/(np.pi*density))**(1/3) * (10**5) # when ug for the mass, convert to nm with 10 to the 5th
        output = diameter
        Xlabel = "Size (nm)"
            
    #Optional plotting        
    if (make_plot):
        sns.set()
        fig = plt.figure(figsize =(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(element)
        ax.set_xlabel(Xlabel) #Need to check units
        ax.set_ylabel("Frequency")
        ax.hist(output,
                linewidth = 0.5,
                edgecolor = 'white',bins=bins)

        plt.savefig(plot_name)
        sns.reset_orig()
    if (export_csv):
        output.to_csv(csv_name+'.csv')  
        
    return output


def biomolecule_mass(datain, biomol_mr, metal_ar, numb_atoms_per_tag, numb_tags_per_biomol, make_plot = False, plot_name = 'name', export_csv = False, csv_name = 'name'):
    """
    Calculates the mass of a biomolecule per event, based on the detected mass per event of a metal tag.
    
    Input:
    - "datain": Mass per event in μg, calculated from previous command
    - "biomol_mr": The molecular weight of the biomolecule.
    - "metal_ar" : The atomic eweight of the tag metal. 
    - "numb_atoms_per_tag": The number of atoms per metal tag.
    - "numb_tags_per_biomol": The number of tags per biomolecule. 
    
    -"make_plot": True/False for a.png file of the histogram of the standard. Default value=False
    -"plot_name":string, to name your exported image.
    -"export_csv": True/False to export your sample information in .csv. Default value = False
    -"csv_name":string, to name your exported csv file. 

    Output: 
    - Calculated biomolecule mass per event.
    - Optional Plot of the biomolecule mass per event.
    - Optional export of the output to a csv file.
    
    """
    
    biom_mass_per_event = datain * (biomol_mr/(metal_ar*numb_atoms_per_tag*numb_tags_per_biomol))
    
    if (make_plot):
        sns.set()
        fig = plt.figure(figsize =(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(plot_name)
        ax.set_xlabel("Biomolecule mass per event (μg)") #Need to check units
        ax.set_ylabel("Frequency")
        ax.hist(biom_mass_per_event,
                linewidth = 0.5,
                edgecolor = 'white',bins=20)

        plt.savefig(plot_name)
        sns.reset_orig()
        
    if (export_csv):
        biom_mass_per_event.to_csv(csv_name+'.csv')  
        
    return biom_mass_per_event