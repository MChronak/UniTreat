This is a program the intents to make the analysis of ICP-MS related data universal. 
The current focus is in the identification of particle- and cell- related pulses, and the multielemental analysis of datasets. 
We will keep this up, as it looks like the most straight and impactfull way to publish.

The import_TofWerk2R function is made to easily import the data of the Tof2R instrument, and reform them in a manner that will make them usable within the programming environment. It also is giving a visual representation of the data, for quick user check.

The deconvolute funtion is made to extract the pulse-related events along with a number of other, useful information from a dataset. 

The find_events is a simplified version of the deconvolute function, focusing only on the event finding. 

The segmenting function does the same work as the deconvolute function, but separates the datatest into segments and looking for events in every individual segment, ergo mitigating for any background drifting. 

The import_simultaneous_events functions imports data of the ToF2R, and then is identifying pulses that contain overlapping elements, for any elements requested by the user. 
The import_plot_simultaneous_events does the same, but also provides a visual repserentation of the data for the user.

The calibration_curve_values function makes a calibration curve of the given values by the user. 
The calibration_curve_datasets does the same, only using datasets for Y values instead of given values. 

Import_elemental_ratio uploads a file, and gives the elemental ratio calculation of the chosen analytes.
Import_plot_elemental_ratio does the same, but also provides a plot of the requested data.

Import_ratios_per_cell uploads a file, and gives the elemental ratio calculation of the chosen analytes on a per identified cell level.
Import_plot_ratios_per_cell does the same, but also provides a plot of the requested data.

Import_ratios_per_cell uploads a file, and gives the elemental ratio calculation of the chosen analytes on a per cell basis.

Import_single_cell_PCA uploads a desired file, and gives back a dataset containing pulses that include all of the requested element events.
In addition, it runs a PCA and creates and saves a multiplot containing a Scree plot, PCA plot and Loading Scores plot. 

import_cell_PCA is an older version of newer functions. 
single_cell_PCA applies PCA on a dataset and provides relevant plotting. 
Import_single_cell_PCA uploads a tofwerk2R file, identifies the overlapping events of the user's given elements, and then applies PCA on the resulting dataset for the identified cells. 

More functions with more capabilities are on the way. 
