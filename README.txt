This is a program the intents to make the analysis of ICP-MS related data universal. 
The current focus is in the identification of particle- and cell- related pulses, and the multielemental analysis of datasets. 

The import_TofWerk2R function is made to easily import the data of the Tof2R instrument, and reform them in a manner that will make them usable within the programming environment.

The deconvolute funtion is made to extract the pulse-related events along with a number of other, useful information from a dataset. 

the find_events is a simplified version of the deconvolute function, focusing only on the event finding. 

The segmenting function does the same work as the deconvolute function, but separates the datatest into segments and looking for events in every individual segment, ergo mitigating for any background drifting. 

The import_Overlapping signals functions imports data of the ToF2R, and then is identifying pulses that contain overlapping elements, for any elements given by the user. 

The calibration_curve_values function makes a calibration curve of the given values by the user. 
The calibration_curve_datasets does the same, only using datasets for Y values instead of given values. 

More functions with more capabilities are on the way. 
