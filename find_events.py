#!/usr/bin/env python
# coding: utf-8

# In[7]:


def find_events (dataset,datapoints_per_segment):
    """Separates the single-moieties related events from the background in a given dataset, and taking into account potential backgeound drifting.
    
    The given *dataset* is split into however many segments of *datapoints_per_segment* length. 
    The 'deconvolute' funtion is then applied on each of the segments. 
    The results for every output value are gathered into their respective groups.
    
    Call by:
    
    "events" = find_events(dataset,datapoints_per_segment)
    
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


# In[ ]:




