def deconvolute (dataset):
    """Separates the single-moieties related events from the background in a given dataset.
    
    The average and standard deviation of the datset are calculated.
    Events are identified as datapoints that exceed the Poisson threshold, and are removed from the dataset:
    Threshold = average + 2.72 + 3.29*stdev
    The average and standard deviation of the resulting dataset are recalculated.
    the procedure repeats itself until there are no more datapoints to be excluded.
    
    Output: 
    background - A dataset containing the datapoints not identified as events.
    events - A dataset containing the datapoints identified as particles.
    dissolved - The average of the background dataset.
    dissolved_std - The standard deviation of the background dataset.
    event_num - The number of events.
    event_mean - The average of the events dataset.
    event_std - The standard deviation of the events dataset.
    loop_count - The number of times the procedure had to be applid before it reached the final dataset.
    threshold - The final threshold value.
    
    Based on the work by:
    Anal. Chem. 1968, 40, 3, 586–593.
    Pure Appl. Chem., 1995, Vol. 67, No. 10, pp. 1699-1723.
    J. Anal. At. Spectrom., 2013, 28, 1220.
    """
    working_dataset = copy.deepcopy(dataset) 
    event_num = -1 # Resetting the output values
    event_count = 0 
    loop_count = 0
    while event_num < event_count:
        event_num = event_count
        threshold = working_dataset.mean() + 2.72 + 3.29*working_dataset.std()
        reduced_dataset = working_dataset.where(working_dataset<=threshold)
        event_count = reduced_dataset.isna().sum()
        working_dataset = reduced_dataset
        loop_count = loop_count+1
    background = dataset.where(dataset<=threshold).dropna()
    events = dataset.where(dataset>threshold).dropna()
    dissolved = background.mean()
    dissolved_std = background.std()
    event_mean = events.mean()
    event_std = events.std()
    return background, events, dissolved, dissolved_std, event_num, event_mean, event_std, loop_count, threshold

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



