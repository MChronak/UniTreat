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

def segmenting (dataset,datapoints_per_segment):
    """Separates the single-moieties related events from the background in a given dataset, and taking into account potential backgeound drifting.
    
    The given *dataset* is split into however many segments of *datapoints_per_segment* length. 
    The 'deconvolute' funtion is then applied on each of the segments. 
    The results for every output value are gathered into their respective groups.
    
    input: dataset, desired value of segment length
    
    output:
    background - A dataset containing the datapoints not identified as events.
    events - A dataset containing the datapoints identified as particles.
    loops - A list of the iterations each segment needed before it was deemed particle-free.
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
    
    series = range(0,seg_number,1)
    
    for i in series:
        split_event_dataset = split_event_dataset
        split_background_dataset = split_background_dataset
        split_total_count = split_total_count
        split_loopcount_dataset = split_loopcount_dataset
        split_threshold_dataset = split_threshold_dataset
        dataset = split[(i)]
        background, events, dissolved, dissolved_std, event_num, event_mean, event_std, loop_count, threshold = deconvolute(dataset)
        split_total_count = split_total_count + event_num
        split_event_dataset = split_event_dataset.append(events)
        split_background_dataset = split_background_dataset.append(background)
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
    loop_mean = mean(loops)
    loop_std = stdev(loops)
    threshold_mean = mean(final_threshold)
    threshold_std = stdev(final_threshold)
    return background, events, loops, final_threshold, dissolved, dissolved_std, event_num, event_mean, event_std, loop_mean, loop_std, threshold_mean, threshold_std




