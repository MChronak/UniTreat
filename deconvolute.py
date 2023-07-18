def deconvolute (dataset):
    """Separates the single-moieties related events from the background in a given dataset.
    
    The average and standard deviation of the datset are calculated.
    Events are identified as datapoints that exceed the Poisson threshold, and are removed from the dataset:
    Threshold = average + 2.72 + 3.29*stdev
    The average and standard deviation of the resulting dataset are recalculated.
    the procedure repeats itself until there are no more datapoints to be excluded.
    
    Call it by typing:
    background, events, dissolved, dissolved_std, event_num, event_mean, event_std, loop_count, threshold = deconvolute(example)
    *'example' being the name of the preffered dataset.  
    
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
    import copy
    import pandas as pd
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