import numpy as np
import pandas as pd

def io(h5file):
    import os, sys
    import xarray as xr
    import netCDF4 as nc
    
    nlist = ['6Li', '7Li', '9Be', '10B', '11B', '12C', '13C', '14N', '15N',
             '16O', 'OH', '18O', 'H2O', 'H3O', '23Na', '24Mg', '25Mg', '26Mg', 'C2H2',
             '27Al', 'C2H3', '28Si', 'N2', '29Si', 'HN2', '30Si', 'NO', '31P', '15NO',
             'NOH', '32S', 'O2', '33S', 'O2H', '34S', 'O18O', '35Cl', '36S', '36Ar', 'HCl',
             '37Cl', '36ArH', '38Ar', '39K', '38ArH', '40Ar', '40Ca', '41K', 'ArH', '42Ca',
             'ArH2', '43Ca', '44Ca', 'CO2', '45Sc', 'CHO2', '46Ti', '46Ca', 'NO2', '47Ti',
             '48Ti', '48Ca', '49Ti', '50Ti', '50Cr', '50V', '51V', 'ClO', '52Cr', '36ArO',
             '53Cr', '37ClO', '54Cr', '54Fe', 'ArN', '55Mn', '56Fe', 'ArO', '57Fe', 'ArOH',
             '58Fe', '58Ni', '59Co', '60Ni', '61Ni', '62Ni', '63Cu', '64Ni', '64Zn', '65Cu',
             '66Zn', '67Zn', '68Zn', '137Ba+', '69Ga', '138Ba+', '70Ge', '70Zn', '71Ga',
             '72Ge', '73Ge', '74Ge', '74Se', '75As', 'ArCl', '76Se', '76Ge', 'Ar36Ar',
             '77Se', 'Ar37Cl', '78Se', '78Kr', 'Ar38Ar', '79Br', '80Kr', '80Se', 'Ar2',
             '81Br', '82Kr', '82Se', '83Kr', '84Kr', '84Sr', '85Rb', '86Sr', '86Kr', '87Sr',
             '87Rb', '88Sr', '89Y', '90Zr', '91Zr', '92Zr', '92Mo', '93Nb', '94Mo', '94Zr',
             '95Mo', '96Mo', '96Ru', '96Zr', '97Mo', '98Ru', '98Mo', '99Ru', '100Ru',
             '100Mo', '101Ru', '102Ru', '102Pd', '103Rh', '104Pd', '104Ru', '105Pd',
             '106Pd', '106Cd', '107Ag', '108Pd', '108Cd', '109Ag', '110Cd', '110Pd',
             '111Cd', '112Cd', '112Sn', '113In', '113Cd', '114Sn', '114Cd', '115Sn',
             '115In', '116Sn', '116Cd', '117Sn', '118Sn', '119Sn', '120Sn', '120Te',
             '121Sb', '122Te', '122Sn', '123Sb', '123Te', '124Te', '124Sn', '124Xe',
             '125Te', '126Te', '126Xe', '127I', '128Xe', '128Te', '129Xe', '130Xe', '130Te',
             '130Ba', '131Xe', '132Xe', '132Ba', '133Cs', '134Ba', '134Xe', '135Ba',
             '136Ba', '136Ce', '136Xe', '137Ba', '138Ba', '138Ce', '138La', '139La',
             '140Ce', '141Pr', '142Nd', '142Ce', '143Nd', '144Nd', '144Sm', '145Nd',
             '146Nd', '147Sm', '148Sm', '148Nd', '149Sm', '150Sm', '150Nd', '151Eu',
             '152Sm', '152Gd', '153Eu', '154Gd', '154Sm', '155Gd', 'CeO', '156Gd', '156Dy',
             '157Gd', '158Gd', '158Dy', '159Tb', '160Dy', '160Gd', '161Dy', '162Dy',
             '162Er', '163Dy', '164Dy', '164Er', '165Ho', '166Er', '167Er', '168Er',
             '168Yb', '169Tm', '170Yb', '170Er', '171Yb', '172Yb', '173Yb', '174Yb',
             '174Hf', '175Lu', '176Hf', '176Yb', '176Lu', '177Hf', '178Hf', '179Hf',
             '180Hf', '180W', '180Ta', '181Ta', '182W', '183W', '184W', '184Os', '185Re',
             '186Os', '186W', '187Re', '187Os', '188Os', '189Os', '190Os', '190Pt', '191Ir',
             '192Pt', '192Os', '193Ir', '194Pt', '195Pt', '196Pt', '196Hg', '197Au',
             '198Hg', '198Pt', '199Hg', '200Hg', '201Hg', '202Hg', '203Tl', '204Pb',
             '204Hg', '205Tl', '206Pb', '207Pb', '208Pb', '209Bi', '220Bkg', '232Th',
             '234U', '235U', '238U', 'ThO', 'UO']            

    ndata = nc.Dataset(h5file)

    vnames = [f for f in ndata.groups['ICAP'].variables['TwInfo']]
    vdata = [f for f in ndata.groups['ICAP'].variables['TwData']]

    ds = xr.Dataset()
    for i,d in enumerate(vnames):
        dloc = np.asarray(vdata)[:,i]
        ds[d] = xr.DataArray(dloc, dims=['readings'])

    pdata = ndata.groups['PeakData'].variables['PeakData'][:]
    pdata = pdata.squeeze()
    pdata = pdata.reshape(pdata.shape[0]*pdata.shape[1], pdata.shape[2])

    time = np.arange(pdata.shape[0])
    ds['Data'] = xr.DataArray(pdata, 
                              coords=[time,nlist], 
                              dims=['datapoints','mass'])
    nattrs = list()
    vattrs = list()
    for at in list(ndata.__dict__):
        nattrs.append(at)
        vattrs.append(ndata.__dict__[at])

    attrs = dict()
    for i,v in enumerate(nattrs):
        attrs[v] = vattrs[i]


    del(attrs['Configuration File Contents'])

    ds.attrs = attrs
    
    return ds

def deconvolute (dataset, threshold_model = 'Poisson'):
    """Separates the single-moieties related events from the background in a
    given dataset.
    
    The average and standard deviation of the datset are calculated.
    Events are identified as datapoints that exceed the Poisson threshold, and
    are removed from the dataset:
    Threshold = average + 2.72 + 3.29*stdev
    The average and standard deviation of the resulting dataset are
    recalculated.
    the procedure repeats itself until there are no more datapoints to be
    excluded.
    
    Call it by typing:
    background, events, dissolved, dissolved_std, event_num, event_mean,
    event_std, loop_count, threshold = deconvolute(example)
    *'example' being the name of the preffered dataset.  
    
    Output: 
    background - A dataset containing the datapoints not identified as events.
    events - A dataset containing the datapoints identified as particles.
    dissolved - The average of the background dataset.
    dissolved_std - The standard deviation of the background dataset.
    event_num - The number of events.
    event_mean - The average of the events dataset.
    event_std - The standard deviation of the events dataset.
    loop_count - The number of times the procedure had to be applid before it
                 reached the final dataset.
    threshold - The final threshold value.
    
    Based on the work by:
    Anal. Chem. 1968, 40, 3, 586–593.
    Pure Appl. Chem., 1995, Vol. 67, No. 10, pp. 1699-1723.
    J. Anal. At. Spectrom., 2013, 28, 1220.
    """
    import copy

    working_dataset = copy.deepcopy(dataset) 
    event_num = -1 # Resetting the output values
    event_count = 0 
    loop_count = 0
             
    while event_num < event_count:
        event_num = event_count
        
        if threshold_model == 'Poisson':
            threshold = working_dataset.mean() + 2.72 + 3.29*working_dataset.std()
        elif threshold_model == 'Gaussian3':
            threshold = working_dataset.mean() + 3*working_dataset.std()
        elif threshold_model == 'Gaussian5':
            threshold = working_dataset.mean() + 5*working_dataset.std()
        else:
            threshold = threshold_model
            
        used_threshold = threshold
        reduced_dataset = working_dataset.where(working_dataset<=used_threshold)
        event_count = reduced_dataset.isna().sum()
        working_dataset = reduced_dataset
        loop_count = loop_count+1

    background = dataset.where(dataset<=used_threshold).dropna()
    events = dataset.where(dataset>used_threshold).dropna()
    dissolved = background.mean()
    dissolved_std = background.std()
    event_mean = events.mean()
    event_std = events.std()

    return background, events, dissolved, dissolved_std, event_num, event_mean, event_std, loop_count, used_threshold



def find_events (dataset,datapoints_per_segment,threshold_model = 'Poisson'):
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
    
    for ds in split:
        background, events, dissolved, dissolved_std, event_num, event_mean, event_std, loop_count, used_threshold = deconvolute(ds,threshold_model)

        if not (events.empty):
            split_event_dataset = pd.concat((split_event_dataset,events))

    return split_event_dataset