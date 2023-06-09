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

def import_plot_simultaneous_events(waveforms,*elements):
    """Imports data exported from the TofPilot software of TofWerk2R. Exports histograms of the chosen events for the selected elements.
    
    Call by:
    dataset = import_tofwerk2R(waveforms,element1, element2,....)
    
    -"dataset" the desired name of the dataset.
    -"waveforms" is the number of waveforms used during the data acquisition. Necessary for the conversion to cps.
    -"element" is the desired element to be used. Use the symbol and mass without space, and in quotes, e.g. "Li6","C12". Use "All" if you just want the entire acquired dataset.
    
    
    Browse files, click and wait for the dataset to load.
    
    DO NOT close the tkinter window that appears, else the program will crush. 
    Minimize it until your work is done.
    """
    filepath = filedialog.askopenfilename(title='Choose file to open',
                                         filetypes = (("Text Files","*.txt"),
                                                      ("Python Files","*.py")))
    file = open(filepath,'r')
    data = waveforms*pd.read_csv(file,
                                 skiprows=7,
                                 delimiter=',\s+|\t+|\s+\t+|\t+',
                                 engine = 'python',
                                 header =0,
                                 index_col= 'index'
                                )
    file.close()
    
    output = pd.DataFrame()
    
    for element in elements:
        if element == "Al27":
            Al27 = data['[27Al]+ mass 26.981'] # 100%
            events = find_events(Al27,100)
            output = output.assign(Al27 = events)
            
            
        if element == "As75":
            As75 = data['[75As]+ mass 74.9211'] # 100%
            events = find_events(As75,100)
            output = output.assign(As75 = events)
            
        
        if element == "Ba130":
            Ba130 = data['[130Ba]+ mass 129.906'] # 0.106%
            events = find_events(Ba130,100)
            output = output.assign(Ba130 = events)
        if element == "Ba132":
            Ba132 = data['[132Ba]+ mass 131.905'] # 0.101%
            events = find_events(Ba132,100)
            output = output.assign(Ba132 = events)
        if element == "Ba134":
            Ba134 = data['[134Ba]+ mass 133.904'] # 2.417%
            events = find_events(Ba134,100)
            output = output.assign(Ba134 = events)
        if element == "Ba135":
            Ba135 = data['[135Ba]+ mass 134.905'] # 6.592%
            events = find_events(Ba135,100)
            output = output.assign(Ba135 = events)
        if element == "Ba136":
            Ba136 = data['[136Ba]+ mass 135.904'] # 7.854%
            events = find_events(Ba136,100)
            output = output.assign(Ba136 = events)
        if element == "Ba137":
            Ba137 = data['[137Ba]+ mass 136.905'] # 11.232%
            events = find_events(Ba137,100)
            output = output.assign(Ba137 = events)
        if element == "Ba138":
            Ba138 = data['[138Ba]+ mass 137.905'] # 71.698%
            events = find_events(Ba138,100)
            output = output.assign(Ba138 = events)
            
        if element == "Ba137_dc":
            Ba137_dc = data['[137Ba]++ mass 68.4524']
            events = find_events(Ba137_dc,100)
            output = output.assign(Ba137_dc = events)
        if element == "Ba138_dc":
            Ba138_dc = data['[138Ba]++ mass 68.9521']
            events = find_events(Ba138_dc,100)
            output = output.assign(Ba138_dc = events)    
    
    
        if element == "Pb204":
            Pb204 = data['[204Pb]+ mass 203.973']# 1.4%
            events = find_events(Pb204,100)
            output = output.assign(Pb204 = events)
        if element == "Pb206":
            Pb206 = data['[206Pb]+ mass 205.974']# 24.1%
            events = find_events(Pb206,100)
            output = output.assign(Pb206 = events)
        if element == "Pb207":
            Pb207 = data['[207Pb]+ mass 206.975']# 22.1%
            events = find_events(Pb207,100)
            output = output.assign(Pb207 = events)
        if element == "Pb208":
            Pb208 = data['[208Pb]+ mass 207.976']# 52.4%
            events = find_events(Pb208,100)
            output = output.assign(Pb208 = events)
        
        
        if element == "Cd106":
            Cd106 = data['[106Cd]+ mass 105.906'] # 1.25%
            events = find_events(Cd106,100)
            output = output.assign(Cd106 = events)
        if element == "Cd108":
            Cd108 = data['[108Cd]+ mass 107.904'] # 0.89%
            events = find_events(Cd108,100)
            output = output.assign(Cd108 = events)
        if element == "Cd110":
            Cd110 = data['[110Cd]+ mass 109.902'] # 12.49%
            events = find_events(Cd110,100)
            output = output.assign(Cd110 = events)
        if element == "Cd111":
            Cd111 = data['[111Cd]+ mass 110.904'] # 12.80%
            events = find_events(Cd111,100)
            output = output.assign(Cd111 = events)
        if element == "Cd112":
            Cd112 = data['[112Cd]+ mass 111.902'] # 24.13%
            events = find_events(Cd112,100)
            output = output.assign(Cd112 = events)
        if element == "Cd113":
            Cd113 = data['[113Cd]+ mass 112.904'] # 12.22%
            events = find_events(Cd113,100)
            output = output.assign(Cd113 = events)
        if element == "Cd114":
            Cd114 = data['[114Cd]+ mass 113.903'] # 28.73%
            events = find_events(Cd114,100)
            output = output.assign(Cd114 = events)
        if element == "Cd116":
            Cd116 = data['[116Cd]+ mass 115.904'] # 7.49%
            events = find_events(Cd116,100)
            output = output.assign(Cd116 = events)
            
            
        if element == "Ca40":
            Ca40 = data['[40Ca]+ mass 39.962'] # 96.941%
            events = find_events(Ca40,100)
            output = output.assign(Ca40 = events)
        if element == "Ca42":
            Ca42 = data['[42Ca]+ mass 41.9581'] # 0.647%
            events = find_events(Ca42,100)
            output = output.assign(Ca42 = events)
        if element == "Ca43":
            Ca43 = data['[43Ca]+ mass 42.9582'] # 0.135%
            events = find_events(Ca43,100)
            output = output.assign(Ca43 = events)
        if element == "Ca44":
            Ca44 = data['[44Ca]+ mass 43.9549'] # 2.086%
            events = find_events(Ca44,100)
            output = output.assign(Ca44 = events)
        if element == "Ca46":
            Ca46 = data['[46Ca]+ mass 45.9531'] # 0.004%
            events = find_events(Ca46,100)
            output = output.assign(Ca46 = events)
        if element == "Ca48":
            Ca48 = data['[48Ca]+ mass 47.952'] # 0.187%
            events = find_events(Ca48,100)
            output = output.assign(Ca48 = events)
            
            
        if element == "Cr50":
            Cr50 = data['[50Cr]+ mass 49.9455']# 4.345%
            events = find_events(Cr50,100)
            output = output.assign(Cr50 = events)
        if element == "Cr52":
            Cr52 = data['[52Cr]+ mass 51.94']# 83.789%
            events = find_events(Cr52,100)
            output = output.assign(Cr52 = events)
        if element == "Cr53":
            Cr53 = data['[53Cr]+ mass 52.9401']# 9.501%
            events = find_events(Cr53,100)
            output = output.assign(Cr53 = events)
        if element == "Cr54":
            Cr54 = data['[54Cr]+ mass 53.9383']# 2.365%
            events = find_events(Cr54,100)
            output = output.assign(Cr54 = events)
            
            
        if element == "Cu63":
            Cu63 = data['[63Cu]+ mass 62.9291']# 69.17%
            events = find_events(Cu63,100)
            output = output.assign(Cu63 = events)
        if element == "Cu65":
            Cu65 = data['[65Cu]+ mass 64.9272']# 30.83%
            events = find_events(Cu65,100)
            output = output.assign(Cu65 = events)
            
            
        if element == "Fe54":
            Fe54 = data['[54Fe]+ mass 53.9391']# 5.845%
            events = find_events(Fe54,100)
            output = output.assign(Fe54 = events)
        if element == "Fe56":
            Fe56 = data['[56Fe]+ mass 55.9344']# 91.754%
            events = find_events(Fe56,100)
            output = output.assign(Fe56 = events)
        if element == "Fe57":
            Fe57 = data['[57Fe]+ mass 56.9348']# 2.119%
            events = find_events(Fe57,100)
            output = output.assign(Fe57 = events)
        if element == "Fe58":
            Fe58 = data['[58Fe]+ mass 57.9327']# 0.282%
            events = find_events(Fe58,100)
            output = output.assign(Fe58 = events)
            
            
        if element == "Mg24":
            Mg24 = data['[24Mg]+ mass 23.9845']# 78.99%
            events = find_events(Mg24,100)
            output = output.assign(Mg24 = events)
        if element == "Mg25":
            Mg25 = data['[25Mg]+ mass 24.9853']# 10.00%
            events = find_events(Mg25,100)
            output = output.assign(Mg25 = events)
        if element == "Mg26":
            Mg26 = data['[26Mg]+ mass 25.982']# 11.01%
            events = find_events(Mg26,100)
            output = output.assign(Mg26 = events)
            
            
        if element == "Mn55":
            Mn55 = data['[55Mn]+ mass 54.9375']# 100%
            events = find_events(Mn55,100)
            output = output.assign(Mn55 = events)
            
            
        if element == "Mo92":
            Mo92 = data['[92Mo]+ mass 91.9063']# 14.84%
            events = find_events(Mo92,100)
            output = output.assign(Mo92 = events)
        if element == "Mo94":
            Mo94 = data['[94Mo]+ mass 93.9045']# 9.25%
            events = find_events(Mo94,100)
            output = output.assign(Mo94 = events)
        if element == "Mo95":
            Mo95 = data['[95Mo]+ mass 94.9053']# 15.92%
            events = find_events(Mo95,100)
            output = output.assign(Mo95 = events)   
        if element == "Mo96":
            Mo96 = data['[96Mo]+ mass 95.9041']# 16.68%
            events = find_events(Mo96,100)
            output = output.assign(Mo96 = events)
        if element == "Mo97":
            Mo97 = data['[97Mo]+ mass 96.9055']# 9.55%
            events = find_events(Mo97,100)
            output = output.assign(Mo97 = events)
        if element == "Mo98":
            Mo98 = data['[98Mo]+ mass 97.9049']# 24.13%
            events = find_events(Mo98,100)
            output = output.assign(Mo98 = events)   
        if element == "Mo100":
            Mo100 = data['[100Mo]+ mass 99.9069']# 9.63%
            events = find_events(Mo100,100)
            output = output.assign(Mo100 = events)
            
            
        if element == "Ni58":
            Ni58 = data['[58Ni]+ mass 57.9348']# 68.0769%
            events = find_events(Ni58,100)
            output = output.assign(Ni58 = events)   
        if element == "Ni60":
            Ni60 = data['[60Ni]+ mass 59.9302']# 26.2231%
            events = find_events(Ni60,100)
            output = output.assign(Ni60 = events)
        if element == "Ni61":
            Ni61 = data['[61Ni]+ mass 60.9305']# 1.1399%
            events = find_events(Ni61,100)
            output = output.assign(Ni61 = events)
        if element == "Ni62":
            Ni62 = data['[62Ni]+ mass 61.9278']# 3.6345%
            events = find_events(Ni62,100)
            output = output.assign(Ni62 = events)   
        if element == "Ni64":
            Ni64 = data['[64Ni]+ mass 63.9274']# 0.9256%
            events = find_events(Ni64,100)
            output = output.assign(Ni64 = events)   
            
            
        if element == "K39":
            K39 = data['[39K]+ mass 38.9632']# 93.2581%
            events = find_events(K39,100)
            output = output.assign(K39 = events)    
        if element == "K41":
            K41 = data['[41K]+ mass 40.9613']# 	6.7302%
            events = find_events(K41,100)
            output = output.assign(K41 = events)     
        
        
        if element == "Na23":
            Na23 = data['[23Na]+ mass 22.9892']# 100%
            events = find_events(Na23,100)
            output = output.assign(Na23 = events)
            
            
        if element == "Sr84":
            Sr84 = data['[84Sr]+ mass 83.9129']# 0.56%
            events = find_events(Sr84,100)
            output = output.assign(Sr84 = events)
        if element == "Sr86":
            Sr86 = data['[86Sr]+ mass 85.9087']# 9.86%
            events = find_events(Sr86,100)
            output = output.assign(Sr86 = events)
        if element == "Sr87":
            Sr87 = data['[87Sr]+ mass 86.9083']# 7.00%
            events = find_events(Sr87,100)
            output = output.assign(Sr87 = events)   
        if element == "Sr88":
            Sr88 = data['[88Sr]+ mass 87.9051']# 82.58%
            events = find_events(Sr88,100)
            output = output.assign(Sr88 = events)  
            
            
        if element == "U234":
            U234 = data['[234U]+ mass 234.04']# 0.0055%
            events = find_events(U234,100)
            output = output.assign(U234 = events)
        if element == "U235":
            U235 = data['[235U]+ mass 235.043']# 0.7200%
            events = find_events(U235,100)
            output = output.assign(U235 = events)
        if element == "U238":
            U238 = data['[238U]+ mass 238.05']# 99.2745%
            events = find_events(U238,100)
            output = output.assign(U238 = events)
            
        if element == "UO254":
            UO254 = data['UO+ mass 254.045']
            events = find_events(UO254,100)
            output = output.assign(UO254 = events)  
            
            
        if element == "V50":
            V50 = data['[50V]+ mass 49.9466']# 0.250%
            events = find_events(V50,100)
            output = output.assign(V50 = events)  
        if element == "V51":
            V51 = data['[51V]+ mass 50.9434']# 99.750%
            events = find_events(V51,100)
            output = output.assign(V51 = events)
            
            
        if element == "Zn64":
            Zn64 = data['[64Zn]+ mass 63.9286']# 48.63%
            events = find_events(Zn64,100)
            output = output.assign(Zn64 = events)   
        if element == "Zn66":
            Zn66 = data['[66Zn]+ mass 65.9255']# 27.90%
            events = find_events(Zn66,100)
            output = output.assign(Zn66 = events)
        if element == "Zn67":
            Zn67 = data['[67Zn]+ mass 66.9266']# 4.10 %
            events = find_events(Zn67,100)
            output = output.assign(Zn67 = events)
        if element == "Zn68":
            Zn68 = data['[68Zn]+ mass 67.9243']# 18.75%
            events = find_events(Zn68,100)
            output = output.assign(Zn68 = events)    
        if element == "Zn70":
            Zn70 = data['[70Zn]+ mass 69.9248']# 0.62%
            events = find_events(Zn70,100)
            output = output.assign(Zn70 = events)  
            
            
        if element == "Be9":
            Be9 = data['[9Be]+ mass 9.01163']
            events = find_events(Be9,100)
            output = output.assign(Be9 = events)   
        if element == "Be10":
            Be10 = data['[10B]+ mass 10.0124']
            events = find_events(Be10,100)
            output = output.assign(Be10 = events)
        if element == "Be11":
            Be11 = data['[11B]+ mass 11.0088']
            events = find_events(Be11,100)
            output = output.assign(Be11 = events)
            
            
        if element == "Li6":
            Li6 = data['[6Li]+ mass 6.01546']
            events = find_events(Li6,100)
            output = output.assign(Li6 = events)    
        if element == "Li7":
            Li7 = data['[7Li]+ mass 7.01546']
            events = find_events(Li7,100)
            output = output.assign(Li7 = events)    
        
        
        if element == "C12":
            C12 = data['[12C]+ mass 11.9994']
            events = find_events(C12,100)
            output = output.assign(C12 = events)   
        if element == "C13":
            C13 = data['[13C]+ mass 13.0028']
            events = find_events(C13,100)
            output = output.assign(C13 = events)
        if element == "CO2":
            CO2 = data['CO2+ mass 43.9893']
            events = find_events(CO2,100)
            output = output.assign(CO2 = events)
        if element == "CO2H":
            CO2H = data['CHO2+ mass 44.9971']
            events = find_events(CO2H,100)
            output = output.assign(CO2H = events)    
        if element == "C2H226":
            C2H226 = data['C2H2+ mass 26.0151']
            events = find_events(C2H226,100)
            output = output.assign(C2H226 = events)
        if element == "C2H327":
            C2H327 = data['C2H3+ mass 27.0229']
            events = find_events(C2H327,100)
            output = output.assign(C2H327 = events)
            
            
        if element == "N14":
            N14 = data['[14N]+ mass 14.0025']
            events = find_events(N14,100)
            output = output.assign(N14 = events)   
        if element == "N15":
            N15 = data['[15N]+ mass 14.9996']
            events = find_events(N15,100)
            output = output.assign(N15 = events)
        if element == "N2":
            N2 = data['N2+ mass 28.0056']
            events = find_events(N2,100)
            output = output.assign(N2 = events)
        if element == "N2H29":
            N2H29 = data['HN2+ mass 29.0134']
            events = find_events(N2H29,100)
            output = output.assign(N2H29 = events)  
        if element == "NO30":
            NO30 = data['NO+ mass 29.9974']
            events = find_events(NO30,100)
            output = output.assign(NO30 = events)
        if element == "NO31":
            NO31 = data['[15N]O+ mass 30.9945']
            events = find_events(NO31,100)
            output = output.assign(NO31 = events)  
        if element == "NO246":
            NO246 = data['NO2+ mass 45.9924']
            events = find_events(NO246,100)
            output = output.assign(NO246 = events)
            
        if element == "O16":
            O16 = data['[16O]+ mass 15.9944']
            events = find_events(O16,100)
            output = output.assign(O16 = events)   
        if element == "O18":
            O18 = data['[18O]+ mass 17.9986']
            events = find_events(O18,100)
            output = output.assign(O18 = events)
        if element == "OH17":
            OH17 = data['OH+ mass 17.0022']
            events = find_events(OH17,100)
            output = output.assign(OH17 = events)
        if element == "H2O18":
            H2O18 = data['H2O+ mass 18.01']
            events = find_events(H2O18,100)
            output = output.assign(H2O18 = events)    
        if element == "H3O19":
            H3O19 = data['H3O+ mass 19.0178']
            events = find_events(H3O19,100)
            output = output.assign(H3O19 = events)
        if element == "O232":
            O232 = data['O2+ mass 31.9893']
            events = find_events(O232,100)
            output = output.assign(O232 = events)  
        if element == "O2H33":
            O2H33 = data['O2H+ mass 32.9971']
            events = find_events(O2H33,100)
            output = output.assign(O2H33 = events)   
        if element == "OH17":
            OH17 = data['OH+ mass 17.0022']
            events = find_events(OH17,100)
            output = output.assign(OH17 = events)  
        if element == "O234":
            O234 = data['O[18O]+ mass 33.9935']
            events = find_events(O234,100)
            output = output.assign(O234 = events)   
        
        
        if element == "Si28":
            Si28 = data['[28Si]+ mass 27.9764']
            events = find_events(Si28,100)
            output = output.assign(Si28 = events)   
        if element == "Si29":
            Si29 = data['[29Si]+ mass 28.976']
            events = find_events(Si29,100)
            output = output.assign(Si29 = events)  
        if element == "Si30":
            Si30 = data['[30Si]+ mass 29.9732']
            events = find_events(Si30,100)
            output = output.assign(Si30 = events)
            
            
        if element == "P31":
            P31 = data['[31P]+ mass 30.9732']
            events = find_events(P31,100)
            output = output.assign(P31 = events)
        
        
        if element == "S32":
            S32 = data['[32S]+ mass 31.9715']
            events = find_events(S32,100)
            output = output.assign(S32 = events)  
        if element == "S33":
            S33 = data['[33S]+ mass 32.9709']
            events = find_events(S33,100)
            output = output.assign(S33 = events)   
        if element == "S34":
            S34 = data['[34S]+ mass 33.9673']
            events = find_events(S34,100)
            output = output.assign(S34 = events)  
        if element == "S36":
            S36 = data['[36S]+ mass 35.9665']
            events = find_events(S36,100)
            output = output.assign(S36 = events)
            
        if element == "Cl35":
            Cl35 = data['[35Cl]+ mass 34.9683']
            events = find_events(Cl35,100)
            output = output.assign(Cl35 = events)
        if element == "Cl37":
            Cl37 = data['[37Cl]+ mass 36.9654']
            events = find_events(Cl37,100)
            output = output.assign(Cl37 = events)  
        if element == "HCl36":
            HCl36 = data['HCl+ mass 35.9761']
            events = find_events(HCl36,100)
            output = output.assign(HCl36 = events)   
        if element == "ClO51":
            ClO51 = data['ClO+ mass 50.9632']
            events = find_events(ClO51,100)
            output = output.assign(ClO51 = events) 
        if element == "ClO53":
            ClO53 = data['[37Cl]O+ mass 52.9603']
            events = find_events(ClO53,100)
            output = output.assign(ClO53 = events)    
        
        
        if element == "Sc45":
            Sc45 = data['[45Sc]+ mass 44.9554']
            events = find_events(Sc45,100)
            output = output.assign(Sc45 = events)
            
            
        if element == "Ti46":
            Ti46 = data['[46Ti]+ mass 45.9521']
            events = find_events(Ti46,100)
            output = output.assign(Ti46 = events)
        if element == "Ti47":
            Ti47 = data['[47Ti]+ mass 46.9512']
            events = find_events(Ti47,100)
            output = output.assign(Ti47 = events)  
        if element == "Ti48":
            Ti48 = data['[48Ti]+ mass 47.9474']
            events = find_events(Ti48,100)
            output = output.assign(Ti48 = events)  
        if element == "Ti49":
            Ti49 = data['[49Ti]+ mass 48.9473']
            events = find_events(Ti49,100)
            output = output.assign(Ti49 = events)  
        if element == "Ti50":
            Ti50 = data['[50Ti]+ mass 49.9442']
            events = find_events(Ti50,100)
            output = output.assign(Ti50 = events)
            
            
        if element == "Ga69":
            Ga69 = data['[69Ga]+ mass 68.925']
            events = find_events(Ga69,100)
            output = output.assign(Ga69 = events)  
        if element == "Ga71":
            Ga71 = data['[71Ga]+ mass 70.9241']
            events = find_events(Ga71,100)
            output = output.assign(Ga71 = events)
            
            
        if element == "Ar36":
            Ar36 = data['[36Ar]+ mass 35.967']
            events = find_events(Ar36,100)
            output = output.assign(Ar36 = events)
        if element == "Ar38":
            Ar38 = data['[38Ar]+ mass 37.9622']
            events = find_events(Ar38,100)
            output = output.assign(Ar38 = events)  
        if element == "Ar40":
            Ar40 = data['[40Ar]+ mass 39.9618']
            events = find_events(Ar40,100)
            output = output.assign(Ar40 = events)   
        if element == "ArH37":
            ArH37 = data['[36Ar]H+ mass 36.9748']
            events = find_events(ArH37,100)
            output = output.assign(ArH37 = events)  
        if element == "ArH39":
            ArH39 = data['[38Ar]H+ mass 38.97']
            events = find_events(ArH39,100)
            output = output.assign(ArH39 = events)    
        if element == "ArH41":
            ArH41 = data['ArH+ mass 40.9697']
            events = find_events(ArH41,100)
            output = output.assign(ArH41 = events)
        if element == "ArH242":
            ArH242 = data['ArH2+ mass 41.9775']
            events = find_events(ArH242,100)
            output = output.assign(ArH242 = events)  
        if element == "ArO52":
            ArO52 = data['[36Ar]O+ mass 51.9619']
            events = find_events(ArO52,100)
            output = output.assign(ArO52 = events)   
        if element == "ArN54":
            ArN54 = data['ArN+ mass 53.9649']
            events = find_events(ArN54,100)
            output = output.assign(ArN54 = events)  
        if element == "ArO56":
            ArO56 = data['ArO+ mass 55.9567']
            events = find_events(ArO56,100)
            output = output.assign(ArO56 = events)    
        if element == "ArOH57":
            ArOH57 = data['ArOH+ mass 56.9646']
            events = find_events(ArOH57,100)
            output = output.assign(ArOH57 = events)
        if element == "Ar280":
            Ar280 = data['Ar2+ mass 79.9242']
            events = find_events(Ar280,100)
            output = output.assign(Ar280 = events)  
        if element == "ArCl":
            ArCl = data['ArCl+ mass 74.9307']
            events = find_events(ArCl,100)
            output = output.assign(ArCl = events)   
        if element == "Ar276":
            Ar276 = data['Ar[36Ar]+ mass 75.9294']
            events = find_events(Ar276,100)
            output = output.assign(Ar276 = events)  
        if element == "ArCl77":
            ArCl77 = data['Ar[37Cl]+ mass 76.9277']
            events = find_events(ArCl77,100)
            output = output.assign(ArCl77 = events)    
        if element == "Ar278":
            Ar278 = data['Ar[38Ar]+ mass 77.9246']
            events = find_events(Ar278,100)
            output = output.assign(Ar278 = events)    
            
            
        if element == "Ge70":
            Ge70 = data['[70Ge]+ mass 69.9237']
            events = find_events(Ge70,100)
            output = output.assign(Ge70 = events)  
        if element == "Ge72":
            Ge72 = data['[72Ge]+ mass 71.9215']
            events = find_events(Ge72,100)
            output = output.assign(Ge72 = events)  
        if element == "Ge73":
            Ge73 = data['[73Ge]+ mass 72.9229']
            events = find_events(Ge73,100)
            output = output.assign(Ge73 = events)  
        if element == "Ge74":
            Ge74 = data['[74Ge]+ mass 73.9206']
            events = find_events(Ge74,100)
            output = output.assign(Ge74 = events)    
        if element == "Ge76":
            Ge76 = data['[76Ge]+ mass 75.9209']
            events = find_events(Ge76,100)
            output = output.assign(Ge76 = events)    
            
        
        if element == "Co59":
            Co59 = data['[59Co]+ mass 58.9327']
            events = find_events(Co59,100)
            output = output.assign(Co59 = events)
            
        
        if element == "Se74":
            Se74 = data['[74Se]+ mass 73.9219']
            events = find_events(Se74,100)
            output = output.assign(Se74 = events)
        if element == "Se76":
            Se76 = data['[76Se]+ mass 75.9187']
            events = find_events(Se76,100)
            output = output.assign(Se76 = events)  
        if element == "Se77":
            Se77 = data['[77Se]+ mass 76.9194']
            events = find_events(Se77,100)
            output = output.assign(Se77 = events)   
        if element == "Se78":
            Se78 = data['[78Se]+ mass 77.9168']
            events = find_events(Se78,100)
            output = output.assign(Se78 = events)  
        if element == "Se80":
            Se80 = data['[80Se]+ mass 79.916']
            events = find_events(Se80,100)
            output = output.assign(Se80 = events)    
        if element == "Se82":
            Se82 = data['[82Se]+ mass 81.9162']
            events = find_events(Se82,100)
            output = output.assign(Se82 = events) 
            
            
        if element == "Kr78":
            Kr78 = data['[78Kr]+ mass 77.9198']
            events = find_events(Kr78,100)
            output = output.assign(Kr78 = events)
        if element == "Kr80":
            Kr80 = data['[80Kr]+ mass 79.9158']
            events = find_events(Kr80,100)
            output = output.assign(Kr80 = events)  
        if element == "Kr82":
            Kr82 = data['[82Kr]+ mass 81.9129']
            events = find_events(Kr82,100)
            output = output.assign(Kr82 = events)   
        if element == "Kr83":
            Kr83 = data['[83Kr]+ mass 82.9136']
            events = find_events(Kr83,100)
            output = output.assign(Kr83 = events)  
        if element == "Kr84":
            Kr84 = data['[84Kr]+ mass 83.911']
            events = find_events(Kr84,100)
            output = output.assign(Kr84 = events)    
        if element == "Kr86":
            Kr86 = data['[86Kr]+ mass 85.9101']
            events = find_events(Kr86,100)
            output = output.assign(Kr86 = events) 
        
        
        if element == "Br79":
            Br79 = data['[79Br]+ mass 78.9178']
            events = find_events(Br79,100)
            output = output.assign(Br79 = events)    
        if element == "Br81":
            Br81 = data['[81Br]+ mass 80.9157']
            events = find_events(Br81,100)
            output = output.assign(Br81 = events)
        
        
        if element == "Rb85":
            Rb85 = data['[85Rb]+ mass 84.9112']
            events = find_events(Rb85,100)
            output = output.assign(Rb85 = events)    
        if element == "Rb87":
            Rb87 = data['[87Rb]+ mass 86.9086']
            events = find_events(Rb87,100)
            output = output.assign(Rb87 = events)
        
        
        if element == "Y89":
            Y89 = data['[89Y]+ mass 88.9053']
            events = find_events(Y89,100)
            output = output.assign(Y89 = events)
        
        
        if element == "Zr90":
            Zr90 = data['[90Zr]+ mass 89.9042']
            events = find_events(Zr90,100)
            output = output.assign(Zr90 = events)  
        if element == "Zr91":
            Zr91 = data['[91Zr]+ mass 90.9051']
            events = find_events(Zr91,100)
            output = output.assign(Zr91 = events)   
        if element == "Zr92":
            Zr92 = data['[92Zr]+ mass 91.9045']
            events = find_events(Zr92,100)
            output = output.assign(Zr92 = events)  
        if element == "Zr94":
            Zr94 = data['[94Zr]+ mass 93.9058']
            events = find_events(Zr94,100)
            output = output.assign(Zr94 = events)    
        if element == "Zr96":
            Zr96 = data['[96Zr]+ mass 95.9077']
            events = find_events(Zr96,100)
            output = output.assign(Zr96 = events)
            
            
        if element == "Nb93":
            Nb93 = data['[93Nb]+ mass 92.9058']
            events = find_events(Nb93,100)
            output = output.assign(Nb93 = events)
            
            
        if element == "Ru96":
            Ru96 = data['[96Ru]+ mass 95.9071']
            events = find_events(Ru96,100)
            output = output.assign(Ru96 = events)  
        if element == "Ru98":
            Ru98 = data['[98Ru]+ mass 97.9047']
            events = find_events(Ru98,100)
            output = output.assign(Ru98 = events)   
        if element == "Ru99":
            Ru99 = data['[99Ru]+ mass 98.9054']
            events = find_events(Ru99,100)
            output = output.assign(Ru99 = events)  
        if element == "Ru100":
            Ru100 = data['[100Ru]+ mass 99.9037']
            events = find_events(Ru100,100)
            output = output.assign(Ru100 = events)   
        if element == "Ru101":
            Ru101 = data['[101Ru]+ mass 100.905']
            events = find_events(Ru101,100)
            output = output.assign(Ru101 = events)
        if element == "Ru102":
            Ru102 = data['[102Ru]+ mass 101.904']
            events = find_events(Ru102,100)
            output = output.assign(Ru102 = events)    
        if element == "Ru104":
            Ru104 = data['[104Ru]+ mass 103.905']
            events = find_events(Ru104,100)
            output = output.assign(Ru104 = events)
            
            
        if element == "Pd102":
            Pd102 = data['[102Pd]+ mass 101.905']
            events = find_events(Pd102,100)
            output = output.assign(Pd102 = events)  
        if element == "Pd104":
            Pd104 = data['[104Pd]+ mass 103.903']
            events = find_events(Pd104,100)
            output = output.assign(Pd104 = events)   
        if element == "Pd105":
            Pd105 = data['[105Pd]+ mass 104.905']
            events = find_events(Pd105,100)
            output = output.assign(Pd105 = events)  
        if element == "Pd106":
            Pd106 = data['[106Pd]+ mass 105.903']
            events = find_events(Pd106,100)
            output = output.assign(Pd106 = events)    
        if element == "Pd108":
            Pd108 = data['[108Pd]+ mass 107.903']
            events = find_events(Pd108,100)
            output = output.assign(Pd108 = events)
        if element == "Pd110":
            Pd110 = data['[110Pd]+ mass 109.905']
            events = find_events(Pd110,100)
            output = output.assign(Pd110 = events)
            
            
        if element == "Rh103":
            Rh103 = data['[103Rh]+ mass 102.905']
            events = find_events(Rh103,100)
            output = output.assign(Rh103 = events)  
            
            
        if element == "Ag107":
            Ag107 = data['[107Ag]+ mass 106.905']
            events = find_events(Ag107,100)
            output = output.assign(Ag107 = events)
        if element == "Ag109":
            Ag109 = data['[109Ag]+ mass 108.904']
            events = find_events(Ag109,100)
            output = output.assign(Ag109 = events)  
            
            
        if element == "Sn112":
            Sn112 = data['[112Sn]+ mass 111.904']
            events = find_events(Sn112,100)
            output = output.assign(Sn112 = events)   
        if element == "Sn114":
            Sn114 = data['[114Sn]+ mass 113.902']
            events = find_events(Sn114,100)
            output = output.assign(Sn114 = events)  
        if element == "Sn115":
            Sn115 = data['[115Sn]+ mass 114.903']
            events = find_events(Sn115,100)
            output = output.assign(Sn115 = events)   
        if element == "Sn116":
            Sn116 = data['[116Sn]+ mass 115.901']
            events = find_events(Sn116,100)
            output = output.assign(Sn116 = events)
        if element == "Sn117":
            Sn117 = data['[117Sn]+ mass 116.902']
            events = find_events(Sn117,100)
            output = output.assign(Sn117 = events)  
        if element == "Sn118":
            Sn118 = data['[118Sn]+ mass 117.901']
            events = find_events(Sn118,100)
            output = output.assign(Sn118 = events)   
        if element == "Sn119":
            Sn119 = data['[119Sn]+ mass 118.903']
            events = find_events(Sn119,100)
            output = output.assign(Sn119 = events)  
        if element == "Sn120":
            Sn120 = data['[120Sn]+ mass 119.902']
            events = find_events(Sn120,100)
            output = output.assign(Sn120 = events)    
        if element == "Sn122":
            Sn122 = data['[122Sn]+ mass 121.903']
            events = find_events(Sn122,100)
            output = output.assign(Sn122 = events)
        if element == "Sn124":
            Sn124 = data['[124Sn]+ mass 123.905']
            events = find_events(Sn124,100)
            output = output.assign(Sn124 = events)  
            
            
        if element == "In113":
            In113 = data['[113In]+ mass 112.904']
            events = find_events(In113,100)
            output = output.assign(In113 = events)
        if element == "In115":
            In115 = data['[115In]+ mass 114.903']
            events = find_events(In115,100)
            output = output.assign(In115 = events)  
            
            
        if element == "Sb121":
            Sb121 = data['[121Sb]+ mass 120.903']
            events = find_events(Sb121,100)
            output = output.assign(Sb121 = events)   
        if element == "Sb123":
            Sb123 = data['[123Sb]+ mass 122.904']
            events = find_events(Sb123,100)
            output = output.assign(Sb123 = events)
            
            
        if element == "Te120":
            Te120 = data['[120Te]+ mass 119.903']
            events = find_events(Te120,100)
            output = output.assign(Te120 = events)    
        if element == "Te122":
            Te122 = data['[122Te]+ mass 121.902']
            events = find_events(Te122,100)
            output = output.assign(Te122 = events)
        if element == "Te123":
            Te123 = data['[123Te]+ mass 122.904']
            events = find_events(Te123,100)
            output = output.assign(Te123 = events)  
        if element == "Te124":
            Te124 = data['[124Te]+ mass 123.902']
            events = find_events(Te124,100)
            output = output.assign(Te124 = events)   
        if element == "Te125":
            Te125 = data['[125Te]+ mass 124.904']
            events = find_events(Te125,100)
            output = output.assign(Te125 = events)  
        if element == "Te126":
            Te126 = data['[126Te]+ mass 125.903']
            events = find_events(Te126,100)
            output = output.assign(Te126 = events)    
        if element == "Te128":
            Te128 = data['[128Te]+ mass 127.904']
            events = find_events(Te128,100)
            output = output.assign(Te128 = events)
        if element == "Te130":
            Te130 = data['[130Te]+ mass 129.906']
            events = find_events(Te130,100)
            output = output.assign(Te130 = events)
            
            
        if element == "Xe124":
            Xe124 = data['[124Xe]+ mass 123.905']
            events = find_events(Xe124,100)
            output = output.assign(Xe124 = events)  
        if element == "Xe126":
            Xe126 = data['[126Xe]+ mass 125.904']
            events = find_events(Xe126,100)
            output = output.assign(Xe126 = events)    
        if element == "Xe128":
            Xe128 = data['[128Xe]+ mass 127.903']
            events = find_events(Xe128,100)
            output = output.assign(Xe128 = events)
        if element == "Xe129":
            Xe129 = data['[129Xe]+ mass 128.904']
            events = find_events(Xe129,100)
            output = output.assign(Xe129 = events)  
        if element == "Xe130":
            Xe130 = data['[130Xe]+ mass 129.903']
            events = find_events(Xe130,100)
            output = output.assign(Xe130 = events)   
        if element == "Xe131":
            Xe131 = data['[131Xe]+ mass 130.905']
            events = find_events(Xe131,100)
            output = output.assign(Xe131 = events)  
        if element == "Xe132":
            Xe132 = data['[132Xe]+ mass 131.904']
            events = find_events(Xe132,100)
            output = output.assign(Xe132 = events)    
        if element == "Xe134":
            Xe134 = data['[134Xe]+ mass 133.905']
            events = find_events(Xe134,100)
            output = output.assign(Xe134 = events)
        if element == "Xe136":
            Xe136 = data['[136Xe]+ mass 135.907']
            events = find_events(Xe136,100)
            output = output.assign(Xe136 = events)
            
            
        if element == "I127":
            I127 = data['[127I]+ mass 126.904']
            events = find_events(I127,100)
            output = output.assign(I127 = events)  
            
            
        if element == "Cs133":
            Cs133 = data['[133Cs]+ mass 132.905']
            events = find_events(Cs133,100)
            output = output.assign(Cs133 = events)   
            
            
        if element == "Ce136":
            Ce136 = data['[136Ce]+ mass 135.907']
            events = find_events(Ce136,100)
            output = output.assign(Ce136 = events)
        if element == "Ce138":
            Ce138 = data['[138Ce]+ mass 137.905']
            events = find_events(Ce138,100)
            output = output.assign(Ce138 = events)  
        if element == "Ce140":
            Ce140 = data['[140Ce]+ mass 139.905']
            events = find_events(Ce140,100)
            output = output.assign(Ce140 = events)   
        if element == "Ce142":
            Ce142 = data['[142Ce]+ mass 141.909']
            events = find_events(Ce142,100)
            output = output.assign(Ce142 = events)  
            
        if element == "CeO156":
            CeO156 = data['CeO+ mass 155.9']
            events = find_events(CeO156,100)
            output = output.assign(CeO156 = events)  
            
            
        if element == "La138":
            La138 = data['[138La]+ mass 137.907']
            events = find_events(La138,100)
            output = output.assign(La138 = events)
        if element == "La139":
            La139 = data['[139La]+ mass 138.906']
            events = find_events(La139,100)
            output = output.assign(La139 = events)
            
            
        if element == "Pr141":
            Pr141 = data['[141Pr]+ mass 140.907']
            events = find_events(Pr141,100)
            output = output.assign(Pr141 = events)
            
            
        if element == "Nd142":
            Nd142 = data['[142Nd]+ mass 141.907']
            events = find_events(Nd142,100)
            output = output.assign(Nd142 = events)
        if element == "Nd143":
            Nd143 = data['[143Nd]+ mass 142.909']
            events = find_events(Nd143,100)
            output = output.assign(Nd143 = events)  
        if element == "Nd144":
            Nd144 = data['[144Nd]+ mass 143.91']
            events = find_events(Nd144,100)
            output = output.assign(Nd144 = events)   
        if element == "Nd145":
            Nd145 = data['[145Nd]+ mass 144.912']
            events = find_events(Nd145,100)
            output = output.assign(Nd145 = events)            
        if element == "Nd146":
            Nd146 = data['[146Nd]+ mass 145.913']
            events = find_events(Nd146,100)
            output = output.assign(Nd146 = events)           
        if element == "Nd148":
            Nd148 = data['[148Nd]+ mass 147.916']
            events = find_events(Nd148,100)
            output = output.assign(Nd148 = events)
        if element == "Nd150":
            Nd150 = data['[150Nd]+ mass 149.92']
            events = find_events(Nd150,100)
            output = output.assign(Nd150 = events)
            
            
        if element == "Sm144":
            Sm144 = data['[144Sm]+ mass 143.911']
            events = find_events(Sm144,100)
            output = output.assign(Sm144 = events)
        if element == "Sm147":
            Sm147 = data['[147Sm]+ mass 146.914']
            events = find_events(Sm147,100)
            output = output.assign(Sm147 = events)  
        if element == "Sm148":
            Sm148 = data['[148Sm]+ mass 147.914']
            events = find_events(Sm148,100)
            output = output.assign(Sm148 = events)   
        if element == "Sm149":
            Sm149 = data['[149Sm]+ mass 148.917']
            events = find_events(Sm149,100)
            output = output.assign(Sm149 = events)             
        if element == "Sm150":
            Sm150 = data['[150Sm]+ mass 149.917']
            events = find_events(Sm150,100)
            output = output.assign(Sm150 = events)           
        if element == "Sm152":
            Sm152 = data['[152Sm]+ mass 151.919']
            events = find_events(Sm152,100)
            output = output.assign(Sm152 = events)
        if element == "Sm154":
            Sm154 = data['[154Sm]+ mass 153.922']
            events = find_events(Sm154,100)
            output = output.assign(Sm154 = events)
            
            
        if element == "Eu151":
            Eu151 = data['[151Eu]+ mass 150.919']
            events = find_events(Eu151,100)
            output = output.assign(Eu151 = events)
        if element == "Eu153":
            Eu153 = data['[153Eu]+ mass 152.921']
            events = find_events(Eu153,100)
            output = output.assign(Eu153 = events)
            
            
        if element == "Gd152":
            Gd152 = data['[152Gd]+ mass 151.919']
            events = find_events(Gd152,100)
            output = output.assign(Gd152 = events)
        if element == "Gd154":
            Gd154 = data['[154Gd]+ mass 153.92']
            events = find_events(Gd154,100)
            output = output.assign(Gd154 = events)  
        if element == "Gd155":
            Gd155 = data['[155Gd]+ mass 154.922']
            events = find_events(Gd155,100)
            output = output.assign(Gd155 = events)   
        if element == "Gd156":
            Gd156 = data['[156Gd]+ mass 155.922']
            events = find_events(Gd156,100)
            output = output.assign(Gd156 = events)             
        if element == "Gd157":
            Gd157 = data['[157Gd]+ mass 156.923']
            events = find_events(Gd157,100)
            output = output.assign(Gd157 = events)           
        if element == "Gd158":
            Gd158 = data['[158Gd]+ mass 157.924']
            events = find_events(Gd158,100)
            output = output.assign(Gd158 = events)
        if element == "Gd160":
            Gd160 = data['[160Gd]+ mass 159.927']
            events = find_events(Gd160,100)
            output = output.assign(Gd160 = events)
            
            
        if element == "Dy156":
            Dy156 = data['[156Dy]+ mass 155.924']
            events = find_events(Dy156,100)
            output = output.assign(Dy156 = events)
        if element == "Dy158":
            Dy158 = data['[158Dy]+ mass 157.924']
            events = find_events(Dy158,100)
            output = output.assign(Dy158 = events)  
        if element == "Dy160":
            Dy160 = data['[160Dy]+ mass 159.925']
            events = find_events(Dy160,100)
            output = output.assign(Dy160 = events)   
        if element == "Dy161":
            Dy161 = data['[161Dy]+ mass 160.926']
            events = find_events(Dy161,100)
            output = output.assign(Dy161 = events)             
        if element == "Dy162":
            Dy162 = data['[162Dy]+ mass 161.926']
            events = find_events(Dy162,100)
            output = output.assign(Dy162 = events)           
        if element == "Dy163":
            Dy162 = data['[163Dy]+ mass 162.928']
            events = find_events(Dy162,100)
            output = output.assign(Dy162 = events)
        if element == "Dy164":
            Dy164 = data['[164Dy]+ mass 163.929']
            oevents = find_events(Dy164,100)
            output = output.assign(Dy164 = events)
            
            
        if element == "Tb159":
            Tb159 = data['[159Tb]+ mass 158.925']
            events = find_events(Tb159,100)
            output = output.assign(Tb159 = events)
            
            
        if element == "Er162":
            Er162 = data['[162Er]+ mass 161.928']
            events = find_events(Er162,100)
            output = output.assign(Er162 = events)  
        if element == "Er164":
            Er164 = data['[164Er]+ mass 163.929']
            events = find_events(Er164,100)
            output = output.assign(Er164 = events)   
        if element == "Er166":
            Er166 = data['[166Er]+ mass 165.93']
            events = find_events(Er166,100)
            output = output.assign(Er166 = events)             
        if element == "Er167":
            Er167 = data['[167Er]+ mass 166.932']
            events = find_events(Er167,100)
            output = output.assign(Er167 = events)           
        if element == "Er168":
            Er168 = data['[168Er]+ mass 167.932']
            events = find_events(Er168,100)
            output = output.assign(Er168 = events)
        if element == "Er170":
            Er170 = data['[170Er]+ mass 169.935']
            events = find_events(Er170,100)
            output = output.assign(Er170 = events)
            
            
        if element == "Ho165":
            Ho165 = data['[165Ho]+ mass 164.93']
            events = find_events(Ho165,100)
            output = output.assign(Ho165 = events)
            
            
        if element == "Yb168":
            Yb168 = data['[168Yb]+ mass 167.933']
            events = find_events(Yb168,100)
            output = output.assign(Yb168 = events)  
        if element == "Yb170":
            Yb170 = data['[170Yb]+ mass 169.934']
            events = find_events(Yb170,100)
            output = output.assign(Yb170 = events)   
        if element == "Yb171":
            Yb171 = data['[171Yb]+ mass 170.936']
            events = find_events(Yb171,100)
            output = output.assign(Yb171 = events)             
        if element == "Yb172":
            Yb172 = data['[172Yb]+ mass 171.936']
            events = find_events(Yb172,100)
            output = output.assign(Yb172 = events)           
        if element == "Yb173":
            Yb173 = data['[173Yb]+ mass 172.938']
            events = find_events(Yb173,100)
            output = output.assign(Yb173 = events)
        if element == "Yb174":
            Yb174 = data['[174Yb]+ mass 173.938']
            events = find_events(Yb174,100)
            output = output.assign(Yb174 = events)    
        if element == "Yb176":
            Yb176 = data['[176Yb]+ mass 175.942']
            events = find_events(Yb176,100)
            output = output.assign(Yb176 = events)
            
            
        if element == "Tm169":
            Tm169 = data['[169Tm]+ mass 168.934']
            events = find_events(Tm169,100)
            output = output.assign(Tm169 = events)  
            
            
        if element == "Hf174":
            Hf174 = data['[174Hf]+ mass 173.939']
            events = find_events(Hf174,100)
            output = output.assign(Hf174 = events)    
        if element == "Hf176":
            Hf176 = data['[176Hf]+ mass 175.941']
            events = find_events(Hf176,100)
            output = output.assign(Hf176 = events)   
        if element == "Hf177":
            Hf177 = data['[177Hf]+ mass 176.943']
            events = find_events(Hf177,100)
            output = output.assign(Hf177 = events)             
        if element == "Hf178":
            Hf178 = data['[178Hf]+ mass 177.943']
            events = find_events(Hf178,100)
            output = output.assign(Hf178 = events)           
        if element == "Hf179":
            Hf179 = data['[179Hf]+ mass 178.945']
            events = find_events(Hf179,100)
            output = output.assign(Hf179 = events)
        if element == "Hf180":
            Hf180 = data['[180Hf]+ mass 179.946']
            events = find_events(Hf180,100)
            output = output.assign(Hf180 = events)
            
            
        if element == "Lu175":
            Lu175 = data['[175Lu]+ mass 174.94']
            events = find_events(Lu175,100)
            output = output.assign(Lu175 = events)
        if element == "Lu176":
            Lu176 = data['[176Lu]+ mass 175.942']
            events = find_events(Lu176,100)
            output = output.assign(Lu176 = events) 
            
            
        if element == "W180":
            W180 = data['[180W]+ mass 179.946']
            events = find_events(W180,100)
            output = output.assign(W180 = events)            
        if element == "W182":
            W182 = data['[182W]+ mass 181.948']
            events = find_events(W182,100)
            output = output.assign(W182 = events)           
        if element == "W183":
            W183 = data['[183W]+ mass 182.95']
            events = find_events(W183,100)
            output = output.assign(W183 = events)
        if element == "W184":
            W184 = data['[184W]+ mass 183.95']
            events = find_events(W184,100)
            output = output.assign(W184 = events)    
        if element == "W186":
            W186 = data['[186W]+ mass 185.954']
            events = find_events(W186,100)
            output = output.assign(W186 = events)
            
            
        if element == "Ta180":
            Ta180 = data['[180Ta]+ mass 179.947']
            events = find_events(Ta180,100)
            output = output.assign(Ta180 = events)  
        if element == "Ta181":
            Ta181 = data['[181Ta]+ mass 180.947']
            events = find_events(Ta181,100)
            output = output.assign(Ta181 = events) 
            
            
        if element == "Os184":
            Os184 = data['[184Os]+ mass 183.952']
            events = find_events(Os184,100)
            output = output.assign(Os184 = events)    
        if element == "Os186":
            Os186 = data['[186Os]+ mass 185.953']
            events = find_events(Os186,100)
            output = output.assign(Os186 = events)              
        if element == "Os187":
            Os187 = data['[187Os]+ mass 186.955']
            events = find_events(Os187,100)
            output = output.assign(Os187 = events)            
        if element == "Os188":
            Os188 = data['[188Os]+ mass 187.955']
            events = find_events(Os188,100)
            output = output.assign(Os188 = events) 
        if element == "Os189":
            Os189 = data['[189Os]+ mass 188.958']
            events = find_events(Os189,100)
            output = output.assign(Os189 = events) 
        if element == "Os190":
            Os190 = data['[190Os]+ mass 189.958']
            events = find_events(Os190,100)
            output = output.assign(Os190 = events)   
        if element == "Os192":
            Os192 = data['[192Os]+ mass 191.961']
            events = find_events(Os192,100)
            output = output.assign(Os192 = events)  
            
            
        if element == "Re185":
            Re185 = data['[185Re]+ mass 184.952']
            events = find_events(Re185,100)
            output = output.assign(Re185 = events)   
        if element == "Re187":
            Re187 = data['[187Re]+ mass 186.955']
            events = find_events(Re187,100)
            output = output.assign(Re187 = events)    
            
            
        if element == "Pt190":
            Pt190 = data['[190Pt]+ mass 189.959']
            events = find_events(Pt190,100)
            output = output.assign(Pt190 = events)   
        if element == "Pt192":
            Pt192 = data['[192Pt]+ mass 191.96']
            events = find_events(Pt192,100)
            output = output.assign(Pt192 = events)               
        if element == "Pt194":
            Pt194 = data['[194Pt]+ mass 193.962']
            events = find_events(Pt194,100)
            output = output.assign(Pt194 = events)             
        if element == "Pt195":
            Pt195 = data['[195Pt]+ mass 194.964']
            events = find_events(Pt195,100)
            output = output.assign(Pt195 = events)
        if element == "Pt196":
            Pt196 = data['[196Pt]+ mass 195.964']
            events = find_events(Pt196,100)
            output = output.assign(Pt196 = events)      
        if element == "Pt198":
            Pt198 = data['[198Pt]+ mass 197.967']
            events = find_events(Pt198,100)
            output = output.assign(Pt198 = events)  
            
            
        if element == "Ir191":
            Ir191 = data['[191Ir]+ mass 190.96']
            events = find_events(Ir191,100)
            output = output.assign(Ir191 = events)  
        if element == "Ir193":
            Ir193 = data['[193Ir]+ mass 192.962']
            events = find_events(Ir193,100)
            output = output.assign(Ir193 = events)
            
            
        if element == "Hg196":
            Hg196 = data['[196Hg]+ mass 195.965']
            events = find_events(Hg196,100)
            output = output.assign(Hg196 = events)   
        if element == "Hg198":
            Hg198 = data['[198Hg]+ mass 197.966']
            events = find_events(Hg198,100)
            output = output.assign(Hg198 = events)             
        if element == "Hg199":
            Hg199 = data['[199Hg]+ mass 198.968']
            events = find_events(Hg199,100)
            output = output.assign(Hg199 = events)           
        if element == "Hg200":
            Hg200 = data['[200Hg]+ mass 199.968']
            events = find_events(Hg200,100)
            output = output.assign(Hg200 = events)
        if element == "Hg201":
            Hg201 = data['[201Hg]+ mass 200.97']
            events = find_events(Hg201,100)
            output = output.assign(Hg201 = events)
        if element == "Hg202":
            Hg202 = data['[202Hg]+ mass 201.97']
            events = find_events(Hg202,100)
            output = output.assign(Hg202 = events)  
        if element == "Hg204":
            Hg204 = data['[204Hg]+ mass 203.973']
            events = find_events(Hg204,100)
            output = output.assign(Hg204 = events)
            
            
        if element == "Au197":
            Au197 = data['[197Au]+ mass 196.966']
            events = find_events(Au197,100)
            output = output.assign(Au197 = events)  
            
            
        if element == "Tl203":
            Tl203 = data['[203Tl]+ mass 202.972']
            events = find_events(Tl203,100)
            output = output.assign(Tl203 = events)             
        if element == "Tl205":
            Tl205 = data['[205Tl]+ mass 204.974']
            events = find_events(Tl205,100)
            output = output.assign(Tl205 = events)   
            
            
        if element == "Bi209":
            Bi209 = data['[209Bi]+ mass 208.98']
            events = find_events(Bi209,100)
            output = output.assign(Bi209 = events)
            
            
        if element == "Th232":
            Th232 = data['[232Th]+ mass 232.038']
            events = find_events(Th232,100)
            output = output.assign(Th232 = events)
            
        if element == "ThO248":
            ThO248 = data['ThO+ mass 248.032']
            events = find_events(ThO248,100)
            output = output.assign(ThO248 = events)  
            
        if element == "Background":
            Background = data['[220Bkg]+ mass 220']
            events = find_events(Background,100)
            output = output.assign(Background = events)
            
            
        if element == "All":
            output = data
            
        output = output.dropna()
    
    # Plotting
    
    number = 0   
    
    for element in elements:
        number = number+1
        fig = plt.figure(number,figsize =(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_title(str(element))
        ax.set_xlabel("Intensity (cps)")
        ax.set_ylabel("Frequency")
        ax.hist(output[str(element)]),
                linewidth = 0.5,
                edgecolor = 'white',
                bins=20,
               )
        plt.savefig(element)
    return output




