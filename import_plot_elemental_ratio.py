def import_plot_elemental_ratio(waveforms,element_numerator,element_denominator):
    """Imports data exported from the TofPilot software of TofWerk2R
    
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
    
    #numerator block
        
    if element_numerator == "Al27":
        numerator = data['[27Al]+ mass 26.981'] # 100%
        output = output.assign(Al27 = numerator)


    if element_numerator == "As75":
        numerator = data['[75As]+ mass 74.9211'] # 100%
        output = output.assign(As75 = numerator)


    if element_numerator == "Ba130":
        numerator = data['[130Ba]+ mass 129.906'] # 0.106%
        output = output.assign(Ba130 = numerator)
    if element_numerator == "Ba132":
        numerator = data['[132Ba]+ mass 131.905'] # 0.101%
        output = output.assign(Ba132 = numerator)
    if element_numerator == "Ba134":
        numerator = data['[134Ba]+ mass 133.904'] # 2.417%
        output = output.assign(Ba134 = numerator)
    if element_numerator == "Ba135":
        numerator = data['[135Ba]+ mass 134.905'] # 6.592%
        output = output.assign(Ba135 = numerator)
    if element_numerator == "Ba136":
        numerator = data['[136Ba]+ mass 135.904'] # 7.854%
        output = output.assign(Ba136 = numerator)
    if element_numerator == "Ba137":
        numerator = data['[137Ba]+ mass 136.905'] # 11.232%
        output = output.assign(Ba137 = numerator)
    if element_numerator == "Ba138":
        numerator = data['[138Ba]+ mass 137.905'] # 71.698%
        output = output.assign(Ba138 = numerator)

    if element_numerator == "Ba137_dc":
        numerator = data['[137Ba]++ mass 68.4524']
        output = output.assign(Ba137_dc = numerator)
    if element_numerator == "Ba138_dc":
        numerator = data['[138Ba]++ mass 68.9521']
        output = output.assign(Ba138_dc = numerator)    


    if element_numerator == "Pb204":
        numerator = data['[204Pb]+ mass 203.973']# 1.4%
        output = output.assign(Pb204 = numerator)
    if element_numerator == "Pb206":
        numerator = data['[206Pb]+ mass 205.974']# 24.1%
        output = output.assign(Pb206 = numerator)
    if element_numerator == "Pb207":
        numerator = data['[207Pb]+ mass 206.975']# 22.1%
        output = output.assign(Pb207 = numerator)
    if element_numerator == "Pb208":
        numerator = data['[208Pb]+ mass 207.976']# 52.4%
        output = output.assign(Pb208 = numerator)


    if element_numerator == "Cd106":
        numerator = data['[106Cd]+ mass 105.906'] # 1.25%
        output = output.assign(Cd106 = numerator)
    if element_numerator == "Cd108":
        numerator = data['[108Cd]+ mass 107.904'] # 0.89%
        output = output.assign(Cd108 = numerator)
    if element_numerator == "Cd110":
        numerator = data['[110Cd]+ mass 109.902'] # 12.49%
        output = output.assign(Cd110 = numerator)
    if element_numerator == "Cd111":
        numerator = data['[111Cd]+ mass 110.904'] # 12.80%
        output = output.assign(Cd111 = numerator)
    if element_numerator == "Cd112":
        numerator = data['[112Cd]+ mass 111.902'] # 24.13%
        output = output.assign(Cd112 = numerator)
    if element_numerator == "Cd113":
        numerator = data['[113Cd]+ mass 112.904'] # 12.22%
        output = output.assign(Cd113 = numerator)
    if element_numerator == "Cd114":
        numerator = data['[114Cd]+ mass 113.903'] # 28.73%
        output = output.assign(Cd114 = numerator)
    if element_numerator == "Cd116":
        numerator = data['[116Cd]+ mass 115.904'] # 7.49%
        output = output.assign(Cd116 = numerator)


    if element_numerator == "Ca40":
        numerator = data['[40Ca]+ mass 39.962'] # 96.941%
        output = output.assign(Ca40 = numerator)
    if element_numerator == "Ca42":
        numerator = data['[42Ca]+ mass 41.9581'] # 0.647%
        output = output.assign(Ca42 = numerator)
    if element_numerator == "Ca43":
        numerator = data['[43Ca]+ mass 42.9582'] # 0.135%
        output = output.assign(Ca43 = numerator)
    if element_numerator == "Ca44":
        numerator = data['[44Ca]+ mass 43.9549'] # 2.086%
        output = output.assign(Ca44 = numerator)
    if element_numerator == "Ca46":
        numerator = data['[46Ca]+ mass 45.9531'] # 0.004%
        output = output.assign(Ca46 = numerator)
    if element_numerator == "Ca48":
        numerator = data['[48Ca]+ mass 47.952'] # 0.187%
        output = output.assign(Ca48 = numerator)


    if element_numerator == "Cr50":
        numerator = data['[50Cr]+ mass 49.9455']# 4.345%
        output = output.assign(Cr50 = numerator)
    if element_numerator == "Cr52":
        numerator = data['[52Cr]+ mass 51.94']# 83.789%
        output = output.assign(Cr52 = numerator)
    if element_numerator == "Cr53":
        numerator = data['[53Cr]+ mass 52.9401']# 9.501%
        output = output.assign(Cr53 = numerator)
    if element_numerator == "Cr54":
        numerator = data['[54Cr]+ mass 53.9383']# 2.365%
        output = output.assign(Cr54 = numerator)


    if element_numerator == "Cu63":
        numerator = data['[63Cu]+ mass 62.9291']# 69.17%
        output = output.assign(Cu63 = numerator)
    if element_numerator == "Cu65":
        numerator = data['[65Cu]+ mass 64.9272']# 30.83%
        output = output.assign(Cu65 = numerator)


    if element_numerator == "Fe54":
        numerator = data['[54Fe]+ mass 53.9391']# 5.845%
        output = output.assign(Fe54 = numerator)
    if element_numerator == "Fe56":
        numerator = data['[56Fe]+ mass 55.9344']# 91.754%
        output = output.assign(Fe56 = numerator)
    if element_numerator == "Fe57":
        numerator = data['[57Fe]+ mass 56.9348']# 2.119%
        output = output.assign(Fe57 = numerator)
    if element_numerator == "Fe58":
        numerator = data['[58Fe]+ mass 57.9327']# 0.282%
        output = output.assign(Fe58 = numerator)


    if element_numerator == "Mg24":
        numerator = data['[24Mg]+ mass 23.9845']# 78.99%
        output = output.assign(Mg24 = numerator)
    if element_numerator == "Mg25":
        numerator = data['[25Mg]+ mass 24.9853']# 10.00%
        output = output.assign(Mg25 = numerator)
    if element_numerator == "Mg26":
        numerator = data['[26Mg]+ mass 25.982']# 11.01%
        output = output.assign(Mg26 = numerator)


    if element_numerator == "Mn55":
        numerator = data['[55Mn]+ mass 54.9375']# 100%
        output = output.assign(Mn55 = numerator)


    if element_numerator == "Mo92":
        numerator = data['[92Mo]+ mass 91.9063']# 14.84%
        output = output.assign(Mo92 = numerator)
    if element_numerator == "Mo94":
        numerator = data['[94Mo]+ mass 93.9045']# 9.25%
        output = output.assign(Mo94 = numerator)
    if element_numerator == "Mo95":
        numerator = data['[95Mo]+ mass 94.9053']# 15.92%
        output = output.assign(Mo95 = numerator)   
    if element_numerator == "Mo96":
        numerator = data['[96Mo]+ mass 95.9041']# 16.68%
        output = output.assign(Mo96 = numerator)
    if element_numerator == "Mo97":
        numerator = data['[97Mo]+ mass 96.9055']# 9.55%
        output = output.assign(Mo97 = numerator)
    if element_numerator == "Mo98":
        numerator = data['[98Mo]+ mass 97.9049']# 24.13%
        output = output.assign(Mo98 = numerator)    
    if element_numerator == "Mo100":
        numerator = data['[100Mo]+ mass 99.9069']# 9.63%
        output = output.assign(Mo100 = numerator)


    if element_numerator == "Ni58":
        numerator = data['[58Ni]+ mass 57.9348']# 68.0769%
        output = output.assign(Ni58 = numerator)   
    if element_numerator == "Ni60":
        numerator = data['[60Ni]+ mass 59.9302']# 26.2231%
        output = output.assign(Ni60 = numerator)
    if element_numerator == "Ni61":
        numerator = data['[61Ni]+ mass 60.9305']# 1.1399%
        output = output.assign(Ni61 = numerator)
    if element_numerator == "Ni62":
        numerator = data['[62Ni]+ mass 61.9278']# 3.6345%
        output = output.assign(Ni62 = numerator)    
    if element_numerator == "Ni64":
        numerator = data['[64Ni]+ mass 63.9274']# 0.9256%
        output = output.assign(Ni64 = numerator)   


    if element_numerator == "K39":
        numerator = data['[39K]+ mass 38.9632']# 93.2581%
        output = output.assign(K39 = numerator)    
    if element_numerator == "K41":
        numerator = data['[41K]+ mass 40.9613']# 	6.7302%
        output = output.assign(K41 = numerator)     


    if element_numerator == "Na23":
        numerator = data['[23Na]+ mass 22.9892']# 100%
        output = output.assign(Na23 = numerator)


    if element_numerator == "Sr84":
        numerator = data['[84Sr]+ mass 83.9129']# 0.56%
        output = output.assign(Sr84 = numerator)
    if element_numerator == "Sr86":
        numerator = data['[86Sr]+ mass 85.9087']# 9.86%
        output = output.assign(Sr86 = numerator)
    if element_numerator == "Sr87":
        numerator = data['[87Sr]+ mass 86.9083']# 7.00%
        output = output.assign(Sr87 = numerator)    
    if element_numerator == "Sr88":
        numerator = data['[88Sr]+ mass 87.9051']# 82.58%
        output = output.assign(Sr88 = numerator)  


    if element_numerator == "U234":
        numerator = data['[234U]+ mass 234.04']# 0.0055%
        output = output.assign(U234 = numerator)
    if element_numerator == "U235":
        numerator = data['[235U]+ mass 235.043']# 0.7200%
        output = output.assign(U235 = numerator)
    if element_numerator == "U238":
        numerator = data['[238U]+ mass 238.05']# 99.2745%
        output = output.assign(U238 = numerator)

    if element_numerator == "UO254":
        numerator = data['UO+ mass 254.045']
        output = output.assign(UO254 = numerator)    


    if element_numerator == "V50":
        numerator = data['[50V]+ mass 49.9466']# 0.250%
        output = output.assign(V50 = numerator)    
    if element_numerator == "V51":
        numerator = data['[51V]+ mass 50.9434']# 99.750%
        output = output.assign(V51 = numerator)  


    if element_numerator == "Zn64":
        numerator = data['[64Zn]+ mass 63.9286']# 48.63%
        output = output.assign(Zn64 = numerator)   
    if element_numerator == "Zn66":
        numerator = data['[66Zn]+ mass 65.9255']# 27.90%
        output = output.assign(Zn66 = numerator)
    if element_numerator == "Zn67":
        numerator = data['[67Zn]+ mass 66.9266']# 4.10 %
        output = output.assign(Zn67 = numerator)
    if element_numerator == "Zn68":
        numerator = data['[68Zn]+ mass 67.9243']# 18.75%
        output = output.assign(Zn68 = numerator)    
    if element_numerator == "Zn70":
        numerator = data['[70Zn]+ mass 69.9248']# 0.62%
        output = output.assign(Zn70 = numerator)  


    if element_numerator == "Be9":
        numerator = data['[9Be]+ mass 9.01163']
        output = output.assign(Be9 = numerator)   
    if element_numerator == "Be10":
        numerator = data['[10B]+ mass 10.0124']
        output = output.assign(Be10 = numerator)
    if element_numerator == "Be11":
        numerator = data['[11B]+ mass 11.0088']
        output = output.assign(Be11 = numerator)


    if element_numerator == "Li6":
        numerator = data['[6Li]+ mass 6.01546']
        output = output.assign(Li6 = numerator)    
    if element_numerator == "Li7":
        numerator = data['[7Li]+ mass 7.01546']
        output = output.assign(Li7 = numerator)    


    if element_numerator == "C12":
        numerator = data['[12C]+ mass 11.9994']
        output = output.assign(C12 = numerator)   
    if element_numerator == "C13":
        numerator = data['[13C]+ mass 13.0028']
        output = output.assign(C13 = numerator)
    if element_numerator == "CO2":
        numerator = data['CO2+ mass 43.9893']
        output = output.assign(CO2 = numerator)
    if element_numerator == "CO2H":
        numerator = data['CHO2+ mass 44.9971']
        output = output.assign(CO2H = numerator)    
    if element_numerator == "C2H226":
        numerator = data['C2H2+ mass 26.0151']
        output = output.assign(C2H226 = numerator)
    if element_numerator == "C2H327":
        numerator = data['C2H3+ mass 27.0229']
        output = output.assign(C2H327 = numerator) 


    if element_numerator == "N14":
        numerator = data['[14N]+ mass 14.0025']
        output = output.assign(N14 = numerator)   
    if element_numerator == "N15":
        numerator = data['[15N]+ mass 14.9996']
        output = output.assign(N15 = numerator)
    if element_numerator == "N2":
        numerator = data['N2+ mass 28.0056']
        output = output.assign(N2 = numerator)
    if element_numerator == "N2H29":
        numerator = data['HN2+ mass 29.0134']
        output = output.assign(N2H29 = numerator)    
    if element_numerator == "NO30":
        numerator = data['NO+ mass 29.9974']
        output = output.assign(NO30 = numerator)
    if element_numerator == "NO31":
        numerator = data['[15N]O+ mass 30.9945']
        output = output.assign(NO31 = numerator)  
    if element_numerator == "NO246":
        numerator = data['NO2+ mass 45.9924']
        output = output.assign(NO246 = numerator)

    if element_numerator == "O16":
        numerator = data['[16O]+ mass 15.9944']
        output = output.assign(O16 = numerator)   
    if element_numerator == "O18":
        numerator = data['[18O]+ mass 17.9986']
        output = output.assign(O18 = numerator)
    if element_numerator == "OH17":
        numerator = data['OH+ mass 17.0022']
        output = output.assign(OH17 = numerator)
    if element_numerator == "H2O18":
        numerator = data['H2O+ mass 18.01']
        output = output.assign(H2O18 = numerator)    
    if element_numerator == "H3O19":
        numerator = data['H3O+ mass 19.0178']
        output = output.assign(H3O19 = numerator)
    if element_numerator == "O232":
        numerator = data['O2+ mass 31.9893']
        output = output.assign(O232 = numerator)  
    if element_numerator == "O2H33":
        numerator = data['O2H+ mass 32.9971']
        output = output.assign(O2H33 = numerator)   
    if element_numerator == "OH17":
        numerator = data['OH+ mass 17.0022']
        output = output.assign(OH17 = numerator)  
    if element_numerator == "O234":
        numerator = data['O[18O]+ mass 33.9935']
        output = output.assign(O234 = numerator)   


    if element_numerator == "Si28":
        numerator = data['[28Si]+ mass 27.9764']
        output = output.assign(Si28 = numerator)   
    if element_numerator == "Si29":
        numerator = data['[29Si]+ mass 28.976']
        output = output.assign(Si29 = numerator)  
    if element_numerator == "Si30":
        numerator = data['[30Si]+ mass 29.9732']
        output = output.assign(Si30 = numerator)


    if element_numerator == "P31":
        numerator = data['[31P]+ mass 30.9732']
        output = output.assign(P31 = numerator)


    if element_numerator == "S32":
        numerator = data['[32S]+ mass 31.9715']
        output = output.assign(S32 = numerator)  
    if element_numerator == "S33":
        numerator = data['[33S]+ mass 32.9709']
        output = output.assign(S33 = numerator)   
    if element_numerator == "S34":
        numerator = data['[34S]+ mass 33.9673']
        output = output.assign(S34 = numerator)  
    if element_numerator == "S36":
        numerator = data['[36S]+ mass 35.9665']
        output = output.assign(S36 = numerator)

    if element_numerator == "Cl35":
        numerator = data['[35Cl]+ mass 34.9683']
        output = output.assign(Cl35 = numerator)
    if element_numerator == "Cl37":
        numerator = data['[37Cl]+ mass 36.9654']
        output = output.assign(Cl37 = numerator)  
    if element_numerator == "HCl36":
        numerator = data['HCl+ mass 35.9761']
        output = output.assign(HCl36 = numerator)   
    if element_numerator == "ClO51":
        numerator = data['ClO+ mass 50.9632']
        output = output.assign(ClO51 = numerator)  
    if element_numerator == "ClO53":
        numerator = data['[37Cl]O+ mass 52.9603']
        output = output.assign(ClO53 = numerator)    


    if element_numerator == "Sc45":
        numerator = data['[45Sc]+ mass 44.9554']
        output = output.assign(Sc45 = numerator)


    if element_numerator == "Ti46":
        numerator = data['[46Ti]+ mass 45.9521']
        output = output.assign(Ti46 = numerator)
    if element_numerator == "Ti47":
        numerator = data['[47Ti]+ mass 46.9512']
        output = output.assign(Ti47 = numerator)  
    if element_numerator == "Ti48":
        numerator = data['[48Ti]+ mass 47.9474']
        output = output.assign(Ti48 = numerator)   
    if element_numerator == "Ti49":
        numerator = data['[49Ti]+ mass 48.9473']
        output = output.assign(Ti49 = numerator)  
    if element_numerator == "Ti50":
        numerator = data['[50Ti]+ mass 49.9442']
        output = output.assign(Ti50 = numerator) 


    if element_numerator == "Ga69":
        numerator = data['[69Ga]+ mass 68.925']
        output = output.assign(Ga69 = numerator)  
    if element_numerator == "Ga71":
        numerator = data['[71Ga]+ mass 70.9241']
        output = output.assign(Ga71 = numerator)


    if element_numerator == "Ar36":
        numerator = data['[36Ar]+ mass 35.967']
        output = output.assign(Ar36 = numerator)
    if element_numerator == "Ar38":
        numerator = data['[38Ar]+ mass 37.9622']
        output = output.assign(Ar38 = numerator)  
    if element_numerator == "Ar40":
        numerator = data['[40Ar]+ mass 39.9618']
        output = output.assign(Ar40 = numerator)   
    if element_numerator == "ArH37":
        numerator = data['[36Ar]H+ mass 36.9748']
        output = output.assign(ArH37 = numerator)  
    if element_numerator == "ArH39":
        numerator = data['[38Ar]H+ mass 38.97']
        output = output.assign(ArH39 = numerator)    
    if element_numerator == "ArH41":
        numerator = data['ArH+ mass 40.9697']
        output = output.assign(ArH41 = numerator)
    if element_numerator == "ArH242":
        numerator = data['ArH2+ mass 41.9775']
        output = output.assign(ArH242 = numerator)  
    if element_numerator == "ArO52":
        numerator = data['[36Ar]O+ mass 51.9619']
        output = output.assign(ArO52 = numerator)   
    if element_numerator == "ArN54":
        numerator = data['ArN+ mass 53.9649']
        output = output.assign(ArN54 = numerator)  
    if element_numerator == "ArO56":
        numerator = data['ArO+ mass 55.9567']
        output = output.assign(ArO56 = numerator)    
    if element_numerator == "ArOH57":
        numerator = data['ArOH+ mass 56.9646']
        output = output.assign(ArOH57 = numerator)
    if element_numerator == "Ar280":
        numerator = data['Ar2+ mass 79.9242']
        output = output.assign(Ar280 = numerator)  
    if element_numerator == "ArCl":
        numerator = data['ArCl+ mass 74.9307']
        output = output.assign(ArCl = numerator)   
    if element_numerator == "Ar276":
        numerator = data['Ar[36Ar]+ mass 75.9294']
        output = output.assign(Ar276 = numerator)  
    if element_numerator == "ArCl77":
        numerator = data['Ar[37Cl]+ mass 76.9277']
        output = output.assign(ArCl77 = numerator)    
    if element_numerator == "Ar278":
        numerator = data['Ar[38Ar]+ mass 77.9246']
        output = output.assign(Ar278 = numerator)    


    if element_numerator == "Ge70":
        numerator = data['[70Ge]+ mass 69.9237']
        output = output.assign(Ge70 = numerator)  
    if element_numerator == "Ge72":
        numerator = data['[72Ge]+ mass 71.9215']
        output = output.assign(Ge72 = numerator)   
    if element_numerator == "Ge73":
        numerator = data['[73Ge]+ mass 72.9229']
        output = output.assign(Ge73 = numerator)  
    if element_numerator == "Ge74":
        numerator = data['[74Ge]+ mass 73.9206']
        output = output.assign(Ge74 = numerator)    
    if element_numerator == "Ge76":
        numerator = data['[76Ge]+ mass 75.9209']
        output = output.assign(Ge76 = numerator)    


    if element_numerator == "Co59":
        numerator = data['[59Co]+ mass 58.9327']
        output = output.assign(Co59 = numerator)


    if element_numerator == "Se74":
        numerator = data['[74Se]+ mass 73.9219']
        output = output.assign(Se74 = numerator)
    if element_numerator == "Se76":
        numerator = data['[76Se]+ mass 75.9187']
        output = output.assign(Se76 = numerator)  
    if element_numerator == "Se77":
        numerator = data['[77Se]+ mass 76.9194']
        output = output.assign(Se77 = numerator)   
    if element_numerator == "Se78":
        numerator = data['[78Se]+ mass 77.9168']
        output = output.assign(Se78 = numerator)  
    if element_numerator == "Se80":
        numerator = data['[80Se]+ mass 79.916']
        output = output.assign(Se80 = numerator)    
    if element_numerator == "Se82":
        numerator = data['[82Se]+ mass 81.9162']
        output = output.assign(Se82 = numerator) 


    if element_numerator == "Kr78":
        numerator = data['[78Kr]+ mass 77.9198']
        output = output.assign(Kr78 = numerator)
    if element_numerator == "Kr80":
        numerator = data['[80Kr]+ mass 79.9158']
        output = output.assign(Kr80 = numerator)  
    if element_numerator == "Kr82":
        numerator = data['[82Kr]+ mass 81.9129']
        output = output.assign(Kr82 = numerator)   
    if element_numerator == "Kr83":
        numerator = data['[83Kr]+ mass 82.9136']
        output = output.assign(Kr83 = numerator)  
    if element_numerator == "Kr84":
        numerator = data['[84Kr]+ mass 83.911']
        output = output.assign(Kr84 = numerator)    
    if element_numerator == "Kr86":
        numerator = data['[86Kr]+ mass 85.9101']
        output = output.assign(Kr86 = numerator) 


    if element_numerator == "Br79":
        numerator = data['[79Br]+ mass 78.9178']
        output = output.assign(Br79 = numerator)    
    if element_numerator == "Br81":
        numerator = data['[81Br]+ mass 80.9157']
        output = output.assign(Br81 = numerator)


    if element_numerator == "Rb85":
        numerator = data['[85Rb]+ mass 84.9112']
        output = output.assign(Rb85 = numerator)    
    if element_numerator == "Rb87":
        numerator = data['[87Rb]+ mass 86.9086']
        output = output.assign(Rb87 = numerator)


    if element_numerator == "Y89":
        numerator = data['[89Y]+ mass 88.9053']
        output = output.assign(Y89 = numerator)


    if element_numerator == "Zr90":
        numerator = data['[90Zr]+ mass 89.9042']
        output = output.assign(Zr90 = numerator)  
    if element_numerator == "Zr91":
        numerator = data['[91Zr]+ mass 90.9051']
        output = output.assign(Zr91 = numerator)   
    if element_numerator == "Zr92":
        numerator = data['[92Zr]+ mass 91.9045']
        output = output.assign(Zr92 = numerator)  
    if element_numerator == "Zr94":
        numerator = data['[94Zr]+ mass 93.9058']
        output = output.assign(Zr94 = numerator)    
    if element_numerator == "Zr96":
        numerator = data['[96Zr]+ mass 95.9077']
        output = output.assign(Zr96 = numerator)


    if element_numerator == "Nb93":
        numerator = data['[93Nb]+ mass 92.9058']
        output = output.assign(Nb93 = numerator)


    if element_numerator == "Ru96":
        numerator = data['[96Ru]+ mass 95.9071']
        output = output.assign(Ru96 = numerator)  
    if element_numerator == "Ru98":
        numerator = data['[98Ru]+ mass 97.9047']
        output = output.assign(Ru98 = numerator)   
    if element_numerator == "Ru99":
        numerator = data['[99Ru]+ mass 98.9054']
        output = output.assign(Ru99 = numerator)  
    if element_numerator == "Ru100":
        numerator = data['[100Ru]+ mass 99.9037']
        output = output.assign(Ru100 = numerator)    
    if element_numerator == "Ru101":
        numerator = data['[101Ru]+ mass 100.905']
        output = output.assign(Ru101 = numerator)
    if element_numerator == "Ru102":
        numerator = data['[102Ru]+ mass 101.904']
        output = output.assign(Ru102 = numerator)    
    if element_numerator == "Ru104":
        numerator = data['[104Ru]+ mass 103.905']
        output = output.assign(Ru104 = numerator)


    if element_numerator == "Pd102":
        numerator = data['[102Pd]+ mass 101.905']
        output = output.assign(Pd102 = numerator)  
    if element_numerator == "Pd104":
        numerator = data['[104Pd]+ mass 103.903']
        output = output.assign(Pd104 = numerator)   
    if element_numerator == "Pd105":
        numerator = data['[105Pd]+ mass 104.905']
        output = output.assign(Pd105 = numerator)  
    if element_numerator == "Pd106":
        numerator = data['[106Pd]+ mass 105.903']
        output = output.assign(Pd106 = numerator)    
    if element_numerator == "Pd108":
        numerator = data['[108Pd]+ mass 107.903']
        output = output.assign(Pd108 = numerator)
    if element_numerator == "Pd110":
        numerator = data['[110Pd]+ mass 109.905']
        output = output.assign(Pd110 = numerator)


    if element_numerator == "Rh103":
        numerator = data['[103Rh]+ mass 102.905']
        output = output.assign(Rh103 = numerator)  


    if element_numerator == "Ag107":
        numerator = data['[107Ag]+ mass 106.905']
        output = output.assign(Ag107 = numerator)
    if element_numerator == "Ag109":
        numerator = data['[109Ag]+ mass 108.904']
        output = output.assign(Ag109 = numerator)  


    if element_numerator == "Sn112":
        numerator = data['[112Sn]+ mass 111.904']
        output = output.assign(Sn112 = numerator)   
    if element_numerator == "Sn114":
        numerator = data['[114Sn]+ mass 113.902']
        output = output.assign(Sn114 = numerator)  
    if element_numerator == "Sn115":
        numerator = data['[115Sn]+ mass 114.903']
        output = output.assign(Sn115 = numerator)    
    if element_numerator == "Sn116":
        numerator = data['[116Sn]+ mass 115.901']
        output = output.assign(Sn116 = numerator)
    if element_numerator == "Sn117":
        numerator = data['[117Sn]+ mass 116.902']
        output = output.assign(Sn117 = numerator)  
    if element_numerator == "Sn118":
        numerator = data['[118Sn]+ mass 117.901']
        output = output.assign(Sn118 = numerator)   
    if element_numerator == "Sn119":
        numerator = data['[119Sn]+ mass 118.903']
        output = output.assign(Sn119 = numerator)  
    if element_numerator == "Sn120":
        numerator = data['[120Sn]+ mass 119.902']
        output = output.assign(Sn120 = numerator)    
    if element_numerator == "Sn122":
        numerator = data['[122Sn]+ mass 121.903']
        output = output.assign(Sn122 = numerator)
    if element_numerator == "Sn124":
        numerator = data['[124Sn]+ mass 123.905']
        output = output.assign(Sn124 = numerator)  


    if element_numerator == "In113":
        numerator = data['[113In]+ mass 112.904']
        output = output.assign(In113 = numerator)
    if element_numerator == "In115":
        numerator = data['[115In]+ mass 114.903']
        output = output.assign(In115 = numerator)  


    if element_numerator == "Sb121":
        numerator = data['[121Sb]+ mass 120.903']
        output = output.assign(Sb121 = numerator)   
    if element_numerator == "Sb123":
        numerator = data['[123Sb]+ mass 122.904']
        output = output.assign(Sb123 = numerator)


    if element_numerator == "Te120":
        numerator = data['[120Te]+ mass 119.903']
        output = output.assign(Te120 = numerator)    
    if element_numerator == "Te122":
        numerator = data['[122Te]+ mass 121.902']
        output = output.assign(Te122 = numerator)
    if element_numerator == "Te123":
        numerator = data['[123Te]+ mass 122.904']
        output = output.assign(Te123 = numerator)  
    if element_numerator == "Te124":
        numerator = data['[124Te]+ mass 123.902']
        output = output.assign(Te124 = numerator)   
    if element_numerator == "Te125":
        numerator = data['[125Te]+ mass 124.904']
        output = output.assign(Te125 = numerator)  
    if element_numerator == "Te126":
        numerator = data['[126Te]+ mass 125.903']
        output = output.assign(Te126 = numerator)    
    if element_numerator == "Te128":
        numerator = data['[128Te]+ mass 127.904']
        output = output.assign(Te128 = numerator)
    if element_numerator == "Te130":
        numerator = data['[130Te]+ mass 129.906']
        output = output.assign(Te130 = numerator)


    if element_numerator == "Xe124":
        numerator = data['[124Xe]+ mass 123.905']
        output = output.assign(Xe124 = numerator)  
    if element_numerator == "Xe126":
        numerator = data['[126Xe]+ mass 125.904']
        output = output.assign(Xe126 = numerator)    
    if element_numerator == "Xe128":
        numerator = data['[128Xe]+ mass 127.903']
        output = output.assign(Xe128 = numerator)
    if element_numerator == "Xe129":
        numerator = data['[129Xe]+ mass 128.904']
        output = output.assign(Xe129 = numerator)  
    if element_numerator == "Xe130":
        numerator = data['[130Xe]+ mass 129.903']
        output = output.assign(Xe130 = numerator)   
    if element_numerator == "Xe131":
        numerator = data['[131Xe]+ mass 130.905']
        output = output.assign(Xe131 = numerator)  
    if element_numerator == "Xe132":
        numerator = data['[132Xe]+ mass 131.904']
        output = output.assign(Xe132 = numerator)    
    if element_numerator == "Xe134":
        numerator = data['[134Xe]+ mass 133.905']
        output = output.assign(Xe134 = numerator)
    if element_numerator == "Xe136":
        numerator = data['[136Xe]+ mass 135.907']
        output = output.assign(Xe136 = numerator)


    if element_numerator == "I127":
        numerator = data['[127I]+ mass 126.904']
        output = output.assign(I127 = numerator)  


    if element_numerator == "Cs133":
        numerator = data['[133Cs]+ mass 132.905']
        output = output.assign(Cs133 = numerator)   


    if element_numerator == "Ce136":
        numerator = data['[136Ce]+ mass 135.907']
        output = output.assign(Ce136 = numerator)
    if element_numerator == "Ce138":
        numerator = data['[138Ce]+ mass 137.905']
        output = output.assign(Ce138 = numerator)  
    if element_numerator == "Ce140":
        numerator = data['[140Ce]+ mass 139.905']
        output = output.assign(Ce140 = numerator)   
    if element_numerator == "Ce142":
        numerator = data['[142Ce]+ mass 141.909']
        output = output.assign(Ce142 = numerator)  

    if element_numerator == "CeO156":
        numerator = data['CeO+ mass 155.9']
        output = output.assign(CeO156 = numerator)  


    if element_numerator == "La138":
        numerator = data['[138La]+ mass 137.907']
        output = output.assign(La138 = numerator)
    if element_numerator == "La139":
        numerator = data['[139La]+ mass 138.906']
        output = output.assign(La139 = numerator)


    if element_numerator == "Pr141":
        numerator = data['[141Pr]+ mass 140.907']
        output = output.assign(Pr141 = numerator)


    if element_numerator == "Nd142":
        numerator = data['[142Nd]+ mass 141.907']
        output = output.assign(Nd142 = numerator)
    if element_numerator == "Nd143":
        numerator = data['[143Nd]+ mass 142.909']
        output = output.assign(Nd143 = numerator)  
    if element_numerator == "Nd144":
        numerator = data['[144Nd]+ mass 143.91']
        output = output.assign(Nd144 = numerator)   
    if element_numerator == "Nd145":
        numerator = data['[145Nd]+ mass 144.912']
        output = output.assign(Nd145 = numerator)             
    if element_numerator == "Nd146":
        numerator = data['[146Nd]+ mass 145.913']
        output = output.assign(Nd146 = numerator)           
    if element_numerator == "Nd148":
        numerator = data['[148Nd]+ mass 147.916']
        output = output.assign(Nd148 = numerator)
    if element_numerator == "Nd150":
        numerator = data['[150Nd]+ mass 149.92']
        output = output.assign(Nd150 = numerator)


    if element_numerator == "Sm144":
        numerator = data['[144Sm]+ mass 143.911']
        output = output.assign(Sm144 = numerator)
    if element_numerator == "Sm147":
        numerator = data['[147Sm]+ mass 146.914']
        output = output.assign(Sm147 = numerator)  
    if element_numerator == "Sm148":
        numerator = data['[148Sm]+ mass 147.914']
        output = output.assign(Sm148 = numerator)   
    if element_numerator == "Sm149":
        numerator = data['[149Sm]+ mass 148.917']
        output = output.assign(Sm149 = numerator)             
    if element_numerator == "Sm150":
        numerator = data['[150Sm]+ mass 149.917']
        output = output.assign(Sm150 = numerator)           
    if element_numerator == "Sm152":
        numerator = data['[152Sm]+ mass 151.919']
        output = output.assign(Sm152 = numerator)
    if element_numerator == "Sm154":
        numerator = data['[154Sm]+ mass 153.922']
        output = output.assign(Sm154 = numerator)


    if element_numerator == "Eu151":
        numerator = data['[151Eu]+ mass 150.919']
        output = output.assign(Eu151 = numerator)
    if element_numerator == "Eu153":
        numerator = data['[153Eu]+ mass 152.921']
        output = output.assign(Eu153 = numerator)


    if element_numerator == "Gd152":
        numerator = data['[152Gd]+ mass 151.919']
        output = output.assign(Gd152 = numerator)
    if element_numerator == "Gd154":
        numerator = data['[154Gd]+ mass 153.92']
        output = output.assign(Gd154 = numerator)  
    if element_numerator == "Gd155":
        numerator = data['[155Gd]+ mass 154.922']
        output = output.assign(Gd155 = numerator)   
    if element_numerator == "Gd156":
        numerator = data['[156Gd]+ mass 155.922']
        output = output.assign(Gd156 = numerator)             
    if element_numerator == "Gd157":
        numerator = data['[157Gd]+ mass 156.923']
        output = output.assign(Gd157 = numerator)           
    if element_numerator == "Gd158":
        numerator = data['[158Gd]+ mass 157.924']
        output = output.assign(Gd158 = numerator)
    if element_numerator == "Gd160":
        numerator = data['[160Gd]+ mass 159.927']
        output = output.assign(Gd160 = numerator)


    if element_numerator == "Dy156":
        numerator = data['[156Dy]+ mass 155.924']
        output = output.assign(Dy156 = numerator)
    if element_numerator == "Dy158":
        numerator = data['[158Dy]+ mass 157.924']
        output = output.assign(Dy158 = numerator)  
    if element_numerator == "Dy160":
        numerator = data['[160Dy]+ mass 159.925']
        output = output.assign(Dy160 = numerator)   
    if element_numerator == "Dy161":
        numerator = data['[161Dy]+ mass 160.926']
        output = output.assign(Dy161 = numerator)             
    if element_numerator == "Dy162":
        numerator = data['[162Dy]+ mass 161.926']
        output = output.assign(Dy162 = numerator)           
    if element_numerator == "Dy163":
        numerator = data['[163Dy]+ mass 162.928']
        output = output.assign(Dy163 = numerator)
    if element_numerator == "Dy164":
        numerator = data['[164Dy]+ mass 163.929']
        output = output.assign(Dy164 = numerator)


    if element_numerator == "Tb159":
        numerator = data['[159Tb]+ mass 158.925']
        output = output.assign(Tb159 = numerator)


    if element_numerator == "Er162":
        numerator = data['[162Er]+ mass 161.928']
        output = output.assign(Er162 = numerator)  
    if element_numerator == "Er164":
        numerator = data['[164Er]+ mass 163.929']
        output = output.assign(Er164 = numerator)   
    if element_numerator == "Er166":
        numerator = data['[166Er]+ mass 165.93']
        output = output.assign(Er166 = numerator)             
    if element_numerator == "Er167":
        numerator = data['[167Er]+ mass 166.932']
        output = output.assign(Er167 = numerator)           
    if element_numerator == "Er168":
        numerator = data['[168Er]+ mass 167.932']
        output = output.assign(Er168 = numerator)
    if element_numerator == "Er170":
        numerator = data['[170Er]+ mass 169.935']
        output = output.assign(Er170 = numerator)


    if element_numerator == "Ho165":
        numerator = data['[165Ho]+ mass 164.93']
        output = output.assign(Ho165 = numerator)


    if element_numerator == "Yb168":
        numerator = data['[168Yb]+ mass 167.933']
        output = output.assign(Yb168 = numerator)  
    if element_numerator == "Yb170":
        numerator = data['[170Yb]+ mass 169.934']
        output = output.assign(Yb170 = numerator)   
    if element_numerator == "Yb171":
        numerator = data['[171Yb]+ mass 170.936']
        output = output.assign(Yb171 = numerator)             
    if element_numerator == "Yb172":
        numerator = data['[172Yb]+ mass 171.936']
        output = output.assign(Yb172 = numerator)           
    if element_numerator == "Yb173":
        numerator = data['[173Yb]+ mass 172.938']
        output = output.assign(Yb173 = numerator)
    if element_numerator == "Yb174":
        numerator = data['[174Yb]+ mass 173.938']
        output = output.assign(Yb174 = numerator)    
    if element_numerator == "Yb176":
        numerator = data['[176Yb]+ mass 175.942']
        output = output.assign(Yb176 = numerator)


    if element_numerator == "Tm169":
        numerator = data['[169Tm]+ mass 168.934']
        output = output.assign(Tm169 = numerator)  


    if element_numerator == "Hf174":
        numerator = data['[174Hf]+ mass 173.939']
        output = output.assign(Hf174 = numerator)    
    if element_numerator == "Hf176":
        numerator = data['[176Hf]+ mass 175.941']
        output = output.assign(Hf176 = numerator)   
    if element_numerator == "Hf177":
        numerator = data['[177Hf]+ mass 176.943']
        output = output.assign(Hf177 = numerator)             
    if element_numerator == "Hf178":
        numerator = data['[178Hf]+ mass 177.943']
        output = output.assign(Hf178 = numerator)           
    if element_numerator == "Hf179":
        numerator = data['[179Hf]+ mass 178.945']
        output = output.assign(Hf179 = numerator)
    if element_numerator == "Hf180":
        numerator = data['[180Hf]+ mass 179.946']
        output = output.assign(Hf180 = numerator)


    if element_numerator == "Lu175":
        numerator = data['[175Lu]+ mass 174.94']
        output = output.assign(Lu175 = numerator)  
    if element_numerator == "Lu176":
        numerator = data['[176Lu]+ mass 175.942']
        output = output.assign(Lu176 = numerator)  


    if element_numerator == "W180":
        numerator = data['[180W]+ mass 179.946']
        output = output.assign(W180 = numerator)             
    if element_numerator == "W182":
        numerator = data['[182W]+ mass 181.948']
        output = output.assign(W182 = numerator)           
    if element_numerator == "W183":
        numerator = data['[183W]+ mass 182.95']
        output = output.assign(W183 = numerator)
    if element_numerator == "W184":
        numerator = data['[184W]+ mass 183.95']
        output = output.assign(W184 = numerator)    
    if element_numerator == "W186":
        numerator = data['[186W]+ mass 185.954']
        output = output.assign(W186 = numerator)


    if element_numerator == "Ta180":
        numerator = data['[180Ta]+ mass 179.947']
        output = output.assign(Ta180 = numerator)  
    if element_numerator == "Ta181":
        numerator = data['[181Ta]+ mass 180.947']
        output = output.assign(Ta181 = numerator) 


    if element_numerator == "Os184":
        numerator = data['[184Os]+ mass 183.952']
        output = output.assign(Os184 = numerator)   
    if element_numerator == "Os186":
        numerator = data['[186Os]+ mass 185.953']
        output = output.assign(Os186 = numerator)             
    if element_numerator == "Os187":
        numerator = data['[187Os]+ mass 186.955']
        output = output.assign(Os187 = numerator)           
    if element_numerator == "Os188":
        numerator = data['[188Os]+ mass 187.955']
        output = output.assign(Os188 = numerator)
    if element_numerator == "Os189":
        numerator = data['[189Os]+ mass 188.958']
        output = output.assign(Os189 = numerator)
    if element_numerator == "Os190":
        numerator = data['[190Os]+ mass 189.958']
        output = output.assign(Os190 = numerator)  
    if element_numerator == "Os192":
        numerator = data['[192Os]+ mass 191.961']
        output = output.assign(Os192 = numerator) 


    if element_numerator == "Re185":
        numerator = data['[185Re]+ mass 184.952']
        output = output.assign(Re185 = numerator) 
    if element_numerator == "Re187":
        numerator = data['[187Re]+ mass 186.955']
        output = output.assign(Re187 = numerator)  


    if element_numerator == "Pt190":
        numerator = data['[190Pt]+ mass 189.959']
        output = output.assign(Pt190 = numerator) 
    if element_numerator == "Pt192":
        numerator = data['[192Pt]+ mass 191.96']
        output = output.assign(Pt192 =numerator)             
    if element_numerator == "Pt194":
        numerator = data['[194Pt]+ mass 193.962']
        output = output.assign(Pt194 = numerator)           
    if element_numerator == "Pt195":
        numerator = data['[195Pt]+ mass 194.964']
        output = output.assign(Pt195 = numerator)
    if element_numerator == "Pt196":
        numerator = data['[196Pt]+ mass 195.964']
        output = output.assign(Pt196 = numerator)    
    if element_numerator == "Pt198":
        numerator = data['[198Pt]+ mass 197.967']
        output = output.assign(Pt198 = numerator)


    if element_numerator == "Ir191":
        numerator = data['[191Ir]+ mass 190.96']
        output = output.assign(Ir191 = numerator)  
    if element_numerator == "Ir193":
        numerator = data['[193Ir]+ mass 192.962']
        output = output.assign(Ir193 = numerator)


    if element_numerator == "Hg196":
        numerator = data['[196Hg]+ mass 195.965']
        output = output.assign(Hg196 = numerator)   
    if element_numerator == "Hg198":
        numerator = data['[198Hg]+ mass 197.966']
        output = output.assign(Hg198 = numerator)             
    if element_numerator == "Hg199":
        numerator = data['[199Hg]+ mass 198.968']
        output = output.assign(Hg199 = numerator)           
    if element_numerator == "Hg200":
        numerator = data['[200Hg]+ mass 199.968']
        output = output.assign(Hg200 = numerator)
    if element_numerator == "Hg201":
        numerator = data['[201Hg]+ mass 200.97']
        output = output.assign(Hg201 = numerator)
    if element_numerator == "Hg202":
        numerator = data['[202Hg]+ mass 201.97']
        output = output.assign(Hg202 = numerator)  
    if element_numerator == "Hg204":
        numerator = data['[204Hg]+ mass 203.973']
        output = output.assign(Hg204 = numerator)


    if element_numerator == "Au197":
        numerator = data['[197Au]+ mass 196.966']
        output = output.assign(Au197 = numerator)  


    if element_numerator == "Tl203":
        numerator = data['[203Tl]+ mass 202.972']
        output = output.assign(Tl203 = numerator)             
    if element_numerator == "Tl205":
        numerator = data['[205Tl]+ mass 204.974']
        output = output.assign(Tl205 = numerator)   


    if element_numerator == "Bi209":
        numerator = data['[209Bi]+ mass 208.98']
        output = output.assign(Bi209 = numerator)


    if element_numerator == "Th232":
        numerator = data['[232Th]+ mass 232.038']
        output = output.assign(Th232 = numerator)

    if element_numerator == "ThO248":
        numerator = data['ThO+ mass 248.032']
        output = output.assign(ThO248 = numerator)  

    if element_numerator == "Background":
        numerator = data['[220Bkg]+ mass 220']
        output = output.assign(Background = numerator)
        
        
    #denominator block
        
        
    if element_denominator == "Al27":
        denominator = data['[27Al]+ mass 26.981'] # 100%
        output = output.assign(Al27 = denominator)


    if element_denominator == "As75":
        denominator = data['[75As]+ mass 74.9211'] # 100%
        output = output.assign(As75 = denominator)


    if element_denominator == "Ba130":
        denominator = data['[130Ba]+ mass 129.906'] # 0.106%
        output = output.assign(Ba130 = denominator)
    if element_denominator == "Ba132":
        denominator = data['[132Ba]+ mass 131.905'] # 0.101%
        output = output.assign(Ba132 = denominator)
    if element_denominator == "Ba134":
        denominator = data['[134Ba]+ mass 133.904'] # 2.417%
        output = output.assign(Ba134 = denominator)
    if element_denominator == "Ba135":
        denominator = data['[135Ba]+ mass 134.905'] # 6.592%
        output = output.assign(Ba135 = denominator)
    if element_denominator == "Ba136":
        denominator = data['[136Ba]+ mass 135.904'] # 7.854%
        output = output.assign(Ba136 = denominator)
    if element_denominator == "Ba137":
        denominator = data['[137Ba]+ mass 136.905'] # 11.232%
        output = output.assign(Ba137 = denominator)
    if element_denominator == "Ba138":
        denominator = data['[138Ba]+ mass 137.905'] # 71.698%
        output = output.assign(Ba138 = denominator)

    if element_denominator == "Ba137_dc":
        denominator = data['[137Ba]++ mass 68.4524']
        output = output.assign(Ba137_dc = denominator)
    if element_denominator == "Ba138_dc":
        denominator = data['[138Ba]++ mass 68.9521']
        output = output.assign(Ba138_dc = denominator)    


    if element_denominator == "Pb204":
        denominator = data['[204Pb]+ mass 203.973']# 1.4%
        output = output.assign(Pb204 = denominator)
    if element_denominator == "Pb206":
        denominator = data['[206Pb]+ mass 205.974']# 24.1%
        output = output.assign(Pb206 = denominator)
    if element_denominator == "Pb207":
        denominator = data['[207Pb]+ mass 206.975']# 22.1%
        output = output.assign(Pb207 = denominator)
    if element_denominator == "Pb208":
        denominator = data['[208Pb]+ mass 207.976']# 52.4%
        output = output.assign(Pb208 = denominator)


    if element_denominator == "Cd106":
        denominator = data['[106Cd]+ mass 105.906'] # 1.25%
        output = output.assign(Cd106 = denominator)
    if element_denominator == "Cd108":
        denominator = data['[108Cd]+ mass 107.904'] # 0.89%
        output = output.assign(Cd108 = denominator)
    if element_denominator == "Cd110":
        denominator = data['[110Cd]+ mass 109.902'] # 12.49%
        output = output.assign(Cd110 = denominator)
    if element_denominator == "Cd111":
        denominator = data['[111Cd]+ mass 110.904'] # 12.80%
        output = output.assign(Cd111 = denominator)
    if element_denominator == "Cd112":
        denominator = data['[112Cd]+ mass 111.902'] # 24.13%
        output = output.assign(Cd112 = denominator)
    if element_denominator == "Cd113":
        denominator = data['[113Cd]+ mass 112.904'] # 12.22%
        output = output.assign(Cd113 = denominator)
    if element_denominator == "Cd114":
        denominator = data['[114Cd]+ mass 113.903'] # 28.73%
        output = output.assign(Cd114 = denominator)
    if element_denominator == "Cd116":
        denominator = data['[116Cd]+ mass 115.904'] # 7.49%
        output = output.assign(Cd116 = denominator)


    if element_denominator == "Ca40":
        denominator = data['[40Ca]+ mass 39.962'] # 96.941%
        output = output.assign(Ca40 = denominator)
    if element_denominator == "Ca42":
        denominator = data['[42Ca]+ mass 41.9581'] # 0.647%
        output = output.assign(Ca42 = denominator)
    if element_denominator == "Ca43":
        denominator = data['[43Ca]+ mass 42.9582'] # 0.135%
        output = output.assign(Ca43 = denominator)
    if element_denominator == "Ca44":
        denominator = data['[44Ca]+ mass 43.9549'] # 2.086%
        output = output.assign(Ca44 = denominator)
    if element_denominator == "Ca46":
        denominator = data['[46Ca]+ mass 45.9531'] # 0.004%
        output = output.assign(Ca46 = denominator)
    if element_denominator == "Ca48":
        denominator = data['[48Ca]+ mass 47.952'] # 0.187%
        output = output.assign(Ca48 = denominator)


    if element_denominator == "Cr50":
        denominator = data['[50Cr]+ mass 49.9455']# 4.345%
        output = output.assign(Cr50 = denominator)
    if element_denominator == "Cr52":
        denominator = data['[52Cr]+ mass 51.94']# 83.789%
        output = output.assign(Cr52 = denominator)
    if element_denominator == "Cr53":
        denominator = data['[53Cr]+ mass 52.9401']# 9.501%
        output = output.assign(Cr53 = denominator)
    if element_denominator == "Cr54":
        denominator = data['[54Cr]+ mass 53.9383']# 2.365%
        output = output.assign(Cr54 = denominator)


    if element_denominator == "Cu63":
        denominator = data['[63Cu]+ mass 62.9291']# 69.17%
        output = output.assign(Cu63 = denominator)
    if element_denominator == "Cu65":
        denominator = data['[65Cu]+ mass 64.9272']# 30.83%
        output = output.assign(Cu65 = denominator)


    if element_denominator == "Fe54":
        denominator = data['[54Fe]+ mass 53.9391']# 5.845%
        output = output.assign(Fe54 = denominator)
    if element_denominator == "Fe56":
        denominator = data['[56Fe]+ mass 55.9344']# 91.754%
        output = output.assign(Fe56 = denominator)
    if element_denominator == "Fe57":
        denominator = data['[57Fe]+ mass 56.9348']# 2.119%
        output = output.assign(Fe57 = denominator)
    if element_denominator == "Fe58":
        denominator = data['[58Fe]+ mass 57.9327']# 0.282%
        output = output.assign(Fe58 = denominator)


    if element_denominator == "Mg24":
        denominator = data['[24Mg]+ mass 23.9845']# 78.99%
        output = output.assign(Mg24 = denominator)
    if element_denominator == "Mg25":
        denominator = data['[25Mg]+ mass 24.9853']# 10.00%
        output = output.assign(Mg25 = denominator)
    if element_denominator == "Mg26":
        denominator = data['[26Mg]+ mass 25.982']# 11.01%
        output = output.assign(Mg26 = denominator)


    if element_denominator == "Mn55":
        denominator = data['[55Mn]+ mass 54.9375']# 100%
        output = output.assign(Mn55 = denominator)


    if element_denominator == "Mo92":
        denominator = data['[92Mo]+ mass 91.9063']# 14.84%
        output = output.assign(Mo92 = denominator)
    if element_denominator == "Mo94":
        denominator = data['[94Mo]+ mass 93.9045']# 9.25%
        output = output.assign(Mo94 = denominator)
    if element_denominator == "Mo95":
        denominator = data['[95Mo]+ mass 94.9053']# 15.92%
        output = output.assign(Mo95 = denominator)   
    if element_denominator == "Mo96":
        denominator = data['[96Mo]+ mass 95.9041']# 16.68%
        output = output.assign(Mo96 = denominator)
    if element_denominator == "Mo97":
        denominator = data['[97Mo]+ mass 96.9055']# 9.55%
        output = output.assign(Mo97 = denominator)
    if element_denominator == "Mo98":
        denominator = data['[98Mo]+ mass 97.9049']# 24.13%
        output = output.assign(Mo98 = denominator)    
    if element_denominator == "Mo100":
        denominator = data['[100Mo]+ mass 99.9069']# 9.63%
        output = output.assign(Mo100 = denominator)


    if element_denominator == "Ni58":
        denominator = data['[58Ni]+ mass 57.9348']# 68.0769%
        output = output.assign(Ni58 = denominator)   
    if element_denominator == "Ni60":
        denominator = data['[60Ni]+ mass 59.9302']# 26.2231%
        output = output.assign(Ni60 = denominator)
    if element_denominator == "Ni61":
        denominator = data['[61Ni]+ mass 60.9305']# 1.1399%
        output = output.assign(Ni61 = denominator)
    if element_denominator == "Ni62":
        denominator = data['[62Ni]+ mass 61.9278']# 3.6345%
        output = output.assign(Ni62 = denominator)    
    if element_denominator == "Ni64":
        denominator = data['[64Ni]+ mass 63.9274']# 0.9256%
        output = output.assign(Ni64 = denominator)   


    if element_denominator == "K39":
        denominator = data['[39K]+ mass 38.9632']# 93.2581%
        output = output.assign(K39 = denominator)    
    if element_denominator == "K41":
        denominator = data['[41K]+ mass 40.9613']# 	6.7302%
        output = output.assign(K41 = denominator)     


    if element_denominator == "Na23":
        denominator = data['[23Na]+ mass 22.9892']# 100%
        output = output.assign(Na23 = denominator)


    if element_denominator == "Sr84":
        denominator = data['[84Sr]+ mass 83.9129']# 0.56%
        output = output.assign(Sr84 = denominator)
    if element_denominator == "Sr86":
        denominator = data['[86Sr]+ mass 85.9087']# 9.86%
        output = output.assign(Sr86 = denominator)
    if element_denominator == "Sr87":
        denominator = data['[87Sr]+ mass 86.9083']# 7.00%
        output = output.assign(Sr87 = denominator)    
    if element_denominator == "Sr88":
        denominator = data['[88Sr]+ mass 87.9051']# 82.58%
        output = output.assign(Sr88 = denominator)  


    if element_denominator == "U234":
        denominator = data['[234U]+ mass 234.04']# 0.0055%
        output = output.assign(U234 = denominator)
    if element_denominator == "U235":
        denominator = data['[235U]+ mass 235.043']# 0.7200%
        output = output.assign(U235 = denominator)
    if element_denominator == "U238":
        denominator = data['[238U]+ mass 238.05']# 99.2745%
        output = output.assign(U238 = denominator)

    if element_denominator == "UO254":
        denominator = data['UO+ mass 254.045']
        output = output.assign(UO254 = denominator)    


    if element_denominator == "V50":
        denominator = data['[50V]+ mass 49.9466']# 0.250%
        output = output.assign(V50 = denominator)    
    if element_denominator == "V51":
        denominator = data['[51V]+ mass 50.9434']# 99.750%
        output = output.assign(V51 = denominator)  


    if element_denominator == "Zn64":
        denominator = data['[64Zn]+ mass 63.9286']# 48.63%
        output = output.assign(Zn64 = denominator)   
    if element_denominator == "Zn66":
        denominator = data['[66Zn]+ mass 65.9255']# 27.90%
        output = output.assign(Zn66 = denominator)
    if element_denominator == "Zn67":
        denominator = data['[67Zn]+ mass 66.9266']# 4.10 %
        output = output.assign(Zn67 = denominator)
    if element_denominator == "Zn68":
        denominator = data['[68Zn]+ mass 67.9243']# 18.75%
        output = output.assign(Zn68 = denominator)    
    if element_denominator == "Zn70":
        denominator = data['[70Zn]+ mass 69.9248']# 0.62%
        output = output.assign(Zn70 = denominator)  


    if element_denominator == "Be9":
        denominator = data['[9Be]+ mass 9.01163']
        output = output.assign(Be9 = denominator)   
    if element_denominator == "Be10":
        denominator = data['[10B]+ mass 10.0124']
        output = output.assign(Be10 = denominator)
    if element_denominator == "Be11":
        denominator = data['[11B]+ mass 11.0088']
        output = output.assign(Be11 = denominator)


    if element_denominator == "Li6":
        denominator = data['[6Li]+ mass 6.01546']
        output = output.assign(Li6 = denominator)    
    if element_denominator == "Li7":
        denominator = data['[7Li]+ mass 7.01546']
        output = output.assign(Li7 = denominator)    


    if element_denominator == "C12":
        denominator = data['[12C]+ mass 11.9994']
        output = output.assign(C12 = denominator)   
    if element_denominator == "C13":
        denominator = data['[13C]+ mass 13.0028']
        output = output.assign(C13 = denominator)
    if element_denominator == "CO2":
        denominator = data['CO2+ mass 43.9893']
        output = output.assign(CO2 = denominator)
    if element_denominator == "CO2H":
        denominator = data['CHO2+ mass 44.9971']
        output = output.assign(CO2H = denominator)    
    if element_denominator == "C2H226":
        denominator = data['C2H2+ mass 26.0151']
        output = output.assign(C2H226 = denominator)
    if element_denominator == "C2H327":
        denominator = data['C2H3+ mass 27.0229']
        output = output.assign(C2H327 = denominator) 


    if element_denominator == "N14":
        denominator = data['[14N]+ mass 14.0025']
        output = output.assign(N14 = denominator)   
    if element_denominator == "N15":
        denominator = data['[15N]+ mass 14.9996']
        output = output.assign(N15 = denominator)
    if element_denominator == "N2":
        denominator = data['N2+ mass 28.0056']
        output = output.assign(N2 = denominator)
    if element_denominator == "N2H29":
        denominator = data['HN2+ mass 29.0134']
        output = output.assign(N2H29 = denominator)    
    if element_denominator == "NO30":
        denominator = data['NO+ mass 29.9974']
        output = output.assign(NO30 = denominator)
    if element_denominator == "NO31":
        denominator = data['[15N]O+ mass 30.9945']
        output = output.assign(NO31 = denominator)  
    if element_denominator == "NO246":
        denominator = data['NO2+ mass 45.9924']
        output = output.assign(NO246 = denominator)

    if element_denominator == "O16":
        denominator = data['[16O]+ mass 15.9944']
        output = output.assign(O16 = denominator)   
    if element_denominator == "O18":
        denominator = data['[18O]+ mass 17.9986']
        output = output.assign(O18 = denominator)
    if element_denominator == "OH17":
        denominator = data['OH+ mass 17.0022']
        output = output.assign(OH17 = denominator)
    if element_denominator == "H2O18":
        denominator = data['H2O+ mass 18.01']
        output = output.assign(H2O18 = denominator)    
    if element_denominator == "H3O19":
        denominator = data['H3O+ mass 19.0178']
        output = output.assign(H3O19 = denominator)
    if element_denominator == "O232":
        denominator = data['O2+ mass 31.9893']
        output = output.assign(O232 = denominator)  
    if element_denominator == "O2H33":
        denominator = data['O2H+ mass 32.9971']
        output = output.assign(O2H33 = denominator)   
    if element_denominator == "OH17":
        denominator = data['OH+ mass 17.0022']
        output = output.assign(OH17 = denominator)  
    if element_denominator == "O234":
        denominator = data['O[18O]+ mass 33.9935']
        output = output.assign(O234 = denominator)   


    if element_denominator == "Si28":
        denominator = data['[28Si]+ mass 27.9764']
        output = output.assign(Si28 = denominator)   
    if element_denominator == "Si29":
        denominator = data['[29Si]+ mass 28.976']
        output = output.assign(Si29 = denominator)  
    if element_denominator == "Si30":
        denominator = data['[30Si]+ mass 29.9732']
        output = output.assign(Si30 = denominator)


    if element_denominator == "P31":
        denominator = data['[31P]+ mass 30.9732']
        output = output.assign(P31 = denominator)


    if element_denominator == "S32":
        denominator = data['[32S]+ mass 31.9715']
        output = output.assign(S32 = denominator)  
    if element_denominator == "S33":
        denominator = data['[33S]+ mass 32.9709']
        output = output.assign(S33 = denominator)   
    if element_denominator == "S34":
        denominator = data['[34S]+ mass 33.9673']
        output = output.assign(S34 = denominator)  
    if element_denominator == "S36":
        denominator = data['[36S]+ mass 35.9665']
        output = output.assign(S36 = denominator)

    if element_denominator == "Cl35":
        denominator = data['[35Cl]+ mass 34.9683']
        output = output.assign(Cl35 = denominator)
    if element_denominator == "Cl37":
        denominator = data['[37Cl]+ mass 36.9654']
        output = output.assign(Cl37 = denominator)  
    if element_denominator == "HCl36":
        denominator = data['HCl+ mass 35.9761']
        output = output.assign(HCl36 = denominator)   
    if element_denominator == "ClO51":
        denominator = data['ClO+ mass 50.9632']
        output = output.assign(ClO51 = denominator)  
    if element_denominator == "ClO53":
        denominator = data['[37Cl]O+ mass 52.9603']
        output = output.assign(ClO53 = denominator)    


    if element_denominator == "Sc45":
        denominator = data['[45Sc]+ mass 44.9554']
        output = output.assign(Sc45 = denominator)


    if element_denominator == "Ti46":
        denominator = data['[46Ti]+ mass 45.9521']
        output = output.assign(Ti46 = denominator)
    if element_denominator == "Ti47":
        denominator = data['[47Ti]+ mass 46.9512']
        output = output.assign(Ti47 = denominator)  
    if element_denominator == "Ti48":
        denominator = data['[48Ti]+ mass 47.9474']
        output = output.assign(Ti48 = denominator)   
    if element_denominator == "Ti49":
        denominator = data['[49Ti]+ mass 48.9473']
        output = output.assign(Ti49 = denominator)  
    if element_denominator == "Ti50":
        denominator = data['[50Ti]+ mass 49.9442']
        output = output.assign(Ti50 = denominator) 


    if element_denominator == "Ga69":
        denominator = data['[69Ga]+ mass 68.925']
        output = output.assign(Ga69 = denominator)  
    if element_denominator == "Ga71":
        denominator = data['[71Ga]+ mass 70.9241']
        output = output.assign(Ga71 = denominator)


    if element_denominator == "Ar36":
        denominator = data['[36Ar]+ mass 35.967']
        output = output.assign(Ar36 = denominator)
    if element_denominator == "Ar38":
        denominator = data['[38Ar]+ mass 37.9622']
        output = output.assign(Ar38 = denominator)  
    if element_denominator == "Ar40":
        denominator = data['[40Ar]+ mass 39.9618']
        output = output.assign(Ar40 = denominator)   
    if element_denominator == "ArH37":
        denominator = data['[36Ar]H+ mass 36.9748']
        output = output.assign(ArH37 = denominator)  
    if element_denominator == "ArH39":
        denominator = data['[38Ar]H+ mass 38.97']
        output = output.assign(ArH39 = denominator)    
    if element_denominator == "ArH41":
        denominator = data['ArH+ mass 40.9697']
        output = output.assign(ArH41 = denominator)
    if element_denominator == "ArH242":
        denominator = data['ArH2+ mass 41.9775']
        output = output.assign(ArH242 = denominator)  
    if element_denominator == "ArO52":
        denominator = data['[36Ar]O+ mass 51.9619']
        output = output.assign(ArO52 = denominator)   
    if element_denominator == "ArN54":
        denominator = data['ArN+ mass 53.9649']
        output = output.assign(ArN54 = denominator)  
    if element_denominator == "ArO56":
        denominator = data['ArO+ mass 55.9567']
        output = output.assign(ArO56 = denominator)    
    if element_denominator == "ArOH57":
        denominator = data['ArOH+ mass 56.9646']
        output = output.assign(ArOH57 = denominator)
    if element_denominator == "Ar280":
        denominator = data['Ar2+ mass 79.9242']
        output = output.assign(Ar280 = denominator)  
    if element_denominator == "ArCl":
        denominator = data['ArCl+ mass 74.9307']
        output = output.assign(ArCl = denominator)   
    if element_denominator == "Ar276":
        denominator = data['Ar[36Ar]+ mass 75.9294']
        output = output.assign(Ar276 = denominator)  
    if element_denominator == "ArCl77":
        denominator = data['Ar[37Cl]+ mass 76.9277']
        output = output.assign(ArCl77 = denominator)    
    if element_denominator == "Ar278":
        denominator = data['Ar[38Ar]+ mass 77.9246']
        output = output.assign(Ar278 = denominator)    


    if element_denominator == "Ge70":
        denominator = data['[70Ge]+ mass 69.9237']
        output = output.assign(Ge70 = denominator)  
    if element_denominator == "Ge72":
        denominator = data['[72Ge]+ mass 71.9215']
        output = output.assign(Ge72 = denominator)   
    if element_denominator == "Ge73":
        denominator = data['[73Ge]+ mass 72.9229']
        output = output.assign(Ge73 =denominator)  
    if element_denominator == "Ge74":
        denominator = data['[74Ge]+ mass 73.9206']
        output = output.assign(Ge74 = denominator)    
    if element_denominator == "Ge76":
        denominator = data['[76Ge]+ mass 75.9209']
        output = output.assign(Ge76 = denominator)    


    if element_denominator == "Co59":
        denominator = data['[59Co]+ mass 58.9327']
        output = output.assign(Co59 = denominator)


    if element_denominator == "Se74":
        denominator = data['[74Se]+ mass 73.9219']
        output = output.assign(Se74 = denominator)
    if element_denominator == "Se76":
        denominator = data['[76Se]+ mass 75.9187']
        output = output.assign(Se76 =denominator)  
    if element_denominator == "Se77":
        denominator = data['[77Se]+ mass 76.9194']
        output = output.assign(Se77 = denominator)   
    if element_denominator == "Se78":
        denominator = data['[78Se]+ mass 77.9168']
        output = output.assign(Se78 = denominator)  
    if element_denominator == "Se80":
        denominator = data['[80Se]+ mass 79.916']
        output = output.assign(Se80 = denominator)    
    if element_denominator == "Se82":
        denominator = data['[82Se]+ mass 81.9162']
        output = output.assign(Se82 = denominator) 


    if element_denominator == "Kr78":
        denominator = data['[78Kr]+ mass 77.9198']
        output = output.assign(Kr78 =denominator)
    if element_denominator == "Kr80":
        denominator = data['[80Kr]+ mass 79.9158']
        output = output.assign(Kr80 = denominator)  
    if element_denominator == "Kr82":
        denominator = data['[82Kr]+ mass 81.9129']
        output = output.assign(Kr82 = denominator)   
    if element_denominator == "Kr83":
        denominator = data['[83Kr]+ mass 82.9136']
        output = output.assign(Kr83 = denominator)  
    if element_denominator == "Kr84":
        denominator = data['[84Kr]+ mass 83.911']
        output = output.assign(Kr84 = denominator)    
    if element_denominator == "Kr86":
        denominator = data['[86Kr]+ mass 85.9101']
        output = output.assign(Kr86 = denominator) 


    if element_denominator == "Br79":
        denominator = data['[79Br]+ mass 78.9178']
        output = output.assign(Br79 = denominator)    
    if element_denominator == "Br81":
        denominator = data['[81Br]+ mass 80.9157']
        output = output.assign(Br81 = denominator)


    if element_denominator == "Rb85":
        denominator = data['[85Rb]+ mass 84.9112']
        output = output.assign(Rb85 = denominator)    
    if element_denominator == "Rb87":
        denominator = data['[87Rb]+ mass 86.9086']
        output = output.assign(Rb87 = denominator)


    if element_denominator == "Y89":
        denominator = data['[89Y]+ mass 88.9053']
        output = output.assign(Y89 = denominator)


    if element_denominator == "Zr90":
        denominator = data['[90Zr]+ mass 89.9042']
        output = output.assign(Zr90 = denominator)  
    if element_denominator == "Zr91":
        denominator = data['[91Zr]+ mass 90.9051']
        output = output.assign(Zr91 = denominator)   
    if element_denominator == "Zr92":
        denominator = data['[92Zr]+ mass 91.9045']
        output = output.assign(Zr92 = denominator)  
    if element_denominator == "Zr94":
        denominator = data['[94Zr]+ mass 93.9058']
        output = output.assign(Zr94 = denominator)    
    if element_denominator == "Zr96":
        denominator = data['[96Zr]+ mass 95.9077']
        output = output.assign(Zr96 = denominator)


    if element_denominator == "Nb93":
        denominator = data['[93Nb]+ mass 92.9058']
        output = output.assign(Nb93 = denominator)


    if element_denominator == "Ru96":
        denominator = data['[96Ru]+ mass 95.9071']
        output = output.assign(Ru96 = denominator)  
    if element_denominator == "Ru98":
        denominator = data['[98Ru]+ mass 97.9047']
        output = output.assign(Ru98 = denominator)   
    if element_denominator == "Ru99":
        denominator = data['[99Ru]+ mass 98.9054']
        output = output.assign(Ru99 = denominator)  
    if element_denominator == "Ru100":
        denominator = data['[100Ru]+ mass 99.9037']
        output = output.assign(Ru100 = denominator)    
    if element_denominator == "Ru101":
        denominator = data['[101Ru]+ mass 100.905']
        output = output.assign(Ru101 = denominator)
    if element_denominator == "Ru102":
        denominator = data['[102Ru]+ mass 101.904']
        output = output.assign(Ru102 = denominator)    
    if element_denominator == "Ru104":
        denominator = data['[104Ru]+ mass 103.905']
        output = output.assign(Ru104 = denominator)


    if element_denominator == "Pd102":
        denominator = data['[102Pd]+ mass 101.905']
        output = output.assign(Pd102 = denominator)  
    if element_denominator == "Pd104":
        denominator = data['[104Pd]+ mass 103.903']
        output = output.assign(Pd104 = denominator)   
    if element_denominator == "Pd105":
        denominator = data['[105Pd]+ mass 104.905']
        output = output.assign(Pd105 = denominator)  
    if element_denominator == "Pd106":
        denominator = data['[106Pd]+ mass 105.903']
        output = output.assign(Pd106 = denominator)    
    if element_denominator == "Pd108":
        denominator = data['[108Pd]+ mass 107.903']
        output = output.assign(Pd108 = denominator)
    if element_denominator == "Pd110":
        denominator = data['[110Pd]+ mass 109.905']
        output = output.assign(Pd110 = denominator)


    if element_denominator == "Rh103":
        denominator = data['[103Rh]+ mass 102.905']
        output = output.assign(Rh103 = denominator)  


    if element_denominator == "Ag107":
        denominator = data['[107Ag]+ mass 106.905']
        output = output.assign(Ag107 = denominator)
    if element_denominator == "Ag109":
        denominator = data['[109Ag]+ mass 108.904']
        output = output.assign(Ag109 = denominator)  


    if element_denominator == "Sn112":
        denominator = data['[112Sn]+ mass 111.904']
        output = output.assign(Sn112 = denominator)   
    if element_denominator == "Sn114":
        denominator = data['[114Sn]+ mass 113.902']
        output = output.assign(Sn114 = denominator)  
    if element_denominator == "Sn115":
        denominator = data['[115Sn]+ mass 114.903']
        output = output.assign(Sn115 = denominator)    
    if element_denominator == "Sn116":
        denominator = data['[116Sn]+ mass 115.901']
        output = output.assign(Sn116 = denominator)
    if element_denominator == "Sn117":
        denominator = data['[117Sn]+ mass 116.902']
        output = output.assign(Sn117 = denominator)  
    if element_denominator == "Sn118":
        denominator = data['[118Sn]+ mass 117.901']
        output = output.assign(Sn118 = denominator)   
    if element_denominator == "Sn119":
        denominator = data['[119Sn]+ mass 118.903']
        output = output.assign(Sn119 = denominator)  
    if element_denominator == "Sn120":
        denominator = data['[120Sn]+ mass 119.902']
        output = output.assign(Sn120 = denominator)    
    if element_denominator == "Sn122":
        denominator = data['[122Sn]+ mass 121.903']
        output = output.assign(Sn122 = denominator)
    if element_denominator == "Sn124":
        denominator = data['[124Sn]+ mass 123.905']
        output = output.assign(Sn124 = denominator)  


    if element_denominator == "In113":
        denominator = data['[113In]+ mass 112.904']
        output = output.assign(In113 = denominator)
    if element_denominator == "In115":
        denominator = data['[115In]+ mass 114.903']
        output = output.assign(In115 = denominator)  


    if element_denominator == "Sb121":
        denominator = data['[121Sb]+ mass 120.903']
        output = output.assign(Sb121 = denominator)   
    if element_denominator == "Sb123":
        denominator = data['[123Sb]+ mass 122.904']
        output = output.assign(Sb123 = denominator)


    if element_denominator == "Te120":
        denominator = data['[120Te]+ mass 119.903']
        output = output.assign(Te120 = denominator)    
    if element_denominator == "Te122":
        denominator = data['[122Te]+ mass 121.902']
        output = output.assign(Te122 = denominator)
    if element_denominator == "Te123":
        denominator = data['[123Te]+ mass 122.904']
        output = output.assign(Te123 = denominator)  
    if element_denominator == "Te124":
        denominator = data['[124Te]+ mass 123.902']
        output = output.assign(Te124 = denominator)   
    if element_denominator == "Te125":
        denominator = data['[125Te]+ mass 124.904']
        output = output.assign(Te125 = denominator)  
    if element_denominator == "Te126":
        denominator = data['[126Te]+ mass 125.903']
        output = output.assign(Te126 = denominator)    
    if element_denominator == "Te128":
        denominator = data['[128Te]+ mass 127.904']
        output = output.assign(Te128 = denominator)
    if element_denominator == "Te130":
        denominator = data['[130Te]+ mass 129.906']
        output = output.assign(Te130 = denominator)


    if element_denominator == "Xe124":
        denominator = data['[124Xe]+ mass 123.905']
        output = output.assign(Xe124 = denominator)  
    if element_denominator == "Xe126":
        denominator = data['[126Xe]+ mass 125.904']
        output = output.assign(Xe126 = denominator)    
    if element_denominator == "Xe128":
        denominator = data['[128Xe]+ mass 127.903']
        output = output.assign(Xe128 = denominator)
    if element_denominator == "Xe129":
        denominator = data['[129Xe]+ mass 128.904']
        output = output.assign(Xe129 = denominator)  
    if element_denominator == "Xe130":
        denominator = data['[130Xe]+ mass 129.903']
        output = output.assign(Xe130 = denominator)   
    if element_denominator == "Xe131":
        denominator = data['[131Xe]+ mass 130.905']
        output = output.assign(Xe131 = denominator)  
    if element_denominator == "Xe132":
        denominator = data['[132Xe]+ mass 131.904']
        output = output.assign(Xe132 = denominator)    
    if element_denominator == "Xe134":
        denominator = data['[134Xe]+ mass 133.905']
        output = output.assign(Xe134 = denominator)
    if element_denominator == "Xe136":
        denominator = data['[136Xe]+ mass 135.907']
        output = output.assign(Xe136 = denominator)


    if element_denominator == "I127":
        denominator = data['[127I]+ mass 126.904']
        output = output.assign(I127 = denominator)  


    if element_denominator == "Cs133":
        denominator = data['[133Cs]+ mass 132.905']
        output = output.assign(Cs133 = denominator)   


    if element_denominator == "Ce136":
        denominator = data['[136Ce]+ mass 135.907']
        output = output.assign(Ce136 = denominator)
    if element_denominator == "Ce138":
        denominator = data['[138Ce]+ mass 137.905']
        output = output.assign(Ce138 = denominator)  
    if element_denominator == "Ce140":
        denominator = data['[140Ce]+ mass 139.905']
        output = output.assign(Ce140 = denominator)   
    if element_denominator == "Ce142":
        denominator = data['[142Ce]+ mass 141.909']
        output = output.assign(Ce142 = denominator)  

    if element_denominator == "CeO156":
        denominator = data['CeO+ mass 155.9']
        output = output.assign(CeO156 = denominator)  


    if element_denominator == "La138":
        denominator = data['[138La]+ mass 137.907']
        output = output.assign(La138 = denominator)
    if element_denominator == "La139":
        denominator = data['[139La]+ mass 138.906']
        output = output.assign(La139 = denominator)


    if element_denominator == "Pr141":
        denominator = data['[141Pr]+ mass 140.907']
        output = output.assign(Pr141 = denominator)


    if element_denominator == "Nd142":
        denominator = data['[142Nd]+ mass 141.907']
        output = output.assign(Nd142 = denominator)
    if element_denominator == "Nd143":
        denominator = data['[143Nd]+ mass 142.909']
        output = output.assign(Nd143 = denominator)  
    if element_denominator == "Nd144":
        denominator = data['[144Nd]+ mass 143.91']
        output = output.assign(Nd144 = denominator)   
    if element_denominator == "Nd145":
        denominator = data['[145Nd]+ mass 144.912']
        output = output.assign(Nd145 = denominator)             
    if element_denominator == "Nd146":
        denominator = data['[146Nd]+ mass 145.913']
        output = output.assign(Nd146 = denominator)           
    if element_denominator == "Nd148":
        denominator = data['[148Nd]+ mass 147.916']
        output = output.assign(Nd148 =denominator)
    if element_denominator == "Nd150":
        denominator = data['[150Nd]+ mass 149.92']
        output = output.assign(Nd150 = denominator)


    if element_denominator == "Sm144":
        denominator = data['[144Sm]+ mass 143.911']
        output = output.assign(Sm144 = denominator)
    if element_denominator == "Sm147":
        denominator = data['[147Sm]+ mass 146.914']
        output = output.assign(Sm147 = denominator)  
    if element_denominator == "Sm148":
        denominator = data['[148Sm]+ mass 147.914']
        output = output.assign(Sm148 = denominator)   
    if element_denominator == "Sm149":
        denominator = data['[149Sm]+ mass 148.917']
        output = output.assign(Sm149 = denominator)             
    if element_denominator == "Sm150":
        denominator = data['[150Sm]+ mass 149.917']
        output = output.assign(Sm150 = denominator)           
    if element_denominator == "Sm152":
        denominator = data['[152Sm]+ mass 151.919']
        output = output.assign(Sm152 = denominator)
    if element_denominator == "Sm154":
        denominator = data['[154Sm]+ mass 153.922']
        output = output.assign(Sm154 = denominator)


    if element_denominator == "Eu151":
        denominator = data['[151Eu]+ mass 150.919']
        output = output.assign(Eu151 = denominator)
    if element_denominator == "Eu153":
        denominator = data['[153Eu]+ mass 152.921']
        output = output.assign(Eu153 = denominator)


    if element_denominator == "Gd152":
        denominator = data['[152Gd]+ mass 151.919']
        output = output.assign(Gd152 = denominator)
    if element_denominator == "Gd154":
        denominator = data['[154Gd]+ mass 153.92']
        output = output.assign(Gd154 = denominator)  
    if element_denominator == "Gd155":
        denominator = data['[155Gd]+ mass 154.922']
        output = output.assign(Gd155 = denominator)   
    if element_denominator == "Gd156":
        denominator = data['[156Gd]+ mass 155.922']
        output = output.assign(Gd156 = denominator)             
    if element_denominator == "Gd157":
        denominator = data['[157Gd]+ mass 156.923']
        output = output.assign(Gd157 = denominator)           
    if element_denominator == "Gd158":
        denominator = data['[158Gd]+ mass 157.924']
        output = output.assign(Gd158 = denominator)
    if element_denominator == "Gd160":
        denominator = data['[160Gd]+ mass 159.927']
        output = output.assign(Gd160 = denominator)


    if element_denominator == "Dy156":
        denominator = data['[156Dy]+ mass 155.924']
        output = output.assign(Dy156 = denominator)
    if element_denominator == "Dy158":
        denominator = data['[158Dy]+ mass 157.924']
        output = output.assign(Dy158 = denominator)  
    if element_denominator == "Dy160":
        denominator = data['[160Dy]+ mass 159.925']
        output = output.assign(Dy160 = denominator)   
    if element_denominator == "Dy161":
        denominator = data['[161Dy]+ mass 160.926']
        output = output.assign(Dy161 = denominator)             
    if element_denominator == "Dy162":
        denominator = data['[162Dy]+ mass 161.926']
        output = output.assign(Dy162 = denominator)           
    if element_denominator == "Dy163":
        denominator = data['[163Dy]+ mass 162.928']
        output = output.assign(Dy163 = denominator)
    if element_denominator == "Dy164":
        denominator = data['[164Dy]+ mass 163.929']
        output = output.assign(Dy164 = denominator)


    if element_numerator == "Tb159":
        denominator = data['[159Tb]+ mass 158.925']
        output = output.assign(Tb159 = denominator)


    if element_denominator == "Er162":
        denominator = data['[162Er]+ mass 161.928']
        output = output.assign(Er162 = denominator)  
    if element_denominator == "Er164":
        denominator = data['[164Er]+ mass 163.929']
        output = output.assign(Er164 = denominator)   
    if element_denominator == "Er166":
        denominator = data['[166Er]+ mass 165.93']
        output = output.assign(Er166 = denominator)             
    if element_denominator == "Er167":
        denominator = data['[167Er]+ mass 166.932']
        output = output.assign(Er167 = denominator)           
    if element_denominator == "Er168":
        denominator = data['[168Er]+ mass 167.932']
        output = output.assign(Er168 = denominator)
    if element_denominator == "Er170":
        denominator = data['[170Er]+ mass 169.935']
        output = output.assign(Er170 = denominator)


    if element_denominator == "Ho165":
        denominator = data['[165Ho]+ mass 164.93']
        output = output.assign(Ho165 = denominator)


    if element_denominator == "Yb168":
        denominator = data['[168Yb]+ mass 167.933']
        output = output.assign(Yb168 = denominator)  
    if element_denominator == "Yb170":
        denominator = data['[170Yb]+ mass 169.934']
        output = output.assign(Yb170 = denominator)   
    if element_denominator == "Yb171":
        denominator = data['[171Yb]+ mass 170.936']
        output = output.assign(Yb171 = denominator)             
    if element_denominator == "Yb172":
        denominator= data['[172Yb]+ mass 171.936']
        output = output.assign(Yb172 = denominator)           
    if element_denominator == "Yb173":
        denominator = data['[173Yb]+ mass 172.938']
        output = output.assign(Yb173 = denominator)
    if element_denominator == "Yb174":
        denominator = data['[174Yb]+ mass 173.938']
        output = output.assign(Yb174 = denominator)    
    if element_denominator == "Yb176":
        denominator = data['[176Yb]+ mass 175.942']
        output = output.assign(Yb176 = denominator)


    if element_denominator == "Tm169":
        denominator = data['[169Tm]+ mass 168.934']
        output = output.assign(Tm169 = denominator)  


    if element_denominator == "Hf174":
        denominator = data['[174Hf]+ mass 173.939']
        output = output.assign(Hf174 = denominator)    
    if element_denominator == "Hf176":
        denominator = data['[176Hf]+ mass 175.941']
        output = output.assign(Hf176 = denominator)   
    if element_denominator == "Hf177":
        denominator = data['[177Hf]+ mass 176.943']
        output = output.assign(Hf177 = denominator)             
    if element_denominator == "Hf178":
        denominator = data['[178Hf]+ mass 177.943']
        output = output.assign(Hf178 = denominator)           
    if element_denominator == "Hf179":
        denominator = data['[179Hf]+ mass 178.945']
        output = output.assign(Hf179 = denominator)
    if element_denominator == "Hf180":
        denominator = data['[180Hf]+ mass 179.946']
        output = output.assign(Hf180 = denominator)


    if element_denominator == "Lu175":
        denominator = data['[175Lu]+ mass 174.94']
        output = output.assign(Lu175 = denominator)  
    if element_denominator == "Lu176":
        denominator = data['[176Lu]+ mass 175.942']
        output = output.assign(Lu176 = denominator)  


    if element_denominator == "W180":
        denominator = data['[180W]+ mass 179.946']
        output = output.assign(W180 = denominator)             
    if element_denominator == "W182":
        denominator = data['[182W]+ mass 181.948']
        output = output.assign(W182 =denominator)           
    if element_denominator == "W183":
        denominator = data['[183W]+ mass 182.95']
        output = output.assign(W183 = denominator)
    if element_denominator == "W184":
        denominator = data['[184W]+ mass 183.95']
        output = output.assign(W184 = denominator)    
    if element_denominator == "W186":
        denominator = data['[186W]+ mass 185.954']
        output = output.assign(W186 = denominator)


    if element_denominator == "Ta180":
        denominator = data['[180Ta]+ mass 179.947']
        output = output.assign(Ta180 = denominator)  
    if element_denominator == "Ta181":
        denominator = data['[181Ta]+ mass 180.947']
        output = output.assign(Ta181 = denominator) 


    if element_denominator == "Os184":
        denominator = data['[184Os]+ mass 183.952']
        output = output.assign(Os184 = denominator)   
    if element_denominator == "Os186":
        denominator = data['[186Os]+ mass 185.953']
        output = output.assign(Os186 = denominator)             
    if element_denominator == "Os187":
        denominator = data['[187Os]+ mass 186.955']
        output = output.assign(Os187 = denominator)           
    if element_denominator == "Os188":
        denominator = data['[188Os]+ mass 187.955']
        output = output.assign(Os188 = denominator)
    if element_denominator == "Os189":
        denominator = data['[189Os]+ mass 188.958']
        output = output.assign(Os189 = denominator)
    if element_denominator == "Os190":
        denominator = data['[190Os]+ mass 189.958']
        output = output.assign(Os190 = denominator)  
    if element_denominator == "Os192":
        denominator = data['[192Os]+ mass 191.961']
        output = output.assign(Os192 = denominator) 


    if element_denominator == "Re185":
        denominator = data['[185Re]+ mass 184.952']
        output = output.assign(Re185 = denominator) 
    if element_denominator == "Re187":
        denominator = data['[187Re]+ mass 186.955']
        output = output.assign(Re187 = denominator)  


    if element_denominator == "Pt190":
        denominator = data['[190Pt]+ mass 189.959']
        output = output.assign(Pt190 = denominator) 
    if element_denominator == "Pt192":
        denominator= data['[192Pt]+ mass 191.96']
        output = output.assign(Pt192 =denominator)             
    if element_denominator == "Pt194":
        denominator = data['[194Pt]+ mass 193.962']
        output = output.assign(Pt194 = denominator)           
    if element_denominator == "Pt195":
        denominator = data['[195Pt]+ mass 194.964']
        output = output.assign(Pt195 = denominator)
    if element_denominator == "Pt196":
        denominator = data['[196Pt]+ mass 195.964']
        output = output.assign(Pt196 = denominator)    
    if element_denominator == "Pt198":
        denominator = data['[198Pt]+ mass 197.967']
        output = output.assign(Pt198 = denominator)


    if element_denominator == "Ir191":
        denominator = data['[191Ir]+ mass 190.96']
        output = output.assign(Ir191 = denominator)  
    if element_denominator == "Ir193":
        denominator = data['[193Ir]+ mass 192.962']
        output = output.assign(Ir193 = denominator)


    if element_denominator == "Hg196":
        denominator = data['[196Hg]+ mass 195.965']
        output = output.assign(Hg196 = denominator)   
    if element_denominator == "Hg198":
        denominator = data['[198Hg]+ mass 197.966']
        output = output.assign(Hg198 = denominator)             
    if element_denominator == "Hg199":
        denominator = data['[199Hg]+ mass 198.968']
        output = output.assign(Hg199 = denominator)           
    if element_denominator == "Hg200":
        denominator = data['[200Hg]+ mass 199.968']
        output = output.assign(Hg200 = denominator)
    if element_denominator == "Hg201":
        denominator = data['[201Hg]+ mass 200.97']
        output = output.assign(Hg201 = denominator)
    if element_denominator == "Hg202":
        denominator = data['[202Hg]+ mass 201.97']
        output = output.assign(Hg202 = denominator)  
    if element_denominator == "Hg204":
        denominator = data['[204Hg]+ mass 203.973']
        output = output.assign(Hg204 = denominator)


    if element_denominator == "Au197":
        denominator = data['[197Au]+ mass 196.966']
        output = output.assign(Au197 = denominator)  


    if element_denominator == "Tl203":
        denominator = data['[203Tl]+ mass 202.972']
        output = output.assign(Tl203 = denominator)             
    if element_denominator == "Tl205":
        denominator = data['[205Tl]+ mass 204.974']
        output = output.assign(Tl205 = denominator)   


    if element_denominator == "Bi209":
        denominator = data['[209Bi]+ mass 208.98']
        output = output.assign(Bi209 = denominator)


    if element_denominator == "Th232":
        denominator = data['[232Th]+ mass 232.038']
        output = output.assign(Th232 = denominator)

    if element_denominator == "ThO248":
        denominator = data['ThO+ mass 248.032']
        output = output.assign(ThO248 = denominator)  

    if element_denominator == "Background":
        denominator = data['[220Bkg]+ mass 220']
        output = output.assign(Background = denominator)
        
    Ratio = (numerator/denominator).where(denominator>0)
    output = output.assign(Ratio = Ratio)
    output = output.dropna()
    
    numerator_mean = numerator.mean()
    denominator_mean = denominator.mean()
    mean_ratio = numerator_mean/denominator_mean #to avoid problems of dividing with 0, devides the means and not the dataset point by point
    
    # Plotting
    sns.set()
    fig = plt.figure(figsize =(5,15))
    ax = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    plt.subplots_adjust(hspace=0.3)

    ax.set_title(element_numerator)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity (cps)")

    ax2.set_title(element_denominator)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Intensity (cps)")
    
    ax3.set_title("Ratio")
    ax3.set_xlabel("Ratio")
    ax3.set_ylabel("Frequency")

    ax.plot(output[str(element_numerator)],
            linewidth = 0.7)
    ax2.plot(output[str(element_denominator)],
            linewidth = 0.7)
    ax3.hist(output["Ratio"],
             edgecolor = "white"
            )
    plt.savefig("Elemental Ratio")
    sns.reset_orig()
    return output, numerator_mean, denominator_mean, mean_ratio



