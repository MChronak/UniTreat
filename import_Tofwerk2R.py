def import_tofwerk2R(waveforms,*elements):
    """Imports data exported from the TofPilot software of TofWerk2R, and creates 1) a pandas datasset ready for further use and 2) a plot of the given data
    
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
            output = output.assign(Al27 = Al27)
            
            
        if element == "As75":
            As75 = data['[75As]+ mass 74.9211'] # 100%
            output = output.assign(As75 = As75)
            
        
        if element == "Ba130":
            Ba130 = data['[130Ba]+ mass 129.906'] # 0.106%
            output = output.assign(Ba130 = Ba130)
        if element == "Ba132":
            Ba132 = data['[132Ba]+ mass 131.905'] # 0.101%
            output = output.assign(Ba132 = Ba132)
        if element == "Ba134":
            Ba134 = data['[134Ba]+ mass 133.904'] # 2.417%
            output = output.assign(Ba134 = Ba134)
        if element == "Ba135":
            Ba135 = data['[135Ba]+ mass 134.905'] # 6.592%
            output = output.assign(Ba135 = Ba135)
        if element == "Ba136":
            Ba136 = data['[136Ba]+ mass 135.904'] # 7.854%
            output = output.assign(Ba136 = Ba136)
        if element == "Ba137":
            Ba137 = data['[137Ba]+ mass 136.905'] # 11.232%
            output = output.assign(Ba137 = Ba137)
        if element == "Ba138":
            Ba138 = data['[138Ba]+ mass 137.905'] # 71.698%
            output = output.assign(Ba138 = Ba138)
            
        if element == "Ba137_dc":
            Ba137_dc = data['[137Ba]++ mass 68.4524']
            output = output.assign(Ba137_dc = Ba137_dc)
        if element == "Ba138_dc":
            Ba138_dc = data['[138Ba]++ mass 68.9521']
            output = output.assign(Ba138_dc = Ba138_dc)    
    
    
        if element == "Pb204":
            Pb204 = data['[204Pb]+ mass 203.973']# 1.4%
            output = output.assign(Pb204 = Pb204)
        if element == "Pb206":
            Pb206 = data['[206Pb]+ mass 205.974']# 24.1%
            output = output.assign(Pb206 = Pb206)
        if element == "Pb207":
            Pb207 = data['[207Pb]+ mass 206.975']# 22.1%
            output = output.assign(Pb207 = Pb207)
        if element == "Pb208":
            Pb208 = data['[208Pb]+ mass 207.976']# 52.4%
            output = output.assign(Pb208 = Pb208)
        
        
        if element == "Cd106":
            Cd106 = data['[106Cd]+ mass 105.906'] # 1.25%
            output = output.assign(Cd106 = Cd106)
        if element == "Cd108":
            Cd108 = data['[108Cd]+ mass 107.904'] # 0.89%
            output = output.assign(Cd108 = Cd108)
        if element == "Cd110":
            Cd110 = data['[110Cd]+ mass 109.902'] # 12.49%
            output = output.assign(Cd110 = Cd110)
        if element == "Cd111":
            Cd111 = data['[111Cd]+ mass 110.904'] # 12.80%
            output = output.assign(Cd111 = Cd111)
        if element == "Cd112":
            Cd112 = data['[112Cd]+ mass 111.902'] # 24.13%
            output = output.assign(Cd112 = Cd112)
        if element == "Cd113":
            Cd113 = data['[113Cd]+ mass 112.904'] # 12.22%
            output = output.assign(Cd113 = Cd113)
        if element == "Cd114":
            Cd114 = data['[114Cd]+ mass 113.903'] # 28.73%
            output = output.assign(Cd114 = Cd114)
        if element == "Cd116":
            Cd116 = data['[116Cd]+ mass 115.904'] # 7.49%
            output = output.assign(Cd116 = Cd116)
            
            
        if element == "Ca40":
            Ca40 = data['[40Ca]+ mass 39.962'] # 96.941%
            output = output.assign(Ca40 = Ca40)
        if element == "Ca42":
            Ca42 = data['[42Ca]+ mass 41.9581'] # 0.647%
            output = output.assign(Ca42 = Ca42)
        if element == "Ca43":
            Ca43 = data['[43Ca]+ mass 42.9582'] # 0.135%
            output = output.assign(Ca43 = Ca43)
        if element == "Ca44":
            Ca44 = data['[44Ca]+ mass 43.9549'] # 2.086%
            output = output.assign(Ca44 = Ca44)
        if element == "Ca46":
            Ca46 = data['[46Ca]+ mass 45.9531'] # 0.004%
            output = output.assign(Ca46 = Ca46)
        if element == "Ca48":
            Ca48 = data['[48Ca]+ mass 47.952'] # 0.187%
            output = output.assign(Ca48 = Ca48)
            
            
        if element == "Cr50":
            Cr50 = data['[50Cr]+ mass 49.9455']# 4.345%
            output = output.assign(Cr50 = Cr50)
        if element == "Cr52":
            Cr52 = data['[52Cr]+ mass 51.94']# 83.789%
            output = output.assign(Cr52 = Cr52)
        if element == "Cr53":
            Cr53 = data['[53Cr]+ mass 52.9401']# 9.501%
            output = output.assign(Cr53 = Cr53)
        if element == "Cr54":
            Cr54 = data['[54Cr]+ mass 53.9383']# 2.365%
            output = output.assign(Cr54 = Cr54)
            
            
        if element == "Cu63":
            Cu63 = data['[63Cu]+ mass 62.9291']# 69.17%
            output = output.assign(Cu63 = Cu63)
        if element == "Cu65":
            Cu65 = data['[65Cu]+ mass 64.9272']# 30.83%
            output = output.assign(Cu65 = Cu65)
            
            
        if element == "Fe54":
            Fe54 = data['[54Fe]+ mass 53.9391']# 5.845%
            output = output.assign(Fe54 = Fe54)
        if element == "Fe56":
            Fe56 = data['[56Fe]+ mass 55.9344']# 91.754%
            output = output.assign(Fe56 = Fe56)
        if element == "Fe57":
            Fe57 = data['[57Fe]+ mass 56.9348']# 2.119%
            output = output.assign(Fe57 = Fe57)
        if element == "Fe58":
            Fe58 = data['[58Fe]+ mass 57.9327']# 0.282%
            output = output.assign(Fe58 = Fe58)
            
            
        if element == "Mg24":
            Mg24 = data['[24Mg]+ mass 23.9845']# 78.99%
            output = output.assign(Mg24 = Mg24)
        if element == "Mg25":
            Mg25 = data['[25Mg]+ mass 24.9853']# 10.00%
            output = output.assign(Mg25 = Mg25)
        if element == "Mg26":
            Mg26 = data['[26Mg]+ mass 25.982']# 11.01%
            output = output.assign(Mg26 = Mg26)
            
            
        if element == "Mn55":
            Mn55 = data['[55Mn]+ mass 54.9375']# 100%
            output = output.assign(Mn55 = Mn55)
            
            
        if element == "Mo92":
            Mo92 = data['[92Mo]+ mass 91.9063']# 14.84%
            output = output.assign(Mo92 = Mo92)
        if element == "Mo94":
            Mo94 = data['[94Mo]+ mass 93.9045']# 9.25%
            output = output.assign(Mo94 = Mo94)
        if element == "Mo95":
            Mo95 = data['[95Mo]+ mass 94.9053']# 15.92%
            output = output.assign(Mo95 = Mo95)   
        if element == "Mo96":
            Mo96 = data['[96Mo]+ mass 95.9041']# 16.68%
            output = output.assign(Mo96 = Mo96)
        if element == "Mo97":
            Mo97 = data['[97Mo]+ mass 96.9055']# 9.55%
            output = output.assign(Mo97 = Mo97)
        if element == "Mo98":
            Mo98 = data['[98Mo]+ mass 97.9049']# 24.13%
            output = output.assign(Mo98 = Mo98)    
        if element == "Mo100":
            Mo100 = data['[100Mo]+ mass 99.9069']# 9.63%
            output = output.assign(Mo100 = Mo100)
            
            
        if element == "Ni58":
            Ni58 = data['[58Ni]+ mass 57.9348']# 68.0769%
            output = output.assign(Ni58 = Ni58)   
        if element == "Ni60":
            Ni60 = data['[60Ni]+ mass 59.9302']# 26.2231%
            output = output.assign(Ni60 = Ni60)
        if element == "Ni61":
            Ni61 = data['[61Ni]+ mass 60.9305']# 1.1399%
            output = output.assign(Ni61 = Ni61)
        if element == "Ni62":
            Ni62 = data['[62Ni]+ mass 61.9278']# 3.6345%
            output = output.assign(Ni62 = Ni62)    
        if element == "Ni64":
            Ni64 = data['[64Ni]+ mass 63.9274']# 0.9256%
            output = output.assign(Ni64 = Ni64)   
            
            
        if element == "K39":
            K39 = data['[39K]+ mass 38.9632']# 93.2581%
            output = output.assign(K39 = K39)    
        if element == "K41":
            K41 = data['[41K]+ mass 40.9613']# 	6.7302%
            output = output.assign(K41 = K41)     
        
        
        if element == "Na23":
            Na23 = data['[23Na]+ mass 22.9892']# 100%
            output = output.assign(Na23 = Na23)
            
            
        if element == "Sr84":
            Sr84 = data['[84Sr]+ mass 83.9129']# 0.56%
            output = output.assign(Sr84 = Sr84)
        if element == "Sr86":
            Sr86 = data['[86Sr]+ mass 85.9087']# 9.86%
            output = output.assign(Sr86 = Sr86)
        if element == "Sr87":
            Sr87 = data['[87Sr]+ mass 86.9083']# 7.00%
            output = output.assign(Sr87 = Sr87)    
        if element == "Sr88":
            Sr88 = data['[88Sr]+ mass 87.9051']# 82.58%
            output = output.assign(Sr88 = Sr88)  
            
            
        if element == "U234":
            U234 = data['[234U]+ mass 234.04']# 0.0055%
            output = output.assign(U234 = U234)
        if element == "U235":
            U235 = data['[235U]+ mass 235.043']# 0.7200%
            output = output.assign(U235 = U235)
        if element == "U238":
            U238 = data['[238U]+ mass 238.05']# 99.2745%
            output = output.assign(U238 = U238)
            
        if element == "UO254":
            UO254 = data['UO+ mass 254.045']
            output = output.assign(UO254 = UO254)    
            
            
        if element == "V50":
            V50 = data['[50V]+ mass 49.9466']# 0.250%
            output = output.assign(V50 = V50)    
        if element == "V51":
            V51 = data['[51V]+ mass 50.9434']# 99.750%
            output = output.assign(V51 = V51)  
            
            
        if element == "Zn64":
            Zn64 = data['[64Zn]+ mass 63.9286']# 48.63%
            output = output.assign(Zn64 = Zn64)   
        if element == "Zn66":
            Zn66 = data['[66Zn]+ mass 65.9255']# 27.90%
            output = output.assign(Zn66 = Zn66)
        if element == "Zn67":
            Zn67 = data['[67Zn]+ mass 66.9266']# 4.10 %
            output = output.assign(Zn67 = Zn67)
        if element == "Zn68":
            Zn68 = data['[68Zn]+ mass 67.9243']# 18.75%
            output = output.assign(Zn68 = Zn68)    
        if element == "Zn70":
            Zn70 = data['[70Zn]+ mass 69.9248']# 0.62%
            output = output.assign(Zn70 = Zn70)  
            
            
        if element == "Be9":
            Be9 = data['[9Be]+ mass 9.01163']
            output = output.assign(Be9 = Be9)   
        if element == "Be10":
            Be10 = data['[10B]+ mass 10.0124']
            output = output.assign(Be10 = Be10)
        if element == "Be11":
            Be11 = data['[11B]+ mass 11.0088']
            output = output.assign(Be11 = Be11)
            
            
        if element == "Li6":
            Li6 = data['[6Li]+ mass 6.01546']
            output = output.assign(Li6 = Li6)    
        if element == "Li7":
            Li7 = data['[7Li]+ mass 7.01546']
            output = output.assign(Li7 = Li7)    
        
        
        if element == "C12":
            C12 = data['[12C]+ mass 11.9994']
            output = output.assign(C12 = C12)   
        if element == "C13":
            C13 = data['[13C]+ mass 13.0028']
            output = output.assign(C13 = C13)
        if element == "CO2":
            CO2 = data['CO2+ mass 43.9893']
            output = output.assign(CO2 = CO2)
        if element == "CO2H":
            CO2H = data['CHO2+ mass 44.9971']
            output = output.assign(CO2H = CO2H)    
        if element == "C2H226":
            C2H226 = data['C2H2+ mass 26.0151']
            output = output.assign(C2H226 = C2H226)
        if element == "C2H327":
            C2H327 = data['C2H3+ mass 27.0229']
            output = output.assign(C2H327 = C2H327) 
            
            
        if element == "N14":
            N14 = data['[14N]+ mass 14.0025']
            output = output.assign(N14 = N14)   
        if element == "N15":
            N15 = data['[15N]+ mass 14.9996']
            output = output.assign(N15 = N15)
        if element == "N2":
            N2 = data['N2+ mass 28.0056']
            output = output.assign(N2 = N2)
        if element == "N2H29":
            N2H29 = data['HN2+ mass 29.0134']
            output = output.assign(N2H29 = N2H29)    
        if element == "NO30":
            NO30 = data['NO+ mass 29.9974']
            output = output.assign(NO30 = NO30)
        if element == "NO31":
            NO31 = data['[15N]O+ mass 30.9945']
            output = output.assign(NO31 = NO31)  
        if element == "NO246":
            NO246 = data['NO2+ mass 45.9924']
            output = output.assign(NO246 = NO246)
            
        if element == "O16":
            O16 = data['[16O]+ mass 15.9944']
            output = output.assign(O16 = O16)   
        if element == "O18":
            O18 = data['[18O]+ mass 17.9986']
            output = output.assign(O18 = O18)
        if element == "OH17":
            OH17 = data['OH+ mass 17.0022']
            output = output.assign(OH17 = OH17)
        if element == "H2O18":
            H2O18 = data['H2O+ mass 18.01']
            output = output.assign(H2O18 = H2O18)    
        if element == "H3O19":
            H3O19 = data['H3O+ mass 19.0178']
            output = output.assign(H3O19 = H3O19)
        if element == "O232":
            O232 = data['O2+ mass 31.9893']
            output = output.assign(O232 = O232)  
        if element == "O2H33":
            O2H33 = data['O2H+ mass 32.9971']
            output = output.assign(O2H33 = O2H33)   
        if element == "OH17":
            OH17 = data['OH+ mass 17.0022']
            output = output.assign(OH17 = OH17)  
        if element == "O234":
            O234 = data['O[18O]+ mass 33.9935']
            output = output.assign(O234 = O234)   
        
        
        if element == "Si28":
            Si28 = data['[28Si]+ mass 27.9764']
            output = output.assign(Si28 = Si28)   
        if element == "Si29":
            Si29 = data['[29Si]+ mass 28.976']
            output = output.assign(Si29 = Si29)  
        if element == "Si30":
            Si30 = data['[30Si]+ mass 29.9732']
            output = output.assign(Si30 = Si30)
            
            
        if element == "P31":
            P31 = data['[31P]+ mass 30.9732']
            output = output.assign(P31 = P31)
        
        
        if element == "S32":
            S32 = data['[32S]+ mass 31.9715']
            output = output.assign(S32 = S32)  
        if element == "S33":
            S33 = data['[33S]+ mass 32.9709']
            output = output.assign(S33 = S33)   
        if element == "S34":
            S34 = data['[34S]+ mass 33.9673']
            output = output.assign(S34 = S34)  
        if element == "S36":
            S36 = data['[36S]+ mass 35.9665']
            output = output.assign(S36 = S36)
            
        if element == "Cl35":
            Cl35 = data['[35Cl]+ mass 34.9683']
            output = output.assign(Cl35 = Cl35)
        if element == "Cl37":
            Cl37 = data['[37Cl]+ mass 36.9654']
            output = output.assign(Cl37 = Cl37)  
        if element == "HCl36":
            HCl36 = data['HCl+ mass 35.9761']
            output = output.assign(HCl36 = HCl36)   
        if element == "ClO51":
            ClO51 = data['ClO+ mass 50.9632']
            output = output.assign(ClO51 = ClO51)  
        if element == "ClO53":
            ClO53 = data['[37Cl]O+ mass 52.9603']
            output = output.assign(ClO53 = ClO53)    
        
        
        if element == "Sc45":
            Sc45 = data['[45Sc]+ mass 44.9554']
            output = output.assign(Sc45 = Sc45)
            
            
        if element == "Ti46":
            Ti46 = data['[46Ti]+ mass 45.9521']
            output = output.assign(Ti46 = Ti46)
        if element == "Ti47":
            Ti47 = data['[47Ti]+ mass 46.9512']
            output = output.assign(Ti47 = Ti47)  
        if element == "Ti48":
            Ti48 = data['[48Ti]+ mass 47.9474']
            output = output.assign(Ti48 = Ti48)   
        if element == "Ti49":
            Ti49 = data['[49Ti]+ mass 48.9473']
            output = output.assign(Ti49 = Ti49)  
        if element == "Ti50":
            Ti50 = data['[50Ti]+ mass 49.9442']
            output = output.assign(Ti50 = Ti50) 
            
            
        if element == "Ga69":
            Ga69 = data['[69Ga]+ mass 68.925']
            output = output.assign(Ga69 = Ga69)  
        if element == "Ga71":
            Ga71 = data['[71Ga]+ mass 70.9241']
            output = output.assign(Ga71 = Ga71)
            
            
        if element == "Ar36":
            Ar36 = data['[36Ar]+ mass 35.967']
            output = output.assign(Ar36 = Ar36)
        if element == "Ar38":
            Ar38 = data['[38Ar]+ mass 37.9622']
            output = output.assign(Ar38 = Ar38)  
        if element == "Ar40":
            Ar40 = data['[40Ar]+ mass 39.9618']
            output = output.assign(Ar40 = Ar40)   
        if element == "ArH37":
            ArH37 = data['[36Ar]H+ mass 36.9748']
            output = output.assign(ArH37 = ArH37)  
        if element == "ArH39":
            ArH39 = data['[38Ar]H+ mass 38.97']
            output = output.assign(ArH39 = ArH39)    
        if element == "ArH41":
            ArH41 = data['ArH+ mass 40.9697']
            output = output.assign(ArH41 = ArH41)
        if element == "ArH242":
            ArH242 = data['ArH2+ mass 41.9775']
            output = output.assign(ArH242 = ArH242)  
        if element == "ArO52":
            ArO52 = data['[36Ar]O+ mass 51.9619']
            output = output.assign(ArO52 = ArO52)   
        if element == "ArN54":
            ArN54 = data['ArN+ mass 53.9649']
            output = output.assign(ArN54 = ArN54)  
        if element == "ArO56":
            ArO56 = data['ArO+ mass 55.9567']
            output = output.assign(ArO56 = ArO56)    
        if element == "ArOH57":
            ArOH57 = data['ArOH+ mass 56.9646']
            output = output.assign(ArOH57 = ArOH57)
        if element == "Ar280":
            Ar280 = data['Ar2+ mass 79.9242']
            output = output.assign(Ar280 = Ar280)  
        if element == "ArCl":
            ArCl = data['ArCl+ mass 74.9307']
            output = output.assign(ArCl = ArCl)   
        if element == "Ar276":
            Ar276 = data['Ar[36Ar]+ mass 75.9294']
            output = output.assign(Ar276 = Ar276)  
        if element == "ArCl77":
            ArCl77 = data['Ar[37Cl]+ mass 76.9277']
            output = output.assign(ArCl77 = ArCl77)    
        if element == "Ar278":
            Ar278 = data['Ar[38Ar]+ mass 77.9246']
            output = output.assign(Ar278 = Ar278)    
            
            
        if element == "Ge70":
            Ge70 = data['[70Ge]+ mass 69.9237']
            output = output.assign(Ge70 = Ge70)  
        if element == "Ge72":
            Ge72 = data['[72Ge]+ mass 71.9215']
            output = output.assign(Ge72 = Ge72)   
        if element == "Ge73":
            Ge73 = data['[73Ge]+ mass 72.9229']
            output = output.assign(Ge73 = Ge73)  
        if element == "Ge74":
            Ge74 = data['[74Ge]+ mass 73.9206']
            output = output.assign(Ge74 = Ge74)    
        if element == "Ge76":
            Ge76 = data['[76Ge]+ mass 75.9209']
            output = output.assign(Ge76 = Ge76)    
            
        
        if element == "Co59":
            Co59 = data['[59Co]+ mass 58.9327']
            output = output.assign(Co59 = Co59)
            
        
        if element == "Se74":
            Se74 = data['[74Se]+ mass 73.9219']
            output = output.assign(Se74 = Se74)
        if element == "Se76":
            Se76 = data['[76Se]+ mass 75.9187']
            output = output.assign(Se76 = Se76)  
        if element == "Se77":
            Se77 = data['[77Se]+ mass 76.9194']
            output = output.assign(Se77 = Se77)   
        if element == "Se78":
            Se78 = data['[78Se]+ mass 77.9168']
            output = output.assign(Se78 = Se78)  
        if element == "Se80":
            Se80 = data['[80Se]+ mass 79.916']
            output = output.assign(Se80 = Se80)    
        if element == "Se82":
            Se82 = data['[82Se]+ mass 81.9162']
            output = output.assign(Se82 = Se82) 
            
            
        if element == "Kr78":
            Kr78 = data['[78Kr]+ mass 77.9198']
            output = output.assign(Kr78 = Kr78)
        if element == "Kr80":
            Kr80 = data['[80Kr]+ mass 79.9158']
            output = output.assign(Kr80 = Kr80)  
        if element == "Kr82":
            Kr82 = data['[82Kr]+ mass 81.9129']
            output = output.assign(Kr82 = Kr82)   
        if element == "Kr83":
            Kr83 = data['[83Kr]+ mass 82.9136']
            output = output.assign(Kr83 = Kr83)  
        if element == "Kr84":
            Kr84 = data['[84Kr]+ mass 83.911']
            output = output.assign(Kr84 = Kr84)    
        if element == "Kr86":
            Kr86 = data['[86Kr]+ mass 85.9101']
            output = output.assign(Kr86 = Kr86) 
        
        
        if element == "Br79":
            Br79 = data['[79Br]+ mass 78.9178']
            output = output.assign(Br79 = Br79)    
        if element == "Br81":
            Br81 = data['[81Br]+ mass 80.9157']
            output = output.assign(Br81 = Br81)
        
        
        if element == "Rb85":
            Rb85 = data['[85Rb]+ mass 84.9112']
            output = output.assign(Rb85 = Rb85)    
        if element == "Rb87":
            Rb87 = data['[87Rb]+ mass 86.9086']
            output = output.assign(Rb87 = Rb87)
        
        
        if element == "Y89":
            Y89 = data['[89Y]+ mass 88.9053']
            output = output.assign(Y89 = Y89)
        
        
        if element == "Zr90":
            Zr90 = data['[90Zr]+ mass 89.9042']
            output = output.assign(Zr90 = Zr90)  
        if element == "Zr91":
            Zr91 = data['[91Zr]+ mass 90.9051']
            output = output.assign(Zr91 = Zr91)   
        if element == "Zr92":
            Zr92 = data['[92Zr]+ mass 91.9045']
            output = output.assign(Zr92 = Zr92)  
        if element == "Zr94":
            Zr94 = data['[94Zr]+ mass 93.9058']
            output = output.assign(Zr94 = Zr94)    
        if element == "Zr96":
            Zr96 = data['[96Zr]+ mass 95.9077']
            output = output.assign(Zr96 = Zr96)
            
            
        if element == "Nb93":
            Nb93 = data['[93Nb]+ mass 92.9058']
            output = output.assign(Nb93 = Nb93)
            
            
        if element == "Ru96":
            Ru96 = data['[96Ru]+ mass 95.9071']
            output = output.assign(Ru96 = Ru96)  
        if element == "Ru98":
            Ru98 = data['[98Ru]+ mass 97.9047']
            output = output.assign(Ru98 = Ru98)   
        if element == "Ru99":
            Ru99 = data['[99Ru]+ mass 98.9054']
            output = output.assign(Ru99 = Ru99)  
        if element == "Ru100":
            Ru100 = data['[100Ru]+ mass 99.9037']
            output = output.assign(Ru100 = Ru100)    
        if element == "Ru101":
            Ru101 = data['[101Ru]+ mass 100.905']
            output = output.assign(Ru101 = Ru101)
        if element == "Ru102":
            Ru102 = data['[102Ru]+ mass 101.904']
            output = output.assign(Ru102 = Ru102)    
        if element == "Ru104":
            Ru104 = data['[104Ru]+ mass 103.905']
            output = output.assign(Ru104 = Ru104)
            
            
        if element == "Pd102":
            Pd102 = data['[102Pd]+ mass 101.905']
            output = output.assign(Pd102 = Pd102)  
        if element == "Pd104":
            Pd104 = data['[104Pd]+ mass 103.903']
            output = output.assign(Pd104 = Pd104)   
        if element == "Pd105":
            Pd105 = data['[105Pd]+ mass 104.905']
            output = output.assign(Pd105 = Pd105)  
        if element == "Pd106":
            Pd106 = data['[106Pd]+ mass 105.903']
            output = output.assign(Pd106 = Pd106)    
        if element == "Pd108":
            Pd108 = data['[108Pd]+ mass 107.903']
            output = output.assign(Pd108 = Pd108)
        if element == "Pd110":
            Pd110 = data['[110Pd]+ mass 109.905']
            output = output.assign(Pd110 = Pd110)
            
            
        if element == "Rh103":
            Rh103 = data['[103Rh]+ mass 102.905']
            output = output.assign(Rh103 = Rh103)  
            
            
        if element == "Ag107":
            Ag107 = data['[107Ag]+ mass 106.905']
            output = output.assign(Ag107 = Ag107)
        if element == "Ag109":
            Ag109 = data['[109Ag]+ mass 108.904']
            output = output.assign(Ag109 = Ag109)  
            
            
        if element == "Sn112":
            Sn112 = data['[112Sn]+ mass 111.904']
            output = output.assign(Sn112 = Sn112)   
        if element == "Sn114":
            Sn114 = data['[114Sn]+ mass 113.902']
            output = output.assign(Sn114 = Sn114)  
        if element == "Sn115":
            Sn115 = data['[115Sn]+ mass 114.903']
            output = output.assign(Sn115 = Sn115)    
        if element == "Sn116":
            Sn116 = data['[116Sn]+ mass 115.901']
            output = output.assign(Sn116 = Sn116)
        if element == "Sn117":
            Sn117 = data['[117Sn]+ mass 116.902']
            output = output.assign(Sn117 = Sn117)  
        if element == "Sn118":
            Sn118 = data['[118Sn]+ mass 117.901']
            output = output.assign(Sn118 = Sn118)   
        if element == "Sn119":
            Sn119 = data['[119Sn]+ mass 118.903']
            output = output.assign(Sn119 = Sn119)  
        if element == "Sn120":
            Sn120 = data['[120Sn]+ mass 119.902']
            output = output.assign(Sn120 = Sn120)    
        if element == "Sn122":
            Sn122 = data['[122Sn]+ mass 121.903']
            output = output.assign(Sn122 = Sn122)
        if element == "Sn124":
            Sn124 = data['[124Sn]+ mass 123.905']
            output = output.assign(Sn124 = Sn124)  
            
            
        if element == "In113":
            In113 = data['[113In]+ mass 112.904']
            output = output.assign(In113 = In113)
        if element == "In115":
            In115 = data['[115In]+ mass 114.903']
            output = output.assign(In115 = In115)  
            
            
        if element == "Sb121":
            Sb121 = data['[121Sb]+ mass 120.903']
            output = output.assign(Sb121 = Sb121)   
        if element == "Sb123":
            Sb123 = data['[123Sb]+ mass 122.904']
            output = output.assign(Sb123 = Sb123)
            
            
        if element == "Te120":
            Te120 = data['[120Te]+ mass 119.903']
            output = output.assign(Te120 = Te120)    
        if element == "Te122":
            Te122 = data['[122Te]+ mass 121.902']
            output = output.assign(Te122 = Te122)
        if element == "Te123":
            Te123 = data['[123Te]+ mass 122.904']
            output = output.assign(Te123 = Te123)  
        if element == "Te124":
            Te124 = data['[124Te]+ mass 123.902']
            output = output.assign(Te124 = Te124)   
        if element == "Te125":
            Te125 = data['[125Te]+ mass 124.904']
            output = output.assign(Te125 = Te125)  
        if element == "Te126":
            Te126 = data['[126Te]+ mass 125.903']
            output = output.assign(Te126 = Te126)    
        if element == "Te128":
            Te128 = data['[128Te]+ mass 127.904']
            output = output.assign(Te128 = Te128)
        if element == "Te130":
            Te130 = data['[130Te]+ mass 129.906']
            output = output.assign(Te130 = Te130)
            
            
        if element == "Xe124":
            Xe124 = data['[124Xe]+ mass 123.905']
            output = output.assign(Xe124 = Xe124)  
        if element == "Xe126":
            Xe126 = data['[126Xe]+ mass 125.904']
            output = output.assign(Xe126 = Xe126)    
        if element == "Xe128":
            Xe128 = data['[128Xe]+ mass 127.903']
            output = output.assign(Xe128 = Xe128)
        if element == "Xe129":
            Xe129 = data['[129Xe]+ mass 128.904']
            output = output.assign(Xe129 = Xe129)  
        if element == "Xe130":
            Xe130 = data['[130Xe]+ mass 129.903']
            output = output.assign(Xe130 = Xe130)   
        if element == "Xe131":
            Xe131 = data['[131Xe]+ mass 130.905']
            output = output.assign(Xe131 = Xe131)  
        if element == "Xe132":
            Xe132 = data['[132Xe]+ mass 131.904']
            output = output.assign(Xe132 = Xe132)    
        if element == "Xe134":
            Xe134 = data['[134Xe]+ mass 133.905']
            output = output.assign(Xe134 = Xe134)
        if element == "Xe136":
            Xe136 = data['[136Xe]+ mass 135.907']
            output = output.assign(Xe136 = Xe136)
            
            
        if element == "I127":
            I127 = data['[127I]+ mass 126.904']
            output = output.assign(I127 = I127)  
            
            
        if element == "Cs133":
            Cs133 = data['[133Cs]+ mass 132.905']
            output = output.assign(Cs133 = Cs133)   
            
            
        if element == "Ce136":
            Ce136 = data['[136Ce]+ mass 135.907']
            output = output.assign(Ce136 = Ce136)
        if element == "Ce138":
            Ce138 = data['[138Ce]+ mass 137.905']
            output = output.assign(Ce138 = Ce138)  
        if element == "Ce140":
            Ce140 = data['[140Ce]+ mass 139.905']
            output = output.assign(Ce140 = Ce140)   
        if element == "Ce142":
            Ce142 = data['[142Ce]+ mass 141.909']
            output = output.assign(Ce142 = Ce142)  
            
        if element == "CeO156":
            CeO156 = data['CeO+ mass 155.9']
            output = output.assign(CeO156 = CeO156)  
            
            
        if element == "La138":
            La138 = data['[138La]+ mass 137.907']
            output = output.assign(La138 = La138)
        if element == "La139":
            La139 = data['[139La]+ mass 138.906']
            output = output.assign(La139 = La139)
            
            
        if element == "Pr141":
            Pr141 = data['[141Pr]+ mass 140.907']
            output = output.assign(Pr141 = Pr141)
            
            
        if element == "Nd142":
            Nd142 = data['[142Nd]+ mass 141.907']
            output = output.assign(Nd142 = Nd142)
        if element == "Nd143":
            Nd143 = data['[143Nd]+ mass 142.909']
            output = output.assign(Nd143 = Nd143)  
        if element == "Nd144":
            Nd144 = data['[144Nd]+ mass 143.91']
            output = output.assign(Nd144 = Nd144)   
        if element == "Nd145":
            Nd145 = data['[145Nd]+ mass 144.912']
            output = output.assign(Nd145 = Nd145)             
        if element == "Nd146":
            Nd146 = data['[146Nd]+ mass 145.913']
            output = output.assign(Nd146 = Nd146)           
        if element == "Nd148":
            Nd148 = data['[148Nd]+ mass 147.916']
            output = output.assign(Nd148 = Nd148)
        if element == "Nd150":
            Nd150 = data['[150Nd]+ mass 149.92']
            output = output.assign(Nd150 = Nd150)
            
            
        if element == "Sm144":
            Sm144 = data['[144Sm]+ mass 143.911']
            output = output.assign(Sm144 = Sm144)
        if element == "Sm147":
            Sm147 = data['[147Sm]+ mass 146.914']
            output = output.assign(Sm147 = Sm147)  
        if element == "Sm148":
            Sm148 = data['[148Sm]+ mass 147.914']
            output = output.assign(Sm148 = Sm148)   
        if element == "Sm149":
            Sm149 = data['[149Sm]+ mass 148.917']
            output = output.assign(Sm149 = Sm149)             
        if element == "Sm150":
            Sm150 = data['[150Sm]+ mass 149.917']
            output = output.assign(Sm150 = Sm150)           
        if element == "Sm152":
            Sm152 = data['[152Sm]+ mass 151.919']
            output = output.assign(Sm152 = Sm152)
        if element == "Sm154":
            Sm154 = data['[154Sm]+ mass 153.922']
            output = output.assign(Sm154 = Sm154)
            
            
        if element == "Eu151":
            Eu151 = data['[151Eu]+ mass 150.919']
            output = output.assign(Eu151 = Eu151)
        if element == "Eu153":
            Eu153 = data['[153Eu]+ mass 152.921']
            output = output.assign(Eu153 = Eu153)
            
            
        if element == "Gd152":
            Gd152 = data['[152Gd]+ mass 151.919']
            output = output.assign(Gd152 = Gd152)
        if element == "Gd154":
            Gd154 = data['[154Gd]+ mass 153.92']
            output = output.assign(Gd154 = Gd154)  
        if element == "Gd155":
            Gd155 = data['[155Gd]+ mass 154.922']
            output = output.assign(Gd155 = Gd155)   
        if element == "Gd156":
            Gd156 = data['[156Gd]+ mass 155.922']
            output = output.assign(Gd156 = Gd156)             
        if element == "Gd157":
            Gd157 = data['[157Gd]+ mass 156.923']
            output = output.assign(Gd157 = Gd157)           
        if element == "Gd158":
            Gd158 = data['[158Gd]+ mass 157.924']
            output = output.assign(Gd158 = Gd158)
        if element == "Gd160":
            Gd160 = data['[160Gd]+ mass 159.927']
            output = output.assign(Gd160 = Gd160)
            
            
        if element == "Dy156":
            Dy156 = data['[156Dy]+ mass 155.924']
            output = output.assign(Dy156 = Dy156)
        if element == "Dy158":
            Dy158 = data['[158Dy]+ mass 157.924']
            output = output.assign(Dy158 = Dy158)  
        if element == "Dy160":
            Dy160 = data['[160Dy]+ mass 159.925']
            output = output.assign(Dy160 = Dy160)   
        if element == "Dy161":
            Dy161 = data['[161Dy]+ mass 160.926']
            output = output.assign(Dy161 = Dy161)             
        if element == "Dy162":
            Dy162 = data['[162Dy]+ mass 161.926']
            output = output.assign(Dy162 = Dy162)           
        if element == "Dy163":
            Dy163 = data['[163Dy]+ mass 162.928']
            output = output.assign(Dy163 = Dy163)
        if element == "Dy164":
            Dy164 = data['[164Dy]+ mass 163.929']
            output = output.assign(Dy164 = Dy164)
            
            
        if element == "Tb159":
            Tb159 = data['[159Tb]+ mass 158.925']
            output = output.assign(Tb159 = Tb159)
            
            
        if element == "Er162":
            Er162 = data['[162Er]+ mass 161.928']
            output = output.assign(Er162 = Er162)  
        if element == "Er164":
            Er164 = data['[164Er]+ mass 163.929']
            output = output.assign(Er164 = Er164)   
        if element == "Er166":
            Er166 = data['[166Er]+ mass 165.93']
            output = output.assign(Er166 = Er166)             
        if element == "Er167":
            Er167 = data['[167Er]+ mass 166.932']
            output = output.assign(Er167 = Er167)           
        if element == "Er168":
            Er168 = data['[168Er]+ mass 167.932']
            output = output.assign(Er168 = Er168)
        if element == "Er170":
            Er170 = data['[170Er]+ mass 169.935']
            output = output.assign(Er170 = Er170)
            
            
        if element == "Ho165":
            Ho165 = data['[165Ho]+ mass 164.93']
            output = output.assign(Ho165 = Ho165)
            
            
        if element == "Yb168":
            Yb168 = data['[168Yb]+ mass 167.933']
            output = output.assign(Yb168 = Yb168)  
        if element == "Yb170":
            Yb170 = data['[170Yb]+ mass 169.934']
            output = output.assign(Yb170 = Yb170)   
        if element == "Yb171":
            Yb171 = data['[171Yb]+ mass 170.936']
            output = output.assign(Yb171 = Yb171)             
        if element == "Yb172":
            Yb172 = data['[172Yb]+ mass 171.936']
            output = output.assign(Yb172 = Yb172)           
        if element == "Yb173":
            Yb173 = data['[173Yb]+ mass 172.938']
            output = output.assign(Yb173 = Yb173)
        if element == "Yb174":
            Yb174 = data['[174Yb]+ mass 173.938']
            output = output.assign(Yb174 = Yb174)    
        if element == "Yb176":
            Yb176 = data['[176Yb]+ mass 175.942']
            output = output.assign(Yb176 = Yb176)
            
            
        if element == "Tm169":
            Tm169 = data['[169Tm]+ mass 168.934']
            output = output.assign(Tm169 = Tm169)  
            
            
        if element == "Hf174":
            Hf174 = data['[174Hf]+ mass 173.939']
            output = output.assign(Hf174 = Hf174)    
        if element == "Hf176":
            Hf176 = data['[176Hf]+ mass 175.941']
            output = output.assign(Hf176 = Hf176)   
        if element == "Hf177":
            Hf177 = data['[177Hf]+ mass 176.943']
            output = output.assign(Hf177 = Hf177)             
        if element == "Hf178":
            Hf178 = data['[178Hf]+ mass 177.943']
            output = output.assign(Hf178 = Hf178)           
        if element == "Hf179":
            Hf179 = data['[179Hf]+ mass 178.945']
            output = output.assign(Hf179 = Hf179)
        if element == "Hf180":
            Hf180 = data['[180Hf]+ mass 179.946']
            output = output.assign(Hf180 = Hf180)
            
            
        if element == "Lu175":
            Lu175 = data['[175Lu]+ mass 174.94']
            output = output.assign(Lu175 = Lu175)  
        if element == "Lu176":
            Lu176 = data['[176Lu]+ mass 175.942']
            output = output.assign(Lu176 = Lu176)  
            
            
        if element == "W180":
            W180 = data['[180W]+ mass 179.946']
            output = output.assign(W180 = W180)             
        if element == "W182":
            W182 = data['[182W]+ mass 181.948']
            output = output.assign(W182 = W182)           
        if element == "W183":
            W183 = data['[183W]+ mass 182.95']
            output = output.assign(W183 = W183)
        if element == "W184":
            W184 = data['[184W]+ mass 183.95']
            output = output.assign(W184 = W184)    
        if element == "W186":
            W186 = data['[186W]+ mass 185.954']
            output = output.assign(W186 = W186)
            
            
        if element == "Ta180":
            Ta180 = data['[180Ta]+ mass 179.947']
            output = output.assign(Ta180 = Ta180)  
        if element == "Ta181":
            Ta181 = data['[181Ta]+ mass 180.947']
            output = output.assign(Ta181 = Ta181) 
            
            
        if element == "Os184":
            Os184 = data['[184Os]+ mass 183.952']
            output = output.assign(Os184 = Os184)   
        if element == "Os186":
            Os186 = data['[186Os]+ mass 185.953']
            output = output.assign(Os186 = Os186)             
        if element == "Os187":
            Os187 = data['[187Os]+ mass 186.955']
            output = output.assign(Os187 = Os187)           
        if element == "Os188":
            Os188 = data['[188Os]+ mass 187.955']
            output = output.assign(Os188 = Os188)
        if element == "Os189":
            Os189 = data['[189Os]+ mass 188.958']
            output = output.assign(Os189 = Os189)
        if element == "Os190":
            Os190 = data['[190Os]+ mass 189.958']
            output = output.assign(Os190 = Os190)  
        if element == "Os192":
            Os192 = data['[192Os]+ mass 191.961']
            output = output.assign(Os192 = Os192) 
            
            
        if element == "Re185":
            Re185 = data['[185Re]+ mass 184.952']
            output = output.assign(Re185 = Re185) 
        if element == "Re187":
            Re187 = data['[187Re]+ mass 186.955']
            output = output.assign(Re187 = Re187)  
            
            
        if element == "Pt190":
            Pt190 = data['[190Pt]+ mass 189.959']
            output = output.assign(Pt190 = Pt190) 
        if element == "Pt192":
            Pt192 = data['[192Pt]+ mass 191.96']
            output = output.assign(Pt192 =Pt192)             
        if element == "Pt194":
            Pt194 = data['[194Pt]+ mass 193.962']
            output = output.assign(Pt194 = Pt194)           
        if element == "Pt195":
            Pt195 = data['[195Pt]+ mass 194.964']
            output = output.assign(Pt195 = Pt195)
        if element == "Pt196":
            Pt196 = data['[196Pt]+ mass 195.964']
            output = output.assign(Pt196 = Pt196)    
        if element == "Pt198":
            Pt198 = data['[198Pt]+ mass 197.967']
            output = output.assign(Pt198 = Pt198)
            
            
        if element == "Ir191":
            Ir191 = data['[191Ir]+ mass 190.96']
            output = output.assign(Ir191 = Ir191)  
        if element == "Ir193":
            Ir193 = data['[193Ir]+ mass 192.962']
            output = output.assign(Ir193 = Ir193) 
            
            
        if element == "Hg196":
            Hg196 = data['[196Hg]+ mass 195.965']
            output = output.assign(Hg196 = Hg196)   
        if element == "Hg198":
            Hg198 = data['[198Hg]+ mass 197.966']
            output = output.assign(Hg198 = Hg198)             
        if element == "Hg199":
            Hg199 = data['[199Hg]+ mass 198.968']
            output = output.assign(Hg199 = Hg199)           
        if element == "Hg200":
            Hg200 = data['[200Hg]+ mass 199.968']
            output = output.assign(Hg200 = Hg200)
        if element == "Hg201":
            Hg201 = data['[201Hg]+ mass 200.97']
            output = output.assign(Hg201 = Hg201)
        if element == "Hg202":
            Hg202 = data['[202Hg]+ mass 201.97']
            output = output.assign(Hg202 = Hg202)  
        if element == "Hg204":
            Hg204 = data['[204Hg]+ mass 203.973']
            output = output.assign(Hg204 = Hg204)
            
            
        if element == "Au197":
            Au197 = data['[197Au]+ mass 196.966']
            output = output.assign(Au197 = Au197)   
            
            
        if element == "Tl203":
            Tl203 = data['[203Tl]+ mass 202.972']
            output = output.assign(Tl203 = Tl203)             
        if element == "Tl205":
            Tl205 = data['[205Tl]+ mass 204.974']
            output = output.assign(Tl205 = Tl205)   
            
            
        if element == "Bi209":
            Bi209 = data['[209Bi]+ mass 208.98']
            output = output.assign(Bi209 = Bi209)
            
            
        if element == "Th232":
            Th232 = data['[232Th]+ mass 232.038']
            output = output.assign(Th232 = Th232)
            
        if element == "ThO248":
            ThO248 = data['ThO+ mass 248.032']
            output = output.assign(ThO248 = ThO248)  
            
        if element == "Background":
            Background = data['[220Bkg]+ mass 220']
            output = output.assign(Background = Background)
            
            
        if element == "All":
            output = data
            
    fig = plt.figure(figsize =(15,5))
    ax = fig.add_subplot(1,1,1)
    ax.set_title("TRA")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Intensity (cps)")
    
    for element in elements:
        ax.plot(data['time (s)'],output[element], alpha = 0.8, linewidth = 0.5)
    ax.legend(output)
    
    return output


