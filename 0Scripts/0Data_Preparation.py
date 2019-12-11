# -*- coding: utf-8 -*-
"""
Script to read raw smart meter time series, append to single file and write to numpy array
"""

# Imports
import pandas as pd
import numpy as np

#%%
def read_data(N_start= 6* 96, N_end= (6+ 850)* 96, file1= 1, filen= 6):
    # record starts with Tuesday -> skip first 6 days to start with Mondays
    
    path_read = './../1Raw_Data/SmartMeter/'
    path_save = './../1Raw_Data/SmartMeter/'
    
    print("First File read: "+ str(file1))
    print("Last File read: " + str(filen))
    print(" ")
    
    X = np.empty([N_end- N_start, 0])
    Id = np.empty([0])
    # t = np.linspace(0, N_end- N_start, N_end- N_start)
    
    # loop over files
    for file_idx in range(file1, filen):
        print("Current File read: " + str(file_idx))
        try:
            # read file
            df = pd.read_csv(path_read + "dataset"+ str(file_idx) +".csv", sep= ',')
            data = df.values
            
            if file_idx == file1:
                t = pd.to_datetime(df.time[N_start:N_end])
            
            # append data
            X = np.append(X, data[N_start:N_end, 1:], axis= 1)
                        
            Id = np.append(Id, list(df)[1:], axis = 0)
            
            del data, df
                
        except (FileNotFoundError):
            print('')
    
    # Saving the objects:
    np.save(path_save + 'X_1_6', X)
    np.save(path_save + 'ID_1_6', Id)
    np.save(path_save + 't_1_6', t)
    
    print(X.shape)
    
    del X, t
    
    pass

#%% main
read_data()
