# -*- coding: utf-8 -*-
"""
Script for future prediction
- long term trend of electricity consumption
- inter-daily variations of electricity consumption profiles
- long term trend of photo-voltaic (PV) adaptation
- bootstrapped heat-pump (HP) load profile
- bootstrapped electric vehicle (EV) load profile
- assembly of electriciy consumption, PV, HP, EV profiles of specific node
"""

# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)

import os
from random import shuffle

from Modules import heatpump_generator
from Modules import emobility_generator
from Modules import holidays

#%% 
#plt.close('all')

Nyears = 500 # bootstrapping
n_bins = 35 # density plots
n_cluster = 25 # clusters
n_features = 31

# node ID
# node = ['LJ03510017', 'LJ03510222', 'LJ03510283', 'LJ03512711', 'LJ03512971', 'LJ03513000']
node = ['LJ03513000']
node_norm = np.ndarray([len(node)], dtype = object)
node_cluster = np.ndarray([len(node)], dtype = object)
nodex = np.ndarray([len(node)], dtype = object)

# days hours to investigate
hours = np.array([18, 20])* 4
#np.array([10, 12, 14, 16, 18, 20])* 4

# noise
noise_switch = 1

# PV
pv_switch = 1

# HP
hp_switch = 1

# EV
ev_switch = 1

#%%%%%%%%%%%%%%%%%%%%%%%%% Future Scenarios %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# future demand (ETH study 2011) [year (2015, 2020, 2035), scenario] -> cubic spline
#E_future = np.array([[66, 67, 66], [66, 70, 75], [66, 73, 82]])
#E_year = np.array([2016, 2020, 2035])
E_future = np.array([[66. , 66.3, 66.6, 66.8, 67. , 67.2, 67.3, 67.3, 67.4, 67.4, 67.3,
                          67.2, 67.2, 67. , 66.9, 66.7, 66.6, 66.4, 66.2, 66. ],
                     [66. , 67.1, 68.1, 69.1, 70. , 70.8, 71.5, 72.1, 72.7, 73.1, 73.5,
                          73.8, 74.1, 74.3, 74.5, 74.6, 74.8, 74.9, 74.9, 75.],
                     [66. , 67.9, 69.7, 71.4, 73. , 74.4, 75.7, 76.8, 77.7, 78.5, 79.2,
                          79.8, 80.3, 80.7, 81. , 81.3, 81.5, 81.7, 81.9, 82.]])

# future production PV cell (2015, 2020, 2035) [TWh] -> cubic spline
#PV_future = np.array([[1.55, 1.9 , 5.5], [1.55, 2.5, 8]])
#PV_year = np.array([2016, 2020, 2035])
PV_future = np.array([[1.6, 1.6, 1.7, 1.8, 1.9, 2. , 2.2, 2.4, 2.6, 2.8, 3. , 3.2, 3.5,
                           3.8, 4. , 4.3, 4.6, 4.9, 5.2, 5.5],
                      [1.6, 1.8, 2. , 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.1, 4.4, 4.8, 5.2,
                           5.5, 5.9, 6.3, 6.8, 7.2, 7.6, 8.]])

#%%%%%%%%%%%%%%%%%%%%%%%%%%% load solution %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path_clustering = './../2Clustering/Data/'
path_fit = './../3Least_Squares/Data/'

path_weather = './../1Raw_Data/Weather/'
path_pv = './../3PV/Data/'

path_prediction = './../4Prediction/Data/'
path_prediction_plot = './../4Prediction/Plots/'

title_ext = '' # title extension i.e. Noise, PV, HP, EV

# PV ------------------------------------------
# PV solution vector
PV_w = np.load(path_pv + 'solution_vector_PV.npy')

# Array for prediciton of PV production [#bootstrapps, #data pts in year]
PV_pred = np.zeros([Nyears, 365* 96])

# heat pump
HP_profiles = heatpump_generator.hp_generator(Nyears)

if ev_switch == 1:
    N_EV = 76
    EV_activate = np.arange(0, N_EV) # random order for which EV included next
    shuffle(EV_activate)                    
    EV_portfolio = np.array([0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26])
    EV_profiles = emobility_generator.emob_generator(EV_activate[:EV_portfolio[-1]], Nyears)

#%%
if noise_switch == 1:
    title_ext = title_ext+ 'Noise'
    
if hp_switch == 1:
    title_ext = 'HeatPump_'+ title_ext

if ev_switch == 1:
    title_ext = 'EVehicle_'+ title_ext
    
if pv_switch == 1:
    title_ext = 'Solar_'+ title_ext
  
    
#%% normalization of profiles
# extract normalization  
df = pd.read_csv(path_clustering+ 'Cluster_normalization.csv', sep= ';')
ID_cluster = df.values[:, 1]

# normalization
Profile_mean = df.values[:, 2]

# cluster assignmetn
ID, clust_assignment = np.load(path_clustering+ 'map_ID_cluster.npy')

for no in range(len(node)): 
    # array size
    iter0 = 0
    
    for i in range(len(ID_cluster)):
        if list(ID_cluster)[i][7:17] == node[no]:
            iter0+= 1
    
    print('iter: '+ str(iter0))
            
    node_norm[no] = np.empty([iter0, 2])
    node_cluster[no] = np.empty([iter0, 2])
    
    # normalization
    iter0 = 0
    
    for i in range(len(ID_cluster)):
        if list(ID_cluster)[i][7:17] == node[no]:
            # node_norm[iter0] = np.array([ID_cluster[i][0:6], Profile_min[i], Profile_max[i]])
            node_norm[no][iter0] = np.array([ID_cluster[i][0:6], Profile_mean[i]])
            iter0+= 1
            
    print('iter: '+ str(iter0))
        
    # extract cluster assignment
    iter0 = 0
    
    for i in range(len(ID)):
        if list(ID)[i][7:17] == node[no]:
            node_cluster[no][iter0]= np.array([ID[i][0:6], clust_assignment[i]])
            iter0+= 1
    
    print('iter: '+ str(iter0))
        
    # grid node -> # profiles from each cluster
    nodex[no] = np.bincount(np.int16(node_cluster[no][:, 1]))

#%%%%%%%%%%%%%%%%%%%%%% Holidays and Weekdays %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# length year (time series)
N_TS = 365

# arrays for individual week days
week_day1 = np.zeros([N_TS, 1])
week_day2 = np.zeros([N_TS, 1])
week_day3 = np.zeros([N_TS, 1])
week_day4 = np.zeros([N_TS, 1])
week_day5 = np.zeros([N_TS, 1])
week_day6 = np.zeros([N_TS, 1])
week_day7 = np.zeros([N_TS, 1])

# make sure that time series starts with monday! (for the usual interpretation of
# 2015 starts with a thursday
week_day1[4::7] = 1   # monday
week_day2[5::7] = 1   # tuesday
week_day3[6::7] = 1   # wednesday
week_day4[0::7] = 1   # thursday
week_day5[1::7] = 1   # friday
week_day6[2::7] = 1   # saturday
week_day7[3::7] = 1   # sunday

# holidays
holidays2015 = holidays.holidays(2015)
        
# holiday -> split into pre-holidays, holiday, post-holidays ------------------
hol_hol = np.zeros([len(holidays2015), 1])
hol_hol[np.where(holidays2015 ==0)] = 1
hol_prev = np.zeros([len(holidays2015), 1])
hol_prev[np.where(holidays2015 ==0.5)] = 1
hol_post = np.zeros([len(holidays2015), 1])
hol_post[np.where(holidays2015 ==0.8)] = 1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Read Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

day = 1
week = 7
month = 28

# hours -----------------------------------------------------------------------
for hr in hours:
    print("Hr = "+ str(int(hr/4))+ ":00")
    
    # [#clusters, #features]
    El_w = np.zeros([n_cluster, n_features])
    El_w_regr = np.zeros([n_cluster, n_features])
    El_feature = np.zeros([n_cluster, n_features])
    
    # autoregressive constant
    El_ar = np.zeros([n_cluster, 1])
    
    # noise added to prediction
    El_bins = np.zeros([n_cluster, n_bins])
    El_prob = np.zeros([n_cluster, n_bins]) 

    
    # clusters ----------------------------------------------------------------
    for filename in os.listdir(path_clustering):
        
        main = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[-1].lower()
        
        if ext != '.csv':
            break
        if 'normalization' in filename:
            break
        
        fn = filename[7:9]
        if fn[1] == '_':
            fn = fn[0]
        
        # load solution vector
        loadz = np.load(path_fit+ main+ '_hr_'+ str(int(hr/ 4))+ '_LS.npz')
        El_ar[int(fn)] = loadz['a'].reshape(1, -1) # autoregressive constant
        El_feature[int(fn)] = loadz['d'].reshape(1, -1) # vector of active features
        
        nonz_entry = np.nonzero(El_feature[int(fn)])[0].reshape(-1)
        El_w[int(fn)][nonz_entry] = loadz['b'] # solution vector [# clusters, # features] 
        El_w_regr[int(fn)][nonz_entry] = loadz['c'] # solution vector [# clusters, # features]  
        El_prob[int(fn)] = loadz['e'] # probability
        El_bins[int(fn)] = loadz['f'] # histogram bins

    # predicdiction [#clusters, #bootstrapps]
    El_pred = np.zeros([n_cluster, Nyears], dtype = object)
    
    # noise (sample error)
    El_noise = np.zeros([n_cluster, Nyears], dtype = object)
    
    # years -------------------------------------------------------------------
    for ny in range(Nyears):
        # weather data
        df = pd.read_csv(path_weather+ 'weather_data_year' + str(ny)+ '.csv', sep= ',')
        T = df['temperature'].values
        IR = df['irradiation'].values
        
        # PV prediction [#bootstrapps, time signal length]
        PV_dat = np.concatenate([IR.reshape(1, -1)], axis= 0).T
        PV_pred[ny] = np.dot(PV_dat, PV_w)
        
        #%%%%%%%%%%%%%%%%%%%%%%%% Temperature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
        # temperature fluctuations at t
        T_hr = T[hr::96]
        N_TS = len(T_hr)
        
        # for averaging processes (more than 1 day), need to append temperature
        # before time 0 since interval [t-T, t] is mapped to t
        # hence append end of year to beginning of year
        T_mt = np.concatenate((T[-28*96:], T))
            
        # irradiance
        IR_hr = IR[hr::96]
            
        # past 3 hours Temperature
        T_hr_neg1 = T[hr-1::96]
        T_hr_neg2 = T[hr-2::96]
        T_hr_neg3 = T[hr-4::96]
        T_hr_neg4 = T[hr-6::96]
        T_hr_neg5 = T[hr-8::96]
        T_hr_neg6 = T[hr-10::96]
        
        # past 3 hours Irradiance
        IR_hr_neg1 = IR[hr-1::96]
        IR_hr_neg2 = IR[hr-2::96]
        IR_hr_neg3 = IR[hr-4::96]
        IR_hr_neg4 = IR[hr-6::96]
        IR_hr_neg5 = IR[hr-8::96]
        IR_hr_neg6 = IR[hr-10::96]
        
        # min and max of last 24 hrs (avlues available at beginning of second day due to avg)
        T_hr_max24 = np.zeros([N_TS,  1])
        T_hr_min24 = np.zeros([N_TS,  1])
        T_hr_avg24 = np.zeros([N_TS,  1])
        T_hr_avg24_neg1 = np.zeros([N_TS,  1])
        T_hr_avg24_neg2 = np.zeros([N_TS,  1])
        T_hr_avg8 = np.zeros([N_TS, 1])
         
        for n in range(N_TS):
            # use temperature series extended by last 28 days of the year
            # to obtain a value for the first day, need to average between 
            # the day before the first day (month-1) and the first day
            idx1 = (month-1+ n)* 96+ hr
            idx2 = (month+ n)* 96+ hr
            T_hr_avg8[n] = np.average(np.float64(T_mt[idx1+ 64: idx2]))
            T_hr_max24[n] = np.max(T_mt[idx1: idx2])
            T_hr_min24[n] = np.min(T_mt[idx1: idx2])
            T_hr_avg24[n] = np.average(np.float64(T_mt[idx1: idx2]))
            T_hr_avg24_neg1[n] = np.average(np.float64(T_mt[idx1-96: idx2-96]))
            T_hr_avg24_neg2[n] = np.average(np.float64(T_mt[idx1-192: idx2-192]))

        # average of past 7 days
        T_hr_avg7 = np.zeros([N_TS,  1])
            
        for n in range(N_TS):
            idx1 = (month- week+ n)* 96+ hr
            idx2 = (month+ n)* 96+ hr
            T_hr_avg7[n] = np.average(np.float64(T_mt[idx1: idx2]))
            
        # average of past 28 days
        T_hr_avg28 = np.zeros([N_TS, 1])
        
        for n in range(N_TS):
            idx1 = (month- month+ n)* 96+ hr
            idx2 = (month+ n)* 96+ hr
            T_hr_avg28[n] = np.average(np.float64(T_mt[idx1: idx2]))
        
        #%%%%%%%%%%%%%%%%%%%%%%%%% Prediction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        # clusters
        for nc in range(n_cluster):
            
            # Consumption -> from day 1 until 365
            El_dat = np.concatenate([
                    T_hr.reshape(1, -1),            hol_hol.reshape(1, -1),
                    hol_prev.reshape(1, -1),        hol_post.reshape(1, -1),
                    week_day1.reshape(1,-1),        week_day2.reshape(1,-1),
                    week_day3.reshape(1,-1),        week_day4.reshape(1,-1),
                    week_day5.reshape(1,-1),        week_day6.reshape(1,-1),
                    week_day7.reshape(1,-1),
                    T_hr_neg1.reshape(1, -1),       T_hr_neg2.reshape(1, -1), 
                    T_hr_neg3.reshape(1, -1),       T_hr_neg4.reshape(1, -1), 
                    T_hr_neg5.reshape(1, -1),       T_hr_neg6.reshape(1, -1),
                    IR_hr_neg1.reshape(1, -1),      IR_hr_neg2.reshape(1, -1), 
                    IR_hr_neg3.reshape(1, -1),      IR_hr_neg4.reshape(1, -1), 
                    IR_hr_neg5.reshape(1, -1),
                    T_hr_avg24_neg1.reshape(1, -1), T_hr_avg24_neg2.reshape(1, -1),
                    T_hr_max24.reshape(1, -1),      T_hr_avg24.reshape(1, -1),
                    T_hr_min24.reshape(1, -1),      T_hr_avg8.reshape(1, -1),
                    T_hr_avg7.reshape(1, -1),       T_hr_avg28.reshape(1, -1),
                    np.ones([1, len(T_hr_avg28)])], axis= 0).T
            
            # transformation
            El_dat_new = El_dat[1:]- El_ar[nc]* El_dat[:-1]
            
            if noise_switch == 1:
                # noise probability
                py = El_prob[nc]/ np.sum(El_prob[nc])
                # draw from distribution
                El_noise[nc, ny] = np.random.choice(El_bins[nc], N_TS, p= py)

            else:
                El_noise[nc, ny] = np.zeros([N_TS])
            
            # [#clusters, #bootstrapps] = [#time series, #features] . [#clusters = fixed, #features]'
            Yt = np.zeros([N_TS])
            Yt[0] = np.dot(El_dat[0], El_w[nc])
            Vt = np.dot(El_dat_new, El_w_regr[nc])

            # demand with superimposed noise
            for i in range(1, Yt.shape[0]):
                Yt[i] = El_ar[nc]* Yt[i-1]+ Vt[i-1]+ El_noise[nc, ny][i]
                
            El_pred[nc, ny] = Yt

    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figure %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print("Figures")    
   
    #
    def plot_hist(series_low, series_med, series_high, title, year, node, title_ext):
        season = np.array([15, 100, 190, 280])
        season_str = ['Winter', 'Spring', 'Summer', 'Autumn']
        
        # plot for each season a figure containing the three scenarios ------------
        for k, ss in enumerate(season):
            _ , bin_edges = np.histogram(np.concatenate([series_low[:, ss],
                                                         series_med[:, ss],
                                                         series_high[:, ss]]))
             
            fig, ax = plt.subplots(1, 3, figsize=(12, 5))
            fig.suptitle('Electricity Consumption Histogram '+season_str[k] + ' for t = '+ str(int(hr/4))+ ':00 (' + title+ ')')
            
            ax[0].hist(series_low[:, ss].reshape(-1), bins= n_bins)
            ax[0].set_title('Tief')
            ax[0].set_xlabel('Consumption')
            ax[0].set_ylabel('Count')
            ax[1].hist(series_med[:, ss].reshape(-1), bins= n_bins)
            ax[1].set_title('Mittel')
            ax[1].set_xlabel('Consumption')
            ax[1].set_ylabel('Count')
            ax[2].hist(series_high[:, ss].reshape(-1), bins = n_bins)
            ax[2].set_title('Hoch')
            ax[2].set_xlabel('Consumption')
            ax[2].set_ylabel('Count')
        
            fig.show()
            
            path = path_prediction_plot+ 'Node'+ str(node)+ \
                        '/'+ title_ext+ '_Histogram_'+ str(year)+ '/'
            
            if not os.path.exists(path):
                os.makedirs(path)
                
            fig.savefig(path+ 'Hist_'+season_str[k]+ '_'+ title+ '_t'+ str(int(hr/4)))
            plt.close()
            
            
    def density_plot(ts, hr, scenario, title, year, node, title_ext):
        for d in [2, 7]:
            plt.figure()
            plt.boxplot(ts[:, d::7]* 1E-3)
            plt.xticks(np.arange(0, 52, 5), np.arange(0, 52, 5))
            plt.tick_params(labelright=True)
            plt.title('Density plot for Scenario '+ str(scenario)+ ' day '+ str(d)+ ' time ' +str(int(hr/4))+ ':00 (' + title+ ')')
            plt.xlabel('Time [week]')
            plt.ylabel('Demand [kW]')
            plt.grid(axis = 'y')
            plt.show()
            
            path = path_prediction_plot+ 'Node'+ str(node)+ \
                        '/'+ title_ext+ '_Density_'+ str(year)+ '/'+ scenario+ '/'
            
            if not os.path.exists(path):
                os.makedirs(path)
                
            plt.savefig(path+ 'Density_'+ title+ '_day'+ str(d)+ '_t'+ str(int(hr/4)))
            
            plt.close()
                               
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Node %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # HP load profile [#years, #HP, #data] -> [#years, #data] at given day time
    load_hr_HP = np.sum(HP_profiles[:, :, hr::96], axis = 1)
    
    # loop over nodes
    for no in range(len(node)):
        print('Node '+str(node[no]))
        
        # loop over years (2016 - 2030)
        for i in range(0, 1):
            future_year = 2015+ i
            print('year '+ str(future_year))
            
            # EV load profile [#years, #EV, #data] ----------------------------
            EV_profile_hr = np.sum(EV_profiles[:, :EV_portfolio[i], hr::96], axis = 1)
            
            # electricity demand ----------------------------------------------
            El_node_sz1 = np.zeros([Nyears, N_TS]) # szenario 1
            El_node_sz2 = np.zeros([Nyears, N_TS]) # szenario 2
            El_node_sz3 = np.zeros([Nyears, N_TS]) # szenario 3

            # add electricity consumption
            for k, cn in enumerate(nodex[no]):
                # k: cluster no, cn: #profiles
                
                # array from bootstrapping
                El_pred_boots = np.array([El_pred[k, i] for i in range(Nyears)])
                
                # find consumers belonging to cluster
                consumer = node_cluster[no][np.where(node_cluster[no][:, 1] == k)[0], 0]
                
                # [min, max]
                norm = np.zeros([len(consumer), 1])
                # extract normalization for found consumers
                for c in range(len(consumer)):
                    # normalization
                    norm[c, :] = node_norm[no][np.where(node_norm[no][:, 0] == consumer[c])[0], 1:]
                
                    # add up profiles                           [#clusters, #bootstrapps]                    
                    # log scaler
                    El_node_sz1+= (E_future[0, i]/ E_future[0, 0]* norm[c]* np.exp(El_pred_boots))
                    El_node_sz2+= (E_future[1, i]/ E_future[1, 0]* norm[c]* np.exp(El_pred_boots))
                    El_node_sz3+= (E_future[2, i]/ E_future[2, 0]* norm[c]* np.exp(El_pred_boots))
    
            # add heat pump ---------------------------------------------------
            if hp_switch == 1:
                # [#years, #data]
                El_node_sz1+= load_hr_HP
                El_node_sz2+= load_hr_HP
                El_node_sz3+= load_hr_HP
            
            # add electric vehicles -------------------------------------------
            if ev_switch == 1:
                # [#years, #data]
                El_node_sz1+= EV_profile_hr*1e3
                El_node_sz2+= EV_profile_hr*1e3
                El_node_sz3+= EV_profile_hr*1e3

            
            # PV scenario -----------------------------------------------------
            # electricity demand -> hourly model
            # PV demand -> model for complete time-series
            # Hence, need to extract part that corresponds to hourly model
            PV_hr = PV_pred[:, hr::96]
            pvmin = -2000
            
            El_pv_sz1 = pvmin* PV_future[0, i]/ PV_future[0, 0]* PV_hr[:]
            El_pv_sz2 =  pvmin* PV_future[1, i]/ PV_future[1, 0]* PV_hr[:]
               
            # -----------------------------------------------------------------
            # store solution to post-process (plot) in different script
            path = path_prediction+ 'Node'+str(node[no])+'/'
            
            if not os.path.exists(path):
                os.makedirs(path)
                
            np.save(path+ 'electricity_scenario_'+ 'tief'+ '_year_'+ str(2015+i)+ '_time_'+ str(int(hr/4)), El_node_sz1)
            np.save(path+ 'electricity_scenario_'+ 'mittel'+ '_year_'+ str(2015+i)+ '_time_'+ str(int(hr/4)), El_node_sz2)
            np.save(path+ 'electricity_scenario_'+ 'hoch'+ '_year_'+ str(2015+i)+ '_time_'+ str(int(hr/4)), El_node_sz3)
            
            np.save(path+ 'PV_scenario_'+ 'tief' + '_year_'+ str(2015+i)+ '_time_'+ str(int(hr/4)), El_pv_sz1)
            np.save(path+ 'PV_scenario_'+ 'hoch' + '_year_'+ str(2015+i)+ '_time_'+ str(int(hr/4)), El_pv_sz2)
            
            if pv_switch == 0:
                El_pv_sz1*= 0
                El_pv_sz2*= 0
            
            # plots
            plot_hist(El_node_sz1+ El_pv_sz1, El_node_sz2+ El_pv_sz1, El_node_sz3+ El_pv_sz1, 'lowPV', future_year, node[no], title_ext)
            plot_hist(El_node_sz1+ El_pv_sz2, El_node_sz2+ El_pv_sz2, El_node_sz3+ El_pv_sz2, 'highPV', future_year, node[no], title_ext)
        
            density_plot(El_node_sz1+ El_pv_sz1, hr, 'Tief', 'lowPV', future_year, node[no], title_ext)
            density_plot(El_node_sz2+ El_pv_sz1, hr, 'Mittel', 'lowPV', future_year, node[no], title_ext)
            density_plot(El_node_sz3+ El_pv_sz1, hr, 'Hoch', 'lowPV', future_year, node[no], title_ext)
            
            density_plot(El_node_sz1+ El_pv_sz2, hr, 'Tief', 'highPV', future_year, node[no], title_ext)
            density_plot(El_node_sz2+ El_pv_sz2, hr, 'Mittel', 'highPV', future_year, node[no], title_ext)
            density_plot(El_node_sz3+ El_pv_sz2, hr, 'Hoch', 'highPV', future_year, node[no], title_ext)
            