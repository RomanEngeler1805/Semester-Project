# -*- coding: utf-8 -*-
"""
Script performing following tasks:
- outlier removal
- feature selection
- weigth vector of regularized linear regression
- feature selection based on RMSE
"""

# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
from statsmodels.tsa.ar_model import AR

from pathlib import Path

import time

#%%
from Modules import holidays

#%%
n_bins = 35

path_raw = './../1Raw_Data/'
path_clustering = './../2Clustering/Data/'
path_save = './../3Least_Squares/Data/'
path_performance = './../3Least_Squares/'
path_plot = './../3Least_Squares/Plots/'

# length of time series XXXXXXXXXXXXXXXXXXXXXXXX
N_TS = 850

#%%%%%%%%%%%%%%%%%%%%%% Holidays and Weekdays %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# holiday -> split into pre-holidays, holiday, post-holidays ------------------
hol = np.concatenate((holidays.holidays(2014)[31+ 28+ 31+ 6:],
                            holidays.holidays(2015),
                            holidays.holidays(2016)[:31+ 29+ 31+ 30+ 31 +30+ 31+ 3]), axis = 0)


hol_hol = np.zeros([len(hol), 1])
hol_hol[np.where(hol ==0)] = 1
hol_prev = np.zeros([len(hol), 1])
hol_prev[np.where(hol ==0.5)] = 1
hol_post = np.zeros([len(hol), 1])
hol_post[np.where(hol ==0.8)] = 1

# week days -------------------------------------------------------------------
week_day1 = np.zeros([N_TS, 1])
week_day2 = np.zeros([N_TS, 1])
week_day3 = np.zeros([N_TS, 1])
week_day4 = np.zeros([N_TS, 1])
week_day5 = np.zeros([N_TS, 1])
week_day6 = np.zeros([N_TS, 1])
week_day7 = np.zeros([N_TS, 1])

# make sure that time series starts with monday! (for the usual interpretation of
# week_day1 -> monday etc)
week_day1[0::7] = 1   # monday
week_day2[1::7] = 1   # tuesday
week_day3[2::7] = 1   # wednesday
week_day4[3::7] = 1   # thursday
week_day5[4::7] = 1   # friday
week_day6[5::7] = 1   # saturday
week_day7[6::7] = 1   # sunday

#%%%%%%%%%%%%%%%%%%%%%%%%% Read Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

# Read in Temperature ---------------------------------------------------------
df = pd.read_csv(path_raw + 'Weather/'+ 'weather.csv', sep= ',')
T = df.values[6* 96:,4]
T= T[:850*96]

# Read in Irradiance ----------------------------------------------------------
IR = df.values[6* 96:,2]
IR = IR[:N_TS*96]

del df
plt.close('all')

# clusters --------------------------------------------------------------------
day = 1
week = 7
month = 28
ind_model_hr = 0

# loop over clusters
for filename in ['Cluster15_meanprofile_73.csv', 'Cluster3_meanprofile_33.csv', 'Cluster6_meanprofile_1.csv']:
    
    ext = os.path.splitext(filename)[-1].lower()
    if ext != '.csv':
        break
    if 'normalization' in filename:
        break

    print(filename)
    df = pd.read_csv(path_clustering+ filename, sep= ',')
     
    cluster_load = df.values
        
    del df

    #%% Splitting data
    #E_mean_resolved, E_fluct = YearTrend(cluster_load, time)
    N2014 = 296* 96
    N2015 = N2014+ 365* 96
    N2016 = N2015+ 365* 96
    NTS = len(cluster_load)
    
    #%% Day time splitting
    print('-- Split into times')
    
    # day times considered
    hours = np.array([12, 13, 15, 18, 20])* 4
    
    # arrays for the time series (fluctuations)
    y_pred_regr = np.zeros([len(hours), N_TS- month-400])
    y_real = np.zeros([len(hours), N_TS- month-400])
    
    # keep track of active features
    feat_active = np.zeros([len(hours), 31]) # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx
    
    # store histogram of errors (with regression)
    hist_err_regr = np.zeros([len(hours), n_bins])
    hist_bins_regr = np.zeros([len(hours), n_bins])
    hist_edges_regr = np.zeros([len(hours), n_bins+ 1])
    
    # loop over day times -----------------------------------------------------
    for k, hr in enumerate(hours):
    
        #%%%%%%%%%%%%%%%%%%%%%%%% Temperature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # demand fluctuations XXXXXXXXXXXXXXXXXXXXXXXXX
        TS_hr = np.log(cluster_load[hr::96])
        TS_hr[np.where(TS_hr< -10)[0]] = -5
        
        y_real[k] = TS_hr[month:][400:].reshape(-1)
        y_real[k] = y_real[k].astype("Float64")
        
        # temperature fluctuations
        T_hr = T[hr::96]
        N = len(T_hr)
        
        # past 3 hours Temperature
        T_hr_neg1 = T[hr-1::96]
        T_hr_neg2 = T[hr-2::96]
        T_hr_neg3 = T[hr-4::96]
        T_hr_neg4 = T[hr-6::96]
        T_hr_neg5 = T[hr-8::96]
        T_hr_neg6 = T[hr-10::96]
                
        # min and max of last 24 hrs (values available at beginning of second day due to avg)
        T_hr_max24 = np.zeros([N-1,  1])
        T_hr_min24 = np.zeros([N-1,  1])
        T_hr_avg24 = np.zeros([N-1,  1])
        T_hr_avg8 = np.zeros([N-1, 1])     # track night
        
        for n in range(N-1):
            T_hr_max24[n] = np.max(T[hr+ n*96: hr+ (n+1)* 96])
            T_hr_min24[n] = np.min(T[hr+ n*96: hr+ (n+1)* 96])
            T_hr_avg24[n] = np.average(np.float64(T[hr+ n*96: hr+ (n+1)* 96]))
            T_hr_avg8[n] = np.average(np.float64(T[hr+ 64+ n*96: hr+ (n+1)* 96]))
            
        # average of past 7 days
        T_hr_avg7 = np.zeros([N-7,  1])
        T_hr_avg28 = np.zeros([N-28, 1])
        
        for n in range(N-7):
             T_hr_avg7[n] = np.average(np.float64(T[hr+ n* 96: hr+ (n+7)* 96]))
        
        for n in range(N-28):
             T_hr_avg28[n] = np.average(np.float64(T[hr+ n* 96: hr+ (n+28)* 96]))
        
        # irradiance
        IR_hr = IR[hr::96]
        
        # past 3 hours Irradiance
        IR_hr_neg1 = IR[hr-1::96]
        IR_hr_neg2 = IR[hr-2::96]
        IR_hr_neg3 = IR[hr-4::96]
        IR_hr_neg4 = IR[hr-6::96]
        IR_hr_neg5 = IR[hr-8::96]
        
        #%%%%%%%%%%%%%%%%%%%%%%%%% Fitting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # regularization
        lam = 1E-5
                
        # data vector
        dat = np.concatenate([
                T_hr[month:].reshape(1, -1),            hol_hol[month:].reshape(1, -1), 
                hol_prev[month:].reshape(1, -1),        hol_post[month:].reshape(1, -1),
                week_day1[month:].reshape(1,-1),        week_day2[month:].reshape(1,-1),
                week_day3[month:].reshape(1,-1),        week_day4[month:].reshape(1,-1),
                week_day5[month:].reshape(1,-1),        week_day6[month:].reshape(1,-1),
                week_day7[month:].reshape(1,-1),
                T_hr_neg1[month:].reshape(1, -1),       T_hr_neg2[month:].reshape(1, -1), # T with delay 15 min, 30 min
                T_hr_neg3[month:].reshape(1, -1),       T_hr_neg4[month:].reshape(1, -1), # T with delay 1hr, 1.5hr 
                T_hr_neg5[month:].reshape(1, -1),       T_hr_neg6[month:].reshape(1, -1), # T with delay 2hr, 2.5hr
                IR_hr_neg1[month:].reshape(1, -1),      IR_hr_neg2[month:].reshape(1, -1), # IR with delay 15 min, 30 min
                IR_hr_neg3[month:].reshape(1, -1),      IR_hr_neg4[month:].reshape(1, -1), # IR with delay 1hr, 1.5hr
                IR_hr_neg5[month:].reshape(1, -1),
                T_hr_avg24[month-2*day:-day].reshape(1, -1),  T_hr_avg24[month-3*day:-2*day].reshape(1, -1),
                T_hr_max24[month-day:].reshape(1, -1),      T_hr_avg24[month-day:].reshape(1, -1), # Tmax 24hr, Tavg 24 hr
                T_hr_min24[month-day:].reshape(1, -1),      T_hr_avg8[month-day:].reshape(1, -1), # Tmin 24hr, Tavg 8 hr
                T_hr_avg7[month-week:].reshape(1, -1),      T_hr_avg28[:].reshape(1, -1), # Tavg 7 days, Tavg 28 days
                np.ones([1, len(T_hr_avg28)])], axis= 0).T
    
        dat = dat[400:].astype("Float64")
        
        # start taking time
        start_time = time.time()
        
        # solution vector
        w = np.linalg.solve(np.dot(dat.T, dat)+ lam*np.eye(dat.shape[1]), np.dot(dat.T, y_real[k]))
        
        # prediction
        y_pred = np.dot(dat, w)
        
        # check correlation of error
        eps = y_real[k]- y_pred
        
        #%%%%%%%%%%%%%%%%%%% Feature selection Procedure %%%%%%%%%%%%%%%%%%%%%%
        # vector to keep track on applied features
        active_features = np.ones([dat.shape[1], 1])
            
        # cross-validation
        k_folds = 5
        kf = KFold(len(y_real[k]), n_folds= k_folds)
        
        # Report squared error with all features active -----------------------
        SE = 0
          
        for train, test in kf:
            # splitting of data
            X_train = dat[train]
            y_train = y_real[k][train]
            
            X_test = dat[test]
            y_test = y_real[k][test]
            
            # calculating solution vector and prediction with regularization
            w = np.linalg.solve(np.dot(X_train.T, X_train)+ lam*np.eye(X_train.shape[1]), np.dot(X_train.T, y_train))
            
            y_pred_test = np.dot(X_test, w)
            
            # MSE
            SE += mean_squared_error(y_pred_test, y_test.reshape(-1, 1))* len(y_pred_test)
            
        # Comparison by MSE
        SE_old = 2* SE
        SE_new = SE
        
        # feature selection via MSE
        # ---------------------------------------------------------------------       
        while SE_old > SE_new:
            
            SE_old = SE_new
            
            # RMSE to keep track of features
            SE_features = np.infty* np.ones([len(active_features), 1])
            
            for i in np.nonzero(active_features)[0]:
        
                # feature selection
                features = active_features.copy()
                features[i] = 0
                features = np.nonzero(features)[0]
                
                SE_features[i] = 0                
                
                for train, test in kf:
                    # splitting of data
                    X_train = dat[train.reshape(-1, 1), features.reshape(1, -1)]
                    y_train = y_real[k][train]
                    
                    X_test =  dat[test.reshape(-1, 1), features.reshape(1, -1)]
                    y_test = y_real[k][test]
                    
                    # calculating solution vector and prediction with regularization
                    w = np.linalg.solve(np.dot(X_train.T, X_train)+ lam*np.eye(X_train.shape[1]), np.dot(X_train.T, y_train))
                    
                    y_pred_test = np.dot(X_test, w)
                    
                    # RMSE
                    SE_features[i] += mean_squared_error(y_pred_test, y_test.reshape(-1, 1))* len(y_pred_test)
                
            min_feature = np.argmin(SE_features)
            SE_new = SE_features[min_feature]
            
            if SE_new < SE_old:
                active_features[min_feature] = 0
                
        # solve with de-activated features
        X_dat = dat[:, np.nonzero(active_features)[0]]
        w = np.linalg.solve(np.dot(X_dat.T, X_dat)+ lam*np.eye(X_dat.shape[1]), np.dot(X_dat.T, y_real[k]))
        
        y_train_AR = np.dot(X_dat, w).reshape(-1)
        
        print(np.int16(np.sum(active_features)))

        #%%%%%%%%%%%%%%%%%%%%%%%%% Correlation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        eps = y_train_AR- y_real[k]
        
        model = AR(eps)
        results = model.fit(maxlag = 1)
        
        # correlation coefficient i.e. mu(e_t | e_1, ..., e_(t-1)) = r* e_(t-1)
        r = results.params[1]    

        # transformation
        X_dat_new = X_dat[1:]- r* X_dat[:-1]
        y_dat_new = y_real[k][1:]- r* y_real[k][:-1]
        
        # training and validation set
        Ndat0 = X_dat_new.shape[0]
        Ndat1 = X_dat_new.shape[1]
        
        X_train_AR = X_dat_new[:int(0.9*Ndat0)]
        X_valid_AR = X_dat_new[int(0.9*Ndat0):]
        
        y_train_AR = y_dat_new[:int(0.9*Ndat0)]
        y_valid_AR = y_dat_new[int(0.9*Ndat0):]
        
        # obtain new estimate
        w_regr = np.linalg.solve(np.dot(X_train_AR.T, X_train_AR)+
                                lam*np.eye(X_train_AR.shape[1]), np.dot(X_train_AR.T, y_train_AR))
        
        y_pred_AR = np.dot(X_valid_AR, w_regr).reshape(-1)
        
        # Y_t = r* Y_(t-1)+ Vt, Y_0 obtained by not regressed estimate
        y_pred_regr[k, 0] = y_real[k][0]
        for i in range(1, y_pred_AR.shape[0]):
            y_pred_regr[k, i] = r* y_pred_regr[k, i-1]+ y_pred_AR[i-1]
            
        
        #%%%%%%%%%%%%%%%%%%%%%% Performance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        my_file = Path(path_performance+ 'model_performance_LS.csv')
        
        # take run time
        run_time = time.time()- start_time
        
        # save performance data
        # errors
        RMSE = np.sqrt(mean_squared_error(y_valid_AR, y_pred_AR))
        MASE_norm = (len(y_valid_AR)- 1)/ len(y_valid_AR)
        MASE = MASE_norm* np.sum(np.abs(y_valid_AR- y_pred_AR))/ np.sum(np.abs(y_pred_AR[1:]- y_pred_AR[:-1]))
        MAVE_norm = 1/ (len(y_valid_AR)* 3* np.std(y_valid_AR))
        MAVE = MAVE_norm* np.sum(np.abs(y_valid_AR- y_pred_AR))


        data = {'no_active_feat' : X_dat.shape[1],
                'RMSE' : RMSE, 'MASE' : MASE, 'MAVE' : MAVE,
                'run_time' : run_time, 'time' : int(hr/4), 'Cluster' : filename[:-4]}
            
        df = pd.DataFrame(data, index = [ind_model_hr])
        # df.to_csv('ANN_output.csv', index=False)
    
        if my_file.is_file():        
            with open(my_file, 'a') as f:
                df.to_csv(f, header=False, sep = ';')
            
        else:
            df.to_csv(my_file, sep = ';')
            
        ind_model_hr+= 1
        
        #%% check serial correlation --------------------------------------------
        y_pred_AR_complete = np.dot(X_dat_new, w_regr).reshape(-1)
        
        y_pred_regr[k, 0] = y_real[k][0]
        for i in range(1, y_pred_regr.shape[1]):
            y_pred_regr[k, i] = r* y_pred_regr[k, i-1]+ y_pred_AR_complete[i-1]
            
        # error
        eps_regr = (y_real[k][1:]- r* y_real[k][:-1])-  y_pred_AR_complete
        
        # serial-correlation & variance
        cm = []
        col = ['blue', 'green', 'red', 'black']
        var_idx = np.zeros([eps_regr.shape[0]])
        
        for i in range(eps_regr.shape[0]):
            cm.append(col[int(i/91)% 4])
            
            var_idx[i] = int(i/91)% 4
          
        # variance
        var = np.zeros([4])
        for i in range(4):
            var[i] = np.std(eps_regr[np.where(var_idx ==i)])
            
        print(str(int(hr/4))+ ':00')
        print('Variance for each split (~season): '+ str(np.round(var, decimals = 3)))
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%5 Save %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # but first remove outliers, i.e. pts further spread out than 3*sigma
        eps_regr_mean = np.mean(eps_regr)
        eps_regr_std = np.std(eps_regr)
        
        for i in range(eps_regr.shape[0]):
            
            if eps_regr[i]- eps_regr_mean > 3* eps_regr_std:
                eps_regr[i] = eps_regr_mean + 3* eps_regr_std
                
            elif  eps_regr_mean - eps_regr[i] > 3* eps_regr_std:
                eps_regr[i] = eps_regr_mean - 3* eps_regr_std

        hist_err_regr[k, :], hist_edges_regr[k] = np.histogram(eps_regr, bins = n_bins)
        hist_bins_regr[k] = (hist_edges_regr[k, 1:]+ hist_edges_regr[k, :-1])/ 2

        # save solution vector ------------------------------------------------
        np.savez(path_save+ filename[:-4]+ '_hr_'+ str(int(hr/ 4))+ '_LS',
                a = r, b = w, c = w_regr, d = active_features, e = hist_err_regr[k, :], f = hist_bins_regr[k, :])
        
        feat_active[k] = active_features.reshape(-1)
        
    #%%%%%%%%%%%%%%%%%%%%%%%%% Fitting check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    path_check = './../3Least_Squares/Check/'
    
    np.save(path_check+ filename[:-4], y_pred_regr)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%% plot Fit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure_shape = (2,3)
    fig, ax = plt.subplots(figure_shape[0], figure_shape[1], figsize=(25, 15))
    
    for k, hr in enumerate(hours):
        i, j = np.unravel_index(k, figure_shape)
        ax[i, j].plot(y_real[k], c= 'b')
        #ax[i, j].plot(y_pred[k], c= 'g')
        ax[i, j].plot(y_pred_regr[k], c= 'r')
        ax[i, j].set_title('Fluctuation time= '+ str(int(hr/4))+ ':00')
        ax[i, j].legend(['TS_hr', 'Prediction'])
      
    fig.show()
    fig.savefig(path_plot+ filename[:-4] + '.png') 
    plt.close()
    
    fig, ax = plt.subplots(figsize = (9, 5))
    ax.plot(y_real[0][268: 268+ 366], linewidth = 2, color = [109/ 256, 187/ 256, 248/ 256])
    ax.plot(y_pred_regr[0][268: 268+ 366], linewidth = 2, color = 'tomato')
    ax.legend(['Real Profile', 'Fitted Profile'], loc = 'upper right')
    ax.set_xticks(np.array([31- 15, 60- 15, 91- 15, 121- 15, 152- 15, 182- 15, 213- 15, 244- 15, 274- 15, 305- 15, 335- 15, 366- 15]))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
    ax.grid(axis = 'x', color = 'whitesmoke')
    ax.grid(axis = 'y')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    fig.savefig(path_plot+ 'ClusterYearProcessed/'+ filename[:-4] + '.png')
    plt.show()
    
    # plot Histograms of errors (with regression)
    fig, ax = plt.subplots(figure_shape[0], figure_shape[1], figsize=(25, 15))
        
    for k, hr in enumerate(hours):
        i, j = np.unravel_index(k, figure_shape)
        width = hist_edges_regr[k, 1]- hist_edges_regr[k, 0]
        ax[i, j].bar(hist_edges_regr[k, :-1], hist_err_regr[k], width)
        ax[i, j].set_title('Histogram of Errors time= '+ str(int(hr/4))+ ':00')
    
    fig.show()
    fig.savefig(path_plot+ 'Histogram_regression_'+ filename[:-4] + '.png')
    plt.close()
     
     