# -*- coding: utf-8 -*-
"""
Script performing following tasks:
- outlier removal
- feature selection
- neural network training (selection based on cross-validation)
"""

# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from shutil import copyfile
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR
from pathlib import Path
import time
from Modules import holidays
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.constraints import max_norm
from sklearn.preprocessing import StandardScaler

#%%
# number of bins for error histograms
n_bins = 35

# paths for all kind of stuff
path_raw = './../1Raw_Data/'
path_clustering = './../2Clustering/Data/'
path_save = './../3ANN/Data/'
path_performance = './../3ANN/'
path_plot = './../3ANN/Plots/'

# length of time series
N_TS = 850

# daytimes considered
hours = np.array([10, 12, 14, 16, 18, 20])* 4

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
df = pd.read_csv(path_raw+ 'Weather/'+ 'weather.csv', sep= ',')
T = df.values[6* 96:,4]
T= T[:850*96]

# Read in Irradiance ----------------------------------------------------------
IR = df.values[6* 96:,2]
IR = IR[:850*96]

del df
plt.close('all')

#%% Loading data
week = 7
month = 28
day = 1
ind_model_hr = 0

# loop over clusters ----------------------------------------------------------
for filename in os.listdir(path_clustering):
    
    # check that only csv files and not the normalization file is read in
    ext = os.path.splitext(filename)[-1].lower()
    if ext != '.csv':
        break
    if 'normalization' in filename:
        break

    print(filename)
    df = pd.read_csv(path_clustering+ filename, sep= ',')
    
    # profile for cluster center
    cluster_load = df.values
    
    del df
    
    #%% Day time splitting

    # arrays for the time series (fluctuations)
    y_pred_ANN = np.zeros([len(hours), N_TS- month])  # store prediction for each daytime
    y_real = np.zeros([len(hours), N_TS- month]) # store real profile for each daytime (transformed)
    
    # store histogram of errors (with regression)
    hist_err_regr = np.zeros([len(hours), n_bins])
    hist_bins_regr = np.zeros([len(hours), n_bins])
    hist_edges_regr = np.zeros([len(hours), n_bins+ 1])
    
    # loop over day times -----------------------------------------------------
    for k, hr in enumerate(hours):
    
        #%%%%%%%%%%%%%%%%%%%%%%%% Temperature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # transformation (logarithm) and eliminate "outliers"
        TS_hr = np.log(cluster_load[hr::96])
        TS_hr[np.where(TS_hr< -10)[0]] = -10
        
        y_real[k] = TS_hr[month:].reshape(-1)
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
    
        dat = dat.astype("Float64")
           
        # start taking time            
        start_time = time.time()

        X_train = dat
        y_train = y_real[k]
        
        # scaling of input data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        Ndat1 = dat.shape[1]
        
        K = 1 # number of re-initialization and re-training
            
        # train ANN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        # hyperparameters
        activ_fcn= 'relu'
        learning_rate = 2e-4
        lr_decay = 1e-6
        dropout = 0
        batch = 1
        no_units_L1 = 10
        epochs = 30
        ker_constraint = 0.1
        
        adam_decay = optimizers.Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999,
                                         epsilon = None, decay = lr_decay, amsgrad = False)
            
        
        y_temp = np.zeros([K, X_train.shape[0]])
        performance = np.zeros([K])
        
        # possible re-training of model
        for kk in range(K):
            #
            model = Sequential()
            model.add(Dense(no_units_L1, input_shape = (Ndat1,), activation = activ_fcn, kernel_constraint=max_norm(ker_constraint)))
            model.add(Dropout(dropout))
            model.add(Dense(1))
            
            model.summary()
            
            model.compile(loss = 'mean_squared_error',
                          optimizer = adam_decay,
                          metrics = ['mean_squared_error'])
            
            history = model.fit(X_train, y_train, epochs = epochs, verbose = 0,
                                validation_split = 0.20, batch_size = batch,
                                shuffle = True)
            
            model.save(path_save+ filename[:-4]+ '_hr_'+ str(int(hr/ 4))+ '_no_'+ str(kk)+   '.hdf5')
            
            # prediction
            y_temp[kk] = model.predict(X_train).reshape(-1)
            
            performance[kk] = np.mean(history.history['val_loss'][-10:])
        
        # find best model of the k initializations
        k_best_model = np.argmin(performance)
        #k_worst_model = np.argmax(performance)
        y_pred = y_temp[k_best_model]
        
        # to be activated if re-training is considered -> removes unnecessary models
        #os.remove(path_save+ filename[:-4]+ '_hr_'+ str(int(hr/ 4))+ '_no_'+ str(k_worst_model)+   '.hdf5')
            
        #%%%%%%%%%%%%%%%%%%%%%%%%% Correlation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        eps_ANN = y_train- y_pred
        
        model_AR = AR(eps_ANN)
        results = model_AR.fit(maxlag = 1)
        
        # correlation coefficient i.e. mu(e_t | e_1, ..., e_(t-1)) = r* e_(t-1)
        r = results.params[1]    

        # transformation
        X_dat_new = X_train[1:]- r* X_train[:-1]
        y_new = y_real[k][1:]- r* y_real[k][:-1]
        
        # train ANN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
        Ndat0 = X_dat_new.shape[0]
        Ndat1 = X_dat_new.shape[1]

        X_train_AR = X_dat_new[:365]
        X_valid_AR = X_dat_new[365:730]
        
        y_train_AR = y_new[:365]
        y_valid_AR = y_new[365:730]
        
        y_help_AR = np.zeros([K, X_train_AR.shape[0]])
        
        performance_AR = np.zeros([K])
               
        # for model re-training
        for kk in range(K):
            
            model2 = Sequential()
            model2.add(Dense(no_units_L1, input_shape = (Ndat1,),
                            activation = activ_fcn, kernel_constraint = max_norm(ker_constraint)))
            model2.add(Dropout(dropout))
            model2.add(Dense(1))
            
            model2.summary()
            
            model2.compile(loss = 'mean_squared_error',
                          optimizer = adam_decay,
                          metrics = ['mean_squared_error'])
            
            history = model2.fit(X_train_AR, y_train_AR, epochs = epochs, verbose = 0,
                                validation_split = 0.20, batch_size = batch,
                                shuffle = True)
            
            model2.save_weights(path_save+ filename[:-4]+ '_AR_hr_'+ str(int(hr/ 4))+ '_no_'+ str(kk)+   '.hdf5')
            
            # prediction
            y_help_AR[kk] = model2.predict(X_train_AR).reshape(-1)
            
            performance_AR[kk] = np.mean(history.history['val_loss'][-10])
             
            # summarize history for loss
            fig, ax = plt.subplots(1, 2, figsize=(9, 4)) 
            ax[0].plot(history.history['loss'])
            ax[0].plot(history.history['val_loss'])
            ax[0].set_title('model loss')
            ax[0].set_ylabel('loss')
            ax[0].set_xlabel('epoch')
            ax[0].legend(['train', 'validation'], loc='upper left')
            
            ax[1].plot(y_train_AR, color = [109/ 256, 187/ 256, 248/ 256])
            ax[1].plot(y_help_AR[kk], color = 'tomato')
            ax[1].set_xlabel('time')
            ax[1].set_ylabel('Cosumption')
            ax[1].legend(['True Series', 'Predicted Series'])
            plt.show()
            fig.savefig(path_plot+ filename[:-4]+ "_AR_hr_"+ str(int(hr/ 4)) + '_'+ str(kk) +'.png')
            plt.close()
            
        
        # find best model of the k initializations
        k_best_model = np.argmin(performance_AR)
        #k_worst_model = np.argmax(performance_AR)
        y_pred_AR = y_help_AR[k_best_model]
        
        # to be activated if re-training is considered
        #os.remove(path_save+ filename[:-4]+ '_AR_hr_'+ str(int(hr/ 4))+ '_no_'+ str(k_worst_model)+   '.hdf5')

        
        #%%%%%%%%%%%%%%%%%%%%%% Performance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        my_file = Path(path_performance+ 'model_performance_ANN.csv')
        destination = path_save+ filename[:-4]+ '_AR_hr_'+ str(int(hr/ 4))+ '_no_'+ str(k_best_model)+   '.hdf5'
        
        # load model for prediction
        model = Sequential()
        model.add(Dense(no_units_L1, input_shape = (Ndat1,),
                        activation = activ_fcn, kernel_constraint = max_norm(ker_constraint)))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        
        model.compile(loss = 'mean_squared_error',
                      optimizer = adam_decay,
                      metrics = ['mean_squared_error'])
        
        model.load_weights(destination)
        
        # prediction
        y_pred_AR = model.predict(X_valid_AR)
        y_pred_valid = np.zeros([X_valid_AR.shape[0]])
        y_pred_valid[0] = y_real[k][int(0.9* Ndat0)]
        
        #
        for i in range(1, y_pred_AR.shape[0]):
            y_pred_valid[i] = r* y_pred_valid[i-1]+ y_pred_AR[i-1]
            
        # stop run time 
        run_time = time.time()- start_time
        
        # save performance data
        # errors        
        RMSE = np.sqrt(mean_squared_error(y_valid_AR.reshape(-1), y_pred_AR.reshape(-1)))
        MASE_norm = (len(y_valid_AR)- 1)/ len(y_valid_AR)
        MASE = MASE_norm* np.sum(np.abs(y_valid_AR.reshape(-1)- y_pred_AR.reshape(-1)))/ np.sum(np.abs(y_valid_AR[1:]- y_valid_AR[:-1]))
        MAVE_norm = 1/ (len(y_valid_AR)* 3* np.std(y_valid_AR))
        MAVE = MAVE_norm* np.sum(np.abs(y_valid_AR- y_pred_AR))
        
        data = {'no_units_L1' : no_units_L1, 'lr' : learning_rate,
                'lr_decay' : lr_decay, 'dropout' : dropout, 'batch' : batch,
                'RMSE' : np.round(RMSE, decimals = 3), 'MASE' : np.round(MASE, decimals = 3),
                'MAVE' : np.round(MAVE, decimals = 3), 'run_time' : np.round(run_time, decimals = 3),
                'time' : int(hr/4), 'Cluster' : filename[:-4]}
        
        
        
        df = pd.DataFrame(data, index = [ind_model_hr])
    
        if my_file.is_file():        
            with open(my_file, 'a') as f:
                df.to_csv(f, header=False, sep = ';')
            
        else:
            df.to_csv(my_file, sep = ';')
            
        ind_model_hr+= 1
        
        #%% Error sampling
        # refit model on complete data
        model3 = Sequential()
        model3.add(Dense(no_units_L1, input_shape = (Ndat1,),
                         activation = activ_fcn, kernel_constraint = max_norm(ker_constraint )))
        model3.add(Dropout(dropout))
        model3.add(Dense(1))
            
        model3.summary()
            
        model3.compile(loss = 'mean_squared_error',
                       optimizer = adam_decay,
                       metrics = ['mean_squared_error'])
            
        history = model3.fit(X_dat_new, y_new, epochs = epochs, verbose = 0,
                             validation_split = 0.20, batch_size = batch,
                             shuffle = True)
            
        model3.save_weights(path_save+ filename[:-4]+ '_Final_hr_'+ str(int(hr/ 4))+   '.hdf5')
        
        # assemble predicted time series
        # Y_t = r* Y_(t-1)+ Vt, Y_0 obtained by not regressed estimate
        y_pred_AR_complete = model.predict(X_dat_new).reshape(-1)
        
        y_pred_ANN[k, 0] = y_real[k][0]
        for i in range(1, y_pred_ANN.shape[1]):
            y_pred_ANN[k, i] = r* y_pred_ANN[k, i-1]+ y_pred_AR_complete[i-1]

        # error
        eps_regr = (y_real[k][1:]- r* y_real[k][:-1])-  y_pred_AR_complete
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%5 Save %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # remove outliers, i.e. pts further spread out than 3*sigma
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
        np.savez(path_save+ filename[:-4]+ '_hr_'+ str(int(hr/ 4))+ '_ANN',
                a = r, e = hist_err_regr[k, :], f = hist_bins_regr[k, :])
   
    #%%%%%%%%%%%%%%%%%%%%%%%%% Fitting check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    path_check = './../3ANN/Check/'
    
    np.save(path_check+ filename[:-4], y_pred_ANN)
    
    #%% plot Fit --------------------------------------------------------------
    figure_shape = (2,3)
    fig, ax = plt.subplots(figure_shape[0], figure_shape[1], figsize=(25, 15))
    
    for k, hr in enumerate(hours):
        i, j = np.unravel_index(k, figure_shape)
        ax[i, j].plot(y_real[k], color = [109/ 256, 187/ 256, 248/ 256])
        #ax[i, j].plot(y_pred[k], c= 'g')
        ax[i, j].plot(y_pred_ANN[k], color = 'tomato')
        ax[i, j].set_title('Fluctuation time= '+ str(int(hr/4))+ ':00)')
        ax[i, j].legend(['TS_hr', 'Prediction'])
      
    fig.show()
    fig.savefig(path_plot+ filename[:-4] + '.png') 
    plt.close()
    
    #
    fig, ax = plt.subplots(figsize = (9, 5))
    ax.plot(y_real[0][268: 268+ 366], linewidth = 2, color = [109/ 256, 187/ 256, 248/ 256])
    ax.plot(y_pred_ANN[0][268: 268+ 366], linewidth = 2, color = 'tomato')
    ax.legend(['Real Profile', 'Fitted Profile'], loc = 'upper right')
    ax.set_xticks(np.array([31- 15, 60- 15, 91- 15, 121- 15, 152- 15, 182- 15, 213- 15, 244- 15, 274- 15, 305- 15, 335- 15, 366- 15]))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
    ax.grid(axis = 'x', color = 'whitesmoke')
    ax.grid(axis = 'y')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    fig.savefig(path_plot+ 'ClusterYearProcessed/'+ filename[:-4] + '.png')
    plt.show()
    plt.close()
    
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
