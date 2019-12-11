# -*- coding: utf-8 -*-
"""
Script to analyse data by plotting time series, clustering, test stationarity, etc
"""

# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.linear_model as sklLM
import sklearn.preprocessing as sklPP
import statistics as stat
import re
import sklearn as skl
import statsmodels.api as sm

from DataAnalysis_Plotting import plotting_class
from DataAnalysis_YearTrend import YearTrend

from sklearn.preprocessing import StandardScaler
from sklearn import manifold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller

    
#%% Loading data
path_read = './../1Raw_Data/SmartMeter/'

X= np.load(path_read+ 'all_data_complete_X.npy')
X= X.astype(float)
time = np.load(path_read+ 'all_data_complete_t.npy')

# Test if extraction of small customers worked
plt.figure()
plt.plot(np.linspace(0, X.shape[1]-1, X.shape[1]), np.sort(np.average(X, axis=0)))
plt.title('Overview Mean Load Profiles')
plt.show()

#%% Data preprocessing
# to extract significant patterns, average weekdays over one months period
X_weekday = np.zeros([96, X.shape[1]])

for week in range(0, 52):
    for day in range(5):
        ind1 = (week*7+ day)*96
        ind2  = (week*7+ day+ 1)*96

        X_weekday = np.add(X_weekday, X[ind1: ind2, :])
        
X_weekday/= (20-8)* 5

# plot
plt.figure()
plt.plot(time[0:96], X_weekday[:, 100:110])
plt.show()

# Data standardisation
X_weekday= X_weekday/ np.max(X_weekday, axis= 0)
        

#%% Clustering
# number of clusters (square number)
nclusters= 9

# time horizon for plotting
N= 24* 4

# time horizon for features
start_f = 96
end_f = 96+ 96

# feature extraction (transpose needed to match required dimensions of Kmeans)
features= X_weekday.T

# Kmeans algorithm
myKmeans= KMeans(n_clusters= nclusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001)
myKmeans.fit(features)

# Cluster extraction
sort_index= np.argsort(myKmeans.labels_)
clust_interval = np.zeros([nclusters+ 1])
indicator = 0

#plt.close('all')

for i in range(len(sort_index)):
    if myKmeans.labels_[sort_index[i]] > indicator:
        indicator+= 1
        clust_interval[indicator]= i

clust_interval[-1] = features.shape[0]
  
# Plotting of load profiles belonging to same cluster
fig, ax= plt.subplots(int(np.sqrt(nclusters)), int(np.sqrt(nclusters)), figsize=(12,14))

for i in range(nclusters):
        
    # extract indeces belonging to same cluster
    idx = sort_index[int(clust_interval[i]): int(clust_interval[i+1])]
    
    # plot load profiles and mean of cluster
    ax[i%3, int(i/3)].plot(time[:N], X_weekday[:N, idx])
    ax[i%3, int(i/3)].plot(time[:N], np.mean(X_weekday[:N, idx], axis= 1), c= 'r', linewidth= 3)
    ax[i%3, int(i/3)].set_title('Cluster No '+str(i)+ ' (# '+ str(len(idx))+ ')')
    
plt.pause(0.1)
plt.ion()
plt.show(block=True)

'''
    ax.xaxis.set_major_locator(plt.MultipleLocator(48))
                    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[2] = 'Monday'
    labels[4] = 'Tuesday'
    labels[6] = 'Wednesday'
    labels[8] = 'Thursday'
    labels[10] = 'Friday'
    labels[12] = 'Saturday'
    labels[14] = 'Sunday'
                
    #ax.set_yticks([])
    ax.set_xticklabels(labels, rotation= 45)
    '''
 
# Plot centroids of clusters
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = myKmeans.cluster_centers_
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.plot(time[start_f:end_f], center)
                
    
no_cluster = int(input("Select Cluster for Analysis [0-N-1] \n"))

#%% ARIMA - Analysis
    
def plot_correlation(lag_acf, lag_pacf, lag, length):
    x_plot = np.linspace(0, lag-1, lag)

    plt.figure()
    #Plot ACF: 
    plt.subplot(121) 
    plt.vlines(x= x_plot, ymin= 0, ymax= lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(length),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(length),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    #Plot PACF:
    plt.subplot(122)
    plt.vlines(x= x_plot, ymin= 0, ymax= lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(length),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(length),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    
    plt.show()
    
    pass

# check stationarity with rolling mean
def test_stationarity(timeseries, wdw):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=wdw)
    rolstd = pd.rolling_std(timeseries, window=wdw)

    #Plot rolling statistics:
    plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

     #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff

#%% Cluster Selection & Trend Removal
# Cluster 6 (200 customers, distinct profile)
N_year = 4* 24* 800
Cluster6 = X[:N_year, sort_index[no_cluster: no_cluster+1]]
Cluster6 = np.mean(Cluster6, axis = 1)

# Plot CLuster
plt.figure()
plt.plot(time[:N_year], Cluster6)
plt.show()


E_fluct = YearTrend(Cluster6)

plt.figure()
plt.plot(E_fluct)
plt.plot(2000*hol2015, c='r')
plt.show()

#%% Exogeneous variables --------------------------------------------------------
df = pd.read_csv("weather.csv", sep= ',')
T = df.values[6* 96:,4]

dfT = pd.DataFrame(T[0: 365*96])
#df.loc[0: 365*96-1].temperature
dfE = pd.DataFrame(E_fluct[0: 365*96])
        
#%% Step 1: differencing

horizon = [7, 30, 90, 365]

for ho in horizon:
    time_diff = difference(E_fluct, ho* 96)
    
    lag_acf = acf(time_diff, nlags=20)
    lag_pacf = pacf(time_diff, nlags=20, method='ols')

    plot_correlation(lag_acf, lag_pacf, 20, len(time_diff))
    
    
'''
weekly_diff = difference(E_fluct, 7*96)
plt.figure()
plt.plot(weekly_diff)
plt.show()

lag_acf = acf(weekly_diff, nlags=20)
lag_pacf = pacf(weekly_diff, nlags=20, method='ols')

plot_correlation(lag_acf, lag_pacf, 20, len(weekly_diff))
'''


#%% Step 2: AR and MA terms


dfweek= pd.DataFrame(weekly_diff[0: 365* 96])

plt.close('all')

AIC = np.zeros([6, 6])

for af in range(6):
    for ma in range(6):

        model3 = sm.tsa.ARIMA(np.asarray(dfweek[0:34000].astype(float)), order=(af, 0, ma))
        #sm.tsa.ARIMA(endog= np.asarray(dfweek[0:34000].astype(float)), exog= np.asarray(dfT[0:34000].astype(float)), order=[1, 0, 0])
        results3 = model3.fit()
        
        
        lag_acf = acf(results3.fittedvalues, nlags=20)
        lag_pacf = pacf(results3.fittedvalues, nlags=20, method='ols')
        
        #plot_correlation(lag_acf, lag_pacf, 20, len(results3.fittedvalues))
        
        AIC[af][ma] = results3.aic
        
        
xv, yv = np.meshgrid(np.linspace(0, 5, 6), np.linspace(0, 5, 6))
        
plt.figure()
plt.scatter(xv, yv, c = AIC, cmap= 'jet', s= 200)
plt.colorbar()
            
    
model3 = sm.tsa.ARIMA(np.asarray(dfweek[0:34000].astype(float)), order= (0, 1, 0))
#sm.tsa.ARIMA(endog= np.asarray(dfweek[0:34000].astype(float)), exog= np.asarray(dfT[0:34000].astype(float)), order=[1, 0, 0])
results3 = model3.fit()
    
    
lag_acf = acf(results3.fittedvalues, nlags=20)
lag_pacf = pacf(results3.fittedvalues, nlags=20, method='ols')
    
plot_correlation(lag_acf, lag_pacf, 20, len(results3.fittedvalues))

plt.figure()
plt.plot(weekly_diff, c= 'b')
plt.plot(results3.predict(), c= 'r')
plt.title('Fitting')

#%% Step 3: Consider Exogeneous variables


#%% Holidays
# 2014
hol2014 = np.zeros([269* 96])
dates2014 = np.array([11, 14, 24, 24+ 10, 24+ 28, 24+ 31+ 8, 24+ 31+ 30+ 31, 24+ 31+ 30+ 31+ 31+ 30+ 31+ 30+ 24, 24+ 31+ 30+ 31+ 31+ 30+ 31+ 30+ 25])

for day in dates2014:
    for i in range(96):
        hol2014[day * 96+ i] = 1

# 2015
hol2015 = np.zeros([365* 96])
dates2015 = np.array([0, 31+ 28+ 31+ 2,  31+ 28+ 31+ 5, 31+ 28+ 31+ 30,  31+ 28+ 31+ 30+ 13,  31+ 28+ 31+ 30+ 24,  31+ 28+ 31+ 30+ 31 +30+ 31, 31+ 28+ 31+ 30+ 31 +30+ 31+ 31+ 30+ 31+ 30+ 24, 31+ 28+ 31+ 30+ 31 +30+ 31+ 31+ 30+ 31+ 30+ 25])

for day in dates2015:
    for i in range(96):
        hol2015[day* 96+ i] = 1
        
# 2016
hol2016 = np.zeros([365* 96])
dates2016 = np.array([0, 31+ 28+ 24,  31+ 28+ 28, 31+ 28+ 31+ 30,  31+ 28+ 31+ 30+ 4,  31+ 28+ 31+ 30+ 15,  31+ 28+ 31+ 30+ 31 +30+ 31, 31+ 28+ 31+ 30+ 31 +30+ 31+ 31+ 30+ 31+ 30+ 24, 31+ 28+ 31+ 30+ 31 +30+ 31+ 31+ 30+ 31+ 30+ 25])

for day in dates2016:
    for i in range(96):
        hol2016[day* 96+ i] = 1
        
# 2017
hol2017 = np.zeros([365* 96])
dates2017 = np.array([0, 31+ 28+ 31+ 13,  31+ 28+ 31+ 16, 31+ 28+ 31+ 30,  31+ 28+ 31+ 30+ 25,  31+ 28+ 31+ 30+ 31+ 4,  31+ 28+ 31+ 30+ 31 +30+ 31, 31+ 28+ 31+ 30+ 31 +30+ 31+ 31+ 30+ 31+ 30+ 24, 31+ 28+ 31+ 30+ 31 +30+ 31+ 31+ 30+ 31+ 30+ 25])

for day in dates2017:
    for i in range(96):
        hol2017[day* 96+ i] = 1

# 2018
hol2018 = np.zeros([365* 96])
dates2018 = np.array([0, 31+ 28+ 29, 31+ 28+ 31+ 1, 31+ 28+ 31+ 30, 31+ 28+ 31+ 30+ 9, 31+ 28+ 31+ 30+ 20, 31+ 28+ 31+ 30+ 31+ 30+ 31, 31+ 28+ 31+ 30+ 31+ 30+ 31+ 31+ 30+ 31+ 30+ 24, 31+ 28+ 31+ 30+ 31+ 30+ 31+ 31+ 30+ 31+ 30+ 25])

for day in dates2018:
    for i in range(96):
        hol2018[day* 96+ i] = 1
    
plt.figure()
plt.plot(hol2015, c='r')
plt.plot(hol2016, c='b')
plt.plot(hol2017, c='g')
plt.plot(hol2018, c='c')
plt.show()

plt.figure()
plt.plot(2000* hol2014)
plt.plot(-2000* np.concatenate([hol2014[7*96:], np.zeros([7*96])]))
plt.plot(weekly_diff[0: 269* 96])

'''
dfHolidays = pd.DataFrame(hol2014)

model_holidays = sm.tsa.ARIMA(endog= np.asarray(dfweek[0: 269* 96].astype(float)), exog= np.asarray(dfHolidays[0:269* 96].astype(float)), order=[1, 0, 0])
results_holidays = model_holidays.fit()
model_holidays.predict(params = None , exog = np.asarray(dfHolidays[0:269* 96].astype(float)))

plt.figure()
plt.plot(results_holidays.fittedvalues[:N], c= 'r')
plt.show()
'''

anomalies = np.zeros([365])

weekday = np.zeros([7])

for day in range(365):
    weekday[day%7] += np.average(Cluster6[day*96: (day+1)*96])/ 52
    

for day in range(365):
    if 0.8* weekday[day%7] > np.average(Cluster6[day* 96: (day+1)*96]):
        anomalies[day] = 1
    

#%% Correlation Electricity Demand & Temperature

plt.figure()
plt.plot(results3.fittedvalues[:N], c= 'r')
plt.show()

# temporal averaging
time_horizon_day = [1, 4, 12, 24, 48, 96]
time_horizon_year =  [1, 7, 30, 90, 180]

plt.close('all')

for th in time_horizon_day:
    time_steps = int((365*96) / th)
    T_avg = np.zeros([time_steps-1])
    E_avg = np.zeros([time_steps-1])
    
    for ts in range(time_steps-1):
        T_avg[ts] = np.average(np.float64(T[ts* th: (ts+1)* th]))
        E_avg[ts] = np.average(Cluster6[ts* th: (ts+1)* th])
        
        
    correlation = np.corrcoef([E_avg.T, T_avg.T])
    
    plt.figure()
    plt.plot(E_avg, c= 'b')
    plt.plot(2000+  100*T_avg, c= 'r')
    plt.title('Averaged E and T in '+ str(th)+ 'quarter hours (correlation= '+ str(round(correlation[0, 1], 3))+ ')')
        

for th in time_horizon_year:
    time_steps = int(365*2 / th)
    T_avg = np.zeros([time_steps-1])
    E_avg = np.zeros([time_steps-1])
    
    for ts in range(time_steps-1):
        T_avg[ts] = np.average(np.float64(T[ts* th* 96: (ts+1)* th* 96]))
        E_avg[ts] = np.average(Cluster6[ts* th* 96: (ts+1)* th* 96])
        
        
    correlation = np.corrcoef([E_avg.T, T_avg.T])
    
    plt.figure()
    plt.plot(E_avg, c= 'b')
    plt.plot(2000+  100* T_avg, c= 'r')
    plt.title('Averaged E and T in '+ str(th)+ 'days (correlation= '+ str(round(correlation[0, 1], 3))+ ')')
        
# temporal average
T_week = np.zeros([365])
E_week = np.zeros([365])

for week in range(365):
    th = 96;
    T_week[week] = np.average(np.float64(T[week* th: (week+1)* th]))
    E_week[week] = np.average(Cluster6[week* th: (week+1)* th])
    
    
plt.figure()
plt.plot(335-  4* T_week, c= 'r')
plt.plot(E_week, c= 'b')
plt.title('Daily Averaged E and T')
plt.show()

# correlation between temperature and electricity consumption
correlation = np.corrcoef([Cluster6[0:365*96].T, np.float64(T[0:365*96]).T])
correlation_single = np.corrcoef([X[0:365*96, 10].T, np.float64(T[0:365*96]).T])
correlation_week = np.corrcoef([E_week.T, np.float64(T_week.T)])

correlation[0, 1]
correlation_single[0, 1]
correlation_week[0, 1]

fft_E = np.fft.fftn(E_fluct)
fft_week = np.fft.fftn(weekly_diff)

plt.figure()
plt.plot(fft_E)
plt.plot(fft_week, c='r')
plt.title('Correlation E and T')
plt.show()

#%% SARIMA Try and Error
# differencing to get rid off trend
diff = np.diff(Cluster6)

plt.figure()
plt.plot(diff)
plt.title('Differenced TS')
plt.show()

# calculate auto correlation coefficients
lag_acf = acf(diff, nlags=20)
lag_pacf = pacf(diff, nlags=20, method='ols')

plot_correlation(lag_acf, lag_pacf, 20, len(diff))

seasonal_diff = difference(Cluster6, 8760)
weekly_diff = difference(seasonal_diff, 672)

plt.figure()
plt.plot(seasonal_diff)
plt.title('Seasonally Differenced TS')
plt.show()

plt.figure()
plt.plot(weekly_diff)
plt.title('Seasonally and Weekly Differenced TS')
plt.show()

test_stationarity(np.asarray(Cluster6), 96* 30)
test_stationarity(np.asarray(seasonal_diff), 96* 30)
test_stationarity(np.asarray(weekly_diff), 96* 30)


#%% Plot different level of aggregation
N= 7* 24* 4

# images in x and y direction
kx= 3
ky= 3

# initialize new class
data_analysis= plotting_class(time, X)

# single consumer
data_analysis.plot_consumer(3, 3, 1, "1 consumer", 0)

# 10 consumers
data_analysis.plot_consumer(5, 5, 10, "10 consumer", 0)

# 100 consumers
data_analysis.plot_consumer(5, 5, 100, "100 consumer", 0)

# aggregation of 1'000 load profiles
data_analysis.plot_consumer(2, 0, 1000, "1000 consumer", 1)

#%% time series analysis
t_week= len(X)* 0.25/ 24/ 7

print('Length in weeks')
print(str(int(t_week))+'\n')

t_month= t_week/ 4
print('Length in months')
print(str(int(t_month))+'\n')

t_year= t_week/ 52
print('Length in years')
print(str(int(t_year)))


