# -*- coding: utf-8 -*-
"""
Script for K-Means clustering of customers based on electricity consumption profile
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
#%% Clustering

class clustering():
    """ perform kmeans clustering of time series and return the cluster index
        for each individual load profile
    
    Inputs:
        timeseries: array with load profiles
        datetime: datetime object
        
    Attributes:
        ncluster: number of clusters to be generated
        
    Functions:
        kmeans: perform kmeans clustering
        
    """
    
    def __init__(self, nclusters = 9, flag = False):
        # number of clusters (square number)
        self.nclusters_ = nclusters
        self.flag_ = flag
        
        

    def kmeans(self, timeseries, datetime):
        nclusters = self.nclusters_
        
        # feature extraction (transpose needed to match required dimensions of Kmeans)
        features = timeseries.copy().T
        
        # Kmeans algorithm
        myKmeans= KMeans(n_clusters = nclusters, init='k-means++', n_init=30, max_iter=300, tol=0.0001)
        myKmeans.fit(features)
        cost = myKmeans.inertia_ 
        
        # sort labels and store in array sort_index
        sort_index = np.argsort(myKmeans.labels_)
        clust_interval = np.zeros([nclusters+ 1])
        
        # extract label range corresponding to same cluster
        indicator = 0
        for i in range(len(sort_index)):
            if myKmeans.labels_[sort_index[i]] > indicator:
                indicator+= 1
                clust_interval[indicator]= i

        clust_interval[-1] = features.shape[0]
            
        # Plotting of load profiles belonging to same cluster -----------------        
        if self.flag_ == True:
            # Plot centroids of clusters
            path_save = './../2Clustering/Plots/'
            color = [[109/ 256, 187/ 256, 248/ 256], 'yellowgreen', 'tomato']
                      
            for i in np.unique(myKmeans.labels_):
                print(i)
                max_prof = np.max(timeseries[:, np.where(myKmeans.labels_ == i)[0]])
                step = 1+ np.int16(max_prof/ 6)
                fig, ax = plt.subplots()
                idx = np.where(myKmeans.labels_ == i)[0]
                for j in range(len(idx)):
                    ax.plot(timeseries[:, idx[j]], color = color[int(3*j/len(idx))])
                ax.plot(myKmeans.cluster_centers_[i], c = 'r', linewidth = 6)
                ax.set_xticks(np.arange(0, 97, 24))
                ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
                ax.set_yticks(np.arange(0, np.int16(max_prof)+ 1, step))
                ax.set_yticklabels(np.arange(0, np.int16(max_prof)+ 1, step))
                ax.grid(axis = 'y')
                plt.show()
                plt.savefig(path_save + 'Cluster_'+ str(i))
                plt.close()
        
                    
        return sort_index, clust_interval, cost, myKmeans.labels_
    
#%% Read in Data
path_read = './../1Raw_Data/SmartMeter/'
path_save = './../2Clustering/Data/'
path_plot = './../2Clustering/Plots/'
#
X= np.load(path_read+ 'X_1_6_small.npy')
X= X.astype(float)

fig, ax = plt.subplots(nrows = 1, ncols = 2)
ax[0].scatter(np.mean(X, axis = 0), np.median(X, axis = 0))
ax[0].plot(np.arange(1500), np.arange(1500), c = 'r')
ax[0].set_xlabel('Mean')
ax[0].set_ylabel('Median')
ax[0].set_title('')
ax[1].scatter(np.mean(np.log(X), axis = 0), np.median(np.log(X), axis = 0))
ax[1].plot(np.arange(2, 7), np.arange(2, 7), c = 'r')
ax[1].set_xlabel('Mean')
ax[1].set_ylabel('Median')
ax[1].set_title('Log-Transform')
plt.show()

ID = np.load(path_read+ 'ID_1_6_small.npy')
time = pd.to_datetime(np.load(path_read+ 't_1_6.npy'))


#%% Data preprocessing for Clustering
print('-- Data Preprocessing for Clustering')
# to extract significant patterns, average weekdays over a year
X_weekday = np.zeros([96, X.shape[1]])

# average profile for clustering
for week in range(0, 52):
    for day in range(5):
        # week day
        ind1 = (week*7+ day)*96
        ind2  = (week*7+ day+ 1)*96
        X_weekday = np.add(X_weekday, X[ind1: ind2, :])
        
X_weekday/= (52* 5)

# log scaler
# feature vector is composed of the data from April 14' to April 15' -> divide
# by mean of this period (1 year)
Feature = X_weekday/ np.mean(X_weekday, axis= 0)

#%%
'''
print('-- Data Preprocessing for Clustering')
# to extract significant patterns, average weekdays over one months period
offset = 60* 96 # (start of time series) April -> June (start of summer)
shift = 4* 30* 96 # one season

X_week_Spring = np.zeros([96, X.shape[1]])
X_week_Summer = np.zeros([96, X.shape[1]])
X_week_Autumn = np.zeros([96, X.shape[1]])
X_week_Winter = np.zeros([96, X.shape[1]])

for week in range(12):
    for day in range(5):
        # week day
        ind1 = offset+ (week*7+ day)*96
        ind2 = offset+ (week*7+ day+ 1)*96
        X_week_Summer = np.add(X_week_Summer, X[ind1: ind2, :])
        X_week_Autumn = np.add(X_week_Autumn, X[ind1+shift: ind2+shift, :])
        X_week_Winter = np.add(X_week_Winter, X[ind1+2*shift: ind2+2*shift, :])
        X_week_Spring = np.add(X_week_Spring, X[ind1+3*shift: ind2+3*shift, :]) 
        
X_week_Summer/= (5*12)
X_week_Autumn/= (5*12)
X_week_Winter/= (5*12)
X_week_Spring/= (5*12)

# downsampling factor (fraction of 96)
ds_factor = 4
shape0 = int(X_week_Summer.shape[0]/ ds_factor)

X_week_Summer_ds = np.zeros([shape0, X_week_Summer.shape[1]])
X_week_Autumn_ds = np.zeros([shape0, X_week_Autumn.shape[1]])
X_week_Winter_ds = np.zeros([shape0, X_week_Winter.shape[1]])
X_week_Spring_ds = np.zeros([shape0, X_week_Spring.shape[1]])

for k in range(shape0):
    ind1 = k* ds_factor
    ind2 = (k+ 1)* ds_factor
    X_week_Summer_ds[k] = np.mean(X_week_Summer[ind1: ind2], axis = 0)
    X_week_Autumn_ds[k] = np.mean(X_week_Autumn[ind1: ind2], axis = 0)
    X_week_Winter_ds[k] = np.mean(X_week_Winter[ind1: ind2], axis = 0)
    X_week_Spring_ds[k] = np.mean(X_week_Spring[ind1: ind2], axis = 0)
    
plt.figure()
plt.plot(np.arange(96), X_week_Summer[:, 1], c = 'r')
plt.plot(np.arange(0, 96, ds_factor), X_week_Summer_ds[:, 1], c = 'r', linestyle = ':')

plt.plot(np.arange(96), X_week_Autumn[:, 1], c = 'green')
plt.plot(np.arange(0, 96, ds_factor), X_week_Autumn_ds[:, 1], c = 'green', linestyle = ':')

plt.plot(np.arange(96), X_week_Winter[:, 1], c = 'b')
plt.plot(np.arange(0, 96, ds_factor), X_week_Winter_ds[:, 1], c = 'b', linestyle = ':')

plt.plot(np.arange(96), X_week_Spring[:, 1], c = 'orange')
plt.plot(np.arange(0, 96, ds_factor), X_week_Spring_ds[:, 1], c = 'orange', linestyle = ':')
plt.show()
   
Feature = np.concatenate([X_week_Summer_ds, X_week_Autumn_ds, X_week_Winter_ds, X_week_Spring_ds], axis = 0)

# log scaler
# feature vector is composed of the data from April 14' to April 15' -> divide
# by mean of this period (1 year)
Feature = Feature / np.mean(Feature, axis = 0)
'''
#%%
'''
print('-- Data Preprocessing for Clustering')
hour = np.array([12, 13, 15, 18, 20])* 4
# to extract significant patterns, average weekdays over one months period
X_1200 = np.zeros([18, X.shape[1]])
X_1300 = np.zeros([18, X.shape[1]])
X_1500 = np.zeros([18, X.shape[1]])
X_1800 = np.zeros([18, X.shape[1]])
X_2000 = np.zeros([18, X.shape[1]])

# average profile for clustering
for w in range(18):
    
    for s in range(3):
    
        for d in range(5):
            ind1 = (w*3*7+ s*7+ d)*96
            print(w)
            X_1200[w] += X[ind1+hour[0]]
            X_1300[w] += X[ind1+hour[1]]
            X_1500[w] += X[ind1+hour[2]]
            X_1800[w] += X[ind1+hour[3]]
            X_2000[w] += X[ind1+hour[4]]
            
    X_1200[w]/= 15
    X_1300[w]/= 15
    X_1500[w]/= 15
    X_1800[w]/= 15
    X_2000[w]/= 15
    

time_ass = np.zeros([90])
ind = 800
for i in range(18):
    time_ass[i*5+ 0] = X_1200[i, ind]
    time_ass[i*5+ 1] = X_1300[i, ind]
    time_ass[i*5+ 2] = X_1500[i, ind]
    time_ass[i*5+ 3] = X_1800[i, ind]
    time_ass[i*5+ 4] = X_2000[i, ind]
    
plt.figure()
plt.plot(time_ass)
plt.scatter(np.arange(0, 90, 5), X_1200[:, ind], c = 'b')
plt.scatter(np.arange(1, 91, 5), X_1300[:, ind], c = 'g')
plt.scatter(np.arange(2, 91, 5), X_1500[:, ind], c = 'r')
plt.scatter(np.arange(3, 91, 5), X_1800[:, ind], c = 'orange')
plt.scatter(np.arange(4, 91, 5), X_2000[:, ind], c = 'black')
plt.legend(['time series', '12:00', '13:00', '15:00', '18:00', '20:00'])
plt.show()


Feature = np.concatenate((X_1200, X_1300, X_1500, X_1800, X_2000), axis = 0)

# log scaler
Feature = Feature / np.mean(Feature, axis = 0)
'''
#%% Clustering

print('-- Clustering: Parameter Study')

cost = np.zeros([46, 1])

for nclusters in range(5, 51):
    #nclusters = 10

    clu = clustering(nclusters)
    # -5 since we start with n_clusters= 5
    # sort_index: index of consumers sorted according to the cluster assignments
    # clust_interval: intervals boundaries for assignment to the same cluster
    # cost: clustering cost    
    sort_index, clust_interval, cost[nclusters-5], _ = clu.kmeans(Feature, time)

# plot clustering cost versus number of clsuters
fig, ax = plt.subplots(figsize = (9, 5))
ax.plot(np.arange(5, 51), cost*1e-3, color = [103/256, 184/256, 247/256], linewidth = 5)
ax.set_xlabel('#Cluster')
ax.set_ylabel('Cost [k]')
ax.set_yticks(np.arange(20, 55, 5))
ax.set_yticklabels(np.int32(np.arange(20, 55, 5)))
plt.grid(axis = 'x', color = 'whitesmoke')
plt.grid(axis = 'y')
plt.rcParams.update({'font.size': 20})
plt.tight_layout()
plt.show()


#%% Clustering
print('-- Clustering')

# number of clusters used
nclusters = 25

# initializer kmeans
clu = clustering(nclusters, flag = True)
sort_index, clust_interval, _, clust_assignment = clu.kmeans(Feature, time)

# save mapping ID -> cluster assignment
np.save(path_save+ 'map_ID_cluster', [ID, clust_assignment])

#%% save clusters
for n in range(nclusters):
    Cluster = X[:, sort_index[int(clust_interval[n]): int(clust_interval[n+1])]]
    ID_cluster = ID[sort_index[int(clust_interval[n]): int(clust_interval[n+1])]]
    
    # normalization
    '''
    # min - max scaler
    Profile_min = np.min(Cluster, axis = 0)
    Profile_max = np.max(Cluster, axis = 0)
    Cluster_standard = (Cluster - Profile_min) / (Profile_max - Profile_min)
    '''
    '''
    # standard scaler
    Profile_mean = np.mean(Cluster, axis = 0)
    Profile_std = np.std(Cluster, axis = 0)
    Cluster_standard = (Cluster - Profile_mean) / Profile_std
    '''
    # log scaler
    # profile consists of data from April 14' to Aug 16'
    # mean demand is calculated for April 14' - April 16' to avoid bias
    # due to an unequal number of individual seasons considered
    # Further, it is assumed that the mean demand does not change over the
    # considered period -> no distinction between 14', 15' and 16'
    Profile_mean = np.mean(Cluster[:2*365*96], axis = 0)
    Cluster_standard = Cluster / Profile_mean
    
    # number of points (profiles) in cluster
    C_size = Cluster.shape[1]
    
    # mean profile (in the sense of the average over the profiles) of standardized cluster
    Cluster_standard_mean = np.mean(Cluster_standard, axis = 1)
    
    # save mean profile
    df = pd.DataFrame(data = Cluster_standard_mean)
    df.to_csv(path_save+ 'Cluster'+str(n)+'_meanprofile_'+str(C_size)+'.csv', header= None, index= None)
    del df
    
    # save min/ max for each profile
    # data = {'ID' : ID_cluster, 'min' : Profile_min, 'max' : Profile_max}
    # data = {'ID' : ID_cluster, 'mean' : Profile_mean, 'std' : Profile_std}
    data = {'ID' : ID_cluster, 'mean' : Profile_mean}

    df = pd.DataFrame(data)
    my_file = Path(path_save+ 'Cluster_normalization.csv')
    
    # create folder if not already exist
    if my_file.is_file():        
        with open(my_file, 'a') as f:
            df.to_csv(f, header = None, sep = ';')
            
    else:
        df.to_csv(my_file, sep = ';')
        
        
    # plot profiles and cluster    
    plt.figure()
    plt.plot(Cluster_standard[:30*96])
    plt.plot(Cluster_standard_mean[30* 96], c = 'r', linewidth = 6)
    plt.show()
    plt.savefig(path_plot + 'Cluster_'+ str(n)+ '_long')
    plt.close()
