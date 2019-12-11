# -*- coding: utf-8 -*-
"""
Script to separate large customers (industrial) from small customers (households, shops)
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np


#%% Extract small customers
path_read = './../1Raw_Data/SmartMeter/'
path_save = './../1Raw_Data/SmartMeter/'

# read in data
X = np.load(path_read+ 'X_1_6.npy')
time = np.load(path_read+ 't_1_6.npy')
ID = np.load(path_read+ 'ID_1_6.npy')
#%%
# Averaging
X_avg = np.average(X[0: 2* 365* 96], axis= 0)
X_avg_sort= np.sort(X_avg)

# Prepare x-axis of plot (linear spacing)
no_HH = np.arange(0, X_avg.shape[0])
no_HH= no_HH.astype(int)

# Plot
fig, ax = plt.subplots(figsize = (9, 5))
ax.plot(no_HH, X_avg_sort, color = [103/256, 184/256, 247/256], linewidth = 5)
#plt.xticks([0, 4, 9, 14, 18], [2016, 2020, 2025, 2030, 2034], rotation=45)
ax.set_xlabel('Consumer')
ax.set_ylabel('Average Electricity Demand [W]')
ax.set_yscale('log')
ax.grid(axis = 'x', color = 'whitesmoke')
ax.grid(axis = 'y')
plt.title('Average Electricity Demand ')
plt.tight_layout()
plt.rcParams.update({'font.size': 20})
plt.show()

# Array to store large customers
no_LC= np.empty([1, 0])

thresh = 1.5E3

for x, v in enumerate(X_avg):
    if X_avg[x]> thresh:
        no_LC= np.append(no_LC, x)

# Plot Large customers
N= 7* 24* 4
plt.figure()

no_LC= no_LC.astype(int)

for i in no_LC:
    plt.plot(time[:N], X[:N, int(i)])
     
plt.title('Large Customers')
plt.show()

# Extract small Customers by excluding columns of large customers
no_ALL = np.linspace(0, X.shape[1]- 1, X.shape[1])
no_ALL = no_ALL.astype(int)

no_SC = np.setdiff1d(no_ALL, no_LC)
no_SC = no_SC.astype(int)

# Save results
X_small = X[:, no_SC]
ID_small = ID[no_SC]

X_large = X[:, no_LC]
ID_large = ID[no_LC]

np.save(path_save+ 'X_1_6_small', X_small)
np.save(path_save+ 'ID_1_6_small', ID_small)

np.save(path_save+ 'X_1_6_large', X_large)
np.save(path_save+ 'ID_1_6_large', ID_large)

