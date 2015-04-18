import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.decomposition import PCA, FastICA, ProjectedGradientNMF
from datetime import date, datetime, timedelta
import time
import pytz
from pytz import timezone
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import map_figures as map

# Load Citi Bike data
CBmonths = np.array([np.array([np.ones((6,1))*2013, np.ones((6,1))*2014]).flatten(),     np.array([np.arange(7,13,1), np.arange(1,7,1)]).flatten()], dtype='int').T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(date(CBmonths[i,0], CBmonths[i,1], 1))

startDB_filename = "/home/vagrant/citibike/hourlyDB_byStart_" + mo_strs[0].strftime("%m-%Y") +     "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
hourlyDF_bystart = pd.read_csv(startDB_filename, compression="gzip")

endDB_filename = "/home/vagrant/citibike/hourlyDB_byEnd_" + mo_strs[0].strftime("%m-%Y") +     "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
hourlyDF_byend = pd.read_csv(endDB_filename, compression="gzip")

# Load station info
station_df = pd.read_csv("/home/vagrant/citibike/station_list.csv", index_col="station id")
nstations = station_df.shape[0]

# Trip-end-grouped hourly counts
endDB_filename = "/home/vagrant/citibike/hourlyDB_byEnd_" + mo_strs[0].strftime("%m-%Y") +     "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
hourlyDF_byend = pd.read_csv(endDB_filename, compression="gzip")

# DataFrames for outflow and inflow of bikes into stations
outflow = hourlyDF_bystart.groupby(['startbin', 'start station id']).sum()['total count']
outflow.index.names = ['timebin', 'station id']
outflow = DataFrame({'outgoing': outflow})
inflow = hourlyDF_byend.groupby(['stopbin', 'end station id']).sum()['total count']
inflow.index.names = ['timebin', 'station id']
inflow = DataFrame({'incoming':inflow})

# Merge outflow and inflow data into one DataFrame
station_traffic_df = pd.merge(DataFrame(outflow), DataFrame(inflow), left_index=True, right_index=True, how='outer')
station_traffic_df = station_traffic_df.fillna(0)

# Matrix for decomposition algorithms
station_traffic_matrix = station_traffic_df.unstack().fillna(0) # still type DataFrame

# Centering and normalizing flows at each station prevents decompositions from being 
# dominated by high-traffic stations
traffic_centNorm = np.matrix((station_traffic_matrix - station_traffic_matrix.mean(axis=0))/station_traffic_matrix.std(axis=0))


###########################
###########################
# Map of traffic volumes during morning & evening commutes, and on weekends
fig = plt.figure(figsize=(16,16))
ax1 = plt.subplot(411)
plt.plot(outflow.sum(level='timebin').ix[3300:3800], color='k')
plt.plot(inflow.sum(level='timebin').ix[3300:3800], color='b')
plt.plot(outflow['outgoing'].subtract(inflow['incoming'], fill_value=0).abs().sum(level='timebin').ix[3300:3800], color='r')
ax1 = plt.subplot(411)
###########################
###########################


# PCA decomposition
traffic_PCA = PCA()
traffic_PCs = traffic_PCA.fit_transform(traffic_centNorm)
traffic_PC_loadings = traffic_PCA.inverse_transform(np.diag(np.ones(traffic_PCs.shape[1])))

# Non-negative matrix factorization on scaled (but NOT normalized) traffic matrix
traffic_NMF = ProjectedGradientNMF(n_components=4)
traffic_NMCs = traffic_NMF.fit_transform(station_traffic_matrix/station_traffic_matrix.std(axis=0))

# ICA decomposition
traffic_ICA = FastICA(n_components=4)
traffic_ICs = np.matrix(traffic_ICA.fit_transform(traffic_centNorm)) # (n_samples, n_components)
traffic_IC_loadings = np.matrix(traffic_ICA.mixing_.T) # (n_components, n_features)
traffic_IC_strengths = np.zeros(traffic_IC_loadings.shape[0])
for i in range(traffic_IC_loadings.shape[0]):
    traffic_IC_strengths[i] = np.sum(np.var(np.dot(traffic_IC_loadings[i,:].T, \
        traffic_ICs[:,i].T), axis=1))/traffic_centNorm.var(axis=0).sum()

ICi = 0
fig = plt.figure(figsize=(16,16))
outax = plt.subplot(121)
map_plot('/home/vagrant/citibike/osm_screen1.png', \
    zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    colors=mpl.cm.jet(np.array((traffic_IC_loadings_4[ICi,:nstations] + \
    np.abs(traffic_IC_loadings_4[ICi,:]).max(axis=1))/(2*np.abs(traffic_IC_loadings_4[ICi,:]).max(axis=1))).flatten()), \
    corners_LatLong='/home/vagrant/citibike/osm_screen1_corners_LatLong.txt')
inax = plt.subplot(122)
map_plot('/home/vagrant/citibike/osm_screen1.png', \
    zip(station_df['latitude'].ix[station_traffic_matrix['incoming'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['incoming'].columns]), \
    colors=mpl.cm.jet(np.array((traffic_IC_loadings_4[ICi,nstations:] + \
    np.abs(traffic_IC_loadings_4[ICi,:]).max(axis=1))/(2*np.abs(traffic_IC_loadings_4[ICi,:]).max(axis=1))).flatten()), \
    corners_LatLong='/home/vagrant/citibike/osm_screen1_corners_LatLong.txt')

# Typical daily time patterns for CartoDB animations
wkday_hour_sum = station_traffic_matrix.ix[station_traffic_matrix['weekday'].isin(range(0,5)) \
    ].groupby('hour').apply(sum)[['outgoing', 'incoming']]
wkend_hour_sum = station_traffic_matrix.ix[station_traffic_matrix['weekday'].isin(range(5,7)) \
    ].groupby('hour').apply(sum)[['outgoing', 'incoming']]

station_traffic_imbal = station_traffic_matrix['outgoing'] - station_traffic_matrix['incoming']
station_traffic_imbal['hour'] = station_traffic_matrix['hour']
station_traffic_imbal['weekday'] = station_traffic_matrix['weekday']

wkday_hour_imbal = station_traffic_imbal.ix[station_traffic_matrix['weekday'].isin(range(0,5)) \
    ].groupby('hour').apply(sum)
del wkday_hour_imbal['weekday']
del wkday_hour_imbal['hour']
wkday_hour_imbal.columns = pd.MultiIndex.from_tuples(zip(['out - in' for i in range(len(wkend_hour_sum['outgoing'].columns))], wkend_hour_sum['outgoing'].columns), names=[None,'station id'])
wkday_hour_sum = wkday_hour_sum.merge(wkday_hour_imbal, left_index=True, right_index=True)
wkday_hour_sum['out - in'] = wkday_hour_sum['out - in'] - wkday_hour_sum['out - in'].min()

wkend_hour_imbal = station_traffic_imbal.ix[station_traffic_matrix['weekday'].isin(range(5,7)) \
    ].groupby('hour').apply(sum)
del wkend_hour_imbal['weekday']
del wkend_hour_imbal['hour']
wkend_hour_imbal.columns = pd.MultiIndex.from_tuples(zip(['out - in' for i in range(len(wkend_hour_sum['outgoing'].columns))], wkend_hour_sum['outgoing'].columns), names=[None,'station id'])
wkend_hour_sum = wkend_hour_sum.merge(wkend_hour_imbal, left_index=True, right_index=True)

# wkday_hour_mean = station_traffic_matrix.ix[station_traffic_matrix['weekday'].isin(range(0,5)) \
#     ].groupby('hour').mean()[['outgoing', 'incoming']]
# wkend_hour_mean = station_traffic_matrix.ix[station_traffic_matrix['weekday'].isin(range(5,7)) \
#     ].groupby('hour').mean()[['outgoing', 'incoming']]
# 
# station_traffic_imbal = station_traffic_matrix['outgoing'] - station_traffic_matrix['incoming']
# station_traffic_imbal['hour'] = station_traffic_matrix['hour']
# station_traffic_imbal['weekday'] = station_traffic_matrix['weekday']
# 
# wkday_hour_imbal = station_traffic_imbal.ix[station_traffic_matrix['weekday'].isin(range(0,5)) \
#     ].groupby('hour').mean()
# del wkday_hour_imbal['weekday']
# wkday_hour_imbal.columns = pd.MultiIndex.from_tuples(zip(['out - in' for i in range(len(wkday_hour_mean['outgoing'].columns))], wkday_hour_mean['outgoing'].columns), names=[None,'station id'])
# wkday_hour_mean = wkday_hour_mean.merge(wkday_hour_imbal, left_index=True, right_index=True)
# 
# wkend_hour_imbal = station_traffic_imbal.ix[station_traffic_matrix['weekday'].isin(range(5,7)) \
#     ].groupby('hour').mean()
# del wkend_hour_imbal['weekday']
# wkend_hour_imbal.columns = pd.MultiIndex.from_tuples(zip(['out - in' for i in range(len(wkend_hour_mean['outgoing'].columns))], wkend_hour_mean['outgoing'].columns), names=[None,'station id'])
# wkend_hour_mean = wkend_hour_mean.merge(wkend_hour_imbal, left_index=True, right_index=True)

# Create CartoDB-compatible csv file
wkday_traffic_cartoDB = wkday_hour_mean.stack()
wkday_traffic_cartoDB['latitude'] = station_df['latitude'].ix[np.array(zip(*wkday_traffic_cartoDB.index.values)[1])].values
wkday_traffic_cartoDB['longitude'] = station_df['longitude'].ix[np.array(zip(*wkday_traffic_cartoDB.index.values)[1])].values

