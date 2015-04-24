import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.decomposition import PCA, FastICA, ProjectedGradientNMF
from datetime import date, datetime, timedelta
import time
import pytz
from pytz import timezone
import calendar
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import map_figures as map
import seaborn as sbs


## Load Citi Bike data
CBmonths = np.append(np.append(np.append(np.ones((5,1))*2013, np.ones((12,1))*2014), np.ones((2,1))*2015), \
    np.array(range(8,13,1) + range(1,13,1) + range(1,3,1))).astype('int').reshape(2,19).T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(date(CBmonths[i,0], CBmonths[i,1], 1))

startDB_filename = "/home/vagrant/citibike/hourlyDB_byStart_" + mo_strs[0].strftime("%m-%Y") + \
    "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
hourlyDF_bystart = pd.read_csv(startDB_filename, compression="gzip", \
    index_col=['startbin', 'start station id', 'end station id'])

endDB_filename = "/home/vagrant/citibike/hourlyDB_byEnd_" + mo_strs[0].strftime("%m-%Y") + \
    "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
hourlyDF_byend = pd.read_csv(endDB_filename, compression="gzip", \
    index_col=['stopbin', 'end station id', 'start station id'])

## Load station info
station_df = pd.read_csv("/home/vagrant/citibike/station_list.csv", index_col="station id")
nstations = station_df.shape[0]

## DataFrames for outflow and inflow of bikes into stations
outflow = hourlyDF_bystart['total count'].sum(axis=0, level=['startbin', 'start station id'])
outflow.index.names = ['timebin', 'station id']
outflow = DataFrame({'outgoing': outflow})
inflow = hourlyDF_byend['total count'].sum(axis=0, level=['stopbin', 'end station id'])
inflow.index.names = ['timebin', 'station id']
inflow = DataFrame({'incoming':inflow})

## Merge outflow and inflow data into one DataFrame
station_traffic_df = pd.merge(outflow, inflow, left_index=True, right_index=True, how='outer')
station_traffic_df = station_traffic_df.fillna(0)

## Matrix for decomposition algorithms
station_traffic_matrix = station_traffic_df.unstack().fillna(0) # still type DataFrame

## Add columns for datetime, hour, and weekday/holiday
reftime = time.mktime(time.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'))
binstart = time.mktime(time.strptime(date(CBmonths[0,0], CBmonths[0,1], 1 \
    ).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'))
binend = time.mktime(time.strptime(date(CBmonths[-1,0], CBmonths[-1,1] + 1, 1 \
    ).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'))
hourbins = np.linspace(binstart-reftime, binend-reftime, (binend-binstart)/3600 + 1)
bins_datetime = [datetime.fromtimestamp(x+reftime) for x in hourbins]

file = open('/home/vagrant/citibike/holidays.txt', 'r')
holiday_strs = []
for line in file:
    holiday_strs.append(line.rstrip())
file.close()
file = open('/home/vagrant/citibike/holiday_related.txt', 'r')
holidayRelated_strs = []
for line in file:
    holidayRelated_strs.append(line.rstrip())
file.close()
holidays = [datetime.strptime(x, '%Y-%m-%d').date() for x in holiday_strs]
holidayRelated = [datetime.strptime(x, '%Y-%m-%d').date() for x in holidayRelated_strs]

station_traffic_matrix['datetime'] = [bins_datetime[x] for x in station_traffic_matrix.index.values]
station_traffic_matrix['hour'] = station_traffic_matrix['datetime'].map(lambda x: x.hour)
station_traffic_matrix['WEEKDAY'] = station_traffic_matrix['datetime'].map(lambda x: calendar.weekday(x.year, x.month, x.day) in range(0,5))
station_traffic_matrix['HOLIDAY'] = station_traffic_matrix['datetime'].map(lambda x: x.date() in holidays + holidayRelated)
station_traffic_matrix['WEEKDAY'] = station_traffic_matrix['WEEKDAY'] & ~station_traffic_matrix['HOLIDAY']
station_traffic_matrix['WEEKEND'] = ~station_traffic_matrix['WEEKDAY'] & ~station_traffic_matrix['HOLIDAY']

station_traffic_imbal = station_traffic_matrix['outgoing'] - station_traffic_matrix['incoming']
station_traffic_imbal['datetime'] = station_traffic_matrix['datetime']
station_traffic_imbal['hour'] = station_traffic_matrix['hour']
station_traffic_imbal[['WEEKDAY', 'WEEKEND', 'HOLIDAY']] = station_traffic_matrix[['WEEKDAY', 'WEEKEND', 'HOLIDAY']]

AM_bools = station_traffic_matrix['hour'].isin(range(7,11)) & station_traffic_matrix['WEEKDAY']
PM_bools = station_traffic_matrix['hour'].isin(range(16,20)) & station_traffic_matrix['WEEKDAY']

## Typical daily time patterns
wkday_hour_mean = station_traffic_matrix.ix[station_traffic_matrix['WEEKDAY'] \
    ].groupby('hour').mean()[['outgoing', 'incoming']]
wkend_hour_mean = station_traffic_matrix.ix[station_traffic_matrix['WEEKEND'] \
    ].groupby('hour').mean()[['outgoing', 'incoming']]

wkday_hour_imbal = station_traffic_imbal.ix[station_traffic_matrix['WEEKDAY'] \
    ].groupby('hour').mean()
del wkday_hour_imbal['WEEKDAY'], wkday_hour_imbal['WEEKEND'], wkday_hour_imbal['HOLIDAY']
wkday_hour_imbal.columns = pd.MultiIndex.from_tuples(zip(['out - in' for i in \
    range(len(wkday_hour_mean['outgoing'].columns))], wkday_hour_mean['outgoing'].columns), names=[None,'station id'])
wkday_hour_mean = wkday_hour_mean.merge(wkday_hour_imbal, left_index=True, right_index=True)

wkend_hour_imbal = station_traffic_imbal.ix[station_traffic_matrix['WEEKEND'] \
    ].groupby('hour').mean()
del wkend_hour_imbal['WEEKDAY'], wkend_hour_imbal['WEEKEND'], wkend_hour_imbal['HOLIDAY']
wkend_hour_imbal.columns = pd.MultiIndex.from_tuples(zip(['out - in' for i in \
    range(len(wkend_hour_mean['outgoing'].columns))], wkend_hour_mean['outgoing'].columns), names=[None,'station id'])
wkend_hour_mean = wkend_hour_mean.merge(wkend_hour_imbal, left_index=True, right_index=True)

## Variables for map plotting
figdir = '/home/vagrant/citibike/figures/'
map_img = plt.imread(fname='/home/vagrant/citibike/osm_screen2.png')
map_corners_LatLong = np.loadtxt('/home/vagrant/citibike/osm_screen2_corners_LatLong.txt', delimiter=',')
map_pixel_LatLong = map.gridCoords_fromCorners(map_corners_LatLong, map_img.shape[0:2])
figsize1 = (8,8)
figsize2 = (8,8)
markersize1 = 13
markersize2 = 7
 
##================== FIGURES ==================
# ## In/Out-flow counts & asymmetries for each station during day
# fig = plt.figure()
# ax = plt.subplot(131)
# lin = plt.plot(wkday_hour_mean['incoming'].values)
# ax = plt.subplot(132)
# lin = plt.plot(wkday_hour_mean['outgoing'].values)
# ax = plt.subplot(133)
# lin = plt.plot(wkday_hour_mean['out - in'].values)
# fig.clear()

## Map showing avg bikes in and out for each station
fig = plt.figure(figsize=figsize1)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize1, \
    values=station_traffic_matrix['outgoing'].mean().values, \
    colorMappable=map.colorMappable_from_values(station_traffic_matrix['outgoing'].mean().values, \
        lims=[0,station_traffic_matrix['outgoing'].mean().max()], cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_outgoing.png', bbox_inches='tight', dpi=150)
fig.clear()

fig = plt.figure(figsize=figsize1)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['incoming'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['incoming'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize1, \
    values=station_traffic_matrix['incoming'].mean().values, \
    colorMappable=map.colorMappable_from_values(station_traffic_matrix['incoming'].mean().values, \
        lims=[0,station_traffic_matrix['incoming'].mean().max()], cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_incoming.png', bbox_inches='tight', dpi=150)
fig.clear()

## Traffic imbalances during morning & evening commutes, and on weekends
pltvals = wkday_hour_mean['out - in'].ix[range(7,11)].mean().values
fig = plt.figure(figsize=figsize1)
mfig = mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[wkday_hour_mean['out - in'].columns], \
    station_df['longitude'].ix[wkday_hour_mean['out - in'].columns]), grid_LatLong=map_pixel_LatLong, \
    marker='o', markersize=markersize1, alpha=0.8, values=pltvals, \
    colorMappable=map.colorMappable_from_values(pltvals, \
    lims=np.max(np.abs(pltvals))*np.array([-1,1]), cmap=mpl.cm.seismic) )
fig.savefig(figdir+'map_AM_imbal.png', bbox_inches='tight', dpi=150)
fig.clear()

pltvals = wkday_hour_mean['out - in'].ix[range(16,20)].mean().values
fig = plt.figure(figsize=figsize1)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[wkday_hour_mean['out - in'].columns], \
    station_df['longitude'].ix[wkday_hour_mean['out - in'].columns]), grid_LatLong=map_pixel_LatLong, \
    marker='o', markersize=markersize1, alpha=0.8, values=pltvals, \
    colorMappable=map.colorMappable_from_values(pltvals, \
    lims=np.max(np.abs(pltvals))*np.array([-1,1]), cmap=mpl.cm.seismic) )
fig.savefig(figdir+'map_PM_imbal.png', bbox_inches='tight', dpi=150)
fig.clear()

pltvals = wkend_hour_mean['out - in'].ix[range(12,17)].mean().values
fig = plt.figure(figsize=figsize1)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[wkend_hour_mean['out - in'].columns], \
    station_df['longitude'].ix[wkend_hour_mean['out - in'].columns]), grid_LatLong=map_pixel_LatLong, \
    marker='o', markersize=markersize1, alpha=0.8, values=pltvals, \
    colorMappable=map.colorMappable_from_values(pltvals, \
    lims=np.max(np.abs(pltvals))*np.array([-1,1]), cmap=mpl.cm.seismic) )
fig.savefig(figdir+'map_wkendAfternoon_imbal.png', bbox_inches='tight', dpi=150)
fig.clear()
##============================================


## Centering and normalizing flows at each station prevents decompositions from being 
## dominated by high-traffic stations
traffic_centNorm = np.matrix(station_traffic_matrix[['outgoing', 'incoming']].apply(\
    lambda x: (x - np.mean(x))/np.std(x)).values)

## PCA decomposition
traffic_PCA = PCA()
traffic_PCs = traffic_PCA.fit_transform(traffic_centNorm)
traffic_PC_loadings = traffic_PCA.inverse_transform(np.diag(np.ones(traffic_PCs.shape[1])))
traffic_PC_strengths = traffic_PCA.explained_variance_ratio_

avg_PCs_wkday = np.empty((24, traffic_PCs.shape[1]))
avg_PCs_wkend = np.empty((24, traffic_PCs.shape[1]))
for hour in range(24):
    wkday_hour_bools = station_traffic_matrix['WEEKDAY'] & (station_traffic_matrix['hour'] == hour)
    wkend_hour_bools = station_traffic_matrix['WEEKEND'] & (station_traffic_matrix['hour'] == hour)
    avg_PCs_wkday[hour,:] = traffic_PCs[wkday_hour_bools.values,:].mean(axis=0)
    avg_PCs_wkend[hour,:] = traffic_PCs[wkend_hour_bools.values,:].mean(axis=0)

## Save results
np.savez('/home/vagrant/citibike/traffic_PCs', traffic_PCA=traffic_PCA, \
    traffic_PCs=traffic_PCs, traffic_PC_loadings=traffic_PC_loadings, \
    traffic_PC_strengths=traffic_PC_strengths, avg_PCs_wkday=avg_PCs_wkday, \
    avg_PCs_wkend=avg_PCs_wkend)


##================== FIGURES ==================
PCi = 0
pltvalsOut = np.array(traffic_PC_loadings[PCi,:nstations]).flatten()
pltvalsIn = np.array(traffic_PC_loadings[PCi,nstations:]).flatten()
lims = np.max(np.abs(traffic_PC_loadings[PCi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_PC'+str(PCi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

PCi = 1
pltvalsOut = np.array(traffic_PC_loadings[PCi,:nstations]).flatten()
pltvalsIn = np.array(traffic_PC_loadings[PCi,nstations:]).flatten()
lims = np.max(np.abs(traffic_PC_loadings[PCi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_PC'+str(PCi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

PCi = 2
pltvalsOut = np.array(traffic_PC_loadings[PCi,:nstations]).flatten()
pltvalsIn = np.array(traffic_PC_loadings[PCi,nstations:]).flatten()
lims = np.max(np.abs(traffic_PC_loadings[PCi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_PC'+str(PCi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

# Avg PC activations
fig = plt.figure(figsize=(16.4375, 7.65))
fontsize=14
ax1 = plt.subplot(121)
plt.plot(np.concatenate((avg_PCs_wkday[:,:5], avg_PCs_wkday[0,:5].reshape((1,-1))), axis=0))
plt.xlim([0, 24])
plt.xticks(range(0,23,4)+[24], ['12:00 AM', '4:00 AM', '8:00 AM', '12:00 PM', '4:00 PM', '8:00 PM', '12:00 AM'], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title('Avg. weekday PC activations', fontsize=fontsize)
plt.ylabel('Explained variances: PC0=%.2f, PC1=%.2f, PC2=%.2f, PC3=%.2f, PC4=%.2f' % \
    (traffic_PC_strengths[0], traffic_PC_strengths[1], traffic_PC_strengths[2], \
    traffic_PC_strengths[3], traffic_PC_strengths[4]), fontsize=fontsize)

plt.legend(['PC0', 'PC1', 'PC2', 'PC3', 'PC4'], fontsize=fontsize)
plt.margins(0)

ax2 = plt.subplot(122)
plt.plot(np.concatenate((avg_PCs_wkend[:,:5], avg_PCs_wkend[0,:5].reshape((1,-1))), axis=0))
plt.xlim([0, 24])
plt.xticks(range(0,23,4)+[24], ['12:00 AM', '4:00 AM', '8:00 AM', '12:00 PM', '4:00 PM', '8:00 PM', '12:00 AM'], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title('Avg. weekend PC activations', fontsize=fontsize)
plt.legend(['PC0', 'PC1', 'PC2', 'PC3', 'PC4'], fontsize=fontsize)
plt.margins(0)

ylims = (np.min([ax1.get_ylim()[0], ax2.get_ylim()[0]]), np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]]))
ax1.set_ylim(ylims)
ax2.set_ylim(ylims)
fig.tight_layout()
fig.savefig(figdir+'avg_hourly_PCs.png', bbox_inches='tight')
fig.clear()
##============================================

# ## Non-negative matrix factorization on scaled (but NOT normalized) traffic matrix
# traffic_NMF = ProjectedGradientNMF(n_components=4)
# traffic_NMCs = traffic_NMF.fit_transform(station_traffic_matrix/station_traffic_matrix.std(axis=0))

## ICA decomposition
traffic_ICA = FastICA(n_components=5)
traffic_ICs = np.matrix(traffic_ICA.fit_transform(traffic_centNorm)) # (n_samples, n_components)
traffic_IC_loadings = np.matrix(traffic_ICA.mixing_.T) # (n_components, n_features)
traffic_IC_strengths = np.zeros(traffic_IC_loadings.shape[0])
for i in range(traffic_IC_loadings.shape[0]):
    traffic_IC_strengths[i] = np.sum(np.var(np.dot(traffic_IC_loadings[i,:].T, \
        traffic_ICs[:,i].T), axis=1))/traffic_centNorm.var(axis=0).sum()

## Order by explained variance
order = np.argsort(traffic_IC_strengths)[::-1]
traffic_ICs = traffic_ICs[:,order]
traffic_IC_loadings = traffic_IC_loadings[order,:]
traffic_IC_strengths = traffic_IC_strengths[order]

## Avg hourly PC & IC activations (weekday & weekend)
avg_ICs_wkday = np.empty((24, traffic_ICs.shape[1]))
avg_ICs_wkend = np.empty((24, traffic_ICs.shape[1]))
for hour in range(24):
    wkday_hour_bools = station_traffic_matrix['WEEKDAY'] & (station_traffic_matrix['hour'] == hour)
    wkend_hour_bools = station_traffic_matrix['WEEKEND'] & (station_traffic_matrix['hour'] == hour)
    avg_ICs_wkday[hour,:] = traffic_ICs[wkday_hour_bools.values,:].mean(axis=0)
    avg_ICs_wkend[hour,:] = traffic_ICs[wkend_hour_bools.values,:].mean(axis=0)

## If IC loadings and max activations are both negative, switch the signs
for i in range(traffic_ICs.shape[1]):
    combined_ICs = np.concatenate([avg_ICs_wkday[:,i], avg_ICs_wkend[:,i]], axis=0)
    if (combined_ICs[np.abs(combined_ICs).argmax()] < 0) \
        and (traffic_IC_loadings[i,:].mean() < 0):
        print 'Switching signs for IC', i
        traffic_IC_loadings[i,:] = -traffic_IC_loadings[i,:]
        avg_ICs_wkday[:,i] = -avg_ICs_wkday[:,i]
        avg_ICs_wkend[:,i] = -avg_ICs_wkend[:,i]
        traffic_ICs[:,i] = -traffic_ICs[:,i]

## Save results
np.savez('/home/vagrant/citibike/traffic_ICs', traffic_ICA=traffic_ICA, \
    order=order, traffic_ICs=traffic_ICs, traffic_IC_loadings=traffic_IC_loadings, \
    traffic_IC_strengths=traffic_IC_strengths, avg_ICs_wkday=avg_ICs_wkday, \
    avg_ICs_wkend=avg_ICs_wkend)

##================== FIGURES ==================
ICi = 0
pltvalsOut = np.array(traffic_IC_loadings[ICi,:nstations]).flatten()
pltvalsIn = np.array(traffic_IC_loadings[ICi,nstations:]).flatten()
lims = np.max(np.abs(traffic_IC_loadings[ICi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_IC'+str(ICi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

ICi = 1
pltvalsOut = np.array(traffic_IC_loadings[ICi,:nstations]).flatten()
pltvalsIn = np.array(traffic_IC_loadings[ICi,nstations:]).flatten()
lims = np.max(np.abs(traffic_IC_loadings[ICi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_IC'+str(ICi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

ICi = 2
pltvalsOut = np.array(traffic_IC_loadings[ICi,:nstations]).flatten()
pltvalsIn = np.array(traffic_IC_loadings[ICi,nstations:]).flatten()
lims = np.max(np.abs(traffic_IC_loadings[ICi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_IC'+str(ICi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

ICi = 3
pltvalsOut = np.array(traffic_IC_loadings[ICi,:nstations]).flatten()
pltvalsIn = np.array(traffic_IC_loadings[ICi,nstations:]).flatten()
lims = np.max(np.abs(traffic_IC_loadings[ICi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_IC'+str(ICi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

ICi = 4
pltvalsOut = np.array(traffic_IC_loadings[ICi,:nstations]).flatten()
pltvalsIn = np.array(traffic_IC_loadings[ICi,nstations:]).flatten()
lims = np.max(np.abs(traffic_IC_loadings[ICi,:]))*np.array([-1,1])
fig = plt.figure(figsize=figsize2)
outax = plt.subplot(121)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsOut, colorMappable=map.colorMappable_from_values(pltvalsOut, lims=lims, cmap=mpl.cm.jet) )
inax = plt.subplot(122)
mfig = map.map_plot( map_img, zip(station_df['latitude'].ix[station_traffic_matrix['outgoing'].columns], \
    station_df['longitude'].ix[station_traffic_matrix['outgoing'].columns]), \
    grid_LatLong=map_pixel_LatLong, marker='o', markersize=markersize2, alpha=0.8, \
    values=pltvalsIn, colorMappable=map.colorMappable_from_values(pltvalsIn, lims=lims, cmap=mpl.cm.jet) )
fig.savefig(figdir+'map_IC'+str(ICi)+'_out-in.png', bbox_inches='tight', dpi=150)
fig.clear()

fig = plt.figure(figsize=(16.4375, 7.65))
fontsize=14
ax1 = plt.subplot(121)
plt.plot(np.concatenate((avg_ICs_wkday, avg_ICs_wkday[0,:].reshape((1,-1))), axis=0))
plt.xlim([0, 24])
plt.xticks(range(0,23,4)+[24], ['12:00 AM', '4:00 AM', '8:00 AM', '12:00 PM', '4:00 PM', '8:00 PM', '12:00 AM'], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title('Avg. weekday IC activations', fontsize=fontsize)
plt.legend(['IC0', 'IC1', 'IC2', 'IC3', 'IC4'], fontsize=fontsize)
plt.ylabel('Explained variances: IC0=%.2f, IC1=%.2f, IC2=%.2f, IC3=%.2f, IC4=%.2f' % \
    (traffic_IC_strengths[0], traffic_IC_strengths[1], traffic_IC_strengths[2], \
    traffic_IC_strengths[3], traffic_IC_strengths[4]), fontsize=fontsize)
plt.margins(0)

ax2 = plt.subplot(122)
plt.plot(np.concatenate((avg_ICs_wkend, avg_ICs_wkend[0,:].reshape((1,-1))), axis=0))
plt.xlim([0, 24])
plt.xticks(range(0,23,4)+[24], ['12:00 AM', '4:00 AM', '8:00 AM', '12:00 PM', '4:00 PM', '8:00 PM', '12:00 AM'], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title('Avg. weekend IC activations', fontsize=fontsize)
plt.legend(['IC0', 'IC1', 'IC2', 'IC3', 'IC4'], fontsize=fontsize)
plt.margins(0)

ylims = (np.min([ax1.get_ylim()[0], ax2.get_ylim()[0]]), np.max([ax1.get_ylim()[1], ax2.get_ylim()[1]]))
ax1.set_ylim(ylims)
ax2.set_ylim(ylims)
fig.tight_layout()
fig.savefig(figdir+'avg_hourly_ICs.png', bbox_inches='tight')
fig.clear()
##============================================


# ## Create CartoDB-compatible csv file
# wkday_traffic_cartoDB = wkday_hour_mean.stack()
# wkday_traffic_cartoDB['latitude'] = station_df['latitude'].ix[np.array(zip(*wkday_traffic_cartoDB.index.values)[1])].values
# wkday_traffic_cartoDB['longitude'] = station_df['longitude'].ix[np.array(zip(*wkday_traffic_cartoDB.index.values)[1])].values

