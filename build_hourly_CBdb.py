import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import urllib2 as url
from datetime import date, datetime, timedelta
import time
import pytz
from pytz import timezone
import subprocess

CBmonths = np.append(np.append(np.append(np.ones((5,1))*2013, np.ones((12,1))*2014), np.ones((2,1))*2015), \
    np.array(range(8,13,1) + range(1,13,1) + range(1,3,1))).astype('int').reshape(2,19).T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(date(CBmonths[i,0], CBmonths[i,1], 1))
CBcsv_filename = '/home/vagrant/citibike/trip_logs/citibikeDB_' \
    + mo_strs[0].strftime('%m-%Y') + '_to_' + mo_strs[-1].strftime('%m-%Y') + '.csv.gz'
tripDF = pd.read_csv(CBcsv_filename, compression='gzip', index_col=0)
del tripDF['start station name'], tripDF['end station name'], tripDF['start station latitude'], \
    tripDF['start station longitude'], tripDF['end station latitude'], tripDF['end station longitude'], \
    tripDF['tripduration'], tripDF['bikeid'], tripDF['birth year']

# #*** DEBUG ***
# tripDF = tripDF[0:10000]
# import pdb
# pdb.set_trace()

# Build hourly histograms of trips from and to each station and combine with weather info 
# into one data frame
# NOTE: The time module DOES account for daylight savings by default, but the datetime 
# module DOES NOT
reftime = time.mktime(time.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'))
tripDF['startSecFrom2000'] = tripDF['starttime'].map(lambda x: \
    time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))) - reftime
tripDF['stopSecFrom2000'] = tripDF['stoptime'].map(lambda x: \
    time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S'))) - reftime

binstart = time.mktime(time.strptime(date(CBmonths[0,0], CBmonths[0,1], 1 \
    ).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'))
binend = time.mktime(time.strptime(date(CBmonths[-1,0], CBmonths[-1,1] + 1, 1 \
    ).strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S'))
hourbins = np.linspace(binstart-reftime, binend-reftime, (binend-binstart)/3600 + 1)
bins_datetime = [datetime.fromtimestamp(x+reftime) for x in hourbins]
tripDF['startbin'] = np.digitize(tripDF['startSecFrom2000'], hourbins) - 1
tripDF['stopbin'] = np.digitize(tripDF['stopSecFrom2000'], hourbins) - 1

## Grouping object by start time bin and start/end stations
start_group = tripDF.groupby(['startbin', 'start station id', 'end station id'])

# Create new dataframe for registered hourly stats
hourlyDF_bystart = pd.DataFrame({'total count': start_group['usertype'].count()})
hourlyDF_bystart['start time'] = [bins_datetime[x[0]] for x in hourlyDF_bystart.index.values]
hourlyDF_bystart = hourlyDF_bystart[['start time', 'total count']]
hourlyDF_bystart['male count'] = start_group['gender'].aggregate(lambda x: np.sum(x == 'm'))
hourlyDF_bystart['female count'] = start_group['gender'].aggregate(lambda x: np.sum(x == 'f'))
hourlyDF_bystart['subscriber count'] = start_group['usertype'].aggregate(\
    lambda x: np.sum(x == 'Subscriber'))

# Remove entries outside of hour bins
if hourlyDF_bystart.index.max()[0] == len(hourbins) - 1:
    hourlyDF_bystart = hourlyDF_bystart.drop(len(hourbins) - 1)

# Save results
outfilename = '/home/vagrant/citibike/hourlyDB_byStart_' + mo_strs[0].strftime('%m-%Y') + '_to_' + \
    mo_strs[-1].strftime('%m-%Y') + '.csv'
hourlyDF_bystart.to_csv(outfilename)
psout = subprocess.call(['gzip', outfilename])

# Clear up a bit of space in memory
del hourlyDF_bystart

## Grouping object by end time bin and start/end stations
end_group = tripDF.groupby(['stopbin', 'end station id', 'start station id'])

# Create new dataframe for registered hourly stats
hourlyDF_byend = pd.DataFrame({'total count': end_group['usertype'].count()})
hourlyDF_byend['stop time'] = [bins_datetime[x[0]] for x in hourlyDF_byend.index.values]
hourlyDF_byend = hourlyDF_byend[['stop time', 'total count']]
hourlyDF_byend['male count'] = end_group['gender'].aggregate(lambda x: np.sum(x == 'm'))
hourlyDF_byend['female count'] = end_group['gender'].aggregate(lambda x: np.sum(x == 'f'))
hourlyDF_byend['subscriber count'] = end_group['usertype'].aggregate(\
    lambda x: np.sum(x == 'Subscriber'))

# Remove entries outside of hour bins
if hourlyDF_byend.index.max()[0] == len(hourbins) - 1:
    hourlyDF_byend = hourlyDF_byend.drop(len(hourbins) - 1)

# Save results
outfilename = '/home/vagrant/citibike/hourlyDB_byEnd_' + mo_strs[0].strftime('%m-%Y') + '_to_' + \
    mo_strs[-1].strftime('%m-%Y') + '.csv'
hourlyDF_byend.to_csv(outfilename)
psout = subprocess.call(['gzip', outfilename])
