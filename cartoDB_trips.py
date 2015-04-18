import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from datetime import date, datetime, timedelta
import datetime as dt
import time
from pytz import timezone
import pytz

# Load data
CBmonths = np.array([np.array([np.ones((6,1))*2013, np.ones((6,1))*2014]).flatten(),     np.array([np.arange(7,13,1), np.arange(1,7,1)]).flatten()], dtype='int').T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(date(CBmonths[i,0], CBmonths[i,1], 1))

CBcsv_filename = "/home/vagrant/citibike/trip_logs/citibikeDB_" \
    + mo_strs[0].strftime("%m-%Y") + "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
tripDF = pd.read_csv(CBcsv_filename, compression="gzip", index_col=0)
tripDF["birth year"] = tripDF["birth year"].apply(lambda x: x.replace("\N", "0")).apply(int)
tripDF["birth year"].ix[[i-1 for i, elem in enumerate([(x < 1000) \
    for x in tripDF["birth year"]], 1) if elem]] = None
tripDF["birth year"] = np.array(tripDF["birth year"], dtype=float)

# Build and write for CartoDB
tripStartDF_cartoDB = DataFrame({'datetime': tripDF['starttime'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))})
tripStartDF_cartoDB['WEEKDAY'] = tripStartDF_cartoDB['datetime'].map(lambda x: x.weekday()).isin(range(0,5))
tripStartDF_cartoDB['start station id'] = tripDF['start station id']
tripStartDF_cartoDB['latitude'] = tripDF['start station latitude']
tripStartDF_cartoDB['longitude'] = tripDF['start station longitude']
tripStartDF_cartoDB['end station id'] = tripDF['end station id']
tripStartDF_cartoDB['bike id'] = tripDF['bikeid']
tripStartDF_cartoDB['user type'] = tripDF['usertype']
tripStartDF_cartoDB['gender'] = tripDF['gender']
tripStartDF_cartoDB['birth year'] = tripDF['birth year']

outfilename = '/home/vagrant/citibike/tripStart_cartoDB.csv'
tripStartDF_cartoDB.to_csv(outfilename)
psout = subprocess.call(["gzip", outfilename])

# Smaller version - randomly subsample
nsamples = 5e4
tripStartDF_small = tripStartDF_cartoDB[['datetime', 'WEEKDAY', 'latitude', 'longitude']].ix[\
    tripStartDF_cartoDB.index[np.random.choice(tripStartDF_cartoDB.shape[0], nsamples, replace=False)].values].sort_index()
tripStartDF_wkday_small = tripStartDF_small.ix[tripStartDF_small['WEEKDAY']]
tripStartDF_wkend_small = tripStartDF_small.ix[~tripStartDF_small['WEEKDAY']]
del tripStartDF_wkday_small['WEEKDAY'], tripStartDF_wkend_small['WEEKDAY']
tripStartDF_wkday_small.to_csv('/home/vagrant/citibike/tripStart_wkday_small_cartoDB.csv', index=False)
psout = subprocess.call(["gzip", '/home/vagrant/citibike/tripStart_wkday_small_cartoDB.csv'])
tripStartDF_wkend_small.to_csv('/home/vagrant/citibike/tripStart_wkend_small_cartoDB.csv', index=False)
psout = subprocess.call(["gzip", '/home/vagrant/citibike/tripStart_wkend_small_cartoDB.csv'])


# Now trip endings
tripEndDF_cartoDB = DataFrame({'datetime': tripDF['stoptime'].map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))})
tripEndDF_cartoDB['WEEKDAY'] = tripEndDF_cartoDB['datetime'].map(lambda x: x.weekday()).isin(range(0,5))
tripEndDF_cartoDB['end station id'] = tripDF['end station id']
tripEndDF_cartoDB['latitude'] = tripDF['end station latitude']
tripEndDF_cartoDB['longitude'] = tripDF['end station longitude']
# tripEndDF_cartoDB['start station id'] = tripDF['start station id']
# tripEndDF_cartoDB['bike id'] = tripDF['bikeid']
# tripEndDF_cartoDB['user type'] = tripDF['usertype']
# tripEndDF_cartoDB['gender'] = tripDF['gender']
# tripEndDF_cartoDB['birth year'] = tripDF['birth year']

# outfilename = '/home/vagrant/citibike/tripEnd_cartoDB.csv'
# tripEndDF_cartoDB.to_csv(outfilename)
# psout = subprocess.call(["gzip", outfilename])

# Smaller version - randomly subsample
tripEndDF_small = tripEndDF_cartoDB[['datetime', 'WEEKDAY', 'latitude', 'longitude']].ix[\
    tripEndDF_cartoDB.index[np.random.choice(tripEndDF_cartoDB.shape[0], nsamples, replace=False)].values].sort_index()
tripEndDF_wkday_small = tripEndDF_small.ix[tripEndDF_small['WEEKDAY']]
tripEndDF_wkend_small = tripEndDF_small.ix[~tripEndDF_small['WEEKDAY']]
del tripEndDF_wkday_small['WEEKDAY'], tripEndDF_wkend_small['WEEKDAY']
tripEndDF_wkday_small.to_csv('/home/vagrant/citibike/tripEnd_wkday_small_cartoDB.csv', index=False)
psout = subprocess.call(["gzip", '/home/vagrant/citibike/tripEnd_wkday_small_cartoDB.csv'])
tripEndDF_wkend_small.to_csv('/home/vagrant/citibike/tripEnd_wkend_small_cartoDB.csv', index=False)
psout = subprocess.call(["gzip", '/home/vagrant/citibike/tripEnd_wkend_small_cartoDB.csv'])
