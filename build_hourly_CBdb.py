import numpy as np
import pandas as pd
import urllib2 as url
from datetime import date, datetime, timedelta
import time
import pytz
from pytz import timezone
import subprocess

CBmonths = np.array([np.array([np.ones((6,1))*2013, np.ones((6,1))*2014]).flatten(), \
    np.array([np.arange(7,13,1), np.arange(1,7,1)]).flatten()], dtype='int').T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(datetime.date(CBmonths[i,0], CBmonths[i,1], 1))
CBcsv_filename = "/home/vagrant/citibike/trip_logs/citibikeDB_" \
    + mo_strs[0].strftime("%m-%Y") + "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
tripDF = pd.read_csv(CBcsv_filename, compression="gzip")

# #*** DEBUG ***
# tripDF = tripDF[0:10000]
# import pdb
# pdb.set_trace()

# Build hourly histograms of trips from and to each station and combine with weather info 
# into one data frame
# NOTE: The time module DOES account for daylight savings by default, but the datetime 
# module DOES NOT
reftime = time.mktime(time.strptime("2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"))
tripDF["startSecFrom2000"] = tripDF.starttime.apply(lambda x: \
    time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))) - reftime
tripDF["stopSecFrom2000"] = tripDF.stoptime.apply(lambda x: \
    time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))) - reftime

binstart = time.mktime(time.strptime(date(CBmonths[0,0], CBmonths[0,1], 1 \
    ).strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
binend = time.mktime(time.strptime(date(CBmonths[-1,0], CBmonths[-1,1] + 1, 1 \
    ).strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
hourbins = np.linspace(binstart-reftime, binend-reftime, (binend-binstart)/3600 + 1)
tripDF["startbin"] = np.digitize(tripDF.startSecFrom2000, hourbins) - 1
tripDF["stopbin"] = np.digitize(tripDF.stopSecFrom2000, hourbins) - 1

# For calculating age, convert the birth year strings to numbers
tripDF["birth year"] = tripDF["birth year"].apply(lambda x: x.replace("\N", "0")).apply(int)
tripDF["birth year"].ix[[i-1 for i, elem in enumerate([(x < 1000) \
    for x in tripDF["birth year"]], 1) if elem]] = None
tripDF["birth year"] = np.array(tripDF["birth year"], dtype=float)

## Grouping object by start time bin and start/end stations
start_group = tripDF.groupby(["startbin", "start station id", "end station id"])

# Create new dataframe for registered hourly stats
hourlyDF_bystart = pd.DataFrame({"start time": start_group["tripduration"].count(), "total count": start_group["tripduration"].count()})
hourlyDF_bystart.columns = []
hourlyDF_bystart["male count"] = start_group["gender"].aggregate(lambda x: np.sum(x == 'm'))
hourlyDF_bystart["female count"] = start_group["gender"].aggregate(lambda x: np.sum(x == 'f'))
hourlyDF_bystart["subscriber count"] = start_group["usertype"].aggregate(\
    lambda x: np.sum(x == 'Subscriber'))
hourlyDF_bystart["mean duration"] = start_group["tripduration"].mean()
hourlyDF_bystart["mean age"] = start_group["birth year"].aggregate(\
    lambda year: np.round(2014 - np.nanmean(year)))

# Remove entries outside of hour bins
if hourlyDF_bystart.index.min()[0] == 0:
    hourlyDF_bystart = hourlyDF_bystart.drop(0)
if hourlyDF_bystart.index.max()[0] == len(hourbins):
    hourlyDF_bystart = hourlyDF_bystart.drop(len(hourbins))

# Save results
outfilename = "/home/vagrant/citibike/hourlyDB_byStart_" + mo_strs[0].strftime("%m-%Y") + "_to_" + \
    mo_strs[-1].strftime("%m-%Y") + ".csv"
hourlyDF_bystart.to_csv(outfilename)
psout = subprocess.call(["gzip", outfilename])

## Grouping object by end time bin and start/end stations
end_group = tripDF.groupby(["stopbin", "end station id", "start station id"])

# Create new dataframe for registered hourly stats
hourlyDF_byend = pd.DataFrame(end_group["tripduration"].count())
hourlyDF_byend.columns = ["total count"]
hourlyDF_byend["male count"] = end_group["gender"].aggregate(lambda x: np.sum(x == 'm'))
hourlyDF_byend["female count"] = end_group["gender"].aggregate(lambda x: np.sum(x == 'f'))
hourlyDF_byend["subscriber count"] = end_group["usertype"].aggregate(\
    lambda x: np.sum(x == 'Subscriber'))
hourlyDF_byend["mean duration"] = end_group["tripduration"].mean()
hourlyDF_byend["mean age"] = end_group["birth year"].aggregate(\
    lambda year: np.round(2014 - np.nanmean(year)))

# Remove entries outside of hour bins
if hourlyDF_byend.index.min()[0] == 0:
    hourlyDF_byend = hourlyDF_byend.drop(0)
if hourlyDF_byend.index.max()[0] == len(hourbins):
    hourlyDF_byend = hourlyDF_byend.drop(len(hourbins))

# Save results
outfilename = "/home/vagrant/citibike/hourlyDB_byEnd_" + mo_strs[0].strftime("%m-%Y") + "_to_" + \
    mo_strs[-1].strftime("%m-%Y") + ".csv"
hourlyDF_byend.to_csv(outfilename)
psout = subprocess.call(["gzip", outfilename])
