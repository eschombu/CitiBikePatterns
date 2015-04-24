import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from datetime import datetime, date, timedelta

# Load Citi Bike data
CBmonths = np.array([np.array([np.ones((6,1))*2013, np.ones((6,1))*2014]).flatten(), \
    np.array([np.arange(7,13,1), np.arange(1,7,1)]).flatten()], dtype='int').T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(date(CBmonths[i,0], CBmonths[i,1], 1))

CBcsv_filename = "/home/vagrant/citibike/trip_logs/citibikeDB_" \
    + mo_strs[0].strftime("%m-%Y") + "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
tripDF = pd.read_csv(CBcsv_filename, compression="gzip")

# Starting stations
a = tripDF.groupby("start station id")["start station id", "start station name", \
    "start station latitude", "start station longitude"].first()
a.columns = ["station id", "station name", "latitude", "longitude"]

# Ending stations
b = tripDF.groupby("end station id")["end station id", "end station name", \
    "end station latitude", "end station longitude"].first()
b.columns = ["station id", "station name", "latitude", "longitude"]

# Merge two lists and set index as station id
station_df = pd.merge(a, b, how='outer').set_index("station id")

# Save list to file
station_file = open("/home/vagrant/citibike/station_list.csv", 'w')
station_df.to_csv(station_file)
station_file.close()
