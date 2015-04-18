import numpy as np
import pandas as pd
import urllib2 as url
import datetime
import subprocess

# import pdb
# pdb.set_trace()

CBmonths = np.array([np.array([np.ones((6,1))*2013, np.ones((6,1))*2014]).flatten(), \
    np.array([np.arange(7,13,1), np.arange(1,7,1)]).flatten()], dtype='int').T
CBcsv_filebase = "/home/vagrant/citibike/trip_logs/YYYY-MM - Citi Bike trip data.csv"
CBcsv_filenames = []
mo_strs = []
CBdf = pd.DataFrame()
for i in range(CBmonths.shape[0]):
    mo_strs.append(datetime.date(CBmonths[i,0], CBmonths[i,1], 1))
    CBcsv_filenames.append(CBcsv_filebase.replace("YYYY-MM", mo_strs[i].strftime("%Y-%m")))
    CBdf = CBdf.append(pd.read_csv(CBcsv_filenames[i]), ignore_index=True)

CBdf = CBdf.sort("starttime")
CBdf.gender[CBdf.gender == 0] = "u"
CBdf.gender[CBdf.gender == 1] = "m"
CBdf.gender[CBdf.gender == 2] = "f"

# Save database
outfilename = "/home/vagrant/citibike/trip_logs/citibikeDB_" + mo_strs[0].strftime("%m-%Y") + "_to_" \
    + mo_strs[-1].strftime("%m-%Y") + ".csv"
CBdf.to_csv(outfilename)
psout = subprocess.call(["gzip", outfilename])
