import numpy as np
import pandas as pd
import urllib2 as url
import datetime
import subprocess

# import pdb
# pdb.set_trace()

CBmonths = np.append(np.append(np.append(np.ones((5,1))*2013, np.ones((12,1))*2014), np.ones((2,1))*2015), \
    np.array(range(8,13,1) + range(1,13,1) + range(1,3,1))).astype('int').reshape(2,19).T
CBcsv_filebase = "/home/vagrant/citibike/trip_logs/YYYYMM-citibike-tripdata.csv"
CBcsv_filenames = []
mo_strs = []
CBdf = pd.DataFrame()
for i in range(CBmonths.shape[0]):
    mo_strs.append(datetime.date(CBmonths[i,0], CBmonths[i,1], 1))
    CBcsv_filenames.append(CBcsv_filebase.replace("YYYYMM", mo_strs[i].strftime("%Y%m")))
    CBdf = CBdf.append(pd.read_csv(CBcsv_filenames[i], parse_dates=[1,2]), ignore_index=True)

CBdf.gender[CBdf.gender == 0] = "u"
CBdf.gender[CBdf.gender == 1] = "m"
CBdf.gender[CBdf.gender == 2] = "f"

# Save database
outfilename = "/home/vagrant/citibike/trip_logs/citibikeDB_" + mo_strs[0].strftime("%m-%Y") + "_to_" \
    + mo_strs[-1].strftime("%m-%Y") + ".csv"
CBdf.to_csv(outfilename)
psout = subprocess.call(["gzip", outfilename])
