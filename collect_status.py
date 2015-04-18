import numpy as np
import pandas as pd
import datetime, time, sched

s = sched.scheduler(time.time, time.sleep)

def try_status_pull(status_url="http://www.citibikenyc.com/stations/json", \
                       out_file_base="/home/vagrant/citibike/status_data/status_", \
                       retry_delay=30):
    print "Pulling from", status_url, "at ", time.ctime()
    try:
        pull_status_to_csv(status_url, out_file_base)
    except:
        print "Error accessing status_url... trying again in" + str(retry_delay) + "seconds"
        time.sleep(retry_delay)
        try:
            pull_status_to_csv(status_url, out_file_base)
        except:
            print "Second attempt failed... skipping this status pull"

def pull_status_to_csv(status_url="http://www.citibikenyc.com/stations/json", \
                       out_file_base="/home/vagrant/citibike/status_data/status_"):
    df = pd.read_json(status_url)
    stat_time = df.executionTime[0];
    df = pd.concat([pd.DataFrame.from_dict(item, orient="index").T for item in df.stationBeanList])
    df = df.sort("id")[["id","stationName","stAddress1","totalDocks","availableBikes", \
        "availableDocks","statusValue","statusKey","testStation","latitude","longitude"]]
    df = df.set_index(np.arange(df["id"].count()))
    outfilepath = out_file_base + stat_time.replace(" PM","PM").replace(" ","_").replace(":",".") + \
        u".csv"
    df.to_csv(outfilepath)
        

def periodic_status_pulls(start_time=time.time(), delay_sec=(5*60), stop_time=None):
    if stop_time is None:
        stop_time = start_time + 60*60*24*365 # 1 year
    pull_times = np.arange(start_time, stop_time, delay_sec)
    pull_times = pull_times[pull_times > time.time()]
    print "Starting Citi Bike status pulls every", delay_sec, "seconds at", \
        time.ctime(pull_times[0]), "until", time.ctime(pull_times[-1])
    for event in pull_times:
        s.enterabs(event, 1, try_status_pull, ())
    s.run()

# import pdb
# pdb.set_trace()

start_time = time.mktime(time.strptime("03 25 15, 00 00 EDT", "%m %d %y, %H %M %Z"))
periodic_status_pulls(start_time, 5*60)
