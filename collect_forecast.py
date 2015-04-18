import pandas as pd
import numpy as np
import urllib2 as url
import json
import datetime, time, sched

s = sched.scheduler(time.time, time.sleep)

def pull_forecast_to_csv(out_file_base="/home/vagrant/citibike/forecasts/", \
                         apikey="8ea424e58efa01e7", stationkey="KNYNEWYO270"):
    
    url_str = "http://api.wunderground.com/api/%s/hourly10day/q/pws:%s.json" % (apikey, stationkey)
    f = url.urlopen(url_str)
    json_string = f.read()
    f.close()
    pulltime = time.time()
    
    # Convert downloaded forecast to pandas DataFrame
    json_string = json_string.replace("\n", "").replace("\t", "")
    parsed_json = json.loads(json_string)
    forecast_json = json.dumps(parsed_json['hourly_forecast'])
    forecast_df = pd.read_json(forecast_json, orient="records")
    forecast_df = forecast_df[["FCTTIME", "temp", "feelslike", "windchill", "heatindex", \
        "humidity", "qpf", "pop", "snow", "wspd", "wdir", "condition", "sky", "fctcode"]]
    forecast_df["FCTTIME"] = forecast_df["FCTTIME"].apply(lambda x: \
        datetime.datetime(int(x["year"]), int(x["mon"]), int(x["mday"]), \
        int(x["hour"]), int(x["min"]), int(x["sec"])))
    forecast_df["temp"] = forecast_df["temp"].apply(lambda x: int(x["english"]))
    forecast_df["feelslike"] = forecast_df["feelslike"].apply(lambda x: int(x["english"]))
    forecast_df["heatindex"] = forecast_df["heatindex"].apply(lambda x: int(x["english"]))
    forecast_df["qpf"] = forecast_df["qpf"].apply(lambda x: float(x["english"]))
    forecast_df["snow"] = forecast_df["snow"].apply(lambda x: float(x["english"]))
    forecast_df["wspd"] = forecast_df["wspd"].apply(lambda x: int(x["english"]))
    forecast_df["wdir"] = forecast_df["wdir"].apply(lambda x: x["dir"])
    
    # Calculate number of hours into future for each forecasted time
    forecast_df["hours_ahead"] = forecast_df["FCTTIME"].apply(lambda x: \
        (time.mktime(time.strptime(x.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")) \
        - pulltime)/3600)
    
    # Reorder columns
    forecast_df.columns = [["datetime", "temp", "feelslike", "windchill", "heatindex", \
        "humidity", "precip_inches", "chance_precip", "snow_inches", "windspeed", "winddir", \
        "condition", "sky", "code", "hours_ahead"]]
    forecast_df = forecast_df[["datetime", "hours_ahead", "temp", "feelslike", "windchill", \
        "heatindex", "humidity", "precip_inches", "chance_precip", "snow_inches", "windspeed", \
        "winddir", "condition", "sky", "code"]]
    
    # Save data
    outfilename = out_file_base + stationkey + "_" + \
        time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime(pulltime)) + ".csv"
    f = open(outfilename, "w");
    forecast_df.to_csv(f)
    f.close()


def try_forecast_pull(out_file_base="/home/vagrant/citibike/forecasts/", \
                      apikey="8ea424e58efa01e7", stationkey="KNYNEWYO270", \
                      retry_delay=30):
    
    url_str = "http://api.wunderground.com/api/%s/hourly10day/q/pws:%s.json" % (apikey, stationkey)
    print "Pulling from", url_str, "at ", time.ctime()
    try:
        pull_forecast_to_csv(out_file_base, apikey, stationkey)
    except:
        print "Error accessing url... trying again in" + str(retry_delay) + "seconds"
        time.sleep(retry_delay)
        try:
            pull_forecast_to_csv(out_file_base, apikey, stationkey)
        except:
            print "Second attempt failed... skipping this status pull"


def periodic_forecast_pulls(start_time=time.time(), delay_hours=1, stop_time=None):
    
    if stop_time is None:
        stop_time = start_time + 60*60*24*365 # 1 year
    pull_times = np.arange(start_time, stop_time, delay_hours*3600)
    pull_times = pull_times[pull_times > time.time()]
    print "Starting weather forecast pulls every", delay_hours, "hours at", \
        time.ctime(pull_times[0]), "until", time.ctime(pull_times[-1])
    for event in pull_times:
        s.enterabs(event, 1, try_forecast_pull, ())
    s.run()

# import pdb
# pdb.set_trace()

start_time = time.mktime(time.strptime("04 18 15, 11 00 EDT", "%m %d %y, %H %M %Z"))
periodic_forecast_pulls(start_time)
