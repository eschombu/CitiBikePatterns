import numpy as np
import pandas as pd
import urllib2 as url
import datetime, time
from sqlalchemy import create_engine

def pull_daily_record(date, \
    url_format="http://www.wunderground.com/history/airport/KNYC/YYYY/M/D/DailyHistory.html?format=1", \
    outfile_base="/home/vagrant/citibike/weather_data/knyc_"):
    
    url_path = url_format.replace("YYYY", str(date.year)).replace("M", str(date.month)).replace("D", str(date.day))
    url_string = url.urlopen(url_path).read()
    url_string = url_string[1:]
    url_string = url_string.replace("<br />", "")
    outfile = file(outfile_base + date.strftime("%Y-%m-%d") + ".csv", "w")
    outfile.write(url_string)
    outfile.close()

# Iterator generating function from http://www.ianlewis.org/en/python-date-range-iterator    
def datetimeIterator(from_date=None, to_date=None, delta=datetime.timedelta(days=1)):
    from_date = from_date or datetime.now()
    while to_date is None or from_date <= to_date:
        yield from_date
        from_date = from_date + delta
    return

def pull_weather(start_date, end_date, interval=1):
    datelist = [x for x in datetimeIterator(start_date, end_date, datetime.timedelta(days=interval))]
    print "Downloading weather from", start_date.strftime("%m/%d/%Y"), "to", \
        end_date.strftime("%m/%d/%Y"), "in", str(interval), "day intervals"
    for day in datelist:
        pull_daily_record(day)

def build_weather_dataframe(start_date, end_date, filebase="/home/vagrant/citibike/weather_data/knyc_"):
    all_df = pd.DataFrame()
    for day in datetimeIterator(start_date, end_date):
        filename = filebase + day.strftime("%Y-%m-%d") + ".csv"
        try:
            day_df = pd.read_csv(filename)
            day_df.DateUTC = day_df.DateUTC.apply(pd.datetools.parse)
            day_df['Date'] = pd.Series(day, index=day_df.index)
            
            # Move the Date column to head of list
            cols = list(day_df)
            cols.insert(0, cols.pop(cols.index('Date')))
            day_df = day_df.ix[:, cols]
            all_df = all_df.append(day_df, ignore_index=True)
        except:
            continue
    
    all_df = all_df.sort("DateUTC")
    return all_df

# import pdb
# pdb.set_trace()

start_date = datetime.date(2013,7,1)
end_date = datetime.date(2014,8,31)
df = build_weather_dataframe(start_date, end_date)

# Write database to a single file for quick loading
outfilename = "/home/vagrant/citibike/weatherDB_" + start_date.strftime("%Y-%m-%d") + "_to_" \
    + end_date.strftime("%Y-%m-%d") + ".csv"
df.to_csv(outfilename)

