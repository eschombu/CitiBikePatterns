import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import linear_model
from datetime import date, datetime, timedelta
import datetime as dt
from pytz import timezone
import pytz
import time
import calendar
# import matplotlib
# matplotlib.use(‘Agg’)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Function returning list of boolean values indicating which elements of A are in B
def ismember(A, B):
    return [ bool(np.sum(a == B)) for a in A ]

# Hand-set holidays that I think typically affect travel
holidays = [datetime.strptime(x, '%Y-%m-%d').date() for x in ['2013-07-04','2013-09-02','2013-10-14','2013-11-11', \
    '2013-11-28','2013-12-25','2014-01-01','2014-01-20','2014-02-17','2014-05-26','2014-07-04']]
holidayRelated = [datetime.strptime(x, '%Y-%m-%d').date() for x in ['2013-07-05','2013-11-27','2013-11-29','2013-12-24']]


# Load Citi Bike data
CBmonths = np.array([np.array([np.ones((6,1))*2013, np.ones((6,1))*2014]).flatten(), \
    np.array([np.arange(7,13,1), np.arange(1,7,1)]).flatten()], dtype='int').T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(date(CBmonths[i,0], CBmonths[i,1], 1))

citibike_filename = "/home/vagrant/citibike/hourlyDB_" + mo_strs[0].strftime("%m-%Y") + \
    "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
hourlyDF_bystart = pd.read_csv(citibike_filename, compression="gzip")

# Get proper DateTime objects
hourlyDF_bystart["DateTime"] = hourlyDF_bystart["startbin"].apply(lambda x: \
    datetime(CBmonths[0,0], CBmonths[0,1], 1, 0, 0, 0, tzinfo=timezone('US/Eastern')) \
    + timedelta(hours=(x+0.5)))
hourlyDF_bystart["DateTime_UTC"] = hourlyDF_bystart["DateTime"].apply(lambda x: x.astimezone(pytz.utc))

# Load weather data
dtfmt = "%Y-%m-%d %H:%M:%S"
start_weather = date(2013,7,1)
end_weather = date(2014,8,31)
weather_filename = "/home/vagrant/citibike/weatherDB_" + start_weather.strftime("%Y-%m-%d") \
    + "_to_"     + end_weather.strftime("%Y-%m-%d") + ".csv"
weather_df = pd.read_csv(weather_filename, index_col=0)

# Index by proper UTC DateTimes, using nearest previous 1/2 hours (same as Citi Bike)
weather_df["DateTime_UTC"] = [datetime(*datetime.strptime(x, dtfmt).timetuple()[0:4], \
    minute=int(np.floor(datetime.strptime(x, dtfmt).minute/30.)*30), tzinfo=pytz.utc) \
    for x in weather_df["DateUTC"].values]
del weather_df["Date"], weather_df["DateUTC"], weather_df["TimeEST"], weather_df["TimeEDT"]

# Get rid of bad temperature values
weather_df["TemperatureF"].ix[weather_df["TemperatureF"] < -100] = np.nan

# Determined (arbitrary) numerical mapping of conditions to assess how "pretty" it is out
def condition_number_map(conditions):
    return {
        'Drizzle': 1, 'Rain': 1, 'Snow': 1, 'Snow Grains': 1, 'Ice Crystals': 1, \
        'Ice Pellets': 1, 'Hail': 1, 'Mist': 1, 'Fog': 0, 'Fog Patches': 0.5, 'Smoke': 0, \
        'Volcanic Ash': 0, 'Widespread Dust': 0, 'Sand': 0, 'Haze': 0.5, 'Spray': 0.5, \
        'Dust Whirls': 0.5, 'Sandstorm': 0, 'Low Drifting Snow': 0.5, \
        'Low Drifting Widespread Dust': 0.5, 'Low Drifting Sand': 0.5, 'Blowing Snow': 0.5, \
        'Blowing Widespread Dust': 0.5, 'Blowing Sand': 0.5, 'Rain Mist': 1, 'Rain Showers': 1, \
        'Snow Showers': 1, 'Snow Blowing Snow Mist': 0.5, 'Ice Pellet Showers': 1, 'Hail Showers': 1, \
        'Small Hail Showers': 1, 'Thunderstorm': 0.5, 'Thunderstorms and Rain': 0.5, \
        'Thunderstorms and Snow': 0.5, 'Thunderstorms and Ice Pellets': 0.5, 'Thunderstorms with Hail': 0.5, \
        'Freezing Fog': 0, 'Thunderstorms with Small Hail': 0.5, 'Freezing Drizzle': 1, \
        'Freezing Rain': 1, 'Patches of Fog': 0.5, 'Shallow Fog': 0.5, 'Partial Fog': 0.5, \
        'Overcast': 1, 'Clear': 5, 'Partly Cloudy': 3, 'Mostly Cloudy': 4, 'Scattered Clouds': 2, \
        'Small Hail': 1, 'Squalls': 1, 'Funnel Cloud': 0, 'Unknown Precipitation': 1, 'Unknown': np.nan
    }[conditions]

# Build DataFrame tracking total hourly counts and concurrent weather
hourly_gp = hourlyDF_bystart.groupby("DateTime_UTC")
hourly_df = DataFrame({"DateTime_UTC":hourly_gp["DateTime_UTC"].first().values, \
                          "DateTimeLocal":hourly_gp["DateTime"].first().values, \
                          "count":hourly_gp["total count"].sum().values})
hourly_df["weekday"] = hourly_df["DateTimeLocal"].map(lambda x: calendar.weekday(x.year, x.month, x.day))
hourly_df["hour"] = hourly_df["DateTimeLocal"].map(lambda x: x.hour)
hourly_df["holiday"] = hourly_df["DateTimeLocal"].map(lambda x: x.date() in holidays+holidayRelated)
hourly_df = hourly_df.merge(weather_df[["DateTime_UTC", "TemperatureF", "PrecipitationIn", \
    "Humidity", "Wind SpeedMPH", "Conditions"]], how='left', on="DateTime_UTC")
hourly_df["Wind SpeedMPH"].ix[hourly_df["Wind SpeedMPH"] == '-9999.0'] = 'nan'
hourly_df["Wind SpeedMPH"].ix[hourly_df["Wind SpeedMPH"] == 'Calm'] = '0'
hourly_df["Wind SpeedMPH"] = hourly_df["Wind SpeedMPH"].map(float)
hourly_df['Conditions'].ix[hourly_df['Conditions'].isnull()] = 'Unknown'
hourly_df["Condition index"] = hourly_df["Conditions"].map(lambda x: \
    condition_number_map(x.replace('Light ', '').replace('Heavy ', '')))
hourly_df.head()

# Calculate hourly averages (and distributions)
hourly_df["log count"] = hourly_df["count"].map(np.log10)
tmp_gp = hourly_df.ix[np.array(ismember(hourly_df["weekday"].values, np.arange(0,5))) \
    & ~hourly_df["holiday"]].groupby("hour")
wkday_hourly_min = tmp_gp["count"].min()
wkday_hourly_max = tmp_gp["count"].max()
wkday_hourly_hist = tmp_gp["count"].value_counts(bins=30).reshape(24,30)
wkday_hourly_logmin = tmp_gp["log count"].min()
wkday_hourly_logmax = tmp_gp["log count"].max()
wkday_hourly_loghist = tmp_gp["log count"].value_counts(bins=30).reshape(24,30)
wkday_hourly_avg = tmp_gp["count"].mean()
wkend_hourly_avg = tmp_gp["count"].mean()

tmp_gp = hourly_df.ix[np.array(ismember(hourly_df["weekday"].values, np.arange(5,7)))].groupby("hour")
wkend_hourly_min = tmp_gp["count"].min()
wkend_hourly_max = tmp_gp["count"].max()
wkend_hourly_hist = tmp_gp["count"].value_counts(bins=30).reshape(24,30)
wkend_hourly_logmin = tmp_gp["log count"].min()
wkend_hourly_logmax = tmp_gp["log count"].max()
wkend_hourly_loghist = tmp_gp["log count"].value_counts(bins=30).reshape(24,30)
wkend_hourly_avg = tmp_gp["count"].mean()
wkend_hourly_avg = tmp_gp["count"].mean()

plt.figure()
wkday_hourly_avg.plot(color='b')
wkend_hourly_avg.plot(color='r')


# Calculate various averages for relating counts to different variables
tempF_avg = hourly_df.groupby("TemperatureF")["count"].mean()
plt.figure()
tempF_avg.plot()

precip_avg = DataFrame({"precip":hourly_df.groupby("PrecipitationIn")["PrecipitationIn"].count().index.values, \
                        "n":hourly_df.groupby("PrecipitationIn")["PrecipitationIn"].count().values, \
                        "count mean":hourly_df.groupby("PrecipitationIn")["count"].mean().values, \
                        "count std":hourly_df.groupby("PrecipitationIn")["count"].std().values})
rain_avg = DataFrame({"precip":hourly_df.ix[hourly_df["TemperatureF"] > 35].groupby(\
                      "PrecipitationIn")["PrecipitationIn"].count().index.values, \
                      "n":hourly_df.ix[hourly_df["TemperatureF"] > 35].groupby("PrecipitationIn")["PrecipitationIn"].count().values, \
                      "count mean":hourly_df.ix[hourly_df["TemperatureF"] > 35].groupby("PrecipitationIn")["count"].mean().values, \
                      "count std":hourly_df.ix[hourly_df["TemperatureF"] > 35].groupby("PrecipitationIn")["count"].std().values})
snow_avg = DataFrame({"precip":hourly_df.ix[hourly_df["TemperatureF"] < 35].groupby("PrecipitationIn")["PrecipitationIn"].count().index.values, \
                      "n":hourly_df.ix[hourly_df["TemperatureF"] < 35].groupby("PrecipitationIn")["PrecipitationIn"].count().values, \
                      "count mean":hourly_df.ix[hourly_df["TemperatureF"] < 35].groupby("PrecipitationIn")["count"].mean().values, \
                      "count std":hourly_df.ix[hourly_df["TemperatureF"] < 35].groupby("PrecipitationIn")["count"].std().values})

thresh = 30
plt.figure()
plt.plot(rain_avg["precip"].ix[rain_avg["n"] > thresh], rain_avg["count mean"].ix[rain_avg["n"] > thresh] \
    - rain_avg["count std"].ix[rain_avg["n"] > thresh]/np.sqrt(rain_avg["n"].ix[rain_avg["n"] > thresh]), \
    color = 'r', linestyle='--')
plt.plot(rain_avg["precip"].ix[rain_avg["n"] > thresh], rain_avg["count mean"].ix[rain_avg["n"] > thresh] \
    + rain_avg["count std"].ix[rain_avg["n"] > thresh]/np.sqrt(rain_avg["n"].ix[rain_avg["n"] > thresh]), \
    color = 'r', linestyle='--')
plt.plot(rain_avg["precip"].ix[rain_avg["n"] > thresh], rain_avg["count mean"].ix[rain_avg["n"] > thresh], color = 'r')
plt.plot(snow_avg["precip"].ix[snow_avg["n"] > thresh], snow_avg["count mean"].ix[snow_avg["n"] > thresh] \
    - snow_avg["count std"].ix[snow_avg["n"] > thresh]/np.sqrt(snow_avg["n"].ix[snow_avg["n"] > thresh]), \
    color = 'b', linestyle='--')
plt.plot(snow_avg["precip"].ix[snow_avg["n"] > thresh], snow_avg["count mean"].ix[snow_avg["n"] > thresh] \
    + snow_avg["count std"].ix[snow_avg["n"] > thresh]/np.sqrt(snow_avg["n"].ix[snow_avg["n"] > thresh]), \
    color = 'b', linestyle='--')
plt.plot(snow_avg["precip"].ix[snow_avg["n"] > thresh], snow_avg["count mean"].ix[snow_avg["n"] > thresh], color = 'b')
plt.plot(precip_avg["precip"].ix[precip_avg["n"] > thresh], precip_avg["count mean"].ix[precip_avg["n"] > thresh] \
    - precip_avg["count std"].ix[precip_avg["n"] > thresh]/np.sqrt(precip_avg["n"].ix[precip_avg["n"] > thresh]), \
        color = 'g', linestyle='--')
plt.plot(precip_avg["precip"].ix[precip_avg["n"] > thresh], precip_avg["count mean"].ix[precip_avg["n"] > thresh] \
    + precip_avg["count std"].ix[precip_avg["n"] > thresh]/np.sqrt(precip_avg["n"].ix[precip_avg["n"] > thresh]), \
    color = 'g', linestyle='--')
plt.plot(precip_avg["precip"].ix[precip_avg["n"] > thresh], precip_avg["count mean"].ix[precip_avg["n"] > thresh], \
    color = 'g')

## Train GLM for regression model of counts based on several variables
# train on every other day
train_days = [min(hourly_df.DateTimeLocal).date() + timedelta(days=x) for x in \
    range(0, (max(hourly_df.DateTimeLocal).date() - min(hourly_df.DateTimeLocal).date()).days, 2)]
test_days = [min(hourly_df.DateTimeLocal).date() + timedelta(days=x) for x in \
    range(1, (max(hourly_df.DateTimeLocal).date() - min(hourly_df.DateTimeLocal).date()).days, 2)]

# convert to boolean index vectors
wkday_train_bools = hourly_df["DateTimeLocal"].map(lambda x: x.date()).isin(train_days) & \
    hourly_df["weekday"].isin(range(0,5))
wkday_test_bools = hourly_df["DateTimeLocal"].map(lambda x: x.date()).isin(test_days) & \
    hourly_df["weekday"].isin(range(0,5))
wkend_train_bools = hourly_df["DateTimeLocal"].map(lambda x: x.date()).isin(train_days) & \
    hourly_df["weekday"].isin(range(5,7))
wkend_test_bools = hourly_df["DateTimeLocal"].map(lambda x: x.date()).isin(test_days) & \
    hourly_df["weekday"].isin(range(5,7))

# Function wrapper for building feature vector
def make_features(train_df):
    # Training features
    tempF_array = np.array(train_df["TemperatureF"].values).reshape(train_df.index.size, 1)
    features = np.append(temp_array, temp_array**2, 1)
    features = np.append(features, temp_array**3, 1)
    features = np.append(features, temp_array**4, 1)
    features = np.append(features, \
        np.array(hourly_df["PrecipitationIn"].values).reshape(train_df.index.size, 1), 1)
    
    return features
    
# Function wrapper to calculate model quality metric
def model_quality(prediction, actual):
    return np.sum(np.abs(prediction - actual))

# Fit model for each hour of day
wkday_models = []
wkend_models = []
for i in range(24):
    # Weekdays
    train_bools = wkday_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df.ix[train_bools])
    train_notNA = ~np.isnan(train_features).any(axis=1)
    train_responses = hourly_df["count"].ix[train_bools]
    #train_responses = np.log10(hourly_df["count"].ix[train_bools])
    wkday_models.append( linear_model.LinearRegression().fit(train_features[train_notNA,:], \
        train_responses[train_notNA]) )
    
    # Weekends
    train_bools = wkend_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df.ix[train_bools])
    train_notNA = ~np.isnan(train_features).any(axis=1)
    train_responses = hourly_df["count"].ix[train_bools]
    #train_responses = np.log10(hourly_df["count"].ix[train_bools])
    wkend_models.append( linear_model.LinearRegression().fit(train_features[train_notNA,:], \
        train_responses[train_notNA]) )

# Get model predictions and performances for training and test sets
model_predictions = np.zeros(hourly_df.index.values.shape)
for i in range(24):
    # Training weekdays
    train_bools = wkday_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df.ix[train_bools])
    train_notNA = ~np.isnan(train_features).any(axis=1)
    train_predictions = np.zeros(np.sum(train_bools))
    train_predictions.fill(np.nan)
    train_predictions[train_notNA] = wkday_models[i].predict(train_features[train_notNA])
    #train_predictions[train_notNA] = 10**wkday_models[i].predict(train_features[train_notNA])
    model_predictions[train_bools.values] = train_predictions
    
    # Training weekends
    train_bools = wkend_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df.ix[train_bools])
    train_notNA = ~np.isnan(train_features).any(axis=1)
    train_predictions = np.zeros(np.sum(train_bools))
    train_predictions.fill(np.nan)
    train_predictions[train_notNA] = wkend_models[i].predict(train_features[train_notNA])
    #train_predictions[train_notNA] = 10**wkend_models[i].predict(train_features[train_notNA])
    model_predictions[train_bools.values] = train_predictions
    
    # Training weekdays
    test_bools = wkday_test_bools & (hourly_df["hour"].values == i)
    test_features = make_features(hourly_df.ix[test_bools])
    test_notNA = ~np.isnan(test_features).any(axis=1)
    test_predictions = np.zeros(np.sum(test_bools))
    test_predictions.fill(np.nan)
    test_predictions[test_notNA] = wkday_models[i].predict(test_features[test_notNA])
    #test_predictions[test_notNA] = 10**wkday_models[i].predict(test_features[test_notNA])
    model_predictions[test_bools.values] = test_predictions
    
    # Training weekends
    test_bools = wkend_test_bools & (hourly_df["hour"].values == i)
    test_features = make_features(hourly_df.ix[test_bools])
    test_notNA = ~np.isnan(test_features).any(axis=1)
    test_predictions = np.zeros(np.sum(test_bools))
    test_predictions.fill(np.nan)
    test_predictions[test_notNA] = wkend_models[i].predict(test_features[test_notNA])
    #test_predictions[test_notNA] = 10**wkend_models[i].predict(test_features[test_notNA])
    model_predictions[test_bools.values] = test_predictions


wkday_train_perform = model_quality(model_predictions[wkday_train_bools], \
    hourly_df["count"].ix[wkday_train_bools].values)
wkday_test_perform = model_quality(model_predictions[wkday_test_bools], \
    hourly_df["count"].ix[wkday_test_bools].values)
wkend_train_perform = model_quality(model_predictions[wkend_train_bools], \
    hourly_df["count"].ix[wkend_train_bools].values)
wkend_test_perform = model_quality(model_predictions[wkend_test_bools], \
    hourly_df["count"].ix[wkend_test_bools].values)
    
def multi_color_line(x, y, c, cmap=ListedColormap(['r', 'b']), boundaries=[0, 0.5, 1]):
# Color parts of a line based on its properties, e.g., slope.
# From http://wiki.scipy.org/Cookbook/Matplotlib/MulticoloredLine
    norm = BoundaryNorm(boundaries, cmap.N)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c)
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)
    plt.show()

plt.figure(figsize=(16,8))
plt.plot(hourly_df.index.values, hourly_df['count'], color='k', linewidth=1.5)
plt.ylim([0, 3200])
plt.xlim([4000, 5000])
multi_color_line(hourly_df.index.values, model_predictions, wkday_train_bools | wkend_train_bools)


# 
# # Plot hourly trip counts, temp, precip together
# fig, ax1 = plt.subplots(figsize=(14,10))
# ax1.plot(hourly_df["total count"].index, hourly_df["total count"].values/float(hourly_df["total count"].max()), color='k')
# ax1.plot(weather_df["TemperatureF"].index, weather_df["TemperatureF"].values/float(weather_df["TemperatureF"].max()), \
#     color='r')
# ax1.plot(weather_df["PrecipitationIn"].index, weather_df["PrecipitationIn"].values/float(weather_df["PrecipitationIn"].max()), \
#     color='b')
# plt.xticks(plt.xticks()[0], rotation=30)
# # plt.xlim([datetime(2013,12,15), datetime(2014,1,14)])
