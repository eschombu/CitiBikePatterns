import numpy as np
from scipy import stats
import pandas as pd
from pandas import Series, DataFrame
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from datetime import date, datetime, timedelta
import datetime as dt
from pytz import timezone
import pytz
import time
import calendar
import matplotlib.pyplot as plt
import seaborn as sbs
from matplotlib import gridspec

## Function returning list of boolean values indicating which elements of A are in B
def ismember(A, B):
    return [ bool(np.sum(a == B)) for a in A ]


## Hand-set holidays that I think typically affect travel
file = open('/home/vagrant/citibike/holidays.txt', 'r')
holiday_strs = []
for line in file:
    holiday_strs.append(line.rstrip())

file.close()
file = open('/home/vagrant/citibike/holiday_related.txt', 'r')
holidayRelated_strs = []
for line in file:
    holidayRelated_strs.append(line.rstrip())

file.close()
holidays = [datetime.strptime(x, '%Y-%m-%d').date() for x in holiday_strs]
holidayRelated = [datetime.strptime(x, '%Y-%m-%d').date() for x in holidayRelated_strs]

## Load Citi Bike data
dtfmt = "%Y-%m-%d %H:%M:%S"
CBmonths = np.append(np.append(np.append(np.ones((5,1))*2013, np.ones((12,1))*2014), np.ones((2,1))*2015), \
    np.array(range(8,13,1) + range(1,13,1) + range(1,3,1))).astype('int').reshape(2,19).T
mo_strs = []
for i in range(CBmonths.shape[0]):
    mo_strs.append(date(CBmonths[i,0], CBmonths[i,1], 1))

citibike_filename = "/home/vagrant/citibike/hourlyDB_byStart_" + mo_strs[0].strftime("%m-%Y") + \
    "_to_" + mo_strs[-1].strftime("%m-%Y") + ".csv.gz"
hourlyDF_bystart = pd.read_csv(citibike_filename, compression="gzip")
hourlyDF_bystart['start time'] = [pytz.timezone('US/Eastern').localize(\
    datetime(*datetime.strptime(x, dtfmt).timetuple()[0:4], \
    minute=int(np.floor(datetime.strptime(x, dtfmt).minute/30.)*30))) \
    for x in hourlyDF_bystart['start time'].values]

reftime = time.mktime(time.strptime("2000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"))
binstart = time.mktime(time.strptime(date(CBmonths[0,0], CBmonths[0,1], 1 \
    ).strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
binend = time.mktime(time.strptime(date(CBmonths[-1,0], CBmonths[-1,1] + 1, 1 \
    ).strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
hourbins = np.linspace(binstart-reftime, binend-reftime, (binend-binstart)/3600 + 1)
hourbins_dt = [datetime.fromtimestamp(x + reftime) for x in hourbins]

## Load weather data
start_weather = date(2013,8,1)
end_weather = date(2015,2,28)
weather_filename = "/home/vagrant/citibike/weatherDB_" + start_weather.strftime("%Y-%m-%d") \
    + "_to_" + end_weather.strftime("%Y-%m-%d") + ".csv"
weather_df = pd.read_csv(weather_filename, index_col=0)

## Index by proper UTC DateTimes, using nearest previous 1/2 hours (same as Citi Bike)
weather_df["DateTime_UTC"] = [datetime(*datetime.strptime(x, dtfmt).timetuple()[0:4], tzinfo=pytz.utc) \
    + timedelta(seconds=round(datetime.strptime(x, dtfmt).minute/60.)*3600) for x in weather_df["DateUTC"].values]
weather_df = weather_df.groupby('DateTime_UTC', as_index=False).first()
weather_df["start time"] = weather_df["DateTime_UTC"].map(lambda x: \
    pytz.timezone('UTC').localize(x).astimezone(pytz.timezone('US/Eastern')))

## Get rid of bad temperature values
weather_df["TemperatureF"].ix[weather_df["TemperatureF"] < -100] = np.nan

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

hourly_gp = hourlyDF_bystart.groupby("start time")
hourly_df = DataFrame({"start time":hourly_gp["start time"].first(), "count":hourly_gp["total count"].sum().values})
hourly_df["weekday"] = hourly_df["start time"].map(lambda x: calendar.weekday(x.year, x.month, x.day))
hourly_df["hour"] = hourly_df["start time"].map(lambda x: x.hour)
hourly_df["HOLIDAY"] = hourly_df["start time"].map(lambda x: x.date() in holidays+holidayRelated)
hourly_df = hourly_df.merge(weather_df[["start time", "TemperatureF", "PrecipitationIn", \
    "Humidity", "Wind SpeedMPH", "Conditions"]], how='left', on="start time")
hourly_df["PrecipitationIn"] = hourly_df["PrecipitationIn"].fillna(0)
hourly_df["Wind SpeedMPH"].ix[hourly_df["Wind SpeedMPH"] == '-9999.0'] = 'nan'
hourly_df["Wind SpeedMPH"].ix[hourly_df["Wind SpeedMPH"] == 'Calm'] = '0'
hourly_df["Wind SpeedMPH"] = hourly_df["Wind SpeedMPH"].map(float)
hourly_df['Conditions'].ix[hourly_df['Conditions'].isnull()] = 'Unknown'
hourly_df["Condition index"] = hourly_df["Conditions"].map(lambda x: \
    condition_number_map(x.replace('Light ', '').replace('Heavy ', '')))


## For time points where Temperature not available, interpolate values
null_inds = hourly_df.index[hourly_df['TemperatureF'].isnull()].values
not_null = hourly_df.index[~hourly_df['TemperatureF'].isnull()].values
hourly_df['TemperatureF'].ix[null_inds] = np.interp(null_inds, not_null, hourly_df['TemperatureF'].ix[not_null].values)


## Create feature vectors
def make_features(train_df, trainOn_bools):
    
    # Current precip
    current_precip = train_df["PrecipitationIn"].ix[trainOn_bools].values.reshape(-1, 1)
    
    # Total precip in previous 12, 24, 48 hours, and next 12 hours
    prev12_precip = np.zeros([np.sum(trainOn_bools), 12])
    for i in range(12):
        shifted_bools = np.append(trainOn_bools[i+1:], np.array([False for n in range(i+1)])).astype('bool')
        prev12_precip[:,i] = train_df["PrecipitationIn"].ix[shifted_bools].values
    
    prev12_precip = np.sum(prev12_precip, axis=1).reshape(-1, 1)
    
    prev24_precip = np.zeros([np.sum(trainOn_bools), 12])
    for i in range(12):
        shifted_bools = np.append(trainOn_bools[i+13:], np.array([False for n in range(i+13)])).astype('bool')
        prev24_precip[:,i] = train_df["PrecipitationIn"].ix[shifted_bools].values
    
    prev24_precip = prev12_precip + np.sum(prev24_precip, axis=1).reshape(-1, 1)
    
    prev48_precip = np.zeros([np.sum(trainOn_bools), 24])
    for i in range(24):
        shifted_bools = np.append(trainOn_bools[i+25:], np.array([False for n in range(i+25)])).astype('bool')
        prev48_precip[:,i] = train_df["PrecipitationIn"].ix[shifted_bools].values
    
    prev48_precip = prev24_precip + np.sum(prev48_precip, axis=1).reshape(-1, 1)
    
    prev72_precip = np.zeros([np.sum(trainOn_bools), 24])
    for i in range(24):
        shifted_bools = np.append(trainOn_bools[i+25:], np.array([False for n in range(i+25)])).astype('bool')
        prev72_precip[:,i] = train_df["PrecipitationIn"].ix[shifted_bools].values
    
    prev72_precip = prev48_precip + np.sum(prev72_precip, axis=1).reshape(-1, 1)
    
    next6_precip = np.zeros([np.sum(trainOn_bools), 6])
    for i in range(1,7):
        shifted_bools = np.append(np.array([False for n in range(i)]), trainOn_bools[:-i]).astype('bool')
        next6_precip[:,i-1] = train_df["PrecipitationIn"].ix[shifted_bools].values
    
    next6_precip = np.sum(next6_precip, axis=1).reshape(-1, 1)
    
    # Current temp, and high & low in +/- 12 hours
    current_temp = train_df["TemperatureF"].ix[trainOn_bools].values.reshape(-1, 1)
    
    prev12_temp = np.zeros([np.sum(trainOn_bools), 12])
    for i in range(12):
        shifted_bools = np.append(trainOn_bools[i+1:], np.array([False for n in range(i+1)])).astype('bool')
        prev12_temp[:,i] = train_df["TemperatureF"].ix[shifted_bools].values
        
    next12_temp = np.zeros([np.sum(trainOn_bools), 12])
    for i in range(1,13):
        shifted_bools = np.append(np.array([False for n in range(i)]), trainOn_bools[:-i]).astype('bool')
        next12_temp[:,i-1] = train_df["TemperatureF"].ix[shifted_bools].values
    
    hi24_temp = np.max(np.concatenate((prev12_temp, next12_temp), axis=1), axis=1).reshape(-1, 1)
    lo24_temp = np.min(np.concatenate((prev12_temp, next12_temp), axis=1), axis=1).reshape(-1, 1)
    mean24_temp = np.mean(np.concatenate((current_temp, prev12_temp, next12_temp), axis=1), axis=1).reshape(-1, 1)
    
    # Interactions
    polyFeat = PolynomialFeatures(degree=3, include_bias=True)
    features = polyFeat.fit_transform(np.concatenate((current_precip, prev48_precip, \
        current_temp, mean24_temp), axis=1))
    
    # Add constant
    features = np.concatenate((np.ones([np.sum(trainOn_bools),1]), features), axis=1)
    
    return features   #--- make_features() ---


def model_quality(prediction, actual):
    return np.sum(np.abs(prediction - actual))


train_days = [min(hourly_df['start time']).date() + timedelta(hours=48) + timedelta(days=x) for x in \
    range(0, (max(hourly_df['start time']).date() - min(hourly_df['start time']).date() - timedelta(hours=60)).days, 2)]
test_days = [min(hourly_df['start time']).date() + timedelta(hours=48) + timedelta(days=x) for x in \
    range(1, (max(hourly_df['start time']).date() - min(hourly_df['start time']).date() - timedelta(hours=60)).days, 2)]

## convert to boolean index vectors
wkday_train_bools = hourly_df['start time'].map(lambda x: x.date()).isin(train_days) & \
    hourly_df["weekday"].isin(range(0,5))
wkday_test_bools = hourly_df['start time'].map(lambda x: x.date()).isin(test_days) & \
    hourly_df["weekday"].isin(range(0,5))
wkend_train_bools = hourly_df['start time'].map(lambda x: x.date()).isin(train_days) & \
    hourly_df["weekday"].isin(range(5,7))
wkend_test_bools = hourly_df['start time'].map(lambda x: x.date()).isin(test_days) & \
    hourly_df["weekday"].isin(range(5,7))


## Fit model for each hour of day
LOG = False
wkday_models = []
wkend_models = []
for i in range(24):

    # Weekdays
    train_bools = wkday_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df, train_bools)
    train_notNA = ~np.isnan(train_features).any(axis=1)
    if LOG:
        train_responses = np.log10(hourly_df["count"].ix[train_bools])
    else:
        train_responses = hourly_df["count"].ix[train_bools]

    #print "i=", i, " weekday: Training model on feature matrix of size", train_features[train_notNA,:].shape
    #wkday_models.append( linear_model.LinearRegression().fit(train_features[train_notNA,:], \
    #    train_responses[train_notNA]) )
    wkday_models.append( linear_model.RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]).fit( \
        train_features[train_notNA,:], train_responses[train_notNA]) )
    
    # Weekends
    train_bools = wkend_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df, train_bools)
    train_notNA = ~np.isnan(train_features).any(axis=1)
    if LOG:
        train_responses = np.log10(hourly_df["count"].ix[train_bools])
    else:
        train_responses = hourly_df["count"].ix[train_bools]

    #print "i=", i, " weekend: Training model on feature matrix of size", train_features[train_notNA,:].shape
    #wkend_models.append( linear_model.LinearRegression().fit(train_features[train_notNA,:], \
    #    train_responses[train_notNA]) )
    wkend_models.append( linear_model.RidgeCV(alphas=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]).fit( \
        train_features[train_notNA,:], train_responses[train_notNA]) )


## Get model predictions and performances for training and test sets
model_predictions = hourly_df[['start time', 'hour']].copy()
model_predictions['count'] = np.zeros(hourly_df.index.values.shape)
model_predictions['weekday'] = wkday_train_bools | wkday_test_bools
model_predictions['weekend'] = wkend_train_bools | wkend_test_bools
model_predictions['train'] = wkday_train_bools | wkend_train_bools
model_predictions['test'] = wkday_test_bools | wkend_test_bools
for i in range(24):
    # Training weekdays
    train_bools = wkday_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df, train_bools)
    train_notNA = ~np.isnan(train_features).any(axis=1)
    train_predictions = np.zeros(np.sum(train_bools))
    train_predictions.fill(np.nan)
    if LOG:
        train_predictions[train_notNA] = 10**wkday_models[i].predict(train_features[train_notNA])
    else:
        train_predictions[train_notNA] = wkday_models[i].predict(train_features[train_notNA])
    model_predictions['count'].ix[train_bools] = train_predictions
    
    # Training weekends
    train_bools = wkend_train_bools & (hourly_df["hour"].values == i)
    train_features = make_features(hourly_df, train_bools)
    train_notNA = ~np.isnan(train_features).any(axis=1)
    train_predictions = np.zeros(np.sum(train_bools))
    train_predictions.fill(np.nan)
    if LOG:
        train_predictions[train_notNA] = 10**wkend_models[i].predict(train_features[train_notNA])
    else:
        train_predictions[train_notNA] = wkend_models[i].predict(train_features[train_notNA])
    model_predictions['count'].ix[train_bools.values] = train_predictions
    
    # Training weekdays
    test_bools = wkday_test_bools & (hourly_df["hour"].values == i)
    test_features = make_features(hourly_df, test_bools)
    test_notNA = ~np.isnan(test_features).any(axis=1)
    test_predictions = np.zeros(np.sum(test_bools))
    test_predictions.fill(np.nan)
    if LOG:
        test_predictions[test_notNA] = 10**wkday_models[i].predict(test_features[test_notNA])
    else:
        test_predictions[test_notNA] = wkday_models[i].predict(test_features[test_notNA])
    model_predictions['count'].ix[test_bools.values] = test_predictions
    
    # Training weekends
    test_bools = wkend_test_bools & (hourly_df["hour"].values == i)
    test_features = make_features(hourly_df, test_bools)
    test_notNA = ~np.isnan(test_features).any(axis=1)
    test_predictions = np.zeros(np.sum(test_bools))
    test_predictions.fill(np.nan)
    if LOG:
        test_predictions[test_notNA] = 10**wkend_models[i].predict(test_features[test_notNA])
    else:
        test_predictions[test_notNA] = wkend_models[i].predict(test_features[test_notNA])
    model_predictions['count'].ix[test_bools.values] = test_predictions

model_predictions['count'].ix[model_predictions['count'] < 0] = np.nan


# for i in range(len(wkday_models)):
#     print wkday_models[i].alpha_

train_R2 = hourly_df['count'].ix[wkday_train_bools | wkend_train_bools].corr( \
    model_predictions['count'].ix[wkday_train_bools | wkend_train_bools])**2
test_R2 = hourly_df['count'].ix[wkday_test_bools | wkend_test_bools].corr( \
    model_predictions['count'].ix[wkday_test_bools | wkend_test_bools])**2

## Plot results
figdir = '/home/vagrant/citibike/figures/'
fontsize = 15
figsize1 = (16,9)
figsize2 = (16,4)
figsize3 = (10,8)

## Whole year, using 24-hour rolling avg
fig = plt.figure(figsize=figsize1)
gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
ax1 = plt.subplot(gs[0])
lin1 = plt.plot(hourly_df['start time'], pd.rolling_mean(hourly_df['count'], 24), color='k')
plt.ylim((0, pd.rolling_mean(hourly_df['count'], 24).max()))
plt.ylabel('Count', fontsize=fontsize)
plt.title('Total system use (trips/hour, 24-hour sliding avg)', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize)
ax1.margins(0)

## Weather
ax2a = plt.subplot(gs[1])
lin1 = plt.plot(hourly_df['start time'], pd.rolling_mean(hourly_df['TemperatureF'], 24), 'r')
first = np.nonzero(~pd.rolling_mean(hourly_df['TemperatureF'], 24).isnull().values)[0][0]
last = np.nonzero(~pd.rolling_mean(hourly_df['TemperatureF'], 24).isnull().values)[0][-1]
plt.plot(hourly_df['start time'].ix[[first, last]], [32, 32], 'r:') # Freezing temp
plt.title('Weather (hourly with 24-hour sliding avg)', fontsize=fontsize)
ax2a.tick_params(axis='x', labelsize=fontsize)
ax2a.yaxis.label.set_text('Temperature (F)')
ax2a.yaxis.label.set_fontsize(fontsize)
ax2a.yaxis.label.set_color('r')
ax2a.yaxis.grid(False)
ax2a.spines['bottom'].set_color('r')
ax2a.spines['top'].set_color('r')
ax2a.tick_params(axis='y', colors='r', labelsize=fontsize)
ax2a.margins(0)

ax2b = ax2a.twinx()
lin2 = plt.plot(hourly_df['start time'], pd.rolling_mean(hourly_df['PrecipitationIn'], 24), 'b')
ax2b.yaxis.label.set_text('Precipitation (inches)')
ax2b.yaxis.label.set_fontsize(fontsize)
ax2b.yaxis.label.set_color('b')
ax2b.yaxis.grid(False)
ax2b.spines['bottom'].set_color('b')
ax2b.spines['top'].set_color('b')
ax2b.tick_params(colors='b', labelsize=fontsize)
ax2b.margins(0)
fig.tight_layout()
fig.savefig(figdir+'rides_weather_Aug2013-Feb2015.png', bbox_inches='tight')

## Include model
plt.sca(ax1)
lin2 = plt.plot(hourly_df['start time'], pd.rolling_mean(model_predictions['count'], 24), 'g')
plt.legend(['Data', 'Model ($R^2$=%.2f)' % test_R2], fontsize=fontsize, loc='best')
# ax.margins(0)
# fig.tight_layout()
fig.savefig(figdir+'rides_model_weather_Aug2013-Feb2015.png', bbox_inches='tight')
fig.clear()

## Same plots, zoomed to specified date range
zoom_dates = [datetime(2014, 5, 5), datetime(2014, 5, 26)]
fig = plt.figure(figsize=figsize1)
gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
ax1 = plt.subplot(gs[0])
plt.plot(hourly_df['start time'], hourly_df['count'], color='k')
plt.xlim(zoom_dates)
plt.ylim([0,5000])
plt.ylabel('Count', fontsize=fontsize)
plt.title('Total system trips/hour', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize)
ax1.margins(0)
fig.tight_layout()

ax2a = plt.subplot(gs[1])
lin1 = plt.plot(hourly_df['start time'], hourly_df['TemperatureF'], 'r')
plt.plot(hourly_df['start time'].ix[[0,hourly_df['start time'].size-1]], [32, 32], 'r:') # Freezing temp
plt.title('Weather (hourly)', fontsize=fontsize)
ax2a.tick_params(axis='x', labelsize=fontsize)
ax2a.yaxis.label.set_text('Temperature (F)')
ax2a.yaxis.label.set_fontsize(fontsize)
ax2a.yaxis.label.set_color('r')
ax2a.yaxis.grid(False)
ax2a.spines['bottom'].set_color('r')
ax2a.spines['top'].set_color('r')
ax2a.tick_params(axis='y', colors='r', labelsize=fontsize)
ax2a.margins(0)

ax2b = ax2a.twinx()
lin2 = plt.plot(hourly_df['start time'], hourly_df['PrecipitationIn'], 'b')
ax2b.yaxis.label.set_text('Precipitation (inches)')
ax2b.yaxis.label.set_fontsize(fontsize)
ax2b.yaxis.label.set_color('b')
ax2b.yaxis.grid(False)
ax2b.spines['bottom'].set_color('b')
ax2b.spines['top'].set_color('b')
ax2b.tick_params(colors='b', labelsize=fontsize)
ax2b.margins(0)
ax2a.set_xlim(zoom_dates)
ax2a.set_ylim([40, 85])
ax2b.set_xlim(zoom_dates)
ax2b.set_ylim([ax2b.get_ylim()[0], 0.6])
fig.tight_layout()
fig.savefig(figdir+'rides_weather_' + zoom_dates[0].strftime('%Y-%m-%d') + '_to_' \
    + zoom_dates[1].strftime('%Y-%m-%d') + '.png', bbox_inches='tight')

train_predict = model_predictions['count'].copy()
train_predict.ix[wkday_test_bools | wkend_test_bools] = np.nan
test_predict = model_predictions['count'].copy()
test_predict.ix[wkday_train_bools | wkend_train_bools] = np.nan
plt.sca(ax1)
plt.plot(hourly_df['start time'], train_predict, 'g', linewidth=1)
plt.plot(hourly_df['start time'], test_predict, 'g', linewidth=3)
plt.legend(['Data', 'Training set model ($R^2$= %.2f)' % train_R2, \
    'Test set model ($R^2$= %.2f)' % test_R2], fontsize=fontsize, loc='upper left')
# ax.margins(0)
# fig.tight_layout()
fig.savefig(figdir+'rides_model_weather_' + zoom_dates[0].strftime('%Y-%m-%d') + '_to_' \
    + zoom_dates[1].strftime('%Y-%m-%d') + '.png', bbox_inches='tight')
fig.clear()

## Hourly averages
tmp_gp = hourly_df.ix[np.array(ismember(hourly_df["weekday"].values, np.arange(0,5))) \
    & ~hourly_df["HOLIDAY"]].groupby("hour")
wkday_hourly_min = tmp_gp["count"].min()
wkday_hourly_max = tmp_gp["count"].max()
# wkday_hourly_hist = tmp_gp["count"].value_counts(bins=30).reshape(24,30)
wkday_hourly_avg = tmp_gp["count"].mean()

tmp_gp = hourly_df.ix[np.array(ismember(hourly_df["weekday"].values, np.arange(5,7)))].groupby("hour")
wkend_hourly_min = tmp_gp["count"].min()
wkend_hourly_max = tmp_gp["count"].max()
# wkend_hourly_hist = tmp_gp["count"].value_counts(bins=30).reshape(24,30)
wkend_hourly_avg = tmp_gp["count"].mean()

fig = plt.figure(figsize=figsize3)
plt.plot(np.append(wkday_hourly_avg.values, wkday_hourly_avg.values[0]), 'b')
plt.plot(np.append(wkend_hourly_avg.values, wkend_hourly_avg.values[0]), 'g')
plt.xlim([0, 24])
plt.xticks(range(0,23,4)+[24], ['12:00 AM', '4:00 AM', '8:00 AM', '12:00 PM', '4:00 PM', '8:00 PM', '12:00 AM'], fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('trips', fontsize=fontsize)
plt.title('Avg. hourly system use', fontsize=fontsize)
plt.legend(['weekdays', 'weekends'], fontsize=fontsize)
fig.savefig('/home/vagrant/citibike/figures/avg_hourly_trips.png')
fig.close()
