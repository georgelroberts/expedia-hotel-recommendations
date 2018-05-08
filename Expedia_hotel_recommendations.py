# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:03:36 2018

@author: George
"""

import pandas as pd

nRows = 10 ** 4
trainChunk = pd.read_csv('Data\\train.csv', nrows=nRows)

print(trainChunk.info())
# There are only 7/10000 missing points in srch_ci and srch_co, hence we can
# simply ignore these in training.
# There are 3729/10000 missing in distance from origin to destination, hence
# this may actually encode some information. Perhaps this actually encodes some
# information somehow, so lets set it to zero for the time being.

trainChunk = trainChunk[pd.notnull(trainChunk['srch_ci'])]
trainChunk = trainChunk[pd.notnull(trainChunk['srch_co'])]
trainChunk = trainChunk.fillna(0)

def extractDate(df, colName, returnHour, startStr):
    """ Takes a specific column 'colName' from the dataframe 'df' and replaces
    it with features for the year, month and day of week. Hour is returned
    optionally with the boolean 'returnHour'. The specified string 'startStr'
    is added to the dataframe column name"""

    datetime = pd.to_datetime(df[colName])
    df[startStr + 'year'] = datetime.dt.year
    df[startStr + 'month'] = datetime.dt.month
    df[startStr + 'dayOfWeek'] = datetime.dt.dayofweek
    if returnHour:
        df[startStr + 'hour'] = datetime.dt.hour

    df.drop([colName], axis=1, inplace=True)
    return df


trainChunk = extractDate(trainChunk, 'date_time', True, 'search_')
trainChunk = extractDate(trainChunk, 'srch_ci', False, 'checkin_')
trainChunk = extractDate(trainChunk, 'srch_co', False, 'checkout_')

