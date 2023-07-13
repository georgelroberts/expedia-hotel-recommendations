# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:03:36 2018

@author: George
"""
# %% Import packages

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# %% Read a single chunk of the data
nRows = 10 ** 4
trainChunk = pd.read_csv('Data\\train.csv', nrows=nRows)

# %% Clean a single chunk

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
trainY = trainChunk['hotel_cluster']
trainX = trainChunk.drop('hotel_cluster', axis=1)

# %% Building a scorer
# Scoring allows five predictions for each sample, however for the moment I
# will keep it simple and just do one. I do need to build my own scorer for
# this.


def APatK(truth, prediction):
    """ Takes the real value and the predicted list (1 x m),
    where m <=5 for this project and outputs the average precision. See
    https://www.kaggle.com/c/FacebookRecruiting/discussion/2002 for a good
    explanation of the metric"""

    no_correct = 0
    score = 0

    for num, pred in enumerate(prediction):
        if pred == truth:
            no_correct += 1
            score += no_correct / (num + 1)
    # Would usually divide by the number of elements in the truth, however
    # this is just one in this project (each sample has a single cluster)
    return score


def meanAPatK(actuals, preds):
    """ Uses the APatK function to get the mean average precision.
    Preliminarily I am using just a single prediction, so I am manually turning
    each one into a list here. """

    scoreList = [APatK(truth, list(prediction)) for truth, prediction
                 in zip(actuals, preds)]

    return np.mean(scoreList)


# %% Rudimentary fit of a single chunk

trainX2, testX, trainY2, testY = train_test_split(trainX, trainY)

pipe = Pipeline([
        ('scal', StandardScaler()),
        ('clf', LogisticRegression())])
pipe.fit(trainX2, trainY2.ravel())
yPredict = pipe.predict(testX)
meanAPatK(testY, yPredict)
