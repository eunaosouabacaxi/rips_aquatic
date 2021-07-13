import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# function for linreg()
# sorts tuple (ascending) based on index 1 (R^2 value)
def sort_scores1(scores):
    lst = len(scores) 
    for i in range(0, lst): 

        for j in range(0, lst-i-1): 
            if (scores[j][1] > scores[j + 1][1]): 
                temp = scores[j] 
                scores[j]= scores[j + 1] 
                scores[j + 1]= temp 
    return scores


# function for linreg()
# sorts tuple (ascending) based on index 2 (R^2 value)
def sort_scores2(scores):
    lst = len(scores) 
    for i in range(0, lst): 

        for j in range(0, lst-i-1): 
            if (scores[j][2] > scores[j + 1][2]): 
                temp = scores[j] 
                scores[j]= scores[j + 1] 
                scores[j + 1]= temp 
    return scores
    
    
#computes linear regression scores
def linreg(features_df, targets_df, uni=False, multi=False, year, month, day, squ, hour=None, minute=None, period=None):
    
    if hour = None:
        features = features_df.loc[(features_df['datetime'].dt.date >= datetime.date(year, month, day))
                                           & (features_df['datetime'].dt.date < datetime.date(year, month, day+1))
                                           & (features_df['squ'] == squ)]
        targets = targets_df.loc[(targets_df['datetime'].dt.date >= datetime.date(year, month, day))
                                           & (targets_df['datetime'].dt.date < datetime.date(year, month, day+1))
                                           & (features_df['squ'] == squ)]
        
    else:
        features = features_df.loc[(features_df['datetime'].dt.date >= datetime.date(year, month, day))
                                           & (features_df['datetime'].dt.date < datetime.date(year, month, day+1))
                                           & (features_df['datetime'].dt.time >= datetime.time(hour, minute))
                                           & (features_df['datetime'].dt.time < datetime.time(hour+period, minute+5))
                                           & (features_df['squ'] == squ)]
        targets = targets_df.loc[(targets_df['datetime'].dt.date >= datetime.date(year, month, day))
                                           & (targets_df['datetime'].dt.date < datetime.date(year, month, day+1))
                                           & (targets_df['datetime'].dt.time >= datetime.time(hour, minute))
                                           & (targets_df['datetime'].dt.time < datetime.time(hour+period, minute+5))
                                           & (features_df['squ'] == squ)]
        
    if uni:
        scores = []
        y = np.array(targets_jan_13_2015_30163['y2']).reshape(-1,1)
        for feature in features_jan_13_2015_30163.loc[:,'weights':'z15'].columns:
            x = np.array(features_jan_13_2015_30163[feature]).reshape(-1,1)
            reg = LinearRegression()
            reg.fit(x, y)
            score = reg.score(x , y)
            scores.append((feature, score))
            
        sorted_scores = sort_scores1(scores)[-5:]
        
    if multi:
        scores = []
        y = np.array(targets_jan_13_2015_30163['y2']).reshape(-1,1)
        for feature1 in features_jan_13_2015_30163.loc[:,'weights':'z15'].columns:
            for feature2 in features_jan_13_2015_30163.loc[:,'weights':'z15'].columns:
                feat1 = np.array(features_jan_13_2015_30163[feature1]).reshape(-1, 1)
                feat2 = np.array(features_jan_13_2015_30163[feature2]).reshape(-1, 1)
                x = np.concatenate((feat1, feat2), axis=1)
                reg = LinearRegression()
                reg.fit(x, y)
                score = reg.score(x, y)
                scores.append((feature1, feature2, score))
                
        sorted_scores = sort_scores2(scores)[-10:]
        
    return sorted_scores
                
    
        