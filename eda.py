import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
def linreg(features_df, targets_df, year, month, day, squ, uni=False, multi=False, hour=None, minute=None, period=None):
    
    if hour == None:
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
        y = np.array(targets['y2']).reshape(-1,1)
        for feature in features.loc[:,'weights':'z15'].columns:
            x = np.array(features[feature]).reshape(-1,1)
            reg = LinearRegression()
            reg.fit(x, y)
            score = reg.score(x , y)
            scores.append((feature, score))
            
        sorted_scores = sort_scores1(scores)[-5:]
        
    if multi:
        scores = []
        y = np.array(targets['y2']).reshape(-1,1)
        for feature1 in features.loc[:,'weights':'z15'].columns:
            for feature2 in features.loc[:,'weights':'z15'].columns:
                feat1 = np.array(features[feature1]).reshape(-1, 1)
                feat2 = np.array(features[feature2]).reshape(-1, 1)
                x = np.concatenate((feat1, feat2), axis=1)
                reg = LinearRegression()
                reg.fit(x, y)
                score = reg.score(x, y)
                scores.append((feature1, feature2, score))
                
        sorted_scores = sort_scores2(scores)[-10:]
        
    return sorted_scores


def count_top_feats(feature_df, target_df, year, month, day, hour=None, minute=None, period=None):
    features_dict = {}
    scores_list = []
    squ_list = feature_df['squ'].unique()
    for squ in squ_list:
        scores = linreg(feature_df, target_df, year, month, day, squ, uni=True, hour=hour, minute=minute, period=period)
        print(scores)
        scores_list.append(scores)
    
    for i in range(len(scores_list)):
        for j in range(len(scores_list[i])):
            if scores_list[i][j][0] in features_dict:
                features_dict[scores_list[i][j][0]] += 1
            else:
                features_dict[scores_list[i][j][0]] = 1
                
    return features_dict
                    
    
def plot_all(feature_df, target_df):
    
    figsize = (30,24)
    cols = 5
    rows = len(feature_df.loc[:,'weights':'z15'].columns) // cols + 1

    axs = plt.figure(figsize=figsize).subplots(rows, cols)
    for ax, col in zip(axs.flat, feature_df.loc[:,'weights':'z15'].columns):
        ax.scatter(feature_df[col], target_df['y2'], s=0.5)
        ax.set_xlabel(col)
        ax.set_ylabel('y2')
        
    plt.tight_layout()
    plt.show()
    
    
def plot_3d(feature_df, target_df, feat1, feat2):
    x = feature_df[feat1]
    y = feature_df[feat2]
    z = target_df['y2']

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_zlabel('y2')
    