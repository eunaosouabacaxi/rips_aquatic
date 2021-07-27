import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm


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
                                           & (targets_df['squ'] == squ)]
        
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
                                           & (targets_df['squ'] == squ)]
        
    if uni:
        scores = []
        y = np.array(targets['y2']).reshape(-1,1)
        feat_list = features.loc[:,'weights':'z15'].columns
        for feature in feat_list:
            x = np.array(features[feature]).reshape(-1,1)
            if x.shape == (0,1):
                continue
            reg = LinearRegression()
            reg.fit(x, y)
            score = reg.score(x, y)
            scores.append((feature, score))
            
        scores_sorted = sort_scores1(scores)[-5:]
        
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
        scores_sorted = sort_scores2(scores)[-10:]
        
    return scores_sorted


def count_top_feats(feature_df, target_df, year, month, day, hour=None, minute=None, period=None):
    features_dict = {}
    scores_list = []
    squ_list = feature_df['squ'].unique()
    for squ in squ_list:
        scores = linreg(feature_df, target_df, year, month, day, squ, uni=True, hour=hour, minute=minute, period=period)
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
    
    
def ols_results(feature_df, target_df, x=None, z=None, y='y2'):
    
    q1 = feature_df[z].quantile(.25)
    q2 = feature_df[z].quantile(.5)
    q3 = feature_df[z].quantile(.75)
    
    features_q1 = feature_df[feature_df[z] < q1]
    features_q2 = feature_df[(feature_df[z] >= q1) & (feature_df[z] < q2)]
    features_q3 = feature_df[(feature_df[z] >= q2) & (feature_df[z] < q3)]
    features_q4 = feature_df[feature_df[z] >= q3]

    targets_q1 = target_df[feature_df[z] < q1]
    targets_q2 = target_df[(feature_df[z] >= q1) & (feature_df[z] < q2)]
    targets_q3 = target_df[(feature_df[z] >= q2) & (feature_df[z] < q3)]
    targets_q4 = target_df[feature_df[z] >= q3]
    
    y_q1 = np.array(targets_q1[y]).reshape(-1,1)
    y_q2 = np.array(targets_q2[y]).reshape(-1,1)
    y_q3 = np.array(targets_q3[y]).reshape(-1,1)
    y_q4 = np.array(targets_q4[y]).reshape(-1,1)
    
    zeros_q1 = np.zeros(np.array(features_q1[x]).reshape(-1,1).shape)
    zeros_q2 = np.zeros(np.array(features_q2[x]).reshape(-1,1).shape)
    zeros_q3 = np.zeros(np.array(features_q3[x]).reshape(-1,1).shape)
    zeros_q4 = np.zeros(np.array(features_q4[x]).reshape(-1,1).shape)
    
    x_ = np.array(feature_df[x]).reshape(-1, 1)

    x_q2 = np.array(features_q2[x]).reshape(-1, 1)
    x_q3 = np.array(features_q3[x]).reshape(-1, 1)
    x_q4 = np.array(features_q4[x]).reshape(-1, 1)

    x_q2 = np.concatenate((zeros_q1, x_q2, zeros_q3, zeros_q4), axis=0)
    x_q3 = np.concatenate((zeros_q1, zeros_q2, x_q3, zeros_q4), axis=0)
    x_q4 = np.concatenate((zeros_q1, zeros_q2, zeros_q3, x_q4), axis=0)

    z_q2 = np.array(features_q2[z]).reshape(-1, 1)
    z_q3 = np.array(features_q3[z]).reshape(-1, 1)
    z_q4 = np.array(features_q4[z]).reshape(-1, 1)

    z_q2 = np.concatenate((zeros_q1, z_q2, zeros_q3, zeros_q4), axis=0)
    z_q3 = np.concatenate((zeros_q1, zeros_q2, z_q3, zeros_q4), axis=0)
    z_q4 = np.concatenate((zeros_q1, zeros_q2, zeros_q3, z_q4), axis=0)

    x_z_q2 = x_q2*z_q2
    x_z_q3 = x_q3*z_q3
    x_z_q4 = x_q4*z_q4

    bias = np.ones(x_z_q2.shape)
    
    x__ = np.concatenate((bias, x_, x_z_q2, x_z_q3, x_z_q4), axis=1)
    y_ = np.concatenate((y_q1, y_q2, y_q3, y_q4), axis=0)

    #reg = LinearRegression()
    #reg.fit(x__, y_)
    #coef = reg.coef_
    results = sm.OLS(y_, x__).fit()
    f = results.fvalue
    p = results.f_pvalue
    
    return f, p
    
    #print('The coefficients are: \n', coef[0], '\n')

    #print(results.summary())
    
    
def ols_results_wald(feature_df, target_df, x=None, z=None, y='y2'):
    
    q1 = feature_df[z].quantile(.25)
    q2 = feature_df[z].quantile(.5)
    q3 = feature_df[z].quantile(.75)
    
    features_q1 = feature_df[feature_df[z] < q1]
    features_q2 = feature_df[(feature_df[z] >= q1) & (feature_df[z] < q2)]
    features_q3 = feature_df[(feature_df[z] >= q2) & (feature_df[z] < q3)]
    features_q4 = feature_df[feature_df[z] >= q3]

    targets_q1 = target_df[feature_df[z] < q1]
    targets_q2 = target_df[(feature_df[z] >= q1) & (feature_df[z] < q2)]
    targets_q3 = target_df[(feature_df[z] >= q2) & (feature_df[z] < q3)]
    targets_q4 = target_df[feature_df[z] >= q3]
    
    y_q1 = np.array(targets_q1[y]).reshape(-1,1)
    y_q2 = np.array(targets_q2[y]).reshape(-1,1)
    y_q3 = np.array(targets_q3[y]).reshape(-1,1)
    y_q4 = np.array(targets_q4[y]).reshape(-1,1)
    
    zeros_q1 = np.zeros(np.array(features_q1[x]).reshape(-1,1).shape)
    zeros_q2 = np.zeros(np.array(features_q2[x]).reshape(-1,1).shape)
    zeros_q3 = np.zeros(np.array(features_q3[x]).reshape(-1,1).shape)
    zeros_q4 = np.zeros(np.array(features_q4[x]).reshape(-1,1).shape)
    
    x_ = np.array(feature_df[x]).reshape(-1, 1)

    x_q2 = np.array(features_q2[x]).reshape(-1, 1)
    x_q3 = np.array(features_q3[x]).reshape(-1, 1)
    x_q4 = np.array(features_q4[x]).reshape(-1, 1)

    x_q2 = np.concatenate((zeros_q1, x_q2, zeros_q3, zeros_q4), axis=0)
    x_q3 = np.concatenate((zeros_q1, zeros_q2, x_q3, zeros_q4), axis=0)
    x_q4 = np.concatenate((zeros_q1, zeros_q2, zeros_q3, x_q4), axis=0)

    z_q2 = np.array(features_q2[z]).reshape(-1, 1)
    z_q3 = np.array(features_q3[z]).reshape(-1, 1)
    z_q4 = np.array(features_q4[z]).reshape(-1, 1)

    z_q2 = np.concatenate((zeros_q1, z_q2, zeros_q3, zeros_q4), axis=0)
    z_q3 = np.concatenate((zeros_q1, zeros_q2, z_q3, zeros_q4), axis=0)
    z_q4 = np.concatenate((zeros_q1, zeros_q2, zeros_q3, z_q4), axis=0)

    x_z_q2 = x_q2*z_q2
    x_z_q3 = x_q3*z_q3
    x_z_q4 = x_q4*z_q4

    bias = np.ones(x_z_q2.shape)
    
    x__ = np.concatenate((bias, x_, x_z_q2, x_z_q3, x_z_q4), axis=1)
    y_ = np.concatenate((y_q1, y_q2, y_q3, y_q4), axis=0)

    #reg = LinearRegression()
    #reg.fit(x__, y_)
    #coef = reg.coef_
    results = sm.OLS(y_, x__).fit()
    w = results.wald_test(np.eye(len(results.params)[2:5]))
    f = w.f_value
    p = w.p_value
    
    return f, p
    
    #print('The coefficients are: \n', coef[0], '\n')

    #print(results.summary())