import pandas as pd
import numpy as np
import preprocess as pp
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import eda
import statsmodels.api as sm



def wls_results_wald(feature_df, target_df, x=None, z=None, y='y2'):
    '''
    Computes f-statistic and its p-value from Wald test for B2=B3=B4=0

    Args:
        feature_df : pandas dataframe
        target_df : pandas dataframe
	x : string
	    X (predictor) column to use
	z : string
	    Z (interactor) column to use
        y : string
	    Y (target) column to use

    Returns:
	f : f-statistic from Wald test
	p : p-value of f-statistic from Wald test

    '''

    # find values for quartiles of a given Z
    q1 = feature_df[z].quantile(.25)
    q2 = feature_df[z].quantile(.5)
    q3 = feature_df[z].quantile(.75)
    
    # create vectors with ones corresponding to locations of Z's in given quartiles
    # and zeros elsewhere
    I_q2 = np.where((feature_df[z] >= q1) & (feature_df[z] < q2), 1, 0)
    I_q3 = np.where((feature_df[z] >= q2) & (feature_df[z] < q3), 1, 0)
    I_q4 = np.where(feature_df[z] >= q3, 1, 0)
    
    # create column vectors for a given X and Z
    x_ = np.array(feature_df[x]).reshape(-1,1)
    z_ = np.array(feature_df[z]).reshape(-1,1)
    
    # create column vectors multiplying a given X and Z with the quartile vectors
    x_z_q2 = I_q2.reshape(-1,1) * x_ * z_
    x_z_q3 = I_q3.reshape(-1,1) * x_ * z_
    x_z_q4 = I_q4.reshape(-1,1) * x_ * z_

    bias = np.ones(x_.shape)
    
    # combine the feature vectors into one array
    x__ = np.concatenate((bias, x_, x_z_q2, x_z_q3, x_z_q4), axis=1)

    # create a vector for a given Y
    y_ = target_df[y]

    # create a vector for the weights
    weights = np.array(feature_df['weights']).reshape(-1,1)

    # use WLS from statsmodels to fit a line and compute F-statistic and its p-value
    # from Wald test for B2=B3=B4=0
    results = sm.WLS(y_, x__, weights=weights).fit()
    w = results.wald_test(np.eye(len(results.params))[2:5])
    f = w.fvalue
    p = w.pvalue
    
    return f, p
    
def wls_results(feature_df, target_df, x=None, z=None, y='y2'):
    '''
    Computes f-statistic and its p-value from F-statistic for B1=B2=B3=B4=0

    Args:
        feature_df : pandas dataframe
        target_df : pandas dataframe
	x : string
	    X (predictor) column to use
	z : string
	    Z (interactor) column to use
        y : string
	    Y (target) column to use

    Returns:
	f : f-statistic from Wald test
	p : p-value of f-statistic from Wald test

    '''

    # find values for quartiles of a given Z
    q1 = feature_df[z].quantile(.25)
    q2 = feature_df[z].quantile(.5)
    q3 = feature_df[z].quantile(.75)
    

    # # create vectors with ones corresponding to locations of Z's in given quartiles
    # and zeros elsewhere
    I_q2 = np.where((feature_df[z] >= q1) & (feature_df[z] < q2), 1, 0)
    I_q3 = np.where((feature_df[z] >= q2) & (feature_df[z] < q3), 1, 0)
    I_q4 = np.where(feature_df[z] >= q3, 1, 0)
    

    ## create column vectors for a given X and Z
    x_ = np.array(feature_df[x]).reshape(-1,1)
    z_ = np.array(feature_df[z]).reshape(-1,1)
    

    # create column vectors multiplying a given X and Z with the quartile vectors
    x_z_q2 = I_q2.reshape(-1,1) * x_ * z_
    x_z_q3 = I_q3.reshape(-1,1) * x_ * z_
    x_z_q4 = I_q4.reshape(-1,1) * x_ * z_

    bias = np.ones(x_.shape)
    
    # combine the feature vectors into one array
    x__ = np.concatenate((bias, x_, x_z_q2, x_z_q3, x_z_q4), axis=1)

    # create a vector for a given Y
    y_ = target_df[y]

    # create a vector for the weights
    weights = np.array(feature_df['weights']).reshape(-1,1)

    # use WLS from statsmodels to fit a line and compute F-statistic and its p-value
    results = sm.WLS(y_, x__, weights=weights).fit()
    f = results.fvalue
    p = results.f_pvalue
    
    return f, p


def ranked_from_wald0(features, targets):
    '''
    Ranks top k p-values from F-statistic and Wald test for k=5,10,25,50
    From every X-Z pair in dataset

    Args:
	features : pandas dataframe
        targets : pandas dataframe

    Returns:
	ranked_p_5 : list
	ranked_p_10 : list
	ranked_p_25 : list
	ranked_p_50 : list
    '''

    # create list of strings for X and Z columns
    z_list = features.loc[:,'z1':'z12'].columns
    x_list = features.loc[:,'x1':'x34'].columns

    # create lists to store values for F-statistic and its p-value and 
    # F-statistic and its p-value from Wald test for B2=B3=B4=0
    f_list = []
    p_list = []
    f0_list = []
    p0_list = []

    # iterate through each Z
    for z in z_list:
        flist = []
        plist = []
        f0list = []
        p0list = []
	
	# iterate through each X and compute stats for each X-Z pair
        for feat in x_list:
            f0, p0 = wls_results_wald(features, targets, x=feat, z=z)
            f, p = wls_results(features, targets, x=feat, z=z)
            f0list.append(f0)
            p0list.append(p0)
            flist.append(f)
            plist.append(p)

        f_list.append(flist)
        p_list.append(plist)
        f0_list.append(f0list)
        p0_list.append(p0list)
    
    #f_dict = {'z1':f_list[0], 'z2':f_list[1], 'z3':f_list[2], 'z4':f_list[3],
    #          'z5':f_list[4], 'z6':f_list[5], 'z7':f_list[6], 'z8':f_list[7],
    #          'z9':f_list[8], 'z10':f_list[9], 'z11':f_list[10], 'z12':f_list[11]}

    #f_df = pd.DataFrame.from_dict(f_dict, orient='index',
    #                       columns=x_list)

    p_dict = {'z1':p_list[0], 'z2':p_list[1], 'z3':p_list[2], 'z4':p_list[3],
              'z5':p_list[4], 'z6':p_list[5], 'z7':p_list[6], 'z8':p_list[7],
              'z9':p_list[8], 'z10':p_list[9], 'z11':p_list[10], 'z12':p_list[11]}

    p_df = pd.DataFrame.from_dict(p_dict, orient='index',
                           columns=x_list)
    
    #f0_dict = {'z1':f0_list[0], 'z2':f0_list[1], 'z3':f0_list[2], 'z4':f0_list[3],
    #          'z5':f0_list[4], 'z6':f0_list[5], 'z7':f0_list[6], 'z8':f0_list[7],
    #          'z9':f0_list[8], 'z10':f0_list[9], 'z11':f0_list[10], 'z12':f0_list[11]}

    #f0_df = pd.DataFrame.from_dict(f0_dict, orient='index',
    #                       columns=x_list)

    p0_dict = {'z1':p0_list[0], 'z2':p0_list[1], 'z3':p0_list[2], 'z4':p0_list[3],
              'z5':p0_list[4], 'z6':p0_list[5], 'z7':p0_list[6], 'z8':p0_list[7],
              'z9':p0_list[8], 'z10':p0_list[9], 'z11':p0_list[10], 'z12':p0_list[11]}

    p0_df = pd.DataFrame.from_dict(p0_dict, orient='index',
                           columns=x_list)
    
     
    p_list = []
    for col in p_df.columns:
        for idx in p_df.index:
            val = p_df.loc[idx, col]
            p_list.append(((col, idx), val))
            
    p0_list = []
    for col in p0_df.columns:
        for idx in p0_df.index:
            val = p0_df.loc[idx, col]
            p0_list.append(((col, idx), val))
    
    # ranked pairs with pvals
    ranked_p = eda.sort_scores1(p_list)
    ranked_p0 = eda.sort_scores1(p0_list)
    
    ranked_pair_list = []
    ranked_pair_list0 = []
    
    
    for pair in ranked_p[:30]:
        ranked_pair_list.append(pair[0])
    for pair in ranked_p0:
        ranked_pair_list0.append(pair[0])
        
    ranked_p_5 = ranked_p[:3]
    for i in range(len(ranked_pair_list0)):
        if ranked_pair_list0[i] in ranked_pair_list[:3]:
            continue
        elif len(ranked_p_5) == 5:
            break
        else:
            ranked_p_5.append(ranked_p0[i])
            
    ranked_p_10 = ranked_p[:6]
    for i in range(len(ranked_pair_list0)):
        if ranked_pair_list0[i] in ranked_pair_list:
            continue
        elif len(ranked_p_10) == 10:
            break
        else:
            ranked_p_10.append(ranked_p0[i])
    
    ranked_p_25 = ranked_p[:15]
    for i in range(len(ranked_pair_list0)):
        if ranked_pair_list0[i] in ranked_pair_list[:15]:
            continue
        elif len(ranked_p_25) == 25:
            break
        else:
            ranked_p_25.append(ranked_p0[i])
            
    ranked_p_50 = ranked_p[:30]
    for i in range(len(ranked_pair_list0)):
        if ranked_pair_list0[i] in ranked_pair_list:
            continue
        elif len(ranked_p_50) == 50:
            break
        else:
            ranked_p_50.append(ranked_p0[i])
    
   
    return ranked_p_5, ranked_p_10, ranked_p_25, ranked_p_50

def normalize_and_fill(df):
    '''
    Normalize the columns of a dataframe to have mean 0 and variance 1
    while storing its original mean and standard deviation

    Args:
	df : pandas dataframe

    Returns:
	df : pandas dataframe
	mean : numpy array
	std : numpy array
    '''
    mean = df.loc[:,'x1':'z12'].mean()
    std = df.loc[:,'x1':'z12'].std()
    df.loc[:,'x1':'z12'] = (df.loc[:,'x1':'z12'] - mean)/std
    
    df = df.fillna(0)
    
    return df, mean, std

def train_and_test(features_df, targets_df, tod='early', year=None,
                   train_month_start=None, train_day_start=None,
                   train_month_end=None, train_day_end=None,
                   test_month_start=None, test_day_start=None,
                   test_month_end=None, test_day_end=None):
    '''
    Splits a dataset into training and testing based on input parameters

    Args:
	features_df : pandas dataframe
	targets_df : pandas dataframe
	tod : string
		acceptable inputs : 'early', 'mid', 'late'
	year : int
	train_month_start : int
		acceptable inputs : 1-12
	train_day_start : int
		acceptable inputs : day must fall inside of train_month_start
	train_month_end : int
		acceptable inputs : 1-12
	train_day_end : int
		acceptable inputs : day must fall inside of train_month_end
	test_month_start : int
		acceptable inputs : 1-12
	test_day_start : int
		acceptable inputs : day must fall inside of test_month_start
	test_month_end : int
		acceptable inputs : 1-12
	test_day_end : int
		acceptable inputs : day must fall inside of test_month_end

    Returns:
	train_features : pandas dataframe
        train_targets : pandas dataframe
        test_features : pandas dataframe
        test_targets : pandas dataframe
    '''
    
    train_features = features_df.loc[(features_df['datetime'].dt.date >= datetime.date(year, train_month_start, train_day_start))
                                    & (features_df['datetime'].dt.date <= datetime.date(year, train_month_end, train_day_end))]
    train_targets = targets_df.loc[(targets_df['datetime'].dt.date >= datetime.date(year, train_month_start, train_day_start))
                                    & (targets_df['datetime'].dt.date <= datetime.date(year, train_month_end, train_day_end))]
    
    test_features = features_df.loc[(features_df['datetime'].dt.date >= datetime.date(year, test_month_start, test_day_start))
                                    & (features_df['datetime'].dt.date <= datetime.date(year, test_month_end, test_day_end))]
    test_targets = targets_df.loc[(targets_df['datetime'].dt.date >= datetime.date(year, test_month_start, test_day_start))
                                    & (targets_df['datetime'].dt.date <= datetime.date(year, test_month_end, test_day_end))]
    
    
    if tod == 'all':
        train_features = train_features
        test_features = test_features
        train_targets = train_targets
        test_targets = test_targets
    
    if tod == 'early':
        train_features = train_features.loc[(train_features['datetime'].dt.time >= datetime.time(9, 45))
                                    & (train_features['datetime'].dt.time <= datetime.time(10, 45))]
        test_features = test_features.loc[(test_features['datetime'].dt.time >= datetime.time(9, 45))
                                    & (test_features['datetime'].dt.time <= datetime.time(10, 45))]
        train_targets = train_targets.loc[(train_targets['datetime'].dt.time >= datetime.time(9, 45))
                                    & (train_targets['datetime'].dt.time <= datetime.time(10, 45))]
        test_targets = test_targets.loc[(test_targets['datetime'].dt.time >= datetime.time(9, 45))
                                    & (test_targets['datetime'].dt.time <= datetime.time(10, 45))]
        
    elif tod == 'mid':
        train_features = train_features.loc[(train_features['datetime'].dt.time >= datetime.time(12))
                                    & (train_features['datetime'].dt.time <= datetime.time(13))]
        test_features = test_features.loc[(test_features['datetime'].dt.time >= datetime.time(12))
                                    & (test_features['datetime'].dt.time <= datetime.time(13))]
        train_targets = train_targets.loc[(train_targets['datetime'].dt.time >= datetime.time(12))
                                    & (train_targets['datetime'].dt.time <= datetime.time(13))]
        test_targets = test_targets.loc[(test_targets['datetime'].dt.time >= datetime.time(12))
                                    & (test_targets['datetime'].dt.time <= datetime.time(13))]
        
    elif tod == 'late':
        train_features = train_features.loc[(train_features['datetime'].dt.time >= datetime.time(14, 45))
                                    & (train_features['datetime'].dt.time <= datetime.time(15, 45))]
        test_features = test_features.loc[(test_features['datetime'].dt.time >= datetime.time(14, 45))
                                    & (test_features['datetime'].dt.time <= datetime.time(15, 45))]
        train_targets = train_targets.loc[(train_targets['datetime'].dt.time >= datetime.time(14, 45))
                                    & (train_targets['datetime'].dt.time <= datetime.time(15, 45))]
        test_targets = test_targets.loc[(test_targets['datetime'].dt.time >= datetime.time(14, 45))
                                    & (test_targets['datetime'].dt.time <= datetime.time(15, 45))]
    
    
    return train_features, train_targets, test_features, test_targets



def selection(train_features, train_targets, test_features, test_targets):
    '''
    Creates new data based on all features and top k features for k=5,10,25,50

    Args:
	train_features : pandas dataframe
	train_targets : pandas dataframe
	test_features : pandas dataframe
	test_targets : pandas dataframe

    Returns:
	train_features_all : numpy array
		34 + 408 columns
	train_features_top_5 : numpy array
		34 + 5 columns
	train_features_top_10 : numpy array
		34 + 10 columns
	train_features_top_25 : numpy array
		34 + 25 columns
	train_features_top_50 : numpy array
		34 + 50 columns
	test_features_all : numpy array
		34 + 408 columns
	test_features_top_5 : numpy array
		34 + 5 columns
	test_features_top_10 : numpy array
		34 + 10 columns
	test_features_top_25 : numpy array
		34 + 25 columns
	test_features_top_50 : numpy array
		34 + 50 columns
	df : pandas dataframe
    '''
                
    z_list = train_features.loc[:,'z1':'z12'].columns
    x_list = train_features.loc[:,'x1':'x34'].columns
    
    train_features, mean, std = normalize_and_fill(train_features)
    test_features.loc[:,'x1':] = (test_features.loc[:,'x1':]-mean)/std
    test_features = test_features.fillna(0)
    
    # create lists to store arrays for columns
    pairs_list_train_all = []
    pairs_list_test_all = []
    pairs_list_train_top_5 = []
    pairs_list_test_top_5 = []
    pairs_list_train_top_10 = []
    pairs_list_test_top_10 = []
    pairs_list_train_top_25 = []
    pairs_list_test_top_25 = []
    pairs_list_train_top_50 = []
    pairs_list_test_top_50 = []
    
    
    # iterate through each X and append each X column to each list
    for x in x_list:
        x_col_train = np.array(train_features[x]).reshape(-1,1)
        x_col_test = np.array(test_features[x]).reshape(-1,1)
        pairs_list_train_all.append(x_col_train)
        pairs_list_train_top_5.append(x_col_train)
        pairs_list_train_top_10.append(x_col_train)
        pairs_list_train_top_25.append(x_col_train)
        pairs_list_train_top_50.append(x_col_train)
        pairs_list_test_all.append(x_col_test)
        pairs_list_test_top_5.append(x_col_test)
        pairs_list_test_top_10.append(x_col_test)
        pairs_list_test_top_25.append(x_col_test)
        pairs_list_test_top_50.append(x_col_test)

	# iterate through each Z and multiply an X column and a Z column
	# before appending the new column to the lists with all features
        for z in z_list:
            z_col_train = np.array(train_features[z]).reshape(-1,1)
            x_z_train = x_col_train * z_col_train
            pairs_list_train_all.append(x_z_train)
            z_col_test = np.array(test_features[z]).reshape(-1,1)
            x_z_test = x_col_test * z_col_test
            pairs_list_test_all.append(x_z_test)
            
    # compute and return the top k pairs with the lowest p-values for k=5,10,25,50
    ranked_p_5, ranked_p_10, ranked_p_25, ranked_p_50 = ranked_from_wald0(train_features, train_targets)
    
    # multiply the respective X-Z pair columns and append to the lists for k=5
    pairs_list_5 = []
    for pair in ranked_p_5:
        pairs_list_5.append(pair[0])
        x = pair[0][0]
        z = pair[0][1]
        x_col_train = np.array(train_features[x]).reshape(-1,1)
        z_col_train = np.array(train_features[z]).reshape(-1,1)
        x_z_train = x_col_train * z_col_train
        pairs_list_train_top_5.append(x_z_train)
        x_col_test = np.array(test_features[x]).reshape(-1,1)
        z_col_test = np.array(test_features[z]).reshape(-1,1)
        x_z_test = x_col_test * z_col_test
        pairs_list_test_top_5.append(x_z_test)
        
    # multiply the respective X-Z pair columns and append to the list for k=10
    pairs_list_10 = []
    for pair in ranked_p_10:
        pairs_list_10.append(pair[0])
        x = pair[0][0]
        z = pair[0][1]
        x_col_train = np.array(train_features[x]).reshape(-1,1)
        z_col_train = np.array(train_features[z]).reshape(-1,1)
        x_z_train = x_col_train * z_col_train
        pairs_list_train_top_10.append(x_z_train)
        x_col_test = np.array(test_features[x]).reshape(-1,1)
        z_col_test = np.array(test_features[z]).reshape(-1,1)
        x_z_test = x_col_test * z_col_test
        pairs_list_test_top_10.append(x_z_test)
        
    # multiply the respective X-Z pair columns and append to the list for k=25
    pairs_list_25 = []
    for pair in ranked_p_25:
        pairs_list_25.append(pair[0])
        x = pair[0][0]
        z = pair[0][1]
        x_col_train = np.array(train_features[x]).reshape(-1,1)
        z_col_train = np.array(train_features[z]).reshape(-1,1)
        x_z_train = x_col_train * z_col_train
        pairs_list_train_top_25.append(x_z_train)
        x_col_test = np.array(test_features[x]).reshape(-1,1)
        z_col_test = np.array(test_features[z]).reshape(-1,1)
        x_z_test = x_col_test * z_col_test
        pairs_list_test_top_25.append(x_z_test)
        
    # multiply the respective X-Z pair columns and append to the list for k=50
    pairs_list_50 = []
    print('Top 50 pairs')
    for pair in ranked_p_50:
        print(pair)
        pairs_list_50.append(pair[0])
        x = pair[0][0]
        z = pair[0][1]
        x_col_train = np.array(train_features[x]).reshape(-1,1)
        z_col_train = np.array(train_features[z]).reshape(-1,1)
        x_z_train = x_col_train * z_col_train
        pairs_list_train_top_50.append(x_z_train)
        x_col_test = np.array(test_features[x]).reshape(-1,1)
        z_col_test = np.array(test_features[z]).reshape(-1,1)
        x_z_test = x_col_test * z_col_test
        pairs_list_test_top_50.append(x_z_test)
    print('')
        
    # data frame to store top 50 pairs to be exported to csv
    df = pd.DataFrame(np.array(pairs_list_50))
        
    # combine the arrays in the lists to create complete features arrays
    # for all features and k=5,20,25,50 top features
    train_features_all = np.concatenate(pairs_list_train_all, axis=1)
    test_features_all = np.concatenate(pairs_list_test_all, axis=1)
    train_features_top_5 = np.concatenate(pairs_list_train_top_5, axis=1)
    test_features_top_5 = np.concatenate(pairs_list_test_top_5, axis=1)
    train_features_top_10 = np.concatenate(pairs_list_train_top_10, axis=1)
    test_features_top_10 = np.concatenate(pairs_list_test_top_10, axis=1)
    train_features_top_25 = np.concatenate(pairs_list_train_top_25, axis=1)
    test_features_top_25 = np.concatenate(pairs_list_test_top_25, axis=1)
    train_features_top_50 = np.concatenate(pairs_list_train_top_50, axis=1)
    test_features_top_50 = np.concatenate(pairs_list_test_top_50, axis=1)
    
    return train_features_all, train_features_top_5, train_features_top_10, train_features_top_25, train_features_top_50, test_features_all, test_features_top_5, test_features_top_10, test_features_top_25, test_features_top_50, df
    


def performance(train_features_all, train_features_top_5, train_features_top_10, train_features_top_25, train_features_top_50, train_targets,
               test_features_all, test_features_top_5, test_features_top_10, test_features_top_25, test_features_top_50, test_targets):
    '''
    Computes performance metrics for all models

    Args:
	train_features_all : numpy array
	train_features_top_5 : numpy array
	train_features_top_10 : numpy array
	train_features_top_25 : numpy array
	train_features_top_50 : numpy array
	train_targets : pandas dataframe
	test_features_all : numpy array
	test_features_top_5 : numpy array
	test_features_top_10 : numpy array
	test_features_top_25 : numpy array
	test_features_top_50 : numpy array
	test_targets : pandas dataframe

    Returns:

    '''
    
    
    pca_5 = PCA(n_components=5)
    pca_5.fit(train_features_all)
    train_features_pca_5 = pca_5.transform(train_features_all)
    test_features_pca_5 = pca_5.transform(test_features_all)
    
    pca_10 = PCA(n_components=10)
    pca_10.fit(train_features_all)
    train_features_pca_10 = pca_10.transform(train_features_all)
    test_features_pca_10 = pca_10.transform(test_features_all)
    
    pca_25 = PCA(n_components=25)
    pca_25.fit(train_features_all)
    train_features_pca_25 = pca_25.transform(train_features_all)
    test_features_pca_25 = pca_25.transform(test_features_all)
    
    pca_50 = PCA(n_components=50)
    pca_50.fit(train_features_all)
    train_features_pca_50 = pca_50.transform(train_features_all)
    test_features_pca_50 = pca_50.transform(test_features_all)
    
    lr_all = LinearRegression().fit(train_features_all, np.array(train_targets['y2']).reshape(-1,1))
    lr_top_5 = LinearRegression().fit(train_features_top_5, np.array(train_targets['y2']).reshape(-1,1))
    lr_top_10 = LinearRegression().fit(train_features_top_10, np.array(train_targets['y2']).reshape(-1,1))
    lr_top_25 = LinearRegression().fit(train_features_top_25, np.array(train_targets['y2']).reshape(-1,1))
    lr_top_50 = LinearRegression().fit(train_features_top_50, np.array(train_targets['y2']).reshape(-1,1))
    lr_pca_5 = LinearRegression().fit(train_features_pca_5, np.array(train_targets['y2']).reshape(-1,1))
    lr_pca_10 = LinearRegression().fit(train_features_pca_10, np.array(train_targets['y2']).reshape(-1,1))
    lr_pca_25 = LinearRegression().fit(train_features_pca_25, np.array(train_targets['y2']).reshape(-1,1))
    lr_pca_50 = LinearRegression().fit(train_features_pca_50, np.array(train_targets['y2']).reshape(-1,1))
    
    pred_all = np.array(lr_all.predict(test_features_all)).reshape(-1,1)
    pred_top_5 = np.array(lr_top_5.predict(test_features_top_5)).reshape(-1,1)
    pred_top_10 = np.array(lr_top_10.predict(test_features_top_10)).reshape(-1,1)
    pred_top_25 = np.array(lr_top_25.predict(test_features_top_25)).reshape(-1,1)
    pred_top_50 = np.array(lr_top_50.predict(test_features_top_50)).reshape(-1,1)
    pred_pca_5 = np.array(lr_pca_5.predict(test_features_pca_5)).reshape(-1,1)
    pred_pca_10 = np.array(lr_pca_10.predict(test_features_pca_10)).reshape(-1,1)
    pred_pca_25 = np.array(lr_pca_25.predict(test_features_pca_25)).reshape(-1,1)
    pred_pca_50 = np.array(lr_pca_50.predict(test_features_pca_50)).reshape(-1,1)
    
    
    print('R2 from all features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_all))
    print('R2 from top 5 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_top_5))
    print('R2 from top 10 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_top_10))
    print('R2 from top 25 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_top_25))
    print('R2 from top 50 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_top_50))
    print('R2 from pca 5 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_pca_5))
    print('R2 from pca 10 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_pca_10))
    print('R2 from pca 25 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_pca_25))
    print('R2 from pca 50 features')
    print(r2_score(np.array(test_targets['y2']).reshape(-1,1), pred_pca_50))
    print('')
    
    print('STD from all features')
    print(np.mean(np.std((pred_all, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from top 5 features')
    print(np.mean(np.std((pred_top_5, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from top 10 features')
    print(np.mean(np.std((pred_top_10, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from top 25 features')
    print(np.mean(np.std((pred_top_25, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from top 50 features')
    print(np.mean(np.std((pred_top_50, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from pca 5 features')
    print(np.mean(np.std((pred_pca_5, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from pca 10 features')
    print(np.mean(np.std((pred_pca_10, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from pca 25 features')
    print(np.mean(np.std((pred_pca_25, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    print('STD from pca 50 features')
    print(np.mean(np.std((pred_pca_50, np.array(test_targets['y2']).reshape(-1,1)), axis=0)))
    
    
    lr_all_comp = LinearRegression().fit(pred_all, np.array(test_targets['y2']).reshape(-1,1))
    lr_top_5_comp = LinearRegression().fit(pred_top_5, np.array(test_targets['y2']).reshape(-1,1))
    lr_top_10_comp = LinearRegression().fit(pred_top_10, np.array(test_targets['y2']).reshape(-1,1))
    lr_top_25_comp = LinearRegression().fit(pred_top_25, np.array(test_targets['y2']).reshape(-1,1))
    lr_top_50_comp = LinearRegression().fit(pred_top_50, np.array(test_targets['y2']).reshape(-1,1))
    lr_pca_5_comp = LinearRegression().fit(pred_pca_5, np.array(test_targets['y2']).reshape(-1,1))
    lr_pca_10_comp = LinearRegression().fit(pred_pca_10, np.array(test_targets['y2']).reshape(-1,1))
    lr_pca_25_comp = LinearRegression().fit(pred_pca_25, np.array(test_targets['y2']).reshape(-1,1))
    lr_pca_50_comp = LinearRegression().fit(pred_pca_50, np.array(test_targets['y2']).reshape(-1,1))
    
    print('')
    print('Coef of all')
    print(float(lr_all_comp.coef_[0]))
    print('Coef of top 5')
    print(float(lr_top_5_comp.coef_[0]))
    print('Coef of top 10')
    print(float(lr_top_10_comp.coef_[0]))
    print('Coef of top 25')
    print(float(lr_top_25_comp.coef_[0]))
    print('Coef of top 50')
    print(float(lr_top_50_comp.coef_[0]))
    print('Coef of pca 5')
    print(float(lr_pca_5_comp.coef_[0]))
    print('Coef of pca 10')
    print(float(lr_pca_10_comp.coef_[0]))
    print('Coef of pca 25')
    print(float(lr_pca_25_comp.coef_[0]))
    print('Coef of pca 50')
    print(float(lr_pca_50_comp.coef_[0]))
    print('')
    
    
    x = np.concatenate((pred_all, pred_top_5, pred_top_10, pred_top_25, pred_top_50,
                        pred_pca_5, pred_pca_10, pred_pca_25, pred_pca_50), axis=1)
    df = pd.DataFrame(x, columns=['all', 'top 5', 'top 10', 'top 25', 'top 50', 'pca 5', 'pca 10', 'pca 25', 'pca 50'])
    print(df.corr())
    print('')
    
    #fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9, sharey=True, sharex=True, figsize=(15,8))
    #ax1.scatter(pred_all, test_targets['y2'])
    #ax2.scatter(pred_top_5, test_targets['y2'])
    #ax3.scatter(pred_top_10, test_targets['y2'])
    #ax4.scatter(pred_top_25, test_targets['y2'])
    #ax5.scatter(pred_top_50, test_targets['y2'])
    #ax6.scatter(pred_pca_5, test_targets['y2'])
    #ax7.scatter(pred_pca_10, test_targets['y2'])
    #ax8.scatter(pred_pca_25, test_targets['y2'])
    #ax9.scatter(pred_pca_50, test_targets['y2'])    



jan_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_jan_2015.npy', features=True)
jan_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_jan_2015.npy', targets=True)

feb_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_feb_2015.npy', features=True)
feb_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_feb_2015.npy', targets=True)

mar_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_mar_2015.npy', features=True)
mar_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_mar_2015.npy', targets=True)

z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']

new_jan_features = jan_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_jan_features[z] = jan_features[z]
    
new_feb_features = feb_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_feb_features[z] = feb_features[z]
    
new_mar_features = mar_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_mar_features[z] = mar_features[z]
    
comb_features = pd.concat([new_jan_features, new_feb_features, new_mar_features], ignore_index=True)
comb_targets = pd.concat([jan_targets, feb_targets, mar_targets], ignore_index=True)

train_month_start = [1,1,1,1,2,2,2,2]
train_day_start = [5,12,19,26,2,9,16,23]
train_month_end = [1,2,2,2,2,3,3,3]
train_day_end = [30,6,13,20,27,6,13,20]
test_month_start = [2,2,2,2,3,3,3,3]
test_day_start = [2,9,16,23,2,9,16,23]
test_month_end = [2,2,2,2,3,3,3,3]
test_day_end = [6,13,20,27,6,13,20,27]


for tms, tds, tme, tde, testms, testds, testme, testde in zip(train_month_start, train_day_start, train_month_end, train_day_end, test_month_start, test_day_start, test_month_end, test_day_end):

    train_features, train_targets, test_features, test_targets = train_and_test(comb_features, comb_targets, tod='early', year=2015, 
                   train_month_start=tms, train_day_start=tds,
                   train_month_end=tme, train_day_end=tde,
                   test_month_start=testms, test_day_start=testds,
                   test_month_end=testme, test_day_end=testde)

    train_features_all, train_features_top_5, train_features_top_10, test_features_all, test_features_top_5, test_features_top_10, df = selection(train_features, train_targets, test_features, test_targets)

    print('Train start: {}/{}/2015'.format(tms, tds))
    print('Train end: {}/{}/2015'.format(tme, tde))
    print('Test start: {}/{}/2015'.format(testms, testds))
    print('Test end: {}/{}/2015'.format(testme, testde))
    print('')
    performance(train_features_all, train_features_top_5, train_features_top_10, train_features_top_25, train_features_top_50, train_targets,
               test_features_all, test_features_top_5, test_features_top_10, test_features_top_25, test_features_top_50, test_targets)
    print('')
    


apr_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_apr_2015.npy', features=True)
apr_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_apr_2015.npy', targets=True)

may_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_may_2015.npy', features=True)
may_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_may_2015.npy', targets=True)

jun_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_jun_2015.npy', features=True)
jun_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_jun_2015.npy', targets=True)

z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']    

new_apr_features = apr_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_apr_features[z] = apr_features[z]
    
new_may_features = may_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_may_features[z] = may_features[z]
    
new_jun_features = jun_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_jun_features[z] = jun_features[z]
    
comb_features = pd.concat([new_apr_features, new_may_features, new_jun_features], ignore_index=True)
comb_targets = pd.concat([apr_targets, may_targets, jun_targets], ignore_index=True)

train_month_start = [4,4,4,4,5,5,5,5]
train_day_start = [6,13,20,27,4,11,18,25]
train_month_end = [5,5,5,5,5,6,6,6]
train_day_end = [1,8,15,22,29,5,12,19]
test_month_start = [5,5,5,5,6,6,6,6]
test_day_start = [4,11,18,25,1,8,15,22]
test_month_end = [5,5,5,5,6,6,6,6]
test_day_end = [8,15,22,29,5,12,19,26]


for tms, tds, tme, tde, testms, testds, testme, testde in zip(train_month_start, train_day_start, train_month_end, train_day_end, test_month_start, test_day_start, test_month_end, test_day_end):

    train_features, train_targets, test_features, test_targets = train_and_test(comb_features, comb_targets, tod='early', year=2015, 
                   train_month_start=tms, train_day_start=tds,
                   train_month_end=tme, train_day_end=tde,
                   test_month_start=testms, test_day_start=testds,
                   test_month_end=testme, test_day_end=testde)

    train_features_all, train_features_top_5, train_features_top_10, test_features_all, test_features_top_5, test_features_top_10, df = selection(train_features, train_targets, test_features, test_targets)

    print('Train start: {}/{}/2015'.format(tms, tds))
    print('Train end: {}/{}/2015'.format(tme, tde))
    print('Test start: {}/{}/2015'.format(testms, testds))
    print('Test end: {}/{}/2015'.format(testme, testde))
    print('')
    performance(train_features_all, train_features_top_5, train_features_top_10, train_features_top_25, train_features_top_50, train_targets,
               test_features_all, test_features_top_5, test_features_top_10, test_features_top_25, test_features_top_50, test_targets)
    print('')



jul_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_jul_2015.npy', features=True)
jul_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_jul_2015.npy', targets=True)

aug_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_aug_2015.npy', features=True)
aug_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_aug_2015.npy', targets=True)

sep_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_sep_2015.npy', features=True)
sep_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_sep_2015.npy', targets=True)

z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']

new_jul_features = jul_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_jul_features[z] = jul_features[z]
    
new_aug_features = aug_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_aug_features[z] = aug_features[z]
    
new_sep_features = sep_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_sep_features[z] = sep_features[z]
    
comb_features = pd.concat([new_jul_features, new_aug_features, new_sep_features], ignore_index=True)
comb_targets = pd.concat([jul_targets, aug_targets, sep_targets], ignore_index=True)

train_month_start = [7,7,7,7,8,8,8,8]
train_day_start = [6,13,20,27,3,10,17,24]
train_month_end = [7,8,8,8,8,9,9,9]
train_day_end = [31,7,14,21,28,4,11,18]
test_month_start = [8,8,8,8,8,9,9,9]
test_day_start = [3,10,17,24,31,7,14,21]
test_month_end = [8,8,8,8,9,9,9,9]
test_day_end = [7,14,21,28,4,11,18,25]


for tms, tds, tme, tde, testms, testds, testme, testde in zip(train_month_start, train_day_start, train_month_end, train_day_end, test_month_start, test_day_start, test_month_end, test_day_end):

    train_features, train_targets, test_features, test_targets = train_and_test(comb_features, comb_targets, tod='early', year=2015, 
                   train_month_start=tms, train_day_start=tds,
                   train_month_end=tme, train_day_end=tde,
                   test_month_start=testms, test_day_start=testds,
                   test_month_end=testme, test_day_end=testde)

    train_features_all, train_features_top_5, train_features_top_10, test_features_all, test_features_top_5, test_features_top_10, df = selection(train_features, train_targets, test_features, test_targets)

    print('Train start: {}/{}/2015'.format(tms, tds))
    print('Train end: {}/{}/2015'.format(tme, tde))
    print('Test start: {}/{}/2015'.format(testms, testds))
    print('Test end: {}/{}/2015'.format(testme, testde))
    print('')
    performance(train_features_all, train_features_top_5, train_features_top_10, train_features_top_25, train_features_top_50, train_targets,
               test_features_all, test_features_top_5, test_features_top_10, test_features_top_25, test_features_top_50, test_targets)
    print('')



oct_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_oct_2015.npy', features=True)
oct_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_oct_2015.npy', targets=True)

nov_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_nov_2015.npy', features=True)
nov_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_nov_2015.npy', targets=True)

dec_features = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_dec_2015.npy', features=True)
dec_targets = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_dec_2015.npy', targets=True)

z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']

new_oct_features = oct_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_oct_features[z] = oct_features[z]
    
new_nov_features = nov_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_nov_features[z] = nov_features[z]
    
new_dec_features = dec_features.loc[:, 'datetime':'z1']
for z in z_list:
    new_dec_features[z] = dec_features[z]
    
comb_features = pd.concat([new_oct_features, new_nov_features, new_dec_features], ignore_index=True)
comb_targets = pd.concat([oct_targets, nov_targets, dec_targets], ignore_index=True)

train_month_start = [10,10,10,10,11,11,11,11]
train_day_start = [5,12,19,26,2,9,16,23]
train_month_end = [10,11,11,11,11,12,12,12]
train_day_end = [30,6,13,20,27,4,11,18]
test_month_start = [11,11,11,11,11,12,12,12]
test_day_start = [2,9,16,23,30,7,14,21]
test_month_end = [11,11,11,11,12,12,12,12]
test_day_end = [6,13,20,27,4,11,18,25]


for tms, tds, tme, tde, testms, testds, testme, testde in zip(train_month_start, train_day_start, train_month_end, train_day_end, test_month_start, test_day_start, test_month_end, test_day_end):

    train_features, train_targets, test_features, test_targets = train_and_test(comb_features, comb_targets, tod='early', year=2015, 
                   train_month_start=tms, train_day_start=tds,
                   train_month_end=tme, train_day_end=tde,
                   test_month_start=testms, test_day_start=testds,
                   test_month_end=testme, test_day_end=testde)

    train_features_all, train_features_top_5, train_features_top_10, test_features_all, test_features_top_5, test_features_top_10, df = selection(train_features, train_targets, test_features, test_targets)

    print('Train start: {}/{}/2015'.format(tms, tds))
    print('Train end: {}/{}/2015'.format(tme, tde))
    print('Test start: {}/{}/2015'.format(testms, testds))
    print('Test end: {}/{}/2015'.format(testme, testde))
    print('')
    performance(train_features_all, train_features_top_5, train_features_top_10, train_features_top_25, train_features_top_50, train_targets,
               test_features_all, test_features_top_5, test_features_top_10, test_features_top_25, test_features_top_50, test_targets)
    print('')