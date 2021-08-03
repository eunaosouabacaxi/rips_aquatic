import pandas as pd
import preprocess as pp
import datetime
import numpy as np
import eda
import sklearn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# load in 3 months of data jan feb mar CHANGE FILE NAMES
features_jan = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_jan_2015.npy', features=True)
targets_jan = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_jan_2015.npy', targets=True)

features_feb = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_feb_2015.npy', features=True)
targets_feb = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_feb_2015.npy', targets=True)

features_mar = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/features_mar_2015.npy', features=True)
targets_mar = pp.read_npy1('/u/project/cratsch/tescala/month_split_right/targets_mar_2015.npy', targets=True)

features_jan_2015 = features_jan.loc[:, 'datetime':'z1']
z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']
for z in z_list:
    features_jan_2015[z] = features_jan[z]
    
features_feb_2015 = features_feb.loc[:, 'datetime':'z1']
z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']
for z in z_list:
    features_feb_2015[z] = features_feb[z]
    
features_mar_2015 = features_mar.loc[:, 'datetime':'z1']
z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']
for z in z_list:
    features_mar_2015[z] = features_mar[z]

# center the data around 0 and fill the NaN's with 0 EDIT THIS
def normalize_and_fill(df):
    
    m = df.loc[:,'x1':'z12'].mean()
    std = df.loc[:,'x1':'z12'].std()
    df.loc[:,'x1':'z12'] = (df.loc[:,'x1':'z12']-m) / std

    normalized_df = df.fillna(0)
    
    return normalized_df

# drop the rows with NaN's in the target
# splits the combined dataframe back into features 
# and targets with the datetime and squ and applies 
# normalize_and_fill to the features
def features_and_targets(df):

    df = df.dropna(subset=['fwd_r_1_10_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0'])
    index_df = df.loc[:, 'datetime':'squ']
    features_df = normalize_and_fill(df.loc[:, 'regression_weights':'volume_curve_hist20_interactor'])
    targets_df = df.loc[:, 'fwd_r_10_30_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0':'fwd_smoothed_HL1day_window2_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_skip1']
    
    features_df = index_df.join(features_df)
    targets_df = index_df.join(targets_df)
    
    return features_df, targets_df


# need to load in the data
def hours(feature_df, target_df, year, month, day, hour, period):
    new_features = feature_df.loc[:, 'datetime':'z1']
    z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']
    for z in z_list:
        new_features[z] = feature_df[z]

    features_hour = new_features.loc[(new_features['datetime'].dt.time >= datetime.time(hour,35))
                                              & (new_features['datetime'].dt.time < datetime.time(hour+1,40))
                                              & (new_features['datetime'].dt.date >= datetime.date(year,month,day))
                                              & (new_features['datetime'].dt.date < datetime.date(year,month,day+period))]

    targets_hour = target_df.loc[(target_df['datetime'].dt.time >= datetime.time(hour,35))
                                        & (target_df['datetime'].dt.time < datetime.time(hour+1,40))
                                        & (target_df['datetime'].dt.date >= datetime.date(year,month,day))
                                        & (target_df['datetime'].dt.date < datetime.date(year,month,day+period))]
    return new_features, features_hour, targets_hour


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

    results = sm.WLS(y_, x__, weights=feature_df['weights']).fit()
    a = results.wald_test(np.eye(len(results.params))[2:5])
    f0 = a.fvalue
    p0 = a.pvalue
    
    x_vars = results.summary2().tables[1].index
    wald_str = ' = '.join(list(x_vars[2:]))
    equal_coeffs = results.wald_test(wald_str)
    f_equal = equal_coeffs.fvalue
    p_equal = equal_coeffs.pvalue
    
    # new wald for linearity 
    string = 'x4-x3 = x3-x2 = x2'
    lin = results.wald_test(string)
    f_lin = lin.fvalue
    p_lin = lin.pvalue
    
    return f0, p0, f_equal, p_equal, f_lin, p_lin


def wald_0(new_features_df, features_hour, targets_hour):
    z_list = new_features_df.loc[:,'z1':'z12'].columns
    x_list = features_hour.loc[:,'x1':'x34'].columns

    f_list = []
    p_list = []
    for z in z_list:
        flist = []
        plist = []
        for feat in x_list:
            f0, p0, f_equal, p_equal,f_lin, p_lin = ols_results_wald(features_hour, targets_hour, x=feat, z=z)
            flist.append(f0)
            plist.append(p0)

        f_list.append(flist)
        p_list.append(plist)
    
    f_dict = {'z1':f_list[0], 'z2':f_list[1], 'z3':f_list[2], 'z4':f_list[3],
              'z5':f_list[4], 'z6':f_list[5], 'z7':f_list[6], 'z8':f_list[7],
              'z9':f_list[8], 'z10':f_list[9], 'z11':f_list[10], 'z12':f_list[11]}

    f_df = pd.DataFrame.from_dict(f_dict, orient='index',
                           columns=x_list)

    p_dict = {'z1':p_list[0], 'z2':p_list[1], 'z3':p_list[2], 'z4':p_list[3],
              'z5':p_list[4], 'z6':p_list[5], 'z7':p_list[6], 'z8':p_list[7],
              'z9':p_list[8], 'z10':p_list[9], 'z11':p_list[10], 'z12':p_list[11]}

    p_df = pd.DataFrame.from_dict(p_dict, orient='index',
                           columns=x_list)
    
     
    p_list = []
    for col in p_df.columns:
        for idx in p_df.index:
            val = p_df.loc[idx, col]
            p_list.append(((col, idx), val))
    
    # ranked pairs with pvals
    ranked_p = eda.sort_scores1(p_list)
   
    return p_df, ranked_p[:50]


def wald_lin(new_features_df, features_hour, targets_hour):
    z_list = new_features_df.loc[:,'z1':'z12'].columns
    x_list = features_hour.loc[:,'x1':'x34'].columns

    f_list = []
    p_list = []
    for z in z_list:
        flist = []
        plist = []
        for feat in x_list:
            f0, p0, f_equal, p_equal,f_lin, p_lin = ols_results_wald(features_hour, targets_hour, x=feat, z=z)
            flist.append(f_lin)
            plist.append(p_lin)

        f_list.append(flist)
        p_list.append(plist)
    
    f_dict = {'z1':f_list[0], 'z2':f_list[1], 'z3':f_list[2], 'z4':f_list[3],
              'z5':f_list[4], 'z6':f_list[5], 'z7':f_list[6], 'z8':f_list[7],
              'z9':f_list[8], 'z10':f_list[9], 'z11':f_list[10], 'z12':f_list[11]}

    f_df = pd.DataFrame.from_dict(f_dict, orient='index',
                           columns=x_list)

    p_dict = {'z1':p_list[0], 'z2':p_list[1], 'z3':p_list[2], 'z4':p_list[3],
              'z5':p_list[4], 'z6':p_list[5], 'z7':p_list[6], 'z8':p_list[7],
              'z9':p_list[8], 'z10':p_list[9], 'z11':p_list[10], 'z12':p_list[11]}

    p_df = pd.DataFrame.from_dict(p_dict, orient='index',
                           columns=x_list)
    
     
    p_list = []
    for col in p_df.columns:
        for idx in p_df.index:
            val = p_df.loc[idx, col]
            p_list.append(((col, idx), val))
    
    # ranked pairs with pvals
    ranked_p = eda.sort_scores1(p_list)
            
    return p_df, ranked_p[:50]


# coeffs = 0 

jan = hours(features_jan_2015, targets_jan_2015, 2015, 1, 1, 9, 30)
new_features_jan = jan[0]
features_hour_jan = jan[1]
targets_hour_jan = jan[2]
jan_0 = wald_0(new_features_jan, features_hour_jan, targets_hour_jan)
ranked_jan0 = jan_0[1]
print('coeffs = 0 jan')
print(ranked_jan0)

feb = hours(features_feb_2015, targets_feb_2015, 2015, 2, 2, 9, 26)
new_features_feb = feb[0]
features_hour_feb = feb[1]
targets_hour_feb = feb[2]
feb_0 = wald_0(new_features_feb, features_hour_feb, targets_hour_feb)
ranked_feb0 = feb_0[1]
print('coeffs = 0 Feb')
print(ranked_feb0)

mar = hours(features_mar_2015, targets_mar_2015, 2015, 3, 2, 9, 26)
new_features_mar = mar[0]
features_hour_mar = mar[1]
targets_hour_mar = mar[2]
mar_0 = wald_0(new_features_mar, features_hour_mar, targets_hour_mar)
ranked_mar0 = mar_0[1]
print('coeffs = 0 mar')
print(ranked_mar0)

# linearity

jan =hours(features_jan_2015, targets_jan_2015, 2015, 1, 1, 9, 30)
new_features_jan = jan[0]
features_hour_jan = jan[1]
targets_hour_jan = jan[2]
jan_lin = wald_lin(new_features_jan, features_hour_jan, targets_hour_jan)
ranked_janlin = jan_lin[1]
print('coef lin jan')
print(ranked_janlin)

feb = hours(features_feb_2015, targets_feb_2015, 2015, 2, 2, 9, 26)
new_features_feb = feb[0]
features_hour_feb = feb[1]
targets_hour_feb = feb[2]
feb_lin= wald_lin(new_features_feb, features_hour_feb, targets_hour_feb)
ranked_feblin = feb_lin[1]
print('coeffs lin Feb')
print(ranked_feblin)

mar = hours(features_mar_2015, targets_mar_2015, 2015, 3, 2, 9, 26)
new_features_mar = mar[0]
features_hour_mar = mar[1]
targets_hour_mar = mar[2]
mar_lin = wald_lin(new_features_mar, features_hour_mar, targets_hour_mar)
ranked_marlin = mar_lin[1]
print('coef lin mar')
print(ranked_marlin)
