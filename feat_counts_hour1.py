import pandas as pd
import preprocess as pp
import datetime
import numpy as np
import eda

features_jan_2015 = pp.read_npy1('/u/project/cratsch/tescala/features_jan_2015.npy', features=True)
targets_jan_2015 = pp.read_npy1('/u/project/cratsch/tescala/targets_jan_2015.npy', targets=True)

new_features_jan_2015 = features_jan_2015.loc[:, 'datetime':'z1']
z_list = ['z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15']
for z in z_list:
    new_features_jan_2015[z] = features_jan_2015[z]


feat_counts = eda.count_top_feats(new_features_jan_2015, targets_jan_2015, 2015, 1, 6, hour=9, minute=35, period=1)

print('First hour')
print(feat_counts)


feat_counts = eda.count_top_feats(new_features_jan_2015, targets_jan_2015, 2015, 1, 6, hour=10, minute=40, period=1)

print('Second hour')
print(feat_counts)


feat_counts = eda.count_top_feats(new_features_jan_2015, targets_jan_2015, 2015, 1, 6, hour=11, minute=45, period=1)

print('Third hour')
print(feat_counts)