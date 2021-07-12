# this program should be in the same dir as Tomas' preprocess.py
import csv
import os
import pandas as pd
import preprocess as pp

input_csv = 'aaa.csv'
output_dir = 'result'
output_prefix = 'splitted_aaa'

# you don't have to make a new dir every time you split, putting them in one folder will allow the combine function to access it easily
os.makedirs(output_dir, exist_ok=True)

max_batch_size = 2
batch_id = 0

def get_csv_out(batch_id, output_prefix, output_dir):
    # Get file write object for the output csv
    output_csv = '{output_prefix}_{batch_id}.csv'.format(output_prefix=output_prefix, batch_id=batch_id)
    csv_out = open(os.path.join(output_dir, output_csv), 'w')
    return csv_out
    

with open(input_csv, 'r') as csv_in:
    csv_reader = csv.reader(csv_in, delimiter=',')
    header = next(csv_reader)
    print('header: {}'.format(header))
    cur_batch_size = 0
    csv_out = get_csv_out(batch_id, output_prefix, output_dir)
    csv_writer = csv.writer(csv_out, delimiter=',')
    print("Start writing file number: {}".format(batch_id))
    csv_writer.writerow(header)
    for row in csv_reader:
        if cur_batch_size >= max_batch_size:
            # The batch is full, create new batch (new file)
            csv_out.close()
            batch_id += 1
            csv_out = get_csv_out(batch_id, output_prefix, output_dir)
            csv_writer = csv.writer(csv_out, delimiter=',')
            print("Start writing file number: {}".format(batch_id))
            csv_writer.writerow(header)
            cur_batch_size = 0
        csv_writer.writerow(row)
        cur_batch_size += 1
    csv_out.close()

# only do the following when you have separated all four variables
for i in range(50):
    f_interactors = 'combined_2012/gen1.5_interactors_' + str(i) + '.csv'
    f_predictors = 'combined_2012/gen1.5_predictors_' + str(i) + '.csv'
    f_weights = 'combined_2012/gen1.5_weights_' + str(i) + '.csv'
    f_targets = 'combined_2012/gen1.5_targets_' + str(i) + '.csv'
    interactors_df = pp.read(f_interactors)
    predictors_df = pp.read(f_predictors)
    weights_df = pp.read(f_weights)
    targets_df = pp.read(f_targets)
    list_df = [predictors_df, interactors_df, targets_df]
    comb_df = pp.combine2(weights_df, list_df)
    out_comb = 'combined_2012_' + str(i)
    comb_df.to_csv(out_comb)
# this will save your csv's to the scratch folder, not the combined folder

# center the data around 0 and fill the NaN's with 0
def normalize_and_fill(df):

    normalized_df = (df-df.mean())/df.std()
    normalized_df = normalized_df.fillna(0)
    
    return normalized_df


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

#test_read = pp.read('combined_2012_0')
for i in range(51):
    fin = 'combined_2012_' + str(i)
    df = pp.read(fin)
    processed_features_df, processed_targets_df = features_and_targets(df)
    processed_df = processed_features_df.merge(processed_targets_df, on = ['datetime', 'squ'])
    fout = 'processed_2012_' + str(i) + '.csv'
    processed_df.to_csv(fout)

    
