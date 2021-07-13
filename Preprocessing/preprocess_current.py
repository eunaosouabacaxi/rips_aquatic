import pandas as pd
import numpy as np
import datetime

# 'efficiently' reads csv into dataframe
def read_csv(file):
    TextFileReader = pd.read_csv(file, chunksize=500000)
    dfList = []
    for df in TextFileReader:
        dfList.append(df)
        
    df = pd.concat(dfList)
    
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['datetime'] = df['datetime'].dt.tz_convert(tz='US/Eastern')
    
    return df


# read npy into dataframe with column names    
def read_npy1(file, features=False, targets=False, combined=False):
    np_arr = np.load(file, allow_pickle=True)
    
    feature_cols = ['datetime',
                    'squ',
                    'weights',
                    'x1',
                    'x2',
                    'x3',
                    'x4',
                    'x5',
                    'x6',
                    'x7',
                    'x8',
                    'x9',
                    'x10',
                    'x11',
                    'x12',
                    'x13',
                    'x14',
                    'x15',
                    'x16',
                    'x17',
                    'x18',
                    'x19',
                    'x20',
                    'x21',
                    'x22',
                    'x23',
                    'x24',
                    'x25',
                    'x26',
                    'x27',
                    'x28',
                    'x29',
                    'x30',
                    'x31',
                    'x32',
                    'x33',
                    'x34',
                    'z1',
                    'z1 x z13',
                    'z1 x z14',
                    'z1 x z15',
                    'z2',
                    'z2 x z13',
                    'z2 x z14',
                    'z2 x z15',
                    'z3',
                    'z3 x z13',
                    'z3 x z14',
                    'z3 x z15',
                    'z4',
                    'z4 x z13',
                    'z4 x z14',
                    'z4 x z15',
                    'z5',
                    'z5 x z13',
                    'z5 x z14',
                    'z5 x z15',
                    'z6',
                    'z6 x z13',
                    'z6 x z14',
                    'z6 x z15',
                    'z7',
                    'z7 x z13',
                    'z7 x z14',
                    'z7 x z15',
                    'z8',
                    'z8 x z13',
                    'z8 x z14',
                    'z8 x z15',
                    'z9',
                    'z9 x z13',
                    'z9 x z14',
                    'z9 x z15',
                    'z10',
                    'z10 x z13',
                    'z10 x z14',
                    'z10 x z15',
                    'z11',
                    'z11 x z13',
                    'z11 x z14',
                    'z11 x z15',
                    'z12',
                    'z12 x z13',
                    'z12 x z14',
                    'z12 x z15',
                    'z13',
                    'z14',
                    'z15']
    
    target_cols = ['datetime',
                   'squ',
                   'y1',
                   'y2',
                   'y3',
                   'y4',
                   'y5',
                   'y6']
    
    combined_cols = ['Unnamed: 0', 
                     'Unnamed: 1', 
                     'datetime', 
                     'squ', 
                     'weights',
                     'x1',
                     'x2',
                     'x3',
                     'x4',
                     'x5',
                     'x6',
                     'x7',
                     'x8',
                     'x9',
                     'x10',
                     'x11',
                     'x12',
                     'x13',
                     'x14',
                     'x15',
                     'x16',
                     'x17',
                     'x18',
                     'x19',
                     'x20',
                     'x21',
                     'x22',
                     'x23',
                     'x24',
                     'x25',
                     'x26',
                     'x27',
                     'x28',
                     'x29',
                     'x30',
                     'x31',
                     'x32',
                     'x33',
                     'x34',
                     'z1',
                     'z1 x z13',
                     'z1 x z14',
                     'z1 x z15',
                     'z2',
                     'z2 x z13',
                     'z2 x z14',
                     'z2 x z15',
                     'z3',
                     'z3 x z13',
                     'z3 x z14',
                     'z3 x z15',
                     'z4',
                     'z4 x z13',
                     'z4 x z14',
                     'z4 x z15',
                     'z5',
                     'z5 x z13',
                     'z5 x z14',
                     'z5 x z15',
                     'z6',
                     'z6 x z13',
                     'z6 x z14',
                     'z6 x z15',
                     'z7',
                     'z7 x z13',
                     'z7 x z14',
                     'z7 x z15',
                     'z8',
                     'z8 x z13',
                     'z8 x z14',
                     'z8 x z15',
                     'z9',
                     'z9 x z13',
                     'z9 x z14',
                     'z9 x z15',
                     'z10',
                     'z10 x z13',
                     'z10 x z14',
                     'z10 x z15',
                     'z11',
                     'z11 x z13',
                     'z11 x z14',
                     'z11 x z15',
                     'z12',
                     'z12 x z13',
                     'z12 x z14',
                     'z12 x z15',
                     'z13',
                     'z14',
                     'z15',
                     'y1',
                     'y2',
                     'y3',
                     'y4',
                     'y5',
                     'y6']
    
    if features:
        df = pd.DataFrame(np_arr, columns = feature_cols)
    elif targets:
        df = pd.DataFrame(np_arr, columns = target_cols)
    elif combined:
        df = pd.DataFrame(np_arr, columns = combined_cols)
        df = df.drop(columns = ['Unnamed: 0', 'Unnamed: 1'])
        
    
    return df


# read npy into dataframe with column names    
def read_npy2(file, features=False, targets=False, combined=False):
    np_arr = np.load(file, allow_pickle=True)
    
    feature_cols = ['datetime',
                    'squ',
                    'regression_weights',
                    'ewma_intraday_return_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'ewma_intraday_return_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'ewma_intraday_return_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless)_predictor',
                    'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(intraday_extreme_position)_predictor',
                    'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl120)_predictor',
                    'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl30)_predictor',
                    'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl5)_predictor',
                    'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0)_predictor',
                    'r_BOD_BOD_+120_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_BOD_BOD_+30_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_BOD_BOD_+5_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_BOD_BOD_+60_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_intraday_E4S_degree1_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_intraday_E4S_degree1_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_intraday_E4S_degree2_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_intraday_E4S_degree2_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_intraday_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_intraday_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_intraday_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_overnight_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_historical_overnight_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_volume_weighted_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                    'r_volume_weighted_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'r_volume_weighted_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                    'rev_arrange_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                    'round_price_support_cents1000_predictor',
                    'round_price_support_cents100_predictor',
                    'round_price_support_cents500_predictor',
                    'round_price_support_cents50_predictor',
                    'rsi_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                    'rsi_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                    'rsi_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                    'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor',
                    'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree1_interactor',
                    'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree2_interactor',
                    'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x volume_curve_hist20_interactor',
                    'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor',
                    'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree1_interactor',
                    'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree2_interactor',
                    'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x volume_curve_hist20_interactor',
                    'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'intraday_extreme_position_sq_interactor',
                    'intraday_extreme_position_sq_interactor x tod_kernel_degree1_interactor',
                    'intraday_extreme_position_sq_interactor x tod_kernel_degree2_interactor',
                    'intraday_extreme_position_sq_interactor x volume_curve_hist20_interactor',
                    'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor',
                    'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x tod_kernel_degree1_interactor',
                    'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x tod_kernel_degree2_interactor',
                    'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x volume_curve_hist20_interactor',
                    'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                    'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                    'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                    'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                    'tod_kernel_degree1_interactor',
                    'tod_kernel_degree2_interactor',
                    'volume_curve_hist20_interactor']
    
    target_cols = ['datetime',
                   'squ',
                   'fwd_r_10_30_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                   'fwd_r_1_10_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                   'fwd_r_E4S_days1_embargo_intraday_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                   'fwd_r_E4S_days1_embargo_overnight_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                   'fwd_r_ToClose_E4S_from30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                   'fwd_smoothed_HL1day_window2_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_skip1']
    
    combined_cols = ['Unnamed: 0', 
                     'Unnamed: 1', 
                     'datetime', 
                     'squ', 
                     'regression_weights',
       'ewma_intraday_return_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'ewma_intraday_return_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'ewma_intraday_return_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless)_predictor',
       'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(intraday_extreme_position)_predictor',
       'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl120)_predictor',
       'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl30)_predictor',
       'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl5)_predictor',
       'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0)_predictor',
       'r_BOD_BOD_+120_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_BOD_BOD_+30_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_BOD_BOD_+5_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_BOD_BOD_+60_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_intraday_E4S_degree1_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_intraday_E4S_degree1_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_intraday_E4S_degree2_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_intraday_E4S_degree2_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_intraday_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_intraday_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_intraday_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_overnight_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_historical_overnight_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_volume_weighted_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
       'r_volume_weighted_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'r_volume_weighted_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
       'rev_arrange_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
       'round_price_support_cents1000_predictor',
       'round_price_support_cents100_predictor',
       'round_price_support_cents500_predictor',
       'round_price_support_cents50_predictor',
       'rsi_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
       'rsi_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
       'rsi_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
       'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor',
       'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree1_interactor',
       'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree2_interactor',
       'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x volume_curve_hist20_interactor',
       'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor',
       'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree1_interactor',
       'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree2_interactor',
       'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x volume_curve_hist20_interactor',
       'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'intraday_extreme_position_sq_interactor',
       'intraday_extreme_position_sq_interactor x tod_kernel_degree1_interactor',
       'intraday_extreme_position_sq_interactor x tod_kernel_degree2_interactor',
       'intraday_extreme_position_sq_interactor x volume_curve_hist20_interactor',
       'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor',
       'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x tod_kernel_degree1_interactor',
       'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x tod_kernel_degree2_interactor',
       'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x volume_curve_hist20_interactor',
       'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
       'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
       'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
       'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
       'tod_kernel_degree1_interactor', 'tod_kernel_degree2_interactor',
       'volume_curve_hist20_interactor',
       'fwd_r_10_30_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
       'fwd_r_1_10_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
       'fwd_r_E4S_days1_embargo_intraday_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
       'fwd_r_E4S_days1_embargo_overnight_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
       'fwd_r_ToClose_E4S_from30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
       'fwd_smoothed_HL1day_window2_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_skip1']
    
    if features:
        df = pd.DataFrame(np_arr, columns = feature_cols)
    elif targets:
        df = pd.DataFrame(np_arr, columns = target_cols)
    elif combined:
        df = pd.DataFrame(np_arr, columns = combined_cols)
        df = df.drop(columns = ['Unnamed: 0', 'Unnamed: 1'])
        
    
    return df

# combines the dataframes into one large dataframe
# with the column names left intact and drops
# the rows with NaN's in 'regressions_weights'
def combine1(df_weights, list_df):
    
    predictors_dict = {0: 'Unnamed: 0',
                   1: 'datetime',
                   2: 'squ',
                   3: 'ewma_intraday_return_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   4: 'ewma_intraday_return_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   5: 'ewma_intraday_return_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   6: 'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless)_predictor',
                   7: 'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(intraday_extreme_position)_predictor',
                   8: 'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl120)_predictor',
                   9: 'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl30)_predictor',
                   10: 'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(mid_vwap_hl5)_predictor',
                   11: 'ortho_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0(return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0)_predictor',
                   12: 'r_BOD_BOD_+120_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   13: 'r_BOD_BOD_+30_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   14: 'r_BOD_BOD_+5_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   15: 'r_BOD_BOD_+60_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   16: 'r_historical_intraday_E4S_degree1_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   17: 'r_historical_intraday_E4S_degree1_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   18: 'r_historical_intraday_E4S_degree2_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   19: 'r_historical_intraday_E4S_degree2_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   20: 'r_historical_intraday_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   21: 'r_historical_intraday_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   22: 'r_historical_intraday_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   23: 'r_historical_overnight_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   24: 'r_historical_overnight_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   25: 'r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   26: 'r_volume_weighted_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                   27: 'r_volume_weighted_E4S_hl_days1_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   28: 'r_volume_weighted_E4S_hl_days5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_predictor',
                   29: 'rev_arrange_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                   30: 'round_price_support_cents1000_predictor',
                   31: 'round_price_support_cents100_predictor',
                   32: 'round_price_support_cents500_predictor',
                   33: 'round_price_support_cents50_predictor',
                   34: 'rsi_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                   35: 'rsi_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor',
                   36: 'rsi_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_predictor'
                  }
    
    interactors_dict = {0: 'Unnamed: 0',
                   1: 'datetime',
                   2: 'squ',
                   3: 'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   4: 'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   5: 'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   6: 'abnormal_volatility_E4S_aggw5_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   7: 'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor',
                   8: 'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree1_interactor',
                   9: 'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree2_interactor',
                   10: 'abnormal_volatility_expanding_ema_E4S_aggw5_bod30_hist20_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x volume_curve_hist20_interactor',
                   11: 'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   12: 'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   13: 'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   14: 'abnormal_volume_daysback1_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   15: 'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor',
                   16: 'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree1_interactor',
                   17: 'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x tod_kernel_degree2_interactor',
                   18: 'abnormal_volume_expanding_ema_hist20_hl120_pd30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_sharkless_interactor x volume_curve_hist20_interactor',
                   19: 'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   20: 'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   21: 'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   22: 'bollinger_band_width_E4S_hl120_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   23: 'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   24: 'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   25: 'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   26: 'bollinger_band_width_E4S_hl30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   27: 'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   28: 'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   29: 'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   30: 'bollinger_band_width_E4S_hl5_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   31: 'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   32: 'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   33: 'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   34: 'durbin_watson_30m_E4S_hist20_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   35: 'intraday_extreme_position_sq_interactor',
                   36: 'intraday_extreme_position_sq_interactor x tod_kernel_degree1_interactor',
                   37: 'intraday_extreme_position_sq_interactor x tod_kernel_degree2_interactor',
                   38: 'intraday_extreme_position_sq_interactor x volume_curve_hist20_interactor',
                   39: 'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   40: 'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   41: 'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   42: 'return_skewness_E4S_hist20_min10_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   43: 'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor',
                   44: 'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x tod_kernel_degree1_interactor',
                   45: 'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x tod_kernel_degree2_interactor',
                   46: 'specvol_E4S(abs(r_overnight_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0))_interactor x volume_curve_hist20_interactor',
                   47: 'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor',
                   48: 'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree1_interactor',
                   49: 'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x tod_kernel_degree2_interactor',
                   50: 'sqrt_regression_weights_z_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_interactor x volume_curve_hist20_interactor',
                   51: 'tod_kernel_degree1_interactor',
                   52: 'tod_kernel_degree2_interactor',
                   53: 'volume_curve_hist20_interactor'
                  }

    targets_dict = {0: 'Unnamed: 0',
                1: 'datetime',
                2: 'squ',
                3: 'fwd_r_10_30_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                4: 'fwd_r_1_10_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                5: 'fwd_r_E4S_days1_embargo_intraday_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                6: 'fwd_r_E4S_days1_embargo_overnight_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                7: 'fwd_r_ToClose_E4S_from30_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0',
                8: 'fwd_smoothed_HL1day_window2_E4S_reg_wts_modelINVERSE_IMPACT_USE4S_ADVFIX_V0_skip1'
               }
    
    comb_df = df_weights
    list_dicts = [predictors_dict, interactors_dict, targets_dict]

    for df, dictionary in zip(list_df, list_dicts):
        for i in range(3, len(df.iloc[0,:])):
            comb_df[dictionary[i]] = df.iloc[:,i]
            
    comb_df.dropna(subset=['regression_weights'])
    
    return comb_df


# combines the dataframes by using the merge function
# and drops the rows with NaN's in 'regressions_weights'
def combine2(weights_df, list_df):
    comb_df = weights_df
    for df in list_df:
        comb_df = comb_df.merge(df, on=['Unnamed: 0', 'datetime', 'squ'])
        
    comb_df = comb_df.dropna(subset=['regression_weights'])
    
    return comb_df
 

# splits the data by time of day
def split_by_time(df):

    early_df = df.loc[df['datetime'].dt.time <= datetime.time(11, 40)]
    mid_df = df.loc[(df['datetime'].dt.time > datetime.time(11, 40)) & (df['datetime'].dt.time <= datetime.time(13, 45))]
    late_df = df.loc[df['datetime'].dt.time > datetime.time(13, 45)]
    
    return early_df, mid_df, late_df


# splits the data by month
def split_by_month(df, year, month):

    if (month >= 1) & (month < 12):
        month_df = df.loc[(df['datetime'].dt.date >= datetime.date(year, month, 1)) & (df['datetime'].dt.date < datetime.date(year, month+1, 1))]
    elif month == 12:
        month_df = df.loc[df['datetime'].dt.date >= datetime.date(year, month, 1)]
    
    return month_df
    
    
# center the data around 0 and fill the NaN's with 0
def normalize_and_fill(df):

    normalized_df = (df-df.mean())/df.std()
    normalized_df = normalized_df.fillna(0)
    
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