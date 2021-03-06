import pandas as pd

# 'efficiently' reads csv into dataframe
def read(file):
    TextFileReader = pd.read_csv(file, chunksize=500000)
    dfList = []
    for df in TextFileReader:
        dfList.append(df)
        
    df = pd.concat(dfList)
    return df

# combines the dataframes into one large dataframe
# with the column names left intact and drops
# the rows with NaN's in 'regressions weights'
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

def combine2(weights_df, list_df):
    comb_df = weights_df
    for df in list_df:
        comb_df.merge(df, on=['Unnamed: 0', 'datetime', 'squ'])
    
    comb_df.dropna(subset=['regression_weights'])
    
    return comb_df
