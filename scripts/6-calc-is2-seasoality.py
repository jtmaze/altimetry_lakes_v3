# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

is2_path = './data/lake_timeseries/*_timeseries.csv'
is2_files = glob.glob(is2_path)
is2_dfs = [pd.read_csv(file) for file in is2_files]
is2_data = pd.concat(is2_dfs)

# %% 2. Calculate ICESat-2 seasonality

is2_data['obs_datetime'] = pd.to_datetime(is2_data['obs_date'])
is2_data['obs_month'] = is2_data['obs_datetime'].dt.month.astype(str)
is2_data = is2_data[(is2_data['obs_month'] == '6') | (is2_data['obs_month'] == '8')]

clean_lakes_path = './data/clean_ids.csv'
clean_ids = pd.read_csv(clean_lakes_path)
clean_ids = clean_ids['lake_id']
is2_data = is2_data[is2_data['lake_id'].isin(clean_ids)]
is2_data_clean = is2_data[(is2_data['zdif_date'] > -2.5) & (is2_data['zdif_date'] < 2.5)]
print(len(is2_data['lake_id'].unique()), len(is2_data_clean['lake_id'].unique()))

is2_lake_seasonality = is2_data_clean.groupby(
    by=['lake_id', 'obs_month']
    ).agg(
        mean_zdif = ('zdif_date', 'mean'),
        std_zdif = ('zdif_date', 'std'),
        roi = ('roi_name', 'first')
        ).reset_index()

is2_roi_seasonality = is2_lake_seasonality.groupby(
    by=['roi', 'obs_month']
    ).agg(
        mean_zdif = ('mean_zdif', 'mean'),
    ).reset_index()

