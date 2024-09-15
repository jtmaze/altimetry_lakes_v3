# %% Libraries and directories

import os
import glob
import pandas as pd
import geopandas as gpd

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

is2_lakes = gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')
# The roi_name columnn doesn't sub-dived MRD, TUK and Anderson
is2_lakes.drop(columns=['roi_name'], inplace=True)
is2_timeseries_paths = glob.glob('./data/lake_timeseries/*_timeseries.csv')
is2_timeseries = [pd.read_csv(p) for p in is2_timeseries_paths]
is2_timeseries = pd.concat(is2_timeseries)
is2_timeseries.drop(columns=['Unnamed: 0', 'geometry'], inplace=True)

# %% Generate list of lakes with observations high enough for regression analysis

high_obs_lakes = is2_lakes[is2_lakes['obs_dates_'] >= 10]
high_obs_ids = high_obs_lakes['lake_id']
print(len(high_obs_ids))

high_obs_timeseries = is2_timeseries[is2_timeseries['lake_id'].isin(high_obs_ids)].copy()
high_obs_timeseries['obs_datetime'] = pd.to_datetime(high_obs_timeseries['obs_date'])
high_obs_timeseries['obs_month'] = high_obs_timeseries.obs_datetime.dt.month.astype(str)

months_check = high_obs_timeseries.groupby(by = 'lake_id')['obs_month'].agg(
        june_obs=lambda x: '6' in x.unique(),
        sep_obs=lambda x: '8' in x.unique(),
).reset_index()

aug_sep_months_check = months_check[
        (months_check['june_obs'] == True) & 
        (months_check['sep_obs'] == True)
]

print(len(aug_sep_months_check))

regression_lakes = is2_lakes[is2_lakes['lake_id'].isin(aug_sep_months_check['lake_id'])].copy()
regression_lakes.to_file('./data/lake_summaries/regression_lakes.shp')

# %%
