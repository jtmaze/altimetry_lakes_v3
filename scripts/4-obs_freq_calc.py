#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:36:55 2024

@author: jmaze
"""

# %% 1. Libraries and directories

import glob
from datetime import datetime as dt
import geopandas as gpd
import pandas as pd
#import matplotlib.pyplot as plt
import re
import os

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3/')

# Input paths
matched_pts_path = './data/matchedIS2_pts/'
clipped_lakes_path = './data/pld_clipped/'

# Output paths
lake_summaries_path = './data/lake_summaries/'
lake_ts_path = './data/lake_timeseries/'


# %% 2. List of rois to process

files_list = glob.glob(f'{matched_pts_path}/*.csv')

pattern = r'.*matchedIS2_pts/(.*?)_matched\.csv'
rois_list = []

for file in files_list:
    match = re.search(pattern, file)
    if match:
        rois_list.append(match.group(1))

# Clean up variables
del(match, file, pattern)

# %% 3. Make timeseries and lake summaries

matching_stats = []
obs5_lakes_gdfs = []

for roi_name in rois_list: 
    
    # !!! 3.1 Read IS2 points and lake files
    file = f'{matched_pts_path}/{roi_name}_matched.csv'
    pts_df = pd.read_csv(file)
    lakes_file = f'{clipped_lakes_path}/{roi_name}_pld_clipped.shp'
    clipped_lakes = gpd.read_file(lakes_file)
    clipped_lakes['perim_m'] = clipped_lakes['perim_m2']
    clipped_lakes = clipped_lakes.drop(columns=['perim_m2'])
    
    pts_df['obs_datetime'] = pd.to_datetime(pts_df['obs_date'])
    pts_df['obs_month'] = pts_df['obs_datetime'].dt.month
    pts_df = pts_df[(pts_df['obs_month'] >= 6) & (pts_df['obs_month'] <= 9)]
    
    # !!! 3.2 Generate a timeseries for each lake
    
    # Calculate median z by lake
    quick_summary = pts_df.groupby('lake_id', as_index=False).agg(
        zmed_all_segs=('h_li', 'median'),
        zstd_all_segs=('h_li', 'std')
    )
    
    # Generate stats for each lake on a unique date
    lake_ts = pts_df.groupby(['lake_id', 'obs_date'], as_index=False).agg(
        zmed_date=('h_li', 'median'),
        seg_cnt_date=('h_li', 'count'),
        z10_date=('h_li', lambda x: x.quantile(0.1)),
        z90_date=('h_li', lambda x: x.quantile(0.9)), 
        lat_mean_date=('latitude', 'mean'),
        lon_mean_date=('longitude', 'mean')
    )
    
    # Calculate date's difference from lake's median elevation
    lake_timeseries = pd.merge(lake_ts, quick_summary, on='lake_id')
    lake_timeseries['zdif_date'] = lake_timeseries['zmed_date'] - lake_timeseries['zmed_all_segs']
    
    #del(quick_summary, lake_ts)
    
    # !!! 3.3 Generate summary stats for each lake. 
    
    lake_summary=pts_df.groupby('lake_id', as_index=True).agg(
        zmed_all_segs=('h_li', 'median'),
        zstd_all_segs=('h_li', 'std'),
        z10_all_segs=('h_li', lambda x: x.quantile(0.1)),
        z90_all_segs=('h_li', lambda x: x.quantile(0.9)), 
        zrange_all_segs=('h_li', lambda x: x.max() - x.min()),
        lat_mean_all=('latitude', 'mean'),
        lon_mean_all=('longitude', 'mean'),
        obs_dates_list=('obs_datetime', lambda x: x.dt.strftime('%Y-%m-%d').unique().tolist()),
        obs_dates_cnt=('obs_date', lambda x: len(x.unique().tolist()))             
        )
    
    #del(pts_df)
    
    # Match lake summary stats to PLD geometries
    keep_lake_cols = ['lake_id', 'ref_area', 'Shape_Leng', 'Shape_Area', 'area_m2', 'perim_m', 'pa_ratio' ,'geometry']
    matched_lakes = pd.merge(lake_summary, clipped_lakes[keep_lake_cols], how = 'left', on = 'lake_id')
    
    #del(lake_summary)
    
    # !!! 3.4 Threshold on SUMMER observation count
    # !!! Omits May and October observations here
    obs5_lakes = matched_lakes.query('obs_dates_cnt >= 5').copy()
    obs5_lakeids = obs5_lakes['lake_id'].unique()
    obs5_timeseries = lake_timeseries[lake_timeseries['lake_id'].isin(obs5_lakeids)].copy()
    
    #del(lake_timeseries)
    
    # !!! 3.5 Get percentage matched/thresholded
    total = len(clipped_lakes)
    matched = len(matched_lakes)
    obs5 = len(obs5_lakeids)
    obs5_lake_area_frac = sum(obs5_lakes['area_m2']) / sum(clipped_lakes['area_m2']) * 100
    obs10 = len(matched_lakes.query('obs_dates_cnt >= 10'))
    matched_percent = matched / total * 100
    obs5_percent = obs5 / total * 100
    obs10_percent = obs10 / total * 100
    
    #del(clipped_lakes, matched_lakes, obs5_lakeids)
    
    matching_stats.append({
        'roi_name': roi_name,
        'total_lakes': total, 
        'matched_lakes': matched, 
        'obs5_lakes': obs5,
        'obs5_lake_area_frac': obs5_lake_area_frac,
        'obs10_lakes': obs10,
        'matched_percent': matched_percent,
        'obs5_percent': obs5_percent,
        'obs10_percent': obs10_percent
    })
    
    matching_stats_df = pd.DataFrame(matching_stats)
    
    print(f'{roi_name} summarized. {obs5} of {total} >= 5 obs')
    
    # 3.6 Write the timeseries and rich obs_lakes to csv
    #matched_lakes.to_csv(f'{matched_lakes_path}/{roi_name}_is2matched.csv')
    obs5_lakes.loc[:, 'roi_name'] = roi_name
    obs5_timeseries.loc[:, 'roi_name'] = roi_name
    obs5_lakes_path = f'{lake_summaries_path}/{roi_name}_lakesummary.csv'
    obs5_lakes.to_csv(obs5_lakes_path, index=False)
    obs5_timeseries_path = f'{lake_ts_path}/{roi_name}_timeseries.csv'
    obs5_timeseries.to_csv(obs5_timeseries_path, index=False)
    
    temp_gdf = gpd.GeoDataFrame(data=obs5_lakes, geometry=obs5_lakes['geometry'])
    
    obs5_lakes_gdfs.append(temp_gdf)
    

    del(file, lakes_file, obs5_lakes, obs5_timeseries, total, matched, matched_percent, obs5, obs5_percent)

# %% 4. Save matched/threshold stats

all_obs5_lakes = pd.concat(obs5_lakes_gdfs)
all_obs5_lakes.drop(columns=['obs_dates_list'], inplace=True)
all_obs5_lakes.to_file(f'{lake_summaries_path}/all_lakes.shp')

def format_if_number(value):
    try:
        # Try to convert to float and format if it's a number
        return float('{:.2g}'.format(float(value)))
    except ValueError:
        # If it's not a number, return the value unchanged
        return value
    

matching_stats_df = matching_stats_df.map(format_if_number)

matching_stats_df.to_csv('./data/IS2_obscnts.csv')








