# %% 1.0 Libaries and paths

import glob 
import os
import pandas as pd
import rasterio as rio
import geopandas as gpd

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

sub_roi_names = ['MRD', 'TUK', 'Anderson']
sub_roi_paths = [f'./data/ew_rois/{n}_roi_shape.shp' for n in sub_roi_names]
sub_rois = [gpd.read_file(p) for p in sub_roi_paths]
sub_rois_dict = dict(zip(sub_roi_names, sub_rois))

# %% Split the IS2 Timeseries and lakes by sub roi

all_gdf = gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')

for name, gdf in sub_rois_dict.items():
    clipped_lakes = gpd.clip(all_gdf, gdf)
    clipped_lakes.to_csv(f'./data/lake_summaries/{name}_lakesummary.csv')

all_df = pd.read_csv('./data/lake_timeseries/MRD_TUK_Anderson_timeseries.csv')

# %%
