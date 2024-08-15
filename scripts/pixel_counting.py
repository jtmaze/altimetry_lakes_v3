# %% 1.0 Import libraries and read data

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

buffer_ref = pd.read_csv('data/buffer_ref.csv')
rois_list = pd.read_csv('data/matched_lakes_clean.csv')['roi_name'].unique().to_list()


# %% 2.0 Define functions

def mask_over_matched_lakes(dataset, timeframe, roi_name, band, buffer):

    path_recurrence_raster = f'./data/{dataset}_clean/Recurence/{roi_name}_{timeframe}.tif'
    path_lakes = f'./data/lake_summaries/{dataset}_{roi_name}_rasterized_buffers_{dataset}.tif'
# %% test

src = rio.open('./data/lake_summaries/AKCP_rasterized_buffers_sentinel2.tif')
# %%

print(src.dtype, src.bounds)
# %%
