#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:37:52 2024

@author: jmaze
"""
# %% 1.0 Libraries and directories

import os
import re
import glob
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio import features

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

clean_lakes_path = './data/lake_summaries/matched_lakes_clean.shp'
clean_lakes = gpd.read_file(clean_lakes_path)

rois_list = clean_lakes['roi_name'].unique().tolist()

# %% 2.0 Choose buffer values to apply to lakes.

# Buffer values with corresponding band keys
buffer_vals = [60, 90, 120] # meters
keys = [1, 2, 3] # Band values for each buffer
buffer_ref = {key: value for key, value in zip(keys, buffer_vals)}
buffer_ref = pd.DataFrame(list(buffer_ref.items()), columns=['band', 'buffer']) 
buffer_ref.to_csv('./data/buffer_bands.csv', index=False)

# %% 3.0 Apply buffers and rasterize lakes in each ROI.

"""
To rasterize the buffered lakes, we copy metadata from the Sentinel-2 rasters.
If rasterizing the GSWO lakes, we copy metadata from the GSWO raster.
"""
dataset = 'sentinel2'

for roi_name in rois_list:

    # Read the raster files associated with the ROI
    roi_matched = clean_lakes[clean_lakes['roi_name'] == roi_name]
    # !!! Change path to match GSWO or Sentinel-2 data
    recurrence_path = f'./data/{dataset}_clean/Recurrence_{roi_name}_years*.tif'
    matched_reccurence = glob.glob(recurrence_path) 
    # There will be multiple matches, but they all have the same metadata
    matched_reccurence_first = matched_reccurence[0]
    recurrence_raster = rasterio.open(matched_reccurence_first)
    
    # Reproject ROI to local utm for buffering
    roi_utm = roi_matched.copy()
    est_crs = roi_utm.estimate_utm_crs(datum_name='WGS 84')
    print(f'{roi_name} with {est_crs} estimated UTM')
    roi_utm = roi_utm.to_crs(est_crs)
    

    # Apply buffer values & rasterize lake geoms
    buffered_layers = []
    for buffer_val in buffer_vals:
        
        roi_buffered = roi_utm.copy()
        
        buff_col = f'geom_buff{buffer_val}'
        roi_buffered[buff_col] = roi_utm.geometry.buffer(buffer_val)
        roi_buffered = roi_buffered.set_geometry(buff_col)
        
        # Convert back to GSWO CRS
        roi_buffered = roi_buffered.to_crs(recurrence_raster.crs)
        print(recurrence_raster.crs)
        
        # Rasterize buffered ROI
        roi_rasterized = features.rasterize(
            roi_buffered[f'geom_buff{buffer_val}'],
            out_shape=recurrence_raster.shape,
            fill=0,
            out=None,
            transform= recurrence_raster.transform,
            all_touched=True,
            default_value=buffer_val
        )
        
        buffered_layers.append(roi_rasterized)
        
        
    # Write to memory
    out_path = f'./data/lake_summaries/{roi_name}_rasterized_buffers.tif'
    
    with rasterio.open(
            out_path,
            'w',
            driver='GTiff', 
            height=roi_rasterized.shape[0],
            width=roi_rasterized.shape[1],
            count=len(buffer_vals),
            dtype='uint8',
            crs=recurrence_raster.crs,
            transform=recurrence_raster.transform,
    ) as dst:
        for i, layer in enumerate(buffered_layers, start=1):
            dst.write(layer, i)
            
    print(dst.meta)
    

# %%
