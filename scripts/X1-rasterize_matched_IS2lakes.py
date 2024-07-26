#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:37:52 2024

@author: jmaze
"""
# %% 1.0 Libraries and directories

import os
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio import features

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v2')

rois_path = './inputs/study_regions.shp'
rois_list = gpd.read_file(rois_path).name.to_list()
clean_lakes_path = './output/lake_summaries/matched_lakes_clean.shp'
clean_lakes = gpd.read_file(clean_lakes_path)

# %% 2.0 Choose buffer values to apply to lakes.

# Buffer values with corresponding band keys
buffer_vals = [60, 90, 150] # meters
keys = [1, 2, 3, 4, 5] # Band values for each buffer
buffer_ref = {key: value for key, value in zip(keys, buffer_vals)}
buffer_ref = pd.DataFrame(list(buffer_ref.items()), columns=['band', 'buffer']) 
buffer_ref.to_csv('./output/buffer_bands.csv', index=False)

# %% 3.0 Apply buffers and rasterize lakes in each ROI.

for roi_name in rois_list:

    # Read the raster files associated with the ROI
    roi_matched = clean_lakes[clean_lakes['roi_name'] == roi_name]
    path = f'./inputs/gee_exports/Monthly_Recurrence_aug_{roi_name}.tif'
    gswo = rasterio.open(path)
    
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
        roi_buffered = roi_buffered.to_crs(gswo.crs)
        
        # Rasterize buffered ROI
        roi_rasterized = features.rasterize(
            roi_buffered[f'geom_buff{buffer_val}'],
            out_shape=gswo.shape,
            fill=-1,
            out=None,
            transform= gswo.transform,
            all_touched=True,
            default_value=buffer_val
        )
        
        buffered_layers.append(roi_rasterized)
        
        
    # Write to memory
    out_path = f'./output/is2_lakes_buffered/{roi_name}_rasterized_buffers.tif'
    
    with rasterio.open(
            out_path,
            'w',
            driver='GTiff', 
            height=roi_rasterized.shape[0],
            width=roi_rasterized.shape[1],
            count=len(buffer_vals),
            dtype=buffered_layers[0].dtype,
            crs=gswo.crs,
            transform=gswo.transform,
    ) as dst:
        for i, layer in enumerate(buffered_layers, start=1):
            dst.write(layer, i)
            
    print(dst.meta)
    
    

    
    
