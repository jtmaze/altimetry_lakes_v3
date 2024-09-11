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
import rasterio as rio

from rasterio import features

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

"""
We need two sepperate rasters for matched lakes because GSWO and Sentinel-2 have different
resolutions. Need to run this script multiple times, changing the dataset (gswo or sentinel2)
or the scope (all_pld or matched_is2).
"""
dataset = 'sentinel2'
scope = 'matched_is2'

"""This code for matched ICESat-2 lakes"""
lakes_path = './data/lake_summaries/matched_lakes_clean.shp'
lakes = gpd.read_file(lakes_path)

"""This code for all the clipped PLD lakes"""
# lakes_path = './data/pld_clipped/*.shp'
# pld_files = glob.glob(lakes_path)
# pld_gdfs = [gpd.read_file(file) for file in pld_files]
# lakes = gpd.GeoDataFrame(pd.concat(pld_gdfs, ignore_index=True))
""" Uncomment it !!! here"""

rois_list = lakes['roi_name'].unique().tolist()
rois_remove = ['MRD', 'TUK', 'anderson_plain']
rois_list = [roi for roi in rois_list if roi not in rois_remove]

# %% 2.0 Choose buffer values to apply to lakes.

# Buffer values with corresponding band keys
buffer_vals = [60, 90, 120] # meters
keys = [1, 2, 3] # Band values for each buffer
buffer_ref = {key: value for key, value in zip(keys, buffer_vals)}
buffer_ref = pd.DataFrame(list(buffer_ref.items()), columns=['band', 'buffer']) 
#buffer_ref.to_csv('./data/buffer_bands.csv', index=False)

# %% 3.0 Apply buffers and rasterize lakes in each ROI.

# %% 3.1 Define the functions

def read_recurrence_raster(roi_name, dataset):
    """
    Read the first matching recurrence raster file based on ROI name and dataset.
    """
    recurrence_path = f'./data/recurrence_clean/Recurrence_{roi_name}_*_dataset_{dataset}.tif'
    matched_reccurence = glob.glob(recurrence_path) 
    # There will be multiple matched files, but they all have the same metadata
    matched_reccurence_first = matched_reccurence[0]
    src = rio.open(matched_reccurence_first)
    print(f'SOURCE RASTER META = {src.meta}')
    return src.meta 

def read_matched_lake_shapes(roi_name, lakes_clean_gdf):
    """
    Filter lakes by ROI name and convert geometries to estimated UTM CRS.
    """
    roi_lakes = lakes_clean_gdf[lakes_clean_gdf['roi_name'] == roi_name]
    est_crs = roi_lakes.estimate_utm_crs(datum_name='WGS 84')
    roi_lakes_utm = roi_lakes.copy().to_crs(est_crs)
    return roi_lakes_utm

def buffer_and_rasterize_lakes(roi_lakes_utm, buffer_val, src_meta):
    """
    Buffer lake geometries and rasterize them according to source raster metadata.
    """
    roi_buffered = roi_lakes_utm.copy()
    buff_col = f'geom_buff{buffer_val}'
    roi_buffered = roi_lakes_utm.geometry.buffer(buffer_val)
    # After buffering, convert out of UTM back to coordinates of source raster
    roi_buffered = roi_buffered.to_crs(src_meta['crs'])

    roi_rasterized = features.rasterize(
        roi_buffered,
        out_shape=(src_meta['height'], src_meta['width']),
        fill=0,
        out=None,
        transform=src_meta['transform'],
        all_touched=True,
        default_value=buffer_val
    )
    print(roi_rasterized.max(), roi_rasterized.min())
    return roi_rasterized

def write_output(dataset, roi_name, buffered_layers, src_meta, scope):
    """
    Write rasterized layers to a GeoTIFF file.
    """
    out_path = f'./data/lake_summaries/{scope}_scope_{dataset}_{roi_name}_rasterized_buffers.tif'

    with rio.open(
        out_path,
        'w',
        driver='GTiff',
        width=src_meta['width'],
        height=src_meta['height'],
        count=len(buffer_vals),
        dtype=src_meta['dtype'],
        crs=src_meta['crs'],
        transform=src_meta['transform']
    ) as dst:
        for i, layer in enumerate(buffered_layers, start=1):
            dst.write(layer, i)

    print(f'RASTERIZED OUTPUT META: {dst.meta}')

def rasterize_matched_is2_lakes(roi_name, dataset, lakes_clean_gdf, buffer_vals, scope):
    """
    Coordinate the rasterization of buffered lake geometries for multiple buffer values.
    """
    src_meta = read_recurrence_raster(roi_name, dataset)
    roi_lakes_utm = read_matched_lake_shapes(roi_name, lakes_clean_gdf)

    buffered_layers = [buffer_and_rasterize_lakes(roi_lakes_utm=roi_lakes_utm,
                                                  buffer_val=val,
                                                  src_meta=src_meta
                                                  ) for val in buffer_vals]
    
    write_output(dataset=dataset, 
                 roi_name=roi_name, 
                 buffered_layers=buffered_layers, 
                 src_meta=src_meta,
                 scope=scope
    )

    print(f'{roi_name} is rasterized')

# %% 3.2 Run the functions

for roi_name in rois_list:
    rasterize_matched_is2_lakes(roi_name, 
                                dataset=dataset, 
                                buffer_vals=buffer_vals, 
                                lakes_clean_gdf=lakes,
                                scope=scope
    )

# %%
