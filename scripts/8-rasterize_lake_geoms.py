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
resolutions. Need to run this script multiple times, changing the dataset (landsat or sentinel2)
or the scope (all_pld or matched_is2).

Because GSWO and GLAD have the same resolution, we can use the same rasterized lakes for both.
"""
dataset = 'landsat' # landsat or sentinel2
scope = 'matched_is2' # all_pld or matched_is2

rois_list = ['Anderson','TUK', 'MRD', 'YKdelta', 'YKflats', 'AKCP']


# %% 2.0 Choose buffer values to apply to lakes.

# Buffer values with corresponding band keys
buffer_vals = [60, 120] # meters
keys = [1, 2] # Band values for each buffer
buffer_ref = {key: value for key, value in zip(keys, buffer_vals)}
buffer_ref = pd.DataFrame(list(buffer_ref.items()), columns=['band', 'buffer']) 
buffer_ref.to_csv('./data/buffer_bands.csv', index=False)

# %% 3.0 Apply buffers and rasterize lakes in each ROI.

# %% 3.1 Define the functions

def read_recurrence_raster(roi_name, dataset):
    """
    Read the first matching recurrence raster file based on ROI name and dataset.
    inputs: 1) roi_name to select recurrence raster 2) dataset to specify pixel resolution (landsat or sentinel2)
    outputs: metadata for the recurrence raster
    """
    if dataset == 'landsat':
        ds = 'gswo'
    elif dataset == 'sentinel2':
        ds = 'sentinel2'

    if roi_name == 'Anderson':
        file_roi_name = 'anderson_plain'
    else:
        file_roi_name = roi_name

    if roi_name == 'YKdelta' and dataset == 'sentinel2':
        file_roi_name = 'YKD'
    if roi_name == 'YKflats' and dataset == 'sentinel2':
        file_roi_name = 'YKF'

    recurrence_path = f'./data/recurrence_clean/Recurrence_{file_roi_name}_timeframe*_dataset_{ds}_*.tif'
    print(recurrence_path)
    matched_reccurence = glob.glob(recurrence_path) 
    # There will be multiple matched files, but they all have the same metadata
    matched_reccurence_first = matched_reccurence[0]
    print(matched_reccurence_first)
    src = rio.open(matched_reccurence_first)
    print(f'SOURCE RASTER META = {src.meta}')
    return src.meta 

def read_matched_lake_shapes(roi_name, scope):
    """
    Filter lakes by ROI name and convert geometries to estimated UTM CRS.
    inputs: 1) roi_name to filter lakes 2) GeoDataFrame of all cleaned lakes
    outputs: GeoDataFrame of lakes in UTM CRS
    """
    if scope == 'matched_is2':
        roi_lakes_path = f'./data/lake_summaries/{roi_name}_lakesummary.shp'
    elif scope == 'all_pld':
        if roi_name == 'Anderson':
            file_roi_name = 'anderson_plain'
        else:
            file_roi_name = roi_name

        roi_lakes_path = f'./data/pld_clipped/{file_roi_name}_pld_clipped.shp'

    roi_lakes = gpd.read_file(roi_lakes_path)
    est_crs = roi_lakes.estimate_utm_crs(datum_name='WGS 84')
    roi_lakes_utm = roi_lakes.copy().to_crs(est_crs)
    return roi_lakes_utm

def buffer_and_rasterize_lakes(roi_lakes_utm, buffer_val, src_meta):
    """
    Buffer lake geometries and rasterize them according to source raster metadata.
    inputs: 1) GeoDataFrame of lakes in UTM CRS 2) buffer value in meters 3) source raster metadata
    outputs: rasterized lake geometries
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
    Writes the buffered layers to a GeoTIFF file.
    """
    if roi_name == 'Anderson':
        out_file_roi_name = 'anderson_plain'
    else:
        out_file_roi_name = roi_name

    out_path = f'./data/rasterized_pld/{scope}_scope_{dataset}_dataset_{out_file_roi_name}_rasterized_buffers.tif'

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

def rasterize_matched_lakes(roi_name, dataset, buffer_vals, scope):
    """
    Coordinate the rasterization of buffered lake geometries for multiple buffer values.
    """
    src_meta = read_recurrence_raster(roi_name, dataset)
    roi_lakes_utm = read_matched_lake_shapes(roi_name, scope=scope)

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
    rasterize_matched_lakes(roi_name, 
                            dataset=dataset, 
                            buffer_vals=buffer_vals,
                            scope=scope
    )


# %%
