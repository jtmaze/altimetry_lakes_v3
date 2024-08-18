# %% 1.0 Import libraries and read data

import pprint as pp
import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import re

os.chdir('/Users/jtmaz/Documents/projects/altimetry_lakes_v3')

buffer_ref = pd.read_csv('./data/buffer_bands.csv')
rois_list = list(gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')['roi_name'].unique())

recurrence_files = glob.glob(f'./data/recurrence_clean/*')

timeframe_pattern = r'timeframe_(.*)_dataset'
dataset_pattern = r'dataset_(.*)\.tif'
roi_pattern = r'Recurrence_(.*)_timeframe'

def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)

timeframes = extract_unique(recurrence_files, timeframe_pattern)
datasets = extract_unique(recurrence_files, dataset_pattern)
rois = extract_unique(recurrence_files, roi_pattern)


# %% 2.0 Define functions

def mask_over_matched_lakes(dataset, timeframe, roi_name, band, buffer_val):

    path_recurrence_raster = f'./data/recurrence_clean/Recurrence_{roi_name}_timeframe_{timeframe}_dataset_{dataset}.tif'
    path_lakes = f'./data/lake_summaries/{dataset}_{roi_name}_rasterized_buffers.tif'

    # Quick error handling, some combinations will not exist, because the
    # timeframes are named differently between GSWO and Sentinel-2. 

    if not os.path.exists(path_recurrence_raster):
        print(f"Skipping {path_recurrence_raster} because it does not exist.")
        return None, None, None, None, None

    with rio.open(path_lakes) as mask:
        mask_data = mask.read([band])
        mask_meta = mask.meta
        print(f'MASK META: {mask_meta}')


    with rio.open(path_recurrence_raster) as target:
        target_data = target.read(1)
        target_meta = target.meta
        print(f'TARGET META: {target_meta}')

    if dataset == 'gswo':
        mask_bool = mask_data != 0
        matched_data = np.where(mask_bool, target_data, 0)
        matched_data_squeeze = np.squeeze(matched_data)

    elif dataset == 'sentinel2':
        #target_data = target_data.squeeze()
        mask_bool = mask_data != 0
        matched_data = np.where(mask_bool, target_data, 0)
        matched_data_squeeze = np.squeeze(matched_data)

    print(f'{roi_name} {matched_data.shape}')
            
    return matched_data_squeeze, target_meta, mask_bool, target_data, matched_data


# %% test

buffer_vals = [60, 90, 120] # meters
results = []


for roi in rois:
    for dataset in datasets:
        for timeframe in timeframes:
            for buffer_val in buffer_vals:

                print(f'!!! {roi} {timeframe} {buffer_val} {dataset}')
                band = buffer_ref[buffer_ref['buffer'] == buffer_val]['band'].values[0]
                matched_data_squeeze, target_meta, mask_bool, target_data, matched_data = mask_over_matched_lakes(dataset, timeframe, roi, band, buffer_val)

                if matched_data is None:
                    continue

                flat = matched_data.flatten()

                unique_vals, cnts = np.unique(flat, return_counts=True)
                df = pd.DataFrame({'pix_vals': unique_vals, 'pix_cnts': cnts})
                df['roi_name'] = roi
                df['timeframe'] = timeframe
                df['buffer'] = buffer_val
                df['dataset'] = dataset

                results.append(df)

                print(f'!!! {roi} {timeframe} {buffer_val} {dataset}')

full_results = pd.concat(results)

full_results.to_csv('./data/pixel_counts.csv', index=False)

# %% 3.0 Save results

full_results.to_csv('./data/pixel_counts.csv', index=False)
