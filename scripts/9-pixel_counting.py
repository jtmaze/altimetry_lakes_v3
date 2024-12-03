# %% 1.0 Import libraries and organize directories

import pprint as pp
import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio
import re

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

buffer_ref = pd.read_csv('./data/buffer_bands.csv')
#rois_list = list(gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')['roi_name'].unique())

# %% 2.0 Extract timeframes and rois from recurrence file names

recurrence_files = glob.glob(f'./data/recurrence_clean/*.tif')

timeframe_pattern = r'timeframe_(partial|full)_dataset'
month_pattern = r'(?:gswo|glad|sentinel2)_(.*).tif'
dataset_pattern = r'dataset_(gswo|glad|sentinel2)_.*\.tif'
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
months = extract_unique(recurrence_files, month_pattern)

# %% 3.0 Define functions

def mask_over_matched_lakes(scope, dataset, timeframe, roi_name, month, band, buffer):
    """
    Applies a mask over matched lakes and returns the masked data.
    """

    print(f'Processing {roi_name} {timeframe} {month} {dataset} {scope} {buffer} ')
    path_recurrence_raster = f'./data/recurrence_clean/Recurrence_{roi_name}_timeframe_{timeframe}_dataset_{dataset}_{month}.tif'
    print(path_recurrence_raster)
    # Becuase the gswo and glad datasets need the same PLD mask
    if dataset == 'sentinel2':
        satellite = 'sentinel2'
    else:
        satellite = 'landsat'

    if (dataset == 'sentinel2') and (roi_name == 'YKF'):
        lake_file_roi = 'YKflats'
    elif (dataset == 'sentinel2') and (roi_name == 'YKD'):
        lake_file_roi = 'YKdelta'
    else:
        lake_file_roi = roi_name

    path_lakes = f'./data/rasterized_pld/{scope}_scope_{satellite}_dataset_{lake_file_roi}_rasterized_buffers.tif'

    # Quick error handling, some combinations will not exist, because the
    # rois are named differently (YKflats/YKF & YKdelta/YKD). 

    if not os.path.exists(path_recurrence_raster) or not os.path.exists(path_lakes):
        print(f"!!! Skipping {roi_name}, {timeframe}, {dataset}, {scope}, and {month} because "
              "files are missing.")
        return None

    with rio.open(path_lakes) as mask:
        mask_data = mask.read([band]) # Band should match buffer values
        mask_meta = mask.meta
        #print(mask_meta)

    with rio.open(path_recurrence_raster) as target:
        target_data = target.read(1)
        target_meta = target.meta
        #print(target_meta)

    mask_bool = mask_data != 0
    matched_data = np.where(mask_bool, target_data, -1)
    matched_data = np.squeeze(matched_data) # Need to squeeze because GSWO datasets have an extra dimension (i.e. L, H, W)
    print(matched_data.shape)

    # Write select masked rasters to drive
    if (buffer == 60) and (roi_name in ['MRD', 'TUK', 'YKflats', 'YKF']) and (scope == 'all_pld') and (month == 'late'):
        
        output_path = f'./data/masked_rasters/roi_{roi_name}_timeframe_{timeframe}_month_{month}_dataset_{dataset}_buffer{buffer}.tif'
        with rio.open(output_path, 'w', 
                      driver='GTiff',
                      height=matched_data.shape[0],
                      width=matched_data.shape[1],
                      count=1,  
                      dtype=matched_data.dtype,
                      crs=target_meta['crs'],  
                      transform=target_meta['transform']) as dst:
            dst.write(matched_data, 1)  
        print(f'Wrote change raster to {output_path}')
            
    return matched_data

def create_summary_df(matched_data, roi, timeframe, dataset, scope, buffer_val, month):
    """
    Create a summary DataFrame from the masked data.
    """
    flat = matched_data.flatten()
    unique_vals, cnts = np.unique(flat, return_counts=True)
    df = pd.DataFrame({'pix_vals': unique_vals, 'pix_cnts': cnts})

    # get consistent roi names
    if roi == 'YKflats':
        roi = 'YKF'
    if roi == 'YKdelta':
        roi = 'YKD'
    else:
        roi == roi

    df['roi_name'] = roi
    df['timeframe'] = timeframe
    df['month'] = month
    df['buffer'] = buffer_val
    df['dataset'] = dataset
    df['scope'] = scope

    print(f'Finished with iteration')

    return df

# %% Run the functions

buffer_vals = [60, 120] # meters
scopes = ['all_pld', 'matched_is2']

results = []

for roi in rois:
    for scope in scopes:
        for dataset in datasets:
            for timeframe in timeframes:
                for buffer_val in buffer_vals:
                    for month in months:

                        band = buffer_ref[buffer_ref['buffer'] == buffer_val]['band'].values[0]
                        matched_data = mask_over_matched_lakes(scope, dataset, timeframe, roi, month, band, buffer_val)

                        if matched_data is not None: # Some combinations will not exist
                            results.append(create_summary_df(matched_data,
                                                            roi,
                                                            timeframe,
                                                            dataset,
                                                            scope,
                                                            buffer_val,
                                                            month
                                                            )
                                        )

                        else:
                            print(f'No match for {roi}, {timeframe}, {dataset}, {scope}, and {month}')
                            continue

full_results = pd.concat(results)

full_results.to_csv('./data/pixel_counts.csv', index=False)

# %%
