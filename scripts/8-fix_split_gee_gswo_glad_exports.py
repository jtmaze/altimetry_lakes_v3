#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 09:12:16 2024

@author: jmaze
"""

# %% 1.0 Libraries and directories

import os
import re
import glob
#import numpy as np
import rasterio as rio
from rasterio.merge import merge

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')
split_gswo_dir = './data/gswo_glad_raw/'
output_gswo_dir = './data/recurrence_clean/'

full_file_list = glob.glob(split_gswo_dir + '*')

rois_pattern = r'Monthly_(.*?)_month.*\.tif'
months_pattern = r'_month_(.*?)_'
dataset_pattern = r'/(GSWO|glad).*\.tif'
timeframe_pattern = r'(?:GSWO|glad)([A-Za-z0-9\-]+?)_.*?\.tif$'

def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)

datasets = extract_unique(full_file_list, dataset_pattern)
rois = extract_unique(full_file_list, rois_pattern)
months = extract_unique(full_file_list, months_pattern)
timeframes = extract_unique(full_file_list, timeframe_pattern)


# %% 2.0 Reformat the gswo occurrences
"""
The gswo and glad data are split into multiple files. We need to merge them into a single file.
"""

for roi in rois:
    for month in months:
        for dataset in datasets:
            for timeframe in timeframes:
    
                files = glob.glob(os.path.join(split_gswo_dir, 
                                                f'{dataset}{timeframe}_{roi}_month_{month}*.tif'
                                                )
                                    )
                if not files:
                    continue

                src_files = []
                for path in files:
                    src = rio.open(path)
                    #print(src.meta)
                    src_files.append(src)

                merged, out_transform = merge(src_files)
                print(f'{len(src_files)} to merge')
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": merged.shape[1],
                    "width": merged.shape[2],
                    "dtype": 'uint8',
                    "transform": out_transform,
                    "crs": src_files[0].crs
                })

                print(f'OUT META: {out_meta}')
                # For the love of god, please use consistent naming
                if dataset == 'GSWO':
                    out_dataset = 'gswo'
                elif dataset == 'glad':
                    out_dataset = 'glad'

                if month == 'june':
                    out_month = 'early'
                elif month == 'aug':
                    out_month = 'late'

                if timeframe == 'FullMonthly' or timeframe == 'RecurrenceFullMonthly':
                    out_timeframe = 'full'
                elif timeframe == '2016-2020Monthly' or timeframe == 'Recurrence2016-2020Monthly':
                    out_timeframe = 'partial'

                out_path = os.path.join(output_gswo_dir, 
                                        f'Recurrence_{roi}_timeframe_{out_timeframe}_dataset_{out_dataset}_{out_month}.tif'
                                        )

                with rio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(merged)
            
                for src in src_files:
                    src.close()
            


# %%
