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
split_sentinel2_dir = './data/sentinel2_raw/'
output_sentinel2_dir = './data/recurrence_clean/'

full_file_list = glob.glob(split_sentinel2_dir + '*')

rois_pattern = r'_roi_(.*?)_years.*\.tif'
timeframe_pattern = r's2(partial|full)_roi.*\.tif'
months_pattern = r'_weeks(.*?)-0000.*\.tif'

def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)


rois = extract_unique(full_file_list, rois_pattern)
timeframes = extract_unique(full_file_list, timeframe_pattern)
months = extract_unique(full_file_list, months_pattern)

print(rois, months, timeframes)

# %% 2.0 Reformat the sentinel-2 occurrences
"""
The sentinel-2 masks are split into multiple files. We need to merge them into a single file.
"""

for roi in rois:
    for timeframe in timeframes:
        for month in months:
    
                files = glob.glob(os.path.join(split_sentinel2_dir, 
                                               f's2{timeframe}_roi_{roi}_years*_weeks{months}*.tif'
                                               )
                                  )
        
                src_files = []
                for path in files:
                    src = rio.open(path)
                    #print(src.meta)
                    src_files.append(src)

                print(f'merging total = {len(src_files)}')
                merged, out_transform = merge(src_files)
        
                out_meta = src_files[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": merged.shape[1],
                    "width": merged.shape[2],
                    "transform": out_transform,
                    "dtype": 'uint8',
                    "crs": src_files[0].crs
                })

                print(f'OUT META: {out_meta}')

                if month == '22-26':
                    out_month = 'early'
                elif month == '31-35':
                    out_month = 'late'   
                out_path = os.path.join(output_sentinel2_dir, 
                                        f'Recurrence_{roi}_timeframe_{timeframe}_dataset_sentinel2_{out_month}.tif'
                                        )
        
                with rio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(merged)
            
                for src in src_files:
                    src.close()
            


