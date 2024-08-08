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
import rasterio as rio
from rasterio.merge import merge

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')
split_sentinel2_dir = './data/sentinel2_raw/'
output_sentinel2_dir = './data/sentinel2_clean/'

full_file_list = glob.glob(split_sentinel2_dir + '*')

rois_pattern = r'/sentinel2_raw/(.*?)_yea.*\.tif'
years_pattern = r'_years(.*?)_wee.*\.tif'
weeks_pattern = r'_weeks(.*?)-0000.*\.tif'

def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)


rois = extract_unique(full_file_list, rois_pattern)
year_intervals = extract_unique(full_file_list, years_pattern)
week_intervals = extract_unique(full_file_list, weeks_pattern)

# %% 2.0 Reformat the sentinel-2 masks
"""
The sentinel-2 masks are split into multiple files. We need to merge them into a single file.
Also, we need to scale the values to 0-100, and convert the data type to uint8 for memory efficiency.
"""

for roi in rois:
    for week_interval in week_intervals:
        for year_interval in year_intervals:
    
                files = glob.glob(os.path.join(split_sentinel2_dir, 
                                               f'{roi}_years{year_interval}_weeks{week_interval}*.tif'
                                               )
                                  )
                
                src_files = []
                rescaled_data = []
                out_transform = None
                
                for path in files:
                    with rio.open(path) as src:
                        data = src.read(1)
                        rescaled = (data * 100).round().astype(rio.uint8)
                        rescaled_data.append(rescaled)
                        if out_transform is None:
                            out_transform = src.transform
        
                merged, out_transform = merge(rescaled_data, transform=out_transform)
                print(len(rescaled_data))
        
                out_meta = src.meta.copy()
        
                out_meta.update({
                    "driver": "GTiff",
                    "height": merged.shape[1],
                    "width": merged.shape[2],
                    "transform": out_transform,
                    "crs": src.crs,
                    "dtype": 'uint8'
                })

        
                out_path = os.path.join(output_sentinel2_dir, 
                                        f'Recurrence_{roi}__years{year_interval}_weeks{week_interval}.tif'
                                        )
        
                with rio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(merged)
            

# %%
