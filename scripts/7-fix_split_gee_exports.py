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
import rasterio
from rasterio.merge import merge

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')
split_sentinel2_dir = './data/sentinel2_raw/'
output_sentinel2_dir = './data/sentinel2_clean/'

full_file_list = glob.glob(split_sentinel2_dir + '*')

rois_pattern = r'/sentinel2_raw/(.*?)_weekly.*\.tif'
occurance_cnts_pattern = r'_weekly_(.*?)_years.*\.tif'
weeks_pattern = r'_weeks(.*?)-0000.*\.tif'

def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)


rois = extract_unique(full_file_list, rois_pattern)
occurance_cnts = extract_unique(full_file_list, occurance_cnts_pattern)
timeperiods = extract_unique(full_file_list, weeks_pattern)

# %% 2.0 Merge the split sentinel-2 masks for each roi at conncurrent timeperiods and thresholds. 

for roi in rois:
    for timeperiod in timeperiods:
        for cnt in occurance_cnts:
    
                files = glob.glob(os.path.join(split_sentinel2_dir, 
                                               f'{roi}_weekly_{cnt}_years2019-2023_weeks{timeperiod}*.tif'
                                               )
                                  )
                
                src_files = []
                
                for path in files:
                    print(path)
                    src = rasterio.open(path)
                    #print(src.meta)
                    src_files.append(src)
        
                merged, out_transform = merge(src_files)
                print(len(src_files))
        
                out_meta = src_files[0].meta.copy()
        
                out_meta.update({
                    "driver": "GTiff",
                    "height": merged.shape[1],
                    "width": merged.shape[2],
                    "transform": out_transform,
                    "crs": src_files[0].crs
                })

        
                out_path = os.path.join(output_sentinel2_dir, 
                                        f'Recurrence_{roi}_weeks{timeperiod}_{cnt}.tif'
                                        )
        
                with rasterio.open(out_path, 'w', **out_meta) as dst:
                    dst.write(merged)
            
                for src in src_files:
                    src.close()
            
