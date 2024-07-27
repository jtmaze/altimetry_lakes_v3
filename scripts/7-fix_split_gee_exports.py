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

# %% 

for roi in rois:
    
    jun_files = glob.glob(os.path.join(bad_gee_dir, f'*jun_{roi}*.tif'))
    aug_files = glob.glob(os.path.join(bad_gee_dir, f'*aug_{roi}*.tif'))
    
    months = [jun_files, aug_files]
    month_names = ['jun', 'aug']
    
    for month, month_name in zip(months, month_names):
        
        src_files = []
        
        for path in month:
            
            src = rasterio.open(path)
            src_files.append(src)
            
        merged, out_transform = merge(src_files)
        
        out_meta = src_files[0].meta.copy()
        
        out_meta.update({
            "driver": "GTiff",
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": out_transform,
            "crs": src_files[0].crs
        })
        
        out_path = os.path.join(gee_dir, f'Monthly_Recurrence_{month_name}_{roi}.tif')
    
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(merged)
        
        for src in src_files:
            src.close()
        
        print(f'{roi} {month_name} merged')

print("All processing complete.")