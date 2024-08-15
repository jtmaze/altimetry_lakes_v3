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
import numpy as np
import rasterio as rio
from rasterio.merge import merge

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')
split_gswo_dir = './data/gswo_raw/'
output_gswo_dir = './data/gswo_clean/'

full_file_list = glob.glob(split_gswo_dir + '*')

rois_pattern = r'/gswo_raw/GSWORecurrence_(.*?)_month.*\.tif'
months_pattern = r'_month_(.*?)_'

def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(match.group(1))
    return list(unique_items)


rois = extract_unique(full_file_list, rois_pattern)
months = extract_unique(full_file_list, months_pattern)


# %% 2.0 Reformat the sentinel-2 masks
"""
The gswo masks are split into multiple files. We need to merge them into a single file.
"""

for roi in rois:
    for month in months:
    
        files = glob.glob(os.path.join(split_gswo_dir, 
                                        f'GSWORecurrence_{roi}_month_{month}*.tif'
                                        )
                            )
        
        src_files = []
        
        for path in files:
            src = rio.open(path)
            #print(src.meta)
            src_files.append(src)

        merged, out_transform = merge(src_files)
        print(len(src_files))

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

        out_path = os.path.join(output_gswo_dir, 
                                f'GSWORecurrence_{roi}_months_{month}_.tif'
                                )

        with rio.open(out_path, 'w', **out_meta) as dst:
            dst.write(merged)
    
        for src in src_files:
            src.close()
            

# %%
