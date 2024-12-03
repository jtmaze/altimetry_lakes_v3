#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:48:39 2024

@author: jmaze
"""

# %% 1. Libraries and directories

import geopandas as gpd
import pandas as pd
import glob
import os
import re

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3/')
path_is2_download = './data/IS2raw/'
path_clipped_pld = './data/pld_clipped'
path_matched_pts = './data/matchedIS2_pts/'


# %% 2. Get a list of roi names from downloads
is2_files = glob.glob(f'{path_is2_download}/*.csv')

# Regular expression pattern to extract the roi_name
pattern = r'IS2raw_(.*?)_\d{4}\.csv'
rois = set()

for file in is2_files:
    match = re.search(pattern, file)
    if match:
        rois.add(match.group(1))

rois = list(rois)       

# %% 3. Spatially Join the ICESat-2 points to the PLD lakes

for roi_name in rois:
    
    # Read the lake data
    lake_path = glob.glob(f'{path_clipped_pld}/{roi_name}*.shp')
    lake_data = gpd.read_file(lake_path[0])
    # Select a few columns of interest
    lakes_slim = lake_data[['lake_id', 'geometry']]
    
    roi_files_path = glob.glob(f'{path_is2_download}/*_{roi_name}_*.csv')
    
    # Read each year of IS2 data
    years = []
    for year in range(2019, 2024):
        print(f'{year} in {roi_name}')
    
        specific_file_path = glob.glob(f'{path_is2_download}/*_{roi_name}_{year}.csv')
        dataset = pd.read_csv(specific_file_path[0])
        dataset.drop(columns=['Unnamed: 0'], inplace=True)
        years.append(dataset)
        #del(dataset)
        
    roi_is2_data = pd.concat(years)
    #del(years)

    is2_gdf = gpd.GeoDataFrame(roi_is2_data, 
                               geometry=gpd.points_from_xy(roi_is2_data['longitude'],
                                                           roi_is2_data['latitude']
                                                           ),
                               crs='EPSG:4326'
                               )
    #del(roi_is2_data)
    
    # Spatial join the IS2 points with matched lakes. 
    is2_matched = gpd.sjoin(is2_gdf, lakes_slim, predicate='within', how='inner')
    # Only write a portion of the points for visualization
    is2_matched[0:1_000_000].to_file(f'./temp/{roi_name}_matched_pts.shp', index=False)
    
    # Convert to a df for more efficient file storage. 
    is2_matched_df = pd.DataFrame(is2_matched.drop(columns=['geometry', 'index_right']))
    is2_matched_df.to_csv(f'{path_matched_pts}/{roi_name}_matched.csv', index=False)
    print(f'{roi_name} matched')
    #del(roi_name, lake_path, lake_data, lakes_slim, is2_gdf, is2_matched, is2_matched_df)
    

# %%
