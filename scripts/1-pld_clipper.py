#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:04:01 2024

@author: jmaze
"""

# %% 1. Libraries and directories

import geopandas as gpd
import pandas as pd
import os
import re
import glob

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3/')

path_rois = './data/ew_rois/'
path_pld_in = './data/pld_meh/'
path_pld_out = './data/pld_clipped/'

# %% 2. Read the datasets

roi_files = glob.glob(f'{path_rois}/*.shp')
rois = []

for roi_file in roi_files:
    roi = gpd.read_file(roi_file)
    rois.append(roi)

pld_files = glob.glob(f'{path_pld_in}/*.shp')
pld_data = []

for file in pld_files:
    
    gdf = gpd.read_file(file)
    gdf = gdf.drop(columns=['OBJECTID','grand_id','date_t0', 'ds_t0', 
                             'pass_full', 'pass_part','cycle_flag', 'ref_area_u', 
                             'ref_wse_u', 'storage', 'ice_clim_f', 'ice_dyn_fl',
                             'basin_id', 'names', 'ref_wse', 'reach_id_l'])
    
    # Some lake shapes were self-intersecting throwing a topology error
    # the .buffer(0) method expands and recalculates polygon boundaries
    # this recalculation process creates new verticies adhearing to valid
    # geometry rules. Specifically this helps with "hour-glass" shaped polygons
    gdf['geometry'] = gdf['geometry'].apply(
        lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    
    print(gdf.head())
    
    pld_data.append(gdf)
    
pld = pd.concat(pld_data)

del gdf, pld_data

# %% 3. Clip the pld data

    
for roi in rois:
    
    roi_geom = roi.geometry.iloc[0]
    fully_within = pld[pld.geometry.apply(lambda geom: geom.within(roi_geom))]

    # Perform the intersection with only the fully contained geometries
    if not fully_within.empty:
        clipped_lakes = gpd.overlay(fully_within, roi, how='intersection')
    
    pattern = r'(.*?)_weekly'
    clipped_lakes['roi_name'] = clipped_lakes['roi'].apply(
        lambda x: re.search(pattern, x).group(1)
    )
    clipped_lakes = clipped_lakes.drop(columns=['roi'])
    
    # Convert the lake_id to a string for more robust dataset
    roi_name = str(clipped_lakes['roi_name'].unique()[0])
    
    def lakeid_to_string(x):
        return(f'{roi_name}_id_' + str(round(x)))
    
    clipped_lakes['lake_id'] = clipped_lakes['lake_id'].apply(lakeid_to_string)
    
    ## Recalculate area, perimeter and pa-ratio
    est_crs_utm = clipped_lakes.estimate_utm_crs()
    print(est_crs_utm)
    clipped_lakes = clipped_lakes.to_crs(est_crs_utm)
    clipped_lakes['area_m2'] = clipped_lakes.geometry.area
    clipped_lakes['perim_m'] = clipped_lakes.geometry.length
    clipped_lakes['pa_ratio'] = clipped_lakes.perim_m / clipped_lakes.area_m2
    clipped_lakes = clipped_lakes.to_crs('EPSG:4326')
    
    out_path = f'{path_pld_out}/{roi_name}_pld_clipped.shp'
    clipped_lakes.to_file(out_path)
    
    print(f'{roi_name} processed')
    
    


    
        
    
    


