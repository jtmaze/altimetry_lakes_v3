#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:39:27 2024

@author: jmaze
"""

# %% 1. Libraries and directories

import os
import geopandas as gpd

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3/')

# %% 2. Read matched lakes and GeoDAR convert to Azimuthal Equidistant

proj_crs = "+proj=aeqd +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"

matched_lakes = gpd.read_file('./data/lake_summaries/all_lakes.shp')
matched_lakes = matched_lakes.to_crs(proj_crs)

geodar = gpd.read_file('./data/geodar/GeoDAR_v11_dams.shp')
geodar = geodar.to_crs(proj_crs)

# %% 3. Apply buffer to GeoDAR

buffer_dist = 1000 #meters

geodar['buffered_geom'] = geodar.geometry.buffer(buffer_dist)
buffered_geodar = geodar.set_geometry('buffered_geom')
buffered_geodar.drop(
    columns=['OBJECTID', 'id_v11', 'id_v10', 'id_grd_v13', 'lat', 'lon', 
             'geo_mtd', 'qa_rank', 'val_scn', 'val_src', 'rv_mcm_v10', 
             'rv_mcm_v11', 'har_src', 'pnt_src', 'qc'
     ], 
    inplace=True
)

# %% 4. Filter matched lakes based on GeoDAR buffer

matched_lakes_dams = gpd.sjoin(matched_lakes, buffered_geodar, how='left', predicate='intersects')

no_dams = matched_lakes_dams[matched_lakes_dams['index_right'].isna()]

print(len(matched_lakes_dams) - len(no_dams))

no_dams = no_dams.drop(columns=['index_right', 'geometry_right']).rename(columns={
    'geometry_left': 'geometry'
})

# %% 5. Filter on perimeter threshold

perimeter_threshold = 250000 # meters = 250 km
z_std_threshold = 2 # meters

print(len(no_dams[no_dams['perim_m'] < perimeter_threshold]))
print(len(no_dams[no_dams['zstd_all_s'] < z_std_threshold]))

no_dams_clean = no_dams[
    (no_dams['perim_m'] < perimeter_threshold) & 
    (no_dams['zstd_all_s'] < z_std_threshold)
]

print(len(no_dams_clean), len(no_dams))

# %% 6. Write matched lakes (no dams, and thresholded)

# !!! Not sure why sjoin returned a dataframe not a geodataframe???
no_dams_clean = gpd.GeoDataFrame(no_dams_clean, geometry='geometry')
no_dams_clean = no_dams_clean.to_crs(epsg=4326)

no_dams_clean.to_file('./data/lake_summaries/matched_lakes_clean.shp')

# Write no_dams_ids as csv for faster filtering
cols = no_dams_clean.columns
drop_cols = cols[cols != 'lake_id']
clean_ids = no_dams_clean.drop(columns=drop_cols)
clean_ids.to_csv('./data/clean_ids.csv')










