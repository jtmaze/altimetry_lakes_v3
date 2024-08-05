# %% Libraries and set-up

# pip install -U ipykernel

project_dir = 'projects/alpod-412314/assets/' 

region = 'AKCP'

import ee
import os
import geemap
import pprint as pp
#from shapely.geometry import shape
import geopandas as gpd

ee.Authenticate()
ee.Initialize()

# %% Directories and folders

weekly_mosaics = ee.data.listAssets(project_dir + 'region_weekly')

pp.pp(ee.data.listAssets(project_dir))

# %% Import a random image to build workflow
img = ee.Image('projects/alpod-412314/assets/region_weekly/AKCP_2016_29')
lakes_path = f'{project_dir}Lake_extractions/{region}_extraction'
lake_polygons = ee.FeatureCollection(lakes_path)

pp.pp(img.getInfo())
pp.pp(lake_polygons.first().getInfo())

# %% Make masks for all conditions
obs_mask = img.unmask(0)
obs_mask = obs_mask.rename('observed')
#print(obs_mask.getInfo())

img_wtr = ee.Image.constant(0)
img_wtr = img_wtr.where(img.select('water_occurance').eq(1), 2)

img_land_ice_cloud = ee.Image.constant(0)
img_land_ice_cloud = img_land_ice_cloud.where(img.select('water_occurance').eq(0), 1)
