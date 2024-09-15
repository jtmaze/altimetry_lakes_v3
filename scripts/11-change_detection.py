# %% 1.0 Libraries and data
import os
import re
import glob
from collections import defaultdict
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio as rio
import pprint as pp

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

masked_raster_paths = glob.glob('./data/masked_rasters/*.tif')

# %% 2.0 Get file information and make dictionary to contrast time perioeds. 

def parse_file_names(file_path):

    match = re.match(
        r'.*scope_(.*?)__roi_(.*?)__timeframe_(.*?)__dataset_(.*?)__buffer(\d+)\.tif',
        file_path
    )
    if match:
        scope = match.group(1)
        roi = match.group(2)
        timeframe = match.group(3)
        dataset = match.group(4)
        buffer = match.group(5)

        return scope, roi, timeframe, dataset, buffer


# Using the defaultdict becuase it automatically initializes keys for appending
raster_change_dict = defaultdict(list)

for p in masked_raster_paths:
    scope, roi, timeframe, dataset, buffer = parse_file_names(p)
    key = (scope, roi, dataset, buffer)
    # Append the timeframe and file path as a tuple with corresponding key in dictionary
    raster_change_dict[key].append((timeframe, p))

# %% 3.0 Change detection by simply subtracting pixel values. 

def sort_gswo_key(raster):
    """
    Sort function for GSWO rasters based on month names.
    """
    timeframe, _ = raster
    timeframe_map = {'june': 6, 'aug': 8}
    return timeframe_map.get(timeframe, timeframe)

# def simple_raster_subtraction(
#         src_early, src_late, roi, dataset, scope, timeframe1, 
#         timeframe2, buffer, write_file=False
# ):
#     """
#     Perform raster subtraction between two timeframes and optionally
#     write the result to a file.
#     """
#     if write_file:
#         data_early = src_early.read(1)
#         data_late = src_late.read(1)
#         mask = (data_early == -1) | (data_late == -1)
#         change_data = data_late - data_early
#         change_data = np.ma.masked_array(change_data, mask=mask, fill_value=-200)

#         img_meta = src_early.meta

#         output_path = f'./data/change_maps/RasterChange__{roi}__{dataset}__{scope}__change__{timeframe1}_to_{timeframe2}_buffer{buffer}.tif'

#         with rio.open(output_path, 'w', **meta) as dst:
#             dst.write(change_data, 1)
#         #return data_early, data_late, img_meta

def simple_raster_subtraction_rxr(
        src_early_path, src_late_path, roi, dataset, scope, timeframe1,
        timeframe2, buffer, write_file=False
):

    if write_file:
        data_early = rxr.open_rasterio(src_early_path, chunks='auto').astype('int16')
        data_late = rxr.open_rasterio(src_late_path, chunks='auto').astype('int16')

        change_data = data_late - data_early
        change_data = change_data.where((data_early != -1) & (data_late != -1), -110)
        change_data.rio.write_nodata(-110, inplace=True)
        output_path = f'./data/change_maps/RasterChange__{roi}__{dataset}__{scope}__change__{timeframe1}_to_{timeframe2}_buffer{buffer}.tif'
        print(f'processing {output_path}')
        change_data.rio.to_raster(output_path)

def change_by_water_mask(
        src_early_path, src_late_path, threshold, roi, 
        dataset, scope, timeframe1, timeframe2, 
        buffer, write_file=False
):

    if write_file:
        with rio.open(src_early_path) as src_early, rio.open(src_late_path) as src_late:
            data_early = rxr.open_rasterio(src_early_path, chunks='auto').astype('int8')
            data_late = rxr.open_rasterio(src_late_path, chunks='auto').astype('int8')
            data_early_crs = data_early.rio.crs
            print(data_early_crs)
            lake_mask_early = xr.where(data_early >= threshold, 1, 0)
            lake_mask_late = xr.where(data_late >= threshold, 1, 0)
            na_mask = (data_early == -1) | (data_late == -1)
            # Clean up memory by deleting
            del data_early, data_late

               
            change_classified = xr.full_like(lake_mask_early, 0, dtype='uint8')
            
            change_classified = xr.where((lake_mask_early == 1) & (lake_mask_late == 1), 1, change_classified) # 1 = water, no change
            change_classified = xr.where((lake_mask_early == 0) & (lake_mask_late == 1), 2, change_classified) # 2 = water, seasonal increase
            change_classified = xr.where((lake_mask_early == 1) & (lake_mask_late == 0), 3, change_classified) # 3 = water, seasonal decrease
            change_classified = xr.where((lake_mask_early == 0) & (lake_mask_late == 0), 4, change_classified) # 4 = land, no change
            change_classified = change_classified.where(~na_mask)

            change_classified.rio.write_crs(data_early_crs, inplace=True)

            del lake_mask_early, lake_mask_late, na_mask

            output_path = f'./data/change_maps/MaskChange__{roi}__{dataset}__{scope}__change__{timeframe1}_to_{timeframe2}_buffer{buffer}.tif'
            print(f'writing file to {output_path}')
            change_classified.rio.to_raster(output_path)
            

# %% 

for key, rasters in raster_change_dict.items():
    scope, roi, dataset, buffer = key
    # Sort rasters into proper order before change detection
    if dataset == 'gswo':        
        sorted_rasters = sorted(rasters, key=sort_gswo_key)
    else:   
        sorted_rasters = sorted(rasters)  # sorted by timeframe

    # We will process the rasters two at a time (pairwise)
    for i in range(len(sorted_rasters) - 1):
        # Get the current raster and the next one for comparison
        (timeframe1, path1) = sorted_rasters[i]
        (timeframe2, path2) = sorted_rasters[i + 1]

        print(roi)

        simple_raster_subtraction_rxr(
            src_early_path=path1,
            src_late_path=path2, 
            roi=roi, 
            dataset=dataset, 
            scope=scope, 
            timeframe1=timeframe1, 
            timeframe2=timeframe2, 
            buffer=buffer, 
            write_file=False
        )

        change_by_water_mask(
            src_early_path=path1,
            src_late_path=path2, 
            threshold=80,
            roi=roi,
            dataset=dataset,
            scope=scope,
            timeframe1=timeframe1,
            timeframe2=timeframe2,
            buffer=buffer,
            write_file=True
        )
                            

 # %% 4.0 Change detection via thresholds and classifacation
