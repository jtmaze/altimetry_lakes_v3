# %% 1.0 Libraries and data
import os
import re
import glob
from collections import defaultdict
import numpy as np
import rasterio as rio


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

    # return None


# Using the defaultdict becuase it automatically initializes keys for appending
raster_change_dict = defaultdict(list)

for p in masked_raster_paths:
    scope, roi, timeframe, dataset, buffer = parse_file_names(p)
    key = (scope, roi, dataset, buffer)
    # Append the timeframe and file path as a tuple with corresponding key in dictionary
    raster_change_dict[key].append((timeframe, p))

# %% 3.0 Change detection by simply subtracting pixel values. 

def simple_raster_subtraction(
        src_early, src_late, roi, dataset, scope, timeframe1, 
        timeframe2, buffer, write_file=False
):
    """
    Perform raster subtraction between two timeframes and optionally
    write the result to a file.
    """
    data_early = src_early.read(1)
    data_late = src_late.read(1)
    mask = (data_early == -1) | (data_late == -1)
    change_data = data_late - data_early
    change_data = np.ma.masked_array(change_data, mask=mask, fill_value=-200)

    if write_file:
        output_path = f'./data/change_maps/RasterChange__{roi}__{dataset}__{scope}__change__{timeframe1}_to_{timeframe2}_buffer{buffer}.tif'

        with rio.open(output_path, 'w', **src_early.meta) as dst:
            dst.write(change_data, 1)

        print(src_early.meta)

    return change_data

def sort_gswo_key(raster):
    """
    Sort function for GSWO rasters based on month names.
    """
    timeframe, _ = raster
    timeframe_map = {'june': 6, 'aug': 8}
    return timeframe_map.get(timeframe, timeframe)

### Iterate throught the files. 

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

        with rio.open(path1) as src1, rio.open(path2) as src2:

            change_data = simple_raster_subtraction(
                src_early=src1,
                src_late=src2, 
                roi=roi, 
                dataset=dataset, 
                scope=scope, 
                timeframe1=timeframe1, 
                timeframe2=timeframe2, 
                buffer=buffer, 
                write_file=True
            ) 
                                                    


def change_by_thresholds():            

# %% 4.0 Change detection via thresholds and classifacation
