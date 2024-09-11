# %% 1.0 Libraries and data

import os
import re
import glob
from collections import defaultdict
import numpy as np
import rasterio as rio


os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

masked_raster_paths = glob.glob('./data/masked_rasters/*.tif')

# %%

def parse_file_names(file_path):

    match = re.match(
        r'.*scope_(.*?)__roi_(.*?)__timeframe_(.*?)__dataset_(.*?)__band(\d)\.tif',
        file_path
    )
    if match:
        scope = match.group(1)
        roi = match.group(2)
        timeframe = match.group(3)
        dataset = match.group(4)
        band = match.group(5)

        return scope, roi, timeframe, dataset, band

    # return None


# Using the defaultdict becuase it automatically initializes keys for appending
raster_change_dict = defaultdict(list)

for p in masked_raster_paths:
    scope, roi, timeframe, dataset, band = parse_file_names(p)
    key = (scope, roi, dataset, band)
    # Append the timeframe and file path as a tuple with corresponding key in dictionary
    raster_change_dict[key].append((timeframe, p))

# %% Run change detection

for key, rasters in raster_change_dict.items():
    scope, roi, dataset, band = key
    sorted_rasters = sorted(rasters)  # sorted by timeframe

    # We will process the rasters two at a time (pairwise)
    for i in range(len(sorted_rasters) - 1):
        # Get the current raster and the next one for comparison
        (timeframe1, path1) = sorted_rasters[i]
        (timeframe2, path2) = sorted_rasters[i + 1]

        with rio.open(path1) as src1, rio.open(path2) as src2:

            data1 = src1.read(1)
            data2 = src2.read(1)

            mask = (data1 == -1) | (data2 == -1)
            change = data2 - data1
            change = np.ma.masked_array(change, mask=mask)

            meta = src1.meta
            output_path = f'./data/change_maps/{roi}__{dataset}__{scope}__change__{timeframe1}_to_{timeframe2}_band{band}.tif'

            with rio.open(output_path, 'w', **meta) as dst:
                dst.write(change.filled(-200), 1)



# %%
