# %% 1.0 Libaries and paths

import glob 
import os
import re
import pandas as pd
import rasterio as rio
import rasterio.mask
import geopandas as gpd

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

sub_roi_names = ['MRD', 'TUK', 'Anderson']
sub_roi_paths = [f'./data/ew_rois/{n}_roi_shape.shp' for n in sub_roi_names]
sub_rois = [gpd.read_file(p) for p in sub_roi_paths]
sub_rois_dict = dict(zip(sub_roi_names, sub_rois))

# %% Split the IS2 Timeseries and lakes by sub roi

all_gdf = gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')
all_timeseries = pd.read_csv('./data/lake_timeseries/MRD_TUK_Anderson_timeseries.csv')

for name, gdf in sub_rois_dict.items():
    clipped_lakes = gpd.clip(all_gdf, gdf)
    clipped_lakes.to_csv(f'./data/lake_summaries/{name}_lakesummary.csv')

    geoms_with_ids = clipped_lakes[['lake_id', 'geometry']]
    clipped_timeseries = pd.merge(
        all_timeseries,
        geoms_with_ids,
        on='lake_id',
        how='inner'
    )

    clipped_timeseries.to_csv(f'./data/lake_timeseries/{name}_timeseries.csv')
    print(f'{name} lake shapes and timeseries clipped')

# %% Clip the reccurence clean files
def extract_unique(files, pattern):
    unique_items = set()
    for file in files:
        match = re.search(pattern, file)
        if match:
            unique_items.add(file)
    return list(unique_items)

# %% 

recurrence_files = glob.glob(f'./data/recurrence_clean/*')
sub_roi_pattern = r'Recurrence_MRD_TUK_Anderson_(.*)_dataset_sentinel2.tif'

sub_roi_files = extract_unique(recurrence_files, sub_roi_pattern)

for f in sub_roi_files:
    for name, gdf in sub_rois_dict.items():
        geom = [gdf.iloc[0].geometry]

        with rio.open(f) as src:
            print(f'CRS CHECK {gdf.crs}, {src.crs}')

            out_img, out_trans = rio.mask.mask(src, geom, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                'driver':'Gtiff',
                'height': out_img.shape[1],
                'width': out_img.shape[2],
                'transform': out_trans
            })
            print(out_meta)

        roi_pattern = r'Recurrence_(.*)_timeframe'
        out_path = re.sub(r'(Recurrence_)(.*?)(_timeframe)',rf'\1{name}\3' , f)
        print(out_path)

        with rio.open(out_path, 'w', **out_meta) as dst:
            dst.write(out_img)


# %% Clip the rasterized lake files

rasterized_lake_files = glob.glob(f'./data/lake_summaries/*')
sub_roi_pattern = r'.*_MRD_TUK_Anderson_rasterized_buffers.tif'
sub_roi_files = extract_unique(rasterized_lake_files, sub_roi_pattern)

for f in sub_roi_files:
    for name, gdf in sub_rois_dict.items():
        geom = [gdf.iloc[0].geometry]

        with rio.open(f) as src:
            print(f'CRS CHECK {gdf.crs}, {src.crs}')

            out_img, out_trans = rio.mask.mask(src, geom, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                'driver':'Gtiff',
                'height': out_img.shape[1],
                'width': out_img.shape[2],
                'transform': out_trans
            })
            print(out_meta)

        roi_pattern = r'(gswo|sentinel2)_(.*?)(_rasterized_buffers.tif)'
        out_path = re.sub(roi_pattern, rf'\1_{name}\3', f)
        print(out_path)

        with rio.open(out_path, 'w', **out_meta) as dst:
            dst.write(out_img)

# %%
