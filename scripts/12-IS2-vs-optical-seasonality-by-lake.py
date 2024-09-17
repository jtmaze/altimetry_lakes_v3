# %% Libraries and directories

import os
import glob
import re
import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

import skimage as ski
from skimage import morphology
from skimage.measure import label, regionprops_table

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

is2_lakes = gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')
# The roi_name columnn doesn't sub-dived MRD, TUK and Anderson
is2_lakes.drop(columns=['roi_name'], inplace=True)
is2_timeseries_paths = glob.glob('./data/lake_timeseries/*_timeseries.csv')
is2_timeseries = [pd.read_csv(p) for p in is2_timeseries_paths]
is2_timeseries = pd.concat(is2_timeseries)
is2_timeseries.drop(columns=['Unnamed: 0', 'geometry'], inplace=True)

# %% Generate list of lakes with observations high enough for regression analysis

high_obs_lakes = is2_lakes[is2_lakes['obs_dates_'] >= 8]
high_obs_ids = high_obs_lakes['lake_id']
print(len(high_obs_ids))

high_obs_timeseries = is2_timeseries[is2_timeseries['lake_id'].isin(high_obs_ids)].copy()
high_obs_timeseries['obs_datetime'] = pd.to_datetime(high_obs_timeseries['obs_date'])
high_obs_timeseries['obs_month'] = high_obs_timeseries.obs_datetime.dt.month.astype(str)

months_check = high_obs_timeseries.groupby(by = 'lake_id')['obs_month'].agg(
        june_obs=lambda x: '6' in x.unique(),
        sep_obs=lambda x: '8' in x.unique(),
).reset_index()

aug_sep_months_check = months_check[
        (months_check['june_obs'] == True) & 
        (months_check['sep_obs'] == True)
]

print(len(aug_sep_months_check))

regression_lakes = is2_lakes[is2_lakes['lake_id'].isin(aug_sep_months_check['lake_id'])].copy()
regression_lakes = regression_lakes.drop(
        columns=[
                'zmed_all_s', 'zstd_all_s', 'z10_all_se', 'z90_all_se',
                'zrange_all', 'lat_mean_a', 'lon_mean_a', 'ref_area',
                'Shape_Leng', 'Shape_Area'
                ]
        )

def buffer_lakes_by_meters(geom, buffer_m):
        """Estimates each lake's local UTM, and then buffers by 120m
        lastly, converts back to EPSG:4326"""
        geom_series = gpd.GeoSeries([geom], crs='EPSG:4326')
        lake_crs_utm = geom_series.estimate_utm_crs()
        geom_series_utm = geom_series.to_crs(lake_crs_utm)
        buffered = geom_series_utm.buffer(buffer_m)
        buffered = buffered.to_crs(epsg=4326)

        return buffered.iloc[0]

regression_lakes['buffered'] = regression_lakes['geometry'].apply(buffer_lakes_by_meters, buffer_m=120)
regression_lakes = regression_lakes.rename(columns={'geometry': 'original_geom'})
regression_lakes = regression_lakes.set_geometry('buffered')

regression_lakes.drop(columns=['original_geom']).to_file('./data/lake_summaries/regression_lakes.shp', index=False)

# %% Rasterization function

def rasterize_individual_lakes(row, dataset):
        temp = gpd.GeoDataFrame(
                row.to_frame().T,
                geometry='buffered',
                crs='EPSG:4326'
        )

        box = temp.copy()
        box['box_buffer'] = temp['buffered'].apply(buffer_lakes_by_meters, buffer_m=300)
        box = box.set_geometry('box_buffer')
        bounds = box.total_bounds
        lake_id = temp['lake_id'].iloc[0]
        region = lake_id.split('_id_')[0]

        # Find source raster file
        src_pattern = f'./data/change_maps/MaskChange__{region}__{dataset}__matched_is2*.tif'
        src_path = glob.glob(src_pattern)[0]

        with rio.open(src_path) as src:
                window = from_bounds(*bounds, src.transform)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                crs = src.crs


        lake_rasterized = rio.features.rasterize(
                temp.geometry,
                out_shape=data.shape,
                fill=0,
                transform=transform,
                all_touched=True,
                default_value=1,
                dtype='uint8',
        )

        # Write the raster to a file
        if dataset == 'gswo':
                resolution = 30
        elif dataset == 'sentinel2':
                resolution = 10
        out_path =f'./data/regression_lakes_rasters/{lake_id}__resolution{resolution}.tif'

        with rio.open(
                out_path, 
                'w', 
                driver='GTiff', 
                width=data.shape[1], 
                height=data.shape[0], 
                count=1, 
                dtype='uint8', 
                crs=crs, 
                transform=transform
        ) as dst:
                dst.write(lake_rasterized, 1)


# %% Run the rasterization for both datasets

datasets = ['gswo', 'sentinel2']

for dataset in datasets:
        regression_lakes.apply(
                rasterize_individual_lakes, 
                axis=1, 
                dataset=dataset
        )

# %%

regression_lakes_paths = glob.glob('./data/regression_lakes_rasters/*.tif')
test_cases = regression_lakes_paths[:5] + regression_lakes_paths[-5:]

# %%

def parse_file_names(path):
        pattern = r'.*/(.*?)_id_(.*?)__resolution(.*).tif'
        match = re.match(
                pattern,
                path
        )

        region = match.group(1)
        id = match.group(2)
        resolution = match.group(3)
        return region, resolution, id

def read_data_and_masks(path):
    """
    Reads lake mask and change data for a given region and scope.

    Parameters:
        path (str): The file path to the lake mask raster.
        scope (str): The scope of the analysis (e.g., 'local', 'global').

    Returns:
        tuple: A tuple containing the lake mask, full water mask, and change data arrays.
    """

    region, resolution, id = parse_file_names(path)

    with rio.open(path) as src:
        lake_mask = src.read(1)
        bounds = src.bounds
        lake_crs = src.crs

    if resolution == '30':
        dataset = 'gswo'
        min_sieve_size = 10
    elif resolution == '10':
        dataset = 'sentinel2'
        min_sieve_size = 30
    
    change_map_pattern = f'./data/change_maps/MaskChange__{region}__{dataset}__matched_is2_*.tif'
    change_map_path = glob.glob(change_map_pattern)[0]

    with rio.open(change_map_path) as src:
        window = from_bounds(*bounds, src.transform)
        change_data = src.read(1, window=window)
        full_wtr_mask = np.where(np.isin(change_data, [1, 2, 3]), 1, 0).astype('uint8')
        window_transform = src.window_transform(window)
        change_crs = src.crs


    label_data = np.where((lake_mask == 1), full_wtr_mask, 0).astype('uint8')
    label_data = ski.morphology.remove_small_objects(
          label_data.astype('bool'), 
          min_size=min_sieve_size,
          connectivity=1
        ).astype('uint8')
    label_data_skimage = label(label_data)
    
#     test_path = f'./temp/{region}_{id}.tif'

#     out_meta = {
#         'driver': 'GTiff',
#         'height': label_data.shape[0],
#         'width': label_data.shape[1],
#         'count': 1,
#         'dtype': 'uint8', 
#         'dtype': label_data.dtype,
#         'crs': change_crs,
#         'transform': window_transform
#     }

#     with rio.open(test_path, 'w', **out_meta) as dest:
#         dest.write(label_data, 1)

    return label_data_skimage, change_data 

def make_props_table_largest_area(label_data, change_data):
        """
        Computes properties of labeled regions and counts specific pixel values within each region.

        Parameters:
                label_data (ndarray): Labeled image where each region is assigned a unique integer.
                change_data (ndarray): Intensity image containing pixel values to be counted within regions.

        Returns:
                pandas.Series: A Series containing properties of the largest region by area.
        """

        def count_wtr_pix(region, intensity_data): 
                cnt = np.sum(intensity_data[region] == 1)
                return cnt

        def count_increase_pix(region, intensity_data):
                cnt = np.sum(intensity_data[region] == 2)
                return cnt

        def count_decrease_pix(region, intensity_data):
                cnt = np.sum(intensity_data[region] == 3)
                return cnt

        props_tbl = regionprops_table(
                label_data,
                change_data,
                properties=('area', 'label'),
                extra_properties=(
                        count_wtr_pix,
                        count_increase_pix,
                        count_decrease_pix,
                )
        )
        
        df = pd.DataFrame(props_tbl)
        if df.empty:
                print('Empty DataFrame')
                dummy_series = pd.Series({
                        'area': np.nan,
                        'label': np.nan,
                        'count_wtr_pix': np.nan,
                        'count_increase_pix': np.nan,
                        'count_decrease_pix': np.nan
                })
                return dummy_series
        largest_area = df.sort_values(by='area', ascending=False).iloc[0]
        return largest_area

data = []

for i in regression_lakes_paths:
      label_data, change_data = read_data_and_masks(i)
      lake_data = make_props_table_largest_area(label_data, change_data)
      data.append(lake_data)

# %%
area_results = pd.DataFrame(data)

area_results['total_seasonal_fraction'] = area_results['count_increase_pix'] / area_results['area'] * 100
# %%
