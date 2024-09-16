# %% Libraries and directories

import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

is2_lakes = gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')
# The roi_name columnn doesn't sub-dived MRD, TUK and Anderson
is2_lakes.drop(columns=['roi_name'], inplace=True)
is2_timeseries_paths = glob.glob('./data/lake_timeseries/*_timeseries.csv')
is2_timeseries = [pd.read_csv(p) for p in is2_timeseries_paths]
is2_timeseries = pd.concat(is2_timeseries)
is2_timeseries.drop(columns=['Unnamed: 0', 'geometry'], inplace=True)

# %% Generate list of lakes with observations high enough for regression analysis

high_obs_lakes = is2_lakes[is2_lakes['obs_dates_'] >= 10]
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
