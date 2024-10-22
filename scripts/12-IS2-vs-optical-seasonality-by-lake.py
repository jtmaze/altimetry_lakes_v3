# %% 1.0 Libraries and directories

import os
import glob
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio as rio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

import skimage as ski
from skimage import morphology
from skimage.measure import label, regionprops_table

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')


# %% 1.1 Load ICESat-2 data

is2_lakes = gpd.read_file('./data/lake_summaries/matched_lakes_clean.shp')
# The roi_name columnn doesn't sub-dived MRD, TUK and Anderson
is2_lakes.drop(columns=['roi_name'], inplace=True)
is2_timeseries_paths = glob.glob('./data/lake_timeseries/*_timeseries.csv')
is2_timeseries = [pd.read_csv(p) for p in is2_timeseries_paths]
is2_timeseries = pd.concat(is2_timeseries)
is2_timeseries.drop(columns=['geometry'], inplace=True)

lake_ids_clean_path = './data/clean_ids.csv'
lake_ids_clean = pd.read_csv(lake_ids_clean_path)['lake_id']

is2_lakes = is2_lakes[is2_lakes['lake_id'].isin(lake_ids_clean)]
is2_timeseries = is2_timeseries[is2_timeseries['lake_id'].isin(lake_ids_clean)]


# %% 2.0 Generate list of lakes with observations high enough for regression analysis
high_obs_lakes = is2_lakes[is2_lakes['obs_dates_'] >= 10]
high_obs_ids = high_obs_lakes['lake_id']
print(len(high_obs_ids))

high_obs_timeseries = is2_timeseries[is2_timeseries['lake_id'].isin(high_obs_ids)].copy()

# %% 2.1 Check for lakes with observations in June and August

high_obs_timeseries['obs_datetime'] = pd.to_datetime(high_obs_timeseries['obs_date'])
high_obs_timeseries['obs_month'] = high_obs_timeseries.obs_datetime.dt.month.astype(str)

months_check = high_obs_timeseries.groupby(by='lake_id')['obs_month'].agg(
        june_obs=lambda x: '6' in x.unique(),
        sep_obs=lambda x: '8' in x.unique(),
).reset_index()

aug_sep_months_check = months_check[
        (months_check['june_obs'] == True) & 
        (months_check['sep_obs'] == True)
]

print(len(aug_sep_months_check))

# %% 2.2 Select with greater than 8 observations and June / August observations

regression_lakes = is2_lakes[
        (is2_lakes['lake_id'].isin(aug_sep_months_check['lake_id'])) &
        (is2_lakes['lake_id'].isin(high_obs_ids))
].copy()
print(len(regression_lakes))

regression_lakes = regression_lakes.drop(
        columns=[
                'zmed_all_s', 'zstd_all_s', 'z10_all_se', 'z90_all_se',
                'zrange_all', 'lat_mean_a', 'lon_mean_a', 'ref_area',
                'Shape_Leng', 'Shape_Area'
                ]
        )

regression_timeseries = is2_timeseries[
        (is2_timeseries['lake_id'].isin(aug_sep_months_check['lake_id'])) &
        (is2_timeseries['lake_id'].isin(high_obs_ids))
].copy()
print(len(regression_timeseries))

regression_timeseries['obs_datetime'] = pd.to_datetime(regression_timeseries['obs_date'])
regression_timeseries['obs_month'] = regression_timeseries.obs_datetime.dt.month.astype(str)

# %% 2.3 Buffer the regression lakes, and write to file

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

# %% 3.0 Rasterize the high observation lakes for regression analysis
# %% 3.1 Define function to rasterize individual lakes

def rasterize_individual_lakes(row, satellite):
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
        if satellite == 'landsat':
                dataset = 'gswo'
        elif satellite == 'sentinel2':
                dataset = 'sentinel2'

        src_pattern = f'./data/change_maps/MaskChange__{region}__{dataset}__*.tif'
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
        if satellite in ['landsat', 'gswo']:
                resolution = 30
        elif satellite == 'sentinel2':
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

# %% 3.2 Run the rasterization for both satellites (10m and 30m resolution)

satellites = ['landsat', 'sentinel2']

for s in satellites:
        regression_lakes.apply(
                rasterize_individual_lakes, 
                axis=1, 
                satellite=s
        )

# %% 4.0 Generate optical seasonality table for the high observation lakes

# %% 4.1 Get file paths for the rasterized lakes

regression_lakes_paths = glob.glob('./data/regression_lakes_rasters/*.tif')

# %% 4.2 Define function to read, mask, and calculate properties of the largest region

def parse_file_names(path):
        pattern = r'.*/(.*?)_id_(.*?)__resolution(.*).tif'
        match = re.match(
                pattern,
                path
        )

        region = match.group(1)
        lake_id = match.group(2)
        resolution = match.group(3)
        return region, resolution, lake_id

def read_data_and_masks(path, dataset):
    """
    Reads lake mask and change data for a given region and scope.

    Parameters:
        path (str): The file path to the lake mask raster.
        scope (str): The scope of the analysis (e.g., 'local', 'global').

    Returns:
        tuple: A tuple containing the lake mask, full water mask, and change data arrays.
    """

    region, resolution, lake_id = parse_file_names(path)

    if resolution == '10' and dataset in ['gswo', 'landsat']:
        return None, None, None, None
    if resolution == '30' and dataset == 'sentinel2':
        return None, None, None, None
    
    else:

        if resolution == '30':
                min_sieve_size = 10
        elif resolution == '10':
               min_sieve_size = 30

        with rio.open(path) as src:
                lake_mask = src.read(1)
                bounds = src.bounds
                lake_crs = src.crs

        change_map_pattern = f'./data/change_maps/MaskChange__{region}__{dataset}__*.tif'
        change_map_path = glob.glob(change_map_pattern)[0]

        with rio.open(change_map_path) as src:
                window = from_bounds(*bounds, src.transform)
                change_data = src.read(1, window=window)
                full_wtr_mask = np.where(np.isin(change_data, [1, 2, 3]), 1, 0).astype('uint8')
                window_transform = src.window_transform(window)
                change_crs = src.crs

        if lake_mask.shape != full_wtr_mask.shape:
               print('Shapes do not match')
               return None, None, None, None
        
        else:
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

        return label_data_skimage, change_data,  region, lake_id


def make_props_table_largest_area(label_data, change_data):
        """
        Computes properties of labeled regions and counts specific pixel values within each region.

        Parameters:
                label_data (ndarray): Labeled image where each region is assigned a unique integer.
                change_data (ndarray): Intensity image containing pixel values to be counted within regions.

        Returns:
                pandas.Series: A Series containing properties of the largest region by area.
        """

        if label_data is None or change_data is None:
                print('No data')
                dummy_series = pd.Series({
                        'area': np.nan,
                        'label': np.nan,
                        'count_wtr_pix': np.nan,
                        'count_increase_pix': np.nan,
                        'count_decrease_pix': np.nan
                })
                return dummy_series
        
        else:

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

# %% 4.3 Generate the optical seasonality table

data = []
datasets = ['gswo', 'sentinel2', 'glad']

for d in datasets:
        for i in regression_lakes_paths:
                label_data, change_data, region, lake_id = read_data_and_masks(i, d)
                lake_data = make_props_table_largest_area(label_data, change_data)
                lake_data['dataset'] = d
                lake_data['region'] = region
                lake_data['lake_id'] = lake_id
                data.append(lake_data)

# %% 4.4 Calculate % seasonal change from pixel counts

area_results = pd.DataFrame(data)
area_results.drop(columns=['label'], inplace=True)
area_results = area_results.dropna()

area_results['total_seasonal_decrease'] = (
        area_results['count_decrease_pix'] / 
        area_results['area'] * 
        100
)

area_results['total_seasonal_increase'] = (
        area_results['count_increase_pix'] /
        area_results['area'] *
        100
)

area_results['net_seasonal_change'] = (
        area_results['total_seasonal_increase'] - 
        area_results['total_seasonal_decrease']
)

area_results['lake_id_full'] = area_results['region'] + '_id_' + area_results['lake_id'].astype(str)
area_results = area_results.drop(columns=['lake_id'])

area_results.to_csv('./data/high_obs_lake_area_results.csv', index=False)

# %% 5.0 Calculate ICEsat-2 seasonal change by lake

icesat2_seasonal_change = regression_timeseries[
        regression_timeseries['obs_month'].isin(['6', '8'])
]

icesat2_seasonal_change = icesat2_seasonal_change[
        (icesat2_seasonal_change['zdif_date'] > -5) &
        (icesat2_seasonal_change['zdif_date'] < 5)
]

icesat2_seasonal_change = icesat2_seasonal_change.groupby(
        by=['lake_id', 'obs_month']
).agg(
        zdif_mean=('zdif_date', 'mean')
).reset_index()

icesat2_seasonal_change = icesat2_seasonal_change.pivot_table(
        index='lake_id',
        columns='obs_month',
        values='zdif_mean'
).reset_index()

icesat2_seasonal_change['seasonal_change_icesat2'] = (
        icesat2_seasonal_change['8'] - icesat2_seasonal_change['6']
)

icesat2_seasonal_change = icesat2_seasonal_change.drop(columns=['6', '8'])

# 6.0 Merge the optical and ICESat-2 seasonal change tables

seasonal_change = pd.merge(
        area_results,
        icesat2_seasonal_change,
        left_on='lake_id_full',
        right_on='lake_id',
        how='outer'
)

seasonal_change = seasonal_change.drop(
        columns=['area', 'count_wtr_pix', 'count_increase_pix', 
        'count_decrease_pix', 'lake_id_full'])


# %% 6.0 Plot the seasonal change comparison for ICESat-2 and Optical Data

plot_df = seasonal_change.dropna(subset=['seasonal_change_icesat2', 'net_seasonal_change'])
plot_df = plot_df[(plot_df['seasonal_change_icesat2'] > -1) & (plot_df['seasonal_change_icesat2'] < 1)]
plot_df = plot_df[(plot_df['net_seasonal_change'] < 30) & (plot_df['net_seasonal_change'] > -30)]

datasets = plot_df['dataset'].unique()

colors = plt.cm.get_cmap('tab10', len(datasets))

# Create a dictionary to map datasets to colors
color_dict = {dataset: colors(i) for i, dataset in enumerate(datasets)}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each dataset separately
for dataset in datasets:
    subset = plot_df[plot_df['dataset'] == dataset]
    x = subset['seasonal_change_icesat2'].values
    y = subset['net_seasonal_change'].values
    
    # Scatter plot
    ax.scatter(
        x,
        y,
        label=dataset,
        color=color_dict[dataset],
        alpha=0.7,
        edgecolors='w',
        s=50
    )
    
    # Fit linear regression
    if len(x) > 1:
        coefficients = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coefficients)
        
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = poly1d_fn(x_line)
        
        ax.plot(
            x_line,
            y_line,
            color=color_dict[dataset],
            linestyle='--',
            linewidth=2,
            label=f'{dataset} Trendline'
        )

ax.set_xlabel('Lake WSE Change (meters)', fontsize=12)
ax.set_ylabel('Lake Area Change (%)', fontsize=12)
ax.set_title('Seasonality Comparison by Lake', fontsize=14)
ax.legend(title='Dataset and Trendlines')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# %%

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

regions = plot_df['region'].unique()

# Iterate over regions and their corresponding axes
for ax, region in zip(axes.flat, regions):
    # Filter the data for the current region
    region_data = plot_df[plot_df['region'] == region]
    
    # Plot each dataset within the current region
    for dataset in datasets:
        subset = region_data[region_data['dataset'] == dataset]
        ax.scatter(
            subset['seasonal_change_icesat2'],
            subset['net_seasonal_change'],
            label=dataset,
            color=color_dict[dataset],
            alpha=0.7,
            edgecolors='w',
            s=50
        )

        x = subset['seasonal_change_icesat2'].values
        y = subset['net_seasonal_change'].values
        if len(x) > 1:
                coefficients = np.polyfit(x, y, 1)
                poly1d_fn = np.poly1d(coefficients)
                
                x_line = np.linspace(np.min(x), np.max(x), 100)
                y_line = poly1d_fn(x_line)
                
                ax.plot(
                x_line,
                y_line,
                color=color_dict[dataset],
                linestyle='--',
                linewidth=2,
                label=f'{dataset} Trendline'
                )
    
    # Customize the subplot
    ax.set_title(region)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('Lake WSE Change (meters)', fontsize=10)
    ax.set_ylabel('Lake Area Change (%)', fontsize=10)

# Add a global legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title='Dataset', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(datasets))

# Adjust layout
plt.tight_layout()
plt.show()

# %%

datasets = plot_df['dataset'].unique()

colors = plt.cm.get_cmap('tab10', len(datasets))

# Create a dictionary to map datasets to colors
color_dict = {dataset: colors(i) for i, dataset in enumerate(datasets)}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each dataset separately
for dataset in datasets:
    subset = plot_df[plot_df['dataset'] == dataset]
    x = subset['seasonal_change_icesat2'].values
    y = subset['net_seasonal_change'].values
    
    # Scatter plot
    ax.scatter(
        x,
        y,
        label=dataset,
        color=color_dict[dataset],
        alpha=0.7,
        edgecolors='w',
        s=50
    )
    
    # Fit linear regression if there are enough data points
    if len(x) > 1:
        coefficients = np.polyfit(x, y, 1)
        poly1d_fn = np.poly1d(coefficients)
        
        # Generate x values for plotting the trendline
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = poly1d_fn(x_line)
        
        # Plot the trendline
        ax.plot(
            x_line,
            y_line,
            color=color_dict[dataset],
            linestyle='--',
            linewidth=2,
            label=f'{dataset} Trendline'
        )

# Customize the plot
ax.set_xlabel('Seasonal Change ICESat-2 WSE (m)', fontsize=12)
ax.set_ylabel('Net Seasonal Change Area', fontsize=12)
ax.set_title('Seasonality Comparison by Lake', fontsize=14)
ax.legend(title='Dataset and Trendlines')
ax.grid(True, linestyle='--', alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()
# %%

# %%
