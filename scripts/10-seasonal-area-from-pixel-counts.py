# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

pixels_path = './data/pixel_counts.csv'
pixels_summary = pd.read_csv(pixels_path)

# %% 2.0 Clean up the pixel counts datasets

# !!! A few pixels were oddly above the max value of 100, no clue why this happened.
# Rare enough % to not matter. 
over_max = np.sum(
    np.where(pixels_summary['pix_vals'] > 100, pixels_summary['pix_cnts'], 0)
)
valid = np.sum(
    np.where(pixels_summary['pix_vals'] < 100, pixels_summary['pix_cnts'], 0)
)

print(f'{(over_max / (over_max + valid) * 100):.10f} % of pixels over max')

# Swap values > 100 with 100
pixels_clean = pixels_summary.copy()
pixels_clean['pix_vals'] = np.where(
    pixels_clean['pix_vals'] > 100, 100, pixels_clean['pix_vals']
)
# Recalculate the total pixel counts after swapping the values
pixels_clean = pixels_clean.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer', 'pix_vals', 'month']
).agg(
    pix_cnt_clean=('pix_cnts', 'sum')
).reset_index()

del over_max, valid, pixels_summary

# %% 3.0 Calculate total pixels in the image & number measured within masks.

total_pixels = pixels_clean.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer', 'month']
).agg(
    total_pix=('pix_cnt_clean', 'sum')
).reset_index()

only_measured = pixels_clean[pixels_clean['pix_vals'] != -1]
total_measured = only_measured.groupby(
    by=['roi_name', 'timeframe', 'buffer', 'dataset', 'scope', 'month']
).agg(
    cnt_measured=('pix_cnt_clean', 'sum')
).reset_index()

obs_info = total_pixels.merge(
    total_measured,
    how='inner',
    on=['roi_name', 'timeframe', 'buffer', 'dataset', 'scope', 'month']
)

obs_info['pixel_frac_measured'] = (
    obs_info['cnt_measured'] / obs_info['total_pix'] * 100
)
# Merge information on the total pixels and the number of measured pixels
pixels_clean = pixels_clean.merge(
    obs_info,
    how='left',
    on=['roi_name', 'dataset', 'buffer', 'timeframe', 'scope', 'month']
)

#del obs_info, total_measured, total_pixels, only_measured

# %% 4.0 Calculate the water and land fractions by threshold


# %% 4.1 Functions summing the water and land fractions by threshold

def count_above(df, threshold):
    above_series = df['pix_cnt_clean'][df['pix_vals'] >= threshold]
    cnt_above = above_series.sum()

    return cnt_above

def count_below(df, threshold):
    below_series = df['pix_cnt_clean'][(df['pix_vals'] < threshold) & (df['pix_vals'] > -1)]
    cnt_below = below_series.sum()

    return cnt_below

# %% 4.1 Iterate over thresholds, rois, buffers, datasets, timeframes, and scopes

timeframes = pixels_clean['timeframe'].unique().tolist()
roi_names = pixels_clean['roi_name'].unique().tolist()
buffers = pixels_clean['buffer'].unique().tolist()
datasets = pixels_clean['dataset'].unique().tolist()
scopes = pixels_clean['scope'].unique().tolist()
months = pixels_clean['month'].unique().tolist()
thresholds = np.arange(start=70, stop=90, step=2).tolist()

results = []

for roi_name in roi_names:
    for dataset in datasets:
        for scope in scopes:
            for timeframe in timeframes:
                for buffer in buffers: 
                    for threshold in thresholds:
                        for month in months:
                  
                            temp = pixels_clean[pixels_clean['roi_name'] == roi_name]
                            temp = temp[temp['timeframe'] == timeframe]
                            temp = temp[temp['buffer'] == buffer]
                            temp = temp[temp['dataset'] == dataset]
                            temp = temp[temp['scope'] == scope]
                            temp = temp[temp['month'] == month]
                            
                            # Sum all the water/land pixels in the raster.
                            wtr_cnt = count_above(temp, threshold)
                            land_cnt = count_below(temp, threshold)

                            if wtr_cnt == 0 and land_cnt == 0:
                                print(f"""No data for: 
                                    {roi_name}, {dataset}, {timeframe}, {month}""")
                                continue

                            else:
                                # Should only be one unique value for each of these
                                measured_pix = temp['cnt_measured'].unique()
                                total_pix = temp['total_pix'].unique()
                                
                                data = {
                                    'wtr_pix_sum': wtr_cnt,
                                    'land_pix_sum': land_cnt,
                                    'total_pix': total_pix,
                                    'measured_pix': measured_pix,
                                    'roi_name': roi_name,
                                    'timeframe': timeframe,
                                    'month': month,
                                    'buffer': buffer,
                                    'dataset': dataset,
                                    'scope': scope,
                                    'threshold': threshold
                                }
                                
                                data = pd.DataFrame(data)                          
                                results.append(data)


full_results = pd.concat(results, ignore_index=True)
# %% %% 5.0 Calculate differences across time periods

seasonal_results = pd.pivot_table(
    data=full_results,
    index=['roi_name', 'dataset', 'scope', 'buffer', 'threshold', 'timeframe'],
    columns='month',
    values=['wtr_pix_sum', 'land_pix_sum', 'total_pix', 'measured_pix']
).reset_index()

seasonal_results['net_total_sn_frac'] = (
    (seasonal_results[('wtr_pix_sum', 'late')] - 
     seasonal_results[('wtr_pix_sum', 'early')]) / 
    seasonal_results[('total_pix', 'early')] * 100
)

seasonal_results['net_lake_sn_frac'] = (
    (seasonal_results[('wtr_pix_sum', 'late')] - 
     seasonal_results[('wtr_pix_sum', 'early')]) / 
    seasonal_results[('wtr_pix_sum', 'early')] * 100
)


seasonal_results.columns = ['_'.join(filter(None, col)) for col in seasonal_results.columns.values]

seasonal_results.to_csv('./data/area_seasonal_results.csv', index=False)

# %%
# def by_roi_sensitivity_viz(df, roi_name, y_var):
    
#     temp = df.copy()
#     temp.columns = temp.columns.droplevel(1)

#     temp['threshold_int'] = temp['threshold'].astype(int)
#     temp['buffer_int'] = temp['buffer'].astype(int)
    
#     keep_cols = ['roi_name', 'dataset', 'scope', 'buffer_int', 'threshold_int','buffer', y_var] 
#     drop_cols = [col for col in temp.columns if col not in keep_cols]
#     temp.drop(columns=drop_cols, inplace=True)
    
#     temp = temp[temp['roi_name'] == roi_name]
    
#     unique_buffers = temp['buffer'].unique()
#     colors = plt.cm.viridis(np.linspace(0, 1.1, len(unique_buffers)))
#     color_map = dict(zip(unique_buffers, colors))
    
#     # Get the unique datasets
#     unique_datasets = temp['dataset'].unique()
#     ## Incorperate unique 'scopes in plotting
#     unique_scopes = temp['scope'].unique()
    
#     # Create subplots
#     fig, axes = plt.subplots(len(unique_scopes), 
#                              len(unique_datasets), 
#                              figsize=(15, 5 * len(unique_scopes)), 
#                              sharey=True
#                              )
    
#     # Plot each dataset in a separate subplot
#     for i, scope in enumerate(unique_scopes):
#         for j, dataset in enumerate(unique_datasets):
#             ax = axes[i][j] if len(unique_scopes) > 1 and len(unique_datasets) > 1 else axes[j]
#             subset = temp[(temp['dataset'] == dataset) & (temp['scope'] == scope)]
#             for buffer_val in unique_buffers:
#                 buffer_subset = subset[subset['buffer'] == buffer_val]
#                 ax.plot(buffer_subset['threshold_int'], buffer_subset[y_var],
#                         color=color_map[buffer_val], label=buffer_val, marker='o')
#             ax.set_title(f'ROI: {roi_name}, Dataset: {dataset}, Scope: {scope}', fontsize=8)

#             # Create a single legend for all subplots
#             handles, labels = [], []
#             for buffer_val in unique_buffers:
#                 handles.append(plt.Line2D([0], [0], color=color_map[buffer_val], marker='o', linestyle=''))
#                 labels.append(f'{buffer_val} m')

#             fig.legend(handles, labels, title='Lake Buffer Value (m)', loc='upper center', ncol=len(unique_buffers), fontsize=12)
#             fig.subplots_adjust(bottom=0.25)  # Adjust the bottom to make more space for the legend

#     # Set common labels
#     axes[0][0].set_ylabel('All PLD Lakes', fontweight='bold', fontsize=12)
#     axes[1][0].set_ylabel('Matched IS2 Lakes', fontweight='bold', fontsize=12)
#     axes[1][0].set_xlabel('GLAD', fontweight='bold', fontsize=12)
#     axes[1][1].set_xlabel('GSWO', fontweight='bold', fontsize=12)
#     axes[1][2].set_xlabel('Sentinel-2', fontweight='bold', fontsize=12)
#     fig.text(0.5, 0.1, 'Water Occurrence Threshold (0-100)', ha='center', fontweight='bold', fontsize=16)
#     fig.text(0.04, 0.5, f'{y_var} %', va='center', rotation='vertical', fontweight='bold', fontsize=16)

#     # Adjust layout
#     #plt.tight_layout()
#     plt.show()

# y_vars = ['net_total_sn_frac', 'net_lake_sn_frac']
# for roi_name in roi_names:
#     for y_var in y_vars:
#         by_roi_sensitivity_viz(seasonal_results[seasonal_results['timeframe'] == 'full'], roi_name, y_var)
#         #by_roi_sensitivity_viz(seasonal_results[seasonal_results['timeframe'] == 'partial'], roi_name, y_var)
