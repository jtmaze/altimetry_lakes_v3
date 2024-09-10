# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

pixels_summary = pd.read_csv('./data/pixel_counts_working.csv')
#all_lakes_summary = pd.read_csv('./data/pixel_counts_all_lakes.csv')

is2_path = './data/lake_timeseries/*_timeseries.csv'
is2_files = glob.glob(is2_path)
is2_dfs = [pd.read_csv(file) for file in is2_files]
is2_data = pd.concat(is2_dfs)

del is2_dfs, is2_path
# %% 2.0 Clean up the pixel counts datasets

# !!! A few pixels were randomly above the max value of 100, no clue why this happened.
# Rare enough not to matter. 
over_max = np.sum(np.where(pixels_summary['pix_vals'] > 100, 
                           pixels_summary['pix_cnts'], 
                           0
                           )
                 )
valid = np.sum(np.where(pixels_summary['pix_vals'] < 100, 
                        pixels_summary['pix_cnts'], 
                        0
                        )
               )

print(f'{(over_max/(over_max + valid) * 100):.8f} % of pixels over max')

# Swap values > 100 with 100
pixels_summary['pix_vals'] = np.where(pixels_summary['pix_vals'] > 100, 100, pixels_summary['pix_vals'])
pixels_clean = pixels_summary.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer', 'pix_vals']
    ).agg(
        pix_cnt_clean = ('pix_cnts', 'sum')
        ).reset_index()

# %% 3.0 Calculate the 

total_pixels = pixels_clean.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer']
    ).agg(
        total_pix = ('pix_cnt_clean', 'sum')
        ).reset_index()

""" Uncomment this code 
AND merge with obs_info
If new pixel counting runs in time
"""
        
# only_measured = full_data_clean[full_data_clean['pix_vals'] != -1]
# total_measured = only_measured.groupby(
#     by=['roi_name', 'timeframe', 'buffer', 'dataset', 'scope']
#     ).agg(
#         cnt_measured = ('pix_cnt_clean', 'sum')
#         ).reset_index()
        
# obs_info = total_pixels.merge(total_measured, 
#                        how='inner', 
#                        on=['roi_name', 'month', 'buffer_val', 'dataset']
#                        )

# obs_info['pixel_frac_measured'] = obs_info['cnt_measured'] / obs_info['cnt_all'] * 100

pixels_clean = pixels_clean.merge(total_pixels,
                                   how='left',
                                   on=['roi_name', 'dataset', 'buffer', 'timeframe', 'scope']
                                   )

# %% 4.0 Calculate the water fractions

def count_above(df, threshold):
    above_series = df['pix_cnt_clean'][df['pix_vals'] >= threshold]
    cnt_above = above_series.sum()

    return cnt_above

def count_below(df, threshold):
    below_series = df['pix_cnt_clean'][(df['pix_vals'] < threshold)] 
    #!!! & (temp['pix_clean'] > -1)]
    # Filter based on set-logic once new pix counts runs. 
    cnt_below = below_series.sum()

    return cnt_below

# %%

timeframes = pixels_clean['timeframe'].unique().tolist()
roi_names = pixels_clean['roi_name'].unique().tolist()
buffers = pixels_clean['buffer'].unique().tolist()
datasets = pixels_clean['dataset'].unique().tolist()
scopes = pixels_clean['scope'].unique().tolist()
thresholds = np.arange(start=64, stop=98, step=2).tolist()

results = []

for roi_name in roi_names:
    for timeframe in timeframes:
        for buffer in buffers: 
            for dataset in datasets:
                for threshold in thresholds:
                    for scope in scopes:
    # !!!This filtering is computationally inefficient.
    # !!!Need to fix with MultiIndex Dataframe for larger datasets                    
                        temp = pixels_clean[pixels_clean['roi_name'] == roi_name]
                        temp = temp[temp['timeframe'] == timeframe]
                        temp = temp[temp['buffer'] == buffer]
                        temp = temp[temp['dataset'] == dataset]
                        temp = temp[temp['scope'] == scope]
                        
                        wtr_cnt = count_above(temp, threshold)
                        land_cnt = count_below(temp, threshold)
                        #cnt_measured = temp['cnt_measured'].unique()
                        total_pix = temp['total_pix'].unique()
                        #print(total_pix)
                        if len(total_pix) == 0:
                            print(f"Warning: No total_pix found for {roi_name}, {timeframe}, {buffer}, {dataset}, {scope}")
                            continue
                        
                        # Extract the first value (since there seems to be only one unique value)
                        total_pix = total_pix[0]
                        
                        
                        data = {
                            'wtr_pix_sum': [wtr_cnt],
                            'land_pix_sum': [land_cnt],
                            'total_pix': total_pix,
                            'roi_name': [roi_name],
                            'timeframe': [timeframe],
                            'buffer': [buffer],
                            'dataset': [dataset],
                            'scope': [scope],
                            'threshold': [threshold]
                        }
                        
                        data = pd.DataFrame(data)
                        
                        data['wtr_frac_measured'] = data['wtr_pix_sum'] / data['total_pix'] * 100
                        data['land_frac_measured'] = 100 - data['wtr_frac_measured']
                        
                        results.append(data)
                        
    #print(roi_name)

final_results = pd.concat(results, ignore_index=True)

# %% %% 5.0 Calculate differences across time periods

seasonal_diffs = final_results.pivot_table(
    index=['roi_name', 'buffer', 'dataset', 'threshold', 'scope'],
    columns=['timeframe'],
    values=['total_pix', 'wtr_frac_measured', 'land_frac_measured']
).reset_index()

#seasonal_diffs['jun_aug_wtr'] = seasonal_diffs[('wtr_frac_measured', 'jun')] - seasonal_diffs[('wtr_frac_measured', 'aug')]
# %%
