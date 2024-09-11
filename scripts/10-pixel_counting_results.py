# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

pixels_summary = pd.read_csv('./data/pixel_counts.csv')
#all_lakes_summary = pd.read_csv('./data/pixel_counts_all_lakes.csv')

is2_path = './data/lake_timeseries/*_timeseries.csv'
is2_files = glob.glob(is2_path)
is2_dfs = [pd.read_csv(file) for file in is2_files]
is2_data = pd.concat(is2_dfs)

del is2_dfs, is2_path
# %% 2.0 Clean up the pixel counts datasets

# !!! A few pixels were oddly above the max value of 100, no clue why this happened.
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
pixels_clean = pixels_summary.copy()
pixels_clean['pix_vals'] = np.where(pixels_clean['pix_vals'] > 100, 100, pixels_clean['pix_vals'])
pixels_clean = pixels_clean.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer', 'pix_vals']
    ).agg(
        pix_cnt_clean = ('pix_cnts', 'sum')
        ).reset_index()

# %% 3.0 Calculate total pixels in the image & number measured within masks.

total_pixels = pixels_clean.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer']
    ).agg(
        total_pix = ('pix_cnt_clean', 'sum')
        ).reset_index()
        
only_measured = pixels_clean[pixels_clean['pix_vals'] != -1]
total_measured = only_measured.groupby(
    by=['roi_name', 'timeframe', 'buffer', 'dataset', 'scope']
    ).agg(
        cnt_measured = ('pix_cnt_clean', 'sum')
        ).reset_index()
        
obs_info = total_pixels.merge(total_measured, 
                       how='inner', 
                       on=['roi_name', 'timeframe', 'buffer', 'dataset', 'scope']
                       )

obs_info['pixel_frac_measured'] = obs_info['cnt_measured'] / obs_info['total_pix'] * 100

pixels_clean = pixels_clean.merge(obs_info,
                                   how='left',
                                   on=['roi_name', 'dataset', 'buffer', 'timeframe', 'scope']
                                   )

# %% 4.0 Calculate the water fractions

def count_above(df, threshold):
    above_series = df['pix_cnt_clean'][df['pix_vals'] >= threshold]
    cnt_above = above_series.sum()

    return cnt_above

def count_below(df, threshold):
    below_series = df['pix_cnt_clean'][(df['pix_vals'] < threshold) & (df['pix_vals'] > -1)]
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
    for dataset in datasets:
        for scope in scopes:
            for timeframe in timeframes:
                for buffer in buffers: 
                    for threshold in thresholds:
                  
                        temp = pixels_clean[pixels_clean['roi_name'] == roi_name]
                        temp = temp[temp['timeframe'] == timeframe]
                        temp = temp[temp['buffer'] == buffer]
                        temp = temp[temp['dataset'] == dataset]
                        temp = temp[temp['scope'] == scope]
                        
                        wtr_cnt = count_above(temp, threshold)
                        land_cnt = count_below(temp, threshold)
                        if wtr_cnt == 0 and land_cnt == 0:
                            continue

                        else:
                            cnt_measured = temp['cnt_measured'].unique()
                            total_pix = temp['total_pix'].unique()
                            pixel_frac_measured = temp['pixel_frac_measured'].unique()
                            
                            data = {
                                'wtr_pix_sum': [wtr_cnt],
                                'land_pix_sum': [land_cnt],
                                'total_pix': [total_pix],
                                'cnt_measured': [cnt_measured],
                                'pixel_frac_measured': [pixel_frac_measured],
                                'roi_name': [roi_name],
                                'timeframe': [timeframe],
                                'buffer': [buffer],
                                'dataset': [dataset],
                                'scope': [scope],
                                'threshold': [threshold]
                            }
                            
                            data = pd.DataFrame(data)
                            data['wtr_frac_measured'] = data['wtr_pix_sum'] / data['cnt_measured'] * 100
                            data['land_frac_measured'] = 100 - data['wtr_frac_measured']
                            data['wtr_frac_total'] = data['wtr_pix_sum'] / data['total_pix'] * 100
                            data['land_frac_total'] = 100 - data['wtr_frac_total']
                            
                            results.append(data)


final_results = pd.concat(results, ignore_index=True)

# %% %% 5.0 Calculate differences across time periods

seasonal_diffs = []

for dataset in datasets:
    temp = final_results[final_results['dataset'] == dataset]
    if dataset == 'sentinel2':
        temp_s2 = temp.copy()
        temp_s2 = temp_s2[temp_s2['timeframe'] != 'years2016-2023_weeks35-40']
        temp_s2['timeframe'] = temp_s2['timeframe'].replace({
            'years2016-2023_weeks22-26': 'early',
            'years2016-2023_weeks31-35': 'late',
            #'years2016_2023_weeks35-40': 'extra_late'
        })

        df = temp_s2.pivot_table(
            index=['roi_name', 'buffer', 'dataset', 'threshold', 'scope'],
            columns=['timeframe'],
            values=['total_pix', 'wtr_frac_measured', 'land_frac_measured']
            ).reset_index()
        
    elif dataset == 'gswo':
        temp_gswo = temp.copy()
        temp_gswo['timeframe'] = temp_gswo['timeframe'].replace({
            'june': 'early',
            'aug': 'late'
        })

        df2 = temp_gswo.pivot_table(
            index=['roi_name', 'buffer', 'dataset', 'threshold', 'scope'],
            columns=['timeframe'],
            values=['total_pix', 'wtr_frac_measured', 'land_frac_measured']
            ).reset_index()



# %%
