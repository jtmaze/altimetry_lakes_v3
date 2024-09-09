# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

pixels_summary = pd.read_csv('./data/pixel_counts_matched_lakes.csv')
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
    by = ['roi_name', 'dataset', 'scope', 'timeframe', 'buffer']
    ).agg(
        pix_cnt_clean = ('sum', 'pix_cnts')
    ).reset_index()

# %% 3.0 Calculate the 



# %% PLD lakes cleaning



