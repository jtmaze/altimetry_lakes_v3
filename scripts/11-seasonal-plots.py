# %% 1.0 Load the data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

area_seasonality = pd.read_csv('./data/area_seasonal_results.csv')
area_seasonality['roi_name'] = area_seasonality['roi_name'].replace({
    'anderson_plain': 'Anderson',
    'YKF': 'YKflats',
    'YKD': 'YKdelta'
})

# Load the ICESat-2 data
is2_seasonality = pd.read_csv('./data/is2_roi_seasonality.csv')
is2_seasonality.rename(columns={'roi': 'roi_name', 'seasonality': 'wse_seasonality'}, inplace=True)
is2_seasonality.dropna(inplace=True)


# %% 2.0 Plot ICESat-2 Compared to Area

area_df = area_seasonality[area_seasonality['scope'] == 'matched_is2']
area_df = area_df[area_df['buffer'] == 60]
area_df = area_df[area_df['threshold'] == 80]
area_df = area_df[area_df['timeframe'] == 'partial']

keep_cols = ['roi_name', 'dataset', 'net_lake_sn_frac']
drop_cols = [col for col in area_df.columns if col not in keep_cols]
area_df = area_df.drop(columns=drop_cols)

# Get mean seasonality across all datasets for each roi
mean_per_roi = area_df.groupby('roi_name')['net_lake_sn_frac'].mean().reset_index()

# Join with ICESat-2 data
mean_per_roi = pd.merge(mean_per_roi, is2_seasonality, on='roi_name', how='inner')
mean_per_roi['net_lake_sn_frac'] = mean_per_roi['net_lake_sn_frac'] * -1

# Make the plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
n = len(mean_per_roi)
idx = np.arange(n)
bar_width = 0.35

bars_area = ax1.bar(idx - bar_width/2, mean_per_roi['net_lake_sn_frac'], bar_width,
                label='Mean of Area Datasets', color='skyblue', edgecolor='black')

bars_wse = ax2.bar(idx + bar_width/2, mean_per_roi['wse_seasonality'], bar_width,
                label='ICESat-2', color='salmon', edgecolor='black')

# Set the x-axis labels to the ROI names
ax1.set_title('Area vs WSE by region (lakes common to matched ICESat-2 dataset)')
ax1.set_xticks(idx)
ax1.set_xticklabels(mean_per_roi['roi_name'])
ax1.set_ylabel('Seasonal Fraction of Lake Area %')
ax2.set_ylabel('WSE Seasonality (m)')
ax1.tick_params(axis='y')
ax2.tick_params(axis='y')
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

plt.tight_layout()

#del keep_cols, drop_cols, area_df, mean_per_roi


# %%

