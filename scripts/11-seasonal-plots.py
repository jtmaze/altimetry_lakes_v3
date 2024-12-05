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

del keep_cols, drop_cols, area_df, mean_per_roi, bars_area, bars_wse, idx, n, bar_width
del fig, ax1, ax2, handles1, labels1, handles2, labels2, 


# %% Compare optical areas for given timeframe 

y_var = 'net_lake_sn_frac'
area_df = area_seasonality[area_seasonality['timeframe'] == 'partial']
area_df = area_df[area_df['scope'] == 'all_pld']
area_df = area_df[area_df['buffer'] == 60]
area_df = area_df[(area_df['threshold'] < 90) & (area_df['threshold'] > 70)]

keep_cols = ['roi_name', 'dataset', y_var, 'threshold']
drop_cols = [col for col in area_df.columns if col not in keep_cols]
area_df = area_df.drop(columns=drop_cols) 

area_df[y_var] = area_df[y_var] * -1

s2_area = area_df[area_df['dataset'] == 'sentinel2']
glad_area = area_df[area_df['dataset'] == 'glad']
gswo_area = area_df[area_df['dataset'] == 'gswo']

def compute_stats(area_df):
    mean = area_df.groupby('roi_name')[y_var].mean()
    min_val = area_df.groupby('roi_name')[y_var].min()
    max_val = area_df.groupby('roi_name')[y_var].max()
    lower_error = mean - min_val
    upper_error = max_val - mean
    return mean, lower_error, upper_error

# Calculate stats for each dataset
s2_mean, s2_lower, s2_upper = compute_stats(s2_area)
glad_mean, glad_lower, glad_upper = compute_stats(glad_area)
gswo_mean, gswo_lower, gswo_upper = compute_stats(gswo_area)

# Create DataFrames
mean_df = pd.DataFrame({
    'Sentinel2': s2_mean,
    'GLAD': glad_mean,
    'GSWO': gswo_mean
})

lower_errors = pd.DataFrame({
    'Sentinel2': s2_lower,
    'GLAD': glad_lower,
    'GSWO': gswo_lower
})

upper_errors = pd.DataFrame({
    'Sentinel2': s2_upper,
    'GLAD': glad_upper,
    'GSWO': gswo_upper
})

common_rois = mean_df.dropna().index

# Filter DataFrames to include only common rois
mean_df = mean_df.loc[common_rois]
lower_errors = lower_errors.loc[common_rois]
upper_errors = upper_errors.loc[common_rois]

n_rois = len(mean_df.index)
n_datasets = len(mean_df.columns)

# Positions of the bars on the x-axis
ind = np.arange(n_rois)  # the x locations for the groups
width = 0.8 / n_datasets  # the width of the bars

# Colors and markers
colors = {
    'Sentinel2': '#1f77b4',  # Blue
    'GSWO': '#ff7f0e',       # Orange
    'GLAD': '#2ca02c'        # Green
}

fig, ax = plt.subplots(figsize=(14, 6))


for i, dataset in enumerate(mean_df.columns):
    # Calculate positions for each dataset's bars
    positions = ind + (i - n_datasets / 2) * width + width / 2
    
    # Plot the bars with error bars using error_kw
    ax.bar(
        positions,
        mean_df[dataset],
        width,
        yerr=[lower_errors[dataset], upper_errors[dataset]],
        label=dataset,
        color=colors[dataset],
        edgecolor='black',        # Adds a black edge to the bars
        error_kw={
            'capsize': 5,          # Size of the error bar caps
            'elinewidth': 3,       # Width of the error bar lines
            'ecolor': 'black'      # Color of the error bars
        }
    )

# Set the x-axis labels
ax.set_xticks(ind)
ax.set_xticklabels(mean_df.index, rotation=45, fontsize=12)

# Add legend and labels
ax.set_ylabel('Seasonal (%) of Total Lake Area', fontsize=16)
ax.set_title('Seasonal (%) of Total Lake Area by Region', fontsize=20)
ax.legend(loc='upper left', fontsize=16)

plt.tight_layout()
plt.show()

# %% Calculate gross seasonal change?

area_seasonality

