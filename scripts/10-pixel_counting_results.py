# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

pixels_summary = pd.read_csv('./data/pixel_counts.csv')

is2_path = './data/lake_timeseries/*_timeseries.csv'
is2_files = glob.glob(is2_path)
is2_dfs = [pd.read_csv(file) for file in is2_files]
is2_data = pd.concat(is2_dfs)
is2_rois = is2_data['roi_name'].unique()

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
pixels_clean['pix_vals'] = np.where(pixels_clean['pix_vals'] > 100, 
                                    100, 
                                    pixels_clean['pix_vals']
                            )

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

# %% 4.1 For various thresholds calculate the water fractions across rois, buffers, datasets, timeframes, and scopes

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
                            print(f"""No data for: 
                                  {roi_name}, {dataset}, {scope}, 
                                  {timeframe}, {buffer}, {threshold}""")
                            continue

                        else:
                            # Should only be one unique value for each of these
                            cnt_measured = temp['cnt_measured'].unique()
                            total_pix = temp['total_pix'].unique()
                            pixel_frac_measured = temp['pixel_frac_measured'].unique()
                            
                            data = {
                                'wtr_pix_sum': wtr_cnt,
                                'land_pix_sum': land_cnt,
                                'total_pix': total_pix,
                                'cnt_measured': cnt_measured,
                                'pixel_frac_measured': pixel_frac_measured,
                                'roi_name': roi_name,
                                'timeframe': timeframe,
                                'buffer': buffer,
                                'dataset': dataset,
                                'scope': scope,
                                'threshold': threshold
                            }
                            
                            data = pd.DataFrame(data)
                            data['wtr_frac_measured'] = data['wtr_pix_sum'] / data['cnt_measured'] * 100
                            data['land_frac_measured'] = 100 - data['wtr_frac_measured']
                            data['wtr_frac_total'] = data['wtr_pix_sum'] / data['total_pix'] * 100
                            data['land_frac_total'] = 100 - data['wtr_frac_total']
                            
                            results.append(data)


full_results = pd.concat(results, ignore_index=True)

# %% %% 5.0 Calculate differences across time periods

seasonal_diffs = []

for dataset in datasets:
    temp = full_results[full_results['dataset'] == dataset]
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
            values=['total_pix', 'wtr_frac_measured', 'land_frac_measured', 'wtr_frac_total']
            ).reset_index()
        
        df['seasonality'] = df[('wtr_frac_total', 'early')] - df[('wtr_frac_total', 'late')]
        df['satellite'] = 'sentinel2'
        
        seasonal_diffs.append(df)
        
    elif dataset in ['gswo', 'glad']:
        temp_gswo = temp.copy()
        temp_gswo['timeframe'] = temp_gswo['timeframe'].replace({
            'june': 'early',
            'aug': 'late'
        })

        df = temp_gswo.pivot_table(
            index=['roi_name', 'buffer', 'dataset', 'threshold', 'scope'],
            columns=['timeframe'],
            values=['total_pix', 'wtr_frac_measured', 'land_frac_measured', 'wtr_frac_total']
            ).reset_index()

        df['seasonality'] = df[('wtr_frac_total', 'early')] - df[('wtr_frac_total', 'late')]
        df['satellite'] = 'landsat'

        seasonal_diffs.append(df)

seasonal_results = pd.concat(seasonal_diffs)

# %%
def by_roi_sensitivity_viz(df, roi_name):
    
    temp = df.copy()

    temp['threshold_int'] = temp['threshold'].astype(int)
    temp['buffer_int'] = temp['buffer'].astype(int)
    
    temp.drop(columns=[
        ('total_pix',  'late'),
        ('total_pix', 'early'),
        ('land_frac_measured',  'late'),
        ('land_frac_measured', 'early'),
        ('wtr_frac_measured',  'late'),
        ('wtr_frac_measured', 'early'),
    ], inplace=True)
    
    temp = temp[temp['roi_name'] == roi_name]
    temp.columns = temp.columns.droplevel(1) # Drop the 'timeframe' level from the columns
    
    unique_buffers = temp['buffer'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1.1, len(unique_buffers)))
    color_map = dict(zip(unique_buffers, colors))
    
    # Get the unique datasets
    unique_datasets = temp['dataset'].unique()
    ## Incorperate unique 'scopes in plotting
    unique_scopes = temp['scope'].unique()
    
    # Create subplots
    fig, axes = plt.subplots(len(unique_scopes), 
                             len(unique_datasets), 
                             figsize=(15, 5 * len(unique_scopes)), 
                             sharey=True
                             )
    
    # Plot each dataset in a separate subplot
    for i, scope in enumerate(unique_scopes):
        for j, dataset in enumerate(unique_datasets):
            ax = axes[i][j] if len(unique_scopes) > 1 and len(unique_datasets) > 1 else axes[j]
            subset = temp[(temp['dataset'] == dataset) & (temp['scope'] == scope)]
            for buffer_val in unique_buffers:
                buffer_subset = subset[subset['buffer'] == buffer_val]
                ax.plot(buffer_subset['threshold_int'], buffer_subset['seasonality'],
                        color=color_map[buffer_val], label=buffer_val, marker='o')
            ax.set_title(f'ROI: {roi_name}, Dataset: {dataset}, Scope: {scope}', fontsize=8)

            # Create a single legend for all subplots
            handles, labels = [], []
            for buffer_val in unique_buffers:
                handles.append(plt.Line2D([0], [0], color=color_map[buffer_val], marker='o', linestyle=''))
                labels.append(f'{buffer_val} m')

            fig.legend(handles, labels, title='Lake Buffer Value (m)', loc='upper center', ncol=len(unique_buffers), fontsize=12)
            fig.subplots_adjust(bottom=0.25)  # Adjust the bottom to make more space for the legend

    # Set common labels
    axes[0][0].set_ylabel('All PLD Lakes', fontweight='bold', fontsize=12)
    axes[1][0].set_ylabel('Matched IS2 Lakes', fontweight='bold', fontsize=12)
    axes[1][0].set_xlabel('GLAD', fontweight='bold', fontsize=12)
    axes[1][1].set_xlabel('GSWO', fontweight='bold', fontsize=12)
    axes[1][2].set_xlabel('Sentinel-2', fontweight='bold', fontsize=12)
    fig.text(0.5, 0.1, 'Water Occurrence Threshold (0-100)', ha='center', fontweight='bold', fontsize=16)
    fig.text(0.04, 0.5, 'Seasonal Area % (From Masked Lake Pixels)', va='center', rotation='vertical', fontweight='bold', fontsize=16)

    # Adjust layout
    #plt.tight_layout()
    plt.show()

for roi_name in roi_names:
    by_roi_sensitivity_viz(seasonal_results, roi_name)

# %% Calculate the ICESat-2 seasonality

is2_data['obs_datetime'] = pd.to_datetime(is2_data['obs_date'])
is2_data['obs_month'] = is2_data['obs_datetime'].dt.month.astype(str)
is2_data = is2_data[(is2_data['obs_month'] == '6') | (is2_data['obs_month'] == '8')]

clean_lakes_path = './data/clean_ids.csv'
clean_ids = pd.read_csv(clean_lakes_path)
clean_ids = clean_ids['lake_id']
is2_data = is2_data[is2_data['lake_id'].isin(clean_ids)]
is2_data_clean = is2_data[(is2_data['zdif_date'] > -5) & (is2_data['zdif_date'] < 5)]
print(len(is2_data['lake_id'].unique()), len(is2_data_clean['lake_id'].unique()))

is2_seasonality = is2_data_clean.groupby(
    by=['obs_month', 'roi_name']
    ).agg(
        mean_zdif = ('zdif_date', 'mean'),
        std_zdif = ('zdif_date', 'std'),
        lakes_cnt = ('lake_id', lambda x: len(x.unique().tolist()))
        ).reset_index()

is2_seasonality = is2_seasonality.pivot_table(
    index = ['roi_name'],
    columns = ['obs_month'],
    values = ['mean_zdif', 'std_zdif', 'lakes_cnt']
    ).reset_index()

is2_seasonality['seasonality'] = is2_seasonality[('mean_zdif', '6')] - is2_seasonality[('mean_zdif', '8')]

columns_to_drop = [
    ('lakes_cnt', '6'),
    ('lakes_cnt', '8'),
    ('mean_zdif', '6'),
    ('mean_zdif', '8'),
    ('std_zdif', '6'),
    ('std_zdif', '8')
]

is2_seasonality.drop(
    columns=columns_to_drop,
    inplace=True)

is2_seasonality.columns = is2_seasonality.columns.droplevel(1)

# %% 
def side_by_side_bar(df, x_var):
    # Filter data by dataset
    df_matched_sentinel2 = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'sentinel2')]
    ds_matched_sentinel2 = df_matched_sentinel2.groupby(x_var)['seasonality'].mean()
    
    df_pld_sentinel2 = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'sentinel2')]
    ds_pld_sentinel2 = df_pld_sentinel2.groupby(x_var)['seasonality'].mean()

    df_matched_gswo = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'gswo')]
    ds_matched_gswo = df_matched_gswo.groupby(x_var)['seasonality'].mean()

    df_pld_gswo = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'gswo')]
    ds_pld_gswo = df_pld_gswo.groupby(x_var)['seasonality'].mean()
    
    df_matched_glad = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'glad')]
    ds_matched_glad = df_matched_glad.groupby(x_var)['seasonality'].mean()

    df_pld_glad = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'glad')]
    ds_pld_glad = df_pld_glad.groupby(x_var)['seasonality'].mean()
    
    # Prepare for plotting
    unique_xvar = ds_matched_sentinel2.index.union(ds_pld_sentinel2.index)\
                    .union(ds_matched_gswo.index).union(ds_pld_gswo.index)\
                    .union(ds_matched_glad.index).union(ds_pld_glad.index)
    unique_xvar = sorted(unique_xvar)  # Sort the x variables for consistency

    width = 0.13  # Bar width, adjusted for 6 bars
    ind = np.arange(len(unique_xvar))  # the x locations for the groups

    colors = {
        'Matched IS2 Sentinel2': '#aec7e8',  # Light Blue
        'All PLD Sentinel2': '#1f77b4',      # Blue
        'Matched IS2 GSWO': '#ffbb78',       # Light Orange
        'All PLD GSWO': '#ff7f0e',           # Orange
        'Matched IS2 GLAD': '#98df8a',       # Light Green
        'All PLD GLAD': '#2ca02c'            # Green
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotting bars for each dataset
    bars_matched_sentinel2 = ax.bar(ind - 2.5*width, 
                                    ds_matched_sentinel2.reindex(unique_xvar).fillna(0).values, 
                                    width, 
                                    color=colors['Matched IS2 Sentinel2'],
                                    alpha=0.5,
                                    edgecolor='black', 
                                    label="Matched IS2 Sentinel2")
    
    bars_pld_sentinel2 = ax.bar(ind - 1.5*width, 
                                ds_pld_sentinel2.reindex(unique_xvar).fillna(0).values, 
                                width, 
                                color=colors['All PLD Sentinel2'],
                                edgecolor='black', 
                                label="All PLD Sentinel2")
    
    bars_matched_gswo = ax.bar(ind - 0.5*width, 
                               ds_matched_gswo.reindex(unique_xvar).fillna(0).values, 
                               width, 
                               color=colors['Matched IS2 GSWO'],
                               alpha=0.5,
                               edgecolor='black', 
                               label="Matched IS2 GSWO")
    
    bars_pld_gswo = ax.bar(ind + 0.5*width, 
                           ds_pld_gswo.reindex(unique_xvar).fillna(0).values, 
                           width, 
                           color=colors['All PLD GSWO'],
                           edgecolor='black', 
                           label="All PLD GSWO")
    
    bars_matched_glad = ax.bar(ind + 1.5*width, 
                               ds_matched_glad.reindex(unique_xvar).fillna(0).values, 
                               width, 
                               color=colors['Matched IS2 GLAD'],
                               alpha=0.5,
                               edgecolor='black', 
                               label="Matched IS2 GLAD")
    
    bars_pld_glad = ax.bar(ind + 2.5*width, 
                           ds_pld_glad.reindex(unique_xvar).fillna(0).values, 
                           width, 
                           color=colors['All PLD GLAD'],
                           edgecolor='black', 
                           label="All PLD GLAD")
    
    # Set x-axis labels and legend
    ax.set_xticks(ind)
    ax.set_xticklabels(unique_xvar, rotation=0)  # Rotate labels if they are lengthy
    ax.set_xlabel('ROI Name', fontweight='bold', fontsize=14)
    ax.set_ylabel('Percent change (of lake mask) early to late summer')
    ax.legend()

    # Display the plot
    plt.show()

    # Calculate mean values
    mean_values_matched_is2 = (
        ds_matched_sentinel2.reindex(unique_xvar).fillna(0) + 
        #ds_pld_sentinel2.reindex(unique_xvar).fillna(0) + 
        ds_matched_gswo.reindex(unique_xvar).fillna(0) + 
        #ds_pld_gswo.reindex(unique_xvar).fillna(0) + 
        ds_matched_glad.reindex(unique_xvar).fillna(0) 
        #ds_pld_glad.reindex(unique_xvar).fillna(0)
    ) / 3

    sentinel2_mean = (
        ds_matched_sentinel2.reindex(unique_xvar).fillna(0) + 
        ds_pld_sentinel2.reindex(unique_xvar).fillna(0)
    ) / 2

    gswo_mean = (
        ds_matched_gswo.reindex(unique_xvar).fillna(0) + 
        ds_pld_gswo.reindex(unique_xvar).fillna(0)
    ) / 2

    glad_mean = (
        ds_matched_glad.reindex(unique_xvar).fillna(0) + 
        ds_pld_glad.reindex(unique_xvar).fillna(0)
    ) / 2

    return unique_xvar, mean_values_matched_is2, sentinel2_mean, gswo_mean, glad_mean, ds_matched_sentinel2, ds_matched_gswo, ds_matched_glad



# %% Create seasonality df based on criteria

df_analyze = seasonal_results.drop(columns=[
        ('total_pix',  'late'),
        ('total_pix', 'early'),
        ('land_frac_measured',  'late'),
        ('land_frac_measured', 'early'),
        ('wtr_frac_measured',  'late'),
        ('wtr_frac_measured', 'early'),
    ])

df_analyze.columns = df_analyze.columns.droplevel(1)

df_analyze = df_analyze[
    (df_analyze['buffer'] == 120) & (df_analyze['threshold'] == 80)
]

ordered_rois, mean_values_matched_is2, sentinel2_mean, gswo_mean, glad_mean, matched_sentinel2, matched_gswo, matched_glad = side_by_side_bar(df_analyze, 'roi_name')

# %% Visualize just the ICESat-2 WSE (m) seasonality

is2_seasonality['roi_name'] = pd.Categorical(is2_seasonality['roi_name'],
                                             categories=ordered_rois,
                                             ordered=True)
is2_seasonality = is2_seasonality.sort_values('roi_name')

fig, ax = plt.subplots(figsize=(14, 4))
bars = plt.bar(is2_seasonality['roi_name'], 
               is2_seasonality['seasonality'], 
               color='pink',
               edgecolor='black')
ax.set_xticklabels(is2_seasonality['roi_name'], rotation=0)

plt.ylabel('June - August WSE difference (m)')
plt.show()

# %% Compare mean of all optical datasets with ICESat-2

mean_values_df = mean_values_matched_is2.reset_index()
mean_values_df.columns = ['roi_name', 'seasonality_mean']
merged_df = pd.merge(is2_seasonality, mean_values_df, on='roi_name', how='inner')
merged_df.rename(columns={'seasonality': 'seasonality_is2'}, inplace=True)

rescale_factor = 50
merged_df['seasonality_is2_rescaled'] = merged_df['seasonality_is2'] * rescale_factor

# Set up the x-axis positions
x = np.arange(len(merged_df['roi_name']))
width = 0.35  # Width of the bars

fig, ax1 = plt.subplots(figsize=(14, 4))

# Plot the optical dataset bars on ax1
bars_mean = ax1.bar(x - width/2, merged_df['seasonality_mean'], width,
                    label='Optical Area Change (% of Masked Pixels)',
                    color='grey', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('Area Change % of Masked Pixels')

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim()[0] / rescale_factor, ax1.get_ylim()[1] / rescale_factor)
ax2.set_ylabel('WSE Change (meters)')

# Adjust the secondary y-axis ticks to be meaningful
ax2_ticks = ax1.get_yticks()
ax2.set_yticks(ax2_ticks / rescale_factor)
ax2.set_yticklabels([f"{tick:.2f}" for tick in ax2.get_yticks()])

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1, labels1, loc='upper left')

plt.title('Mean Across Optical Datasets and IS2 Seasonality')

# Display the plot
plt.show()

# %% Compare the matched Sentinel-2 lakes and ICESat-2
optical_values_df = matched_sentinel2.reset_index()
optical_values_df.columns = ['roi_name', 'seasonality_mean']
merged_df = pd.merge(is2_seasonality, optical_values_df, on='roi_name', how='inner')
merged_df.rename(columns={'seasonality': 'seasonality_is2'}, inplace=True)

rescale_factor = 50
merged_df['seasonality_is2_rescaled'] = merged_df['seasonality_is2'] * rescale_factor

# Set up the x-axis positions
x = np.arange(len(merged_df['roi_name']))
width = 0.35  # Width of the bars

fig, ax1 = plt.subplots(figsize=(14, 4))

# Plot the optical dataset bars on ax1
bars_mean = ax1.bar(x - width/2, merged_df['seasonality_mean'], width,
                    label='Optical Area Change (% of Masked Pixels)',
                    color='blue', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('Area Change % of Masked Pixels')

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim()[0] / rescale_factor, ax1.get_ylim()[1] / rescale_factor)
ax2.set_ylabel('WSE Change (meters)')

# Adjust the secondary y-axis ticks to be meaningful
ax2_ticks = ax1.get_yticks()
ax2.set_yticks(ax2_ticks / rescale_factor)
ax2.set_yticklabels([f"{tick:.2f}" for tick in ax2.get_yticks()])

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1, labels1, loc='upper left')

plt.title('Matched Sentinel-2 and IS2 Seasonality')

# Display the plot
plt.show()


# %% Compare the matched GSWO lakes and ICESat-2

optical_values_df = matched_gswo.reset_index()
optical_values_df.columns = ['roi_name', 'seasonality_mean']
merged_df = pd.merge(is2_seasonality, optical_values_df, on='roi_name', how='inner')
merged_df.rename(columns={'seasonality': 'seasonality_is2'}, inplace=True)

rescale_factor = 50
merged_df['seasonality_is2_rescaled'] = merged_df['seasonality_is2'] * rescale_factor

# Set up the x-axis positions
x = np.arange(len(merged_df['roi_name']))
width = 0.35  # Width of the bars

fig, ax1 = plt.subplots(figsize=(14, 4))

# Plot the optical dataset bars on ax1
bars_mean = ax1.bar(x - width/2, merged_df['seasonality_mean'], width,
                    label='Optical Area Change (% of Masked Pixels)',
                    color='orange', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('Area Change % of Masked Pixels')

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim()[0] / rescale_factor, ax1.get_ylim()[1] / rescale_factor)
ax2.set_ylabel('WSE Change (meters)')

# Adjust the secondary y-axis ticks to be meaningful
ax2_ticks = ax1.get_yticks()
ax2.set_yticks(ax2_ticks / rescale_factor)
ax2.set_yticklabels([f"{tick:.2f}" for tick in ax2.get_yticks()])

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1, labels1, loc='upper left')

plt.title('GSWO and IS2 Seasonality')

# Display the plot
plt.show()

# %% Compare the matched GLAD lakes and ICESat-2


optical_values_df = matched_glad.reset_index()
optical_values_df.columns = ['roi_name', 'seasonality_mean']
merged_df = pd.merge(is2_seasonality, optical_values_df, on='roi_name', how='inner')
merged_df.rename(columns={'seasonality': 'seasonality_is2'}, inplace=True)

rescale_factor = 50
merged_df['seasonality_is2_rescaled'] = merged_df['seasonality_is2'] * rescale_factor

# Set up the x-axis positions
x = np.arange(len(merged_df['roi_name']))
width = 0.35  # Width of the bars

fig, ax1 = plt.subplots(figsize=(14, 4))

# Plot the optical dataset bars on ax1
bars_mean = ax1.bar(x - width/2, merged_df['seasonality_mean'], width,
                    label='Optical Area Change (% of Masked Pixels)',
                    color='green', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('Area Change % of Masked Pixels')

# Create a secondary y-axis
ax2 = ax1.twinx()
ax2.set_ylim(ax1.get_ylim()[0] / rescale_factor, ax1.get_ylim()[1] / rescale_factor)
ax2.set_ylabel('WSE Change (meters)')

# Adjust the secondary y-axis ticks to be meaningful
ax2_ticks = ax1.get_yticks()
ax2.set_yticks(ax2_ticks / rescale_factor)
ax2.set_yticklabels([f"{tick:.2f}" for tick in ax2.get_yticks()])

# Create a combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(handles1, labels1, loc='upper left')

plt.title('GSWO and IS2 Seasonality')

# Display the plot
plt.show()

# %%
