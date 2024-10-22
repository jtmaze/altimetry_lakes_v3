# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

pixels_path = './data/pixel_counts_Oct21.csv'
pixels_summary = pd.read_csv(pixels_path)

is2_path = './data/lake_timeseries/*_timeseries.csv'
is2_files = glob.glob(is2_path)
is2_dfs = [pd.read_csv(file) for file in is2_files]
is2_data = pd.concat(is2_dfs)

del is2_dfs, is2_path, is2_files, pixels_path

# %% 2.0 Clean up the pixel counts datasets

# !!! A few pixels were oddly above the max value of 100, no clue why this happened.
# Rare enough % to not matter. 
over_max = np.sum(
    np.where(pixels_summary['pix_vals'] > 100, pixels_summary['pix_cnts'], 0)
)
valid = np.sum(
    np.where(pixels_summary['pix_vals'] < 100, pixels_summary['pix_cnts'], 0)
)

print(f'{(over_max / (over_max + valid) * 100):.8f} % of pixels over max')

# Swap values > 100 with 100
pixels_clean = pixels_summary.copy()
pixels_clean['pix_vals'] = np.where(
    pixels_clean['pix_vals'] > 100, 100, pixels_clean['pix_vals']
)

pixels_clean = pixels_clean.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer', 'pix_vals']
).agg(
    pix_cnt_clean=('pix_cnts', 'sum')
).reset_index()

del over_max, valid, pixels_summary

# %% 3.0 Calculate total pixels in the image & number measured within masks.

total_pixels = pixels_clean.groupby(
    by=['roi_name', 'dataset', 'scope', 'timeframe', 'buffer']
).agg(
    total_pix=('pix_cnt_clean', 'sum')
).reset_index()

only_measured = pixels_clean[pixels_clean['pix_vals'] != -1]
total_measured = only_measured.groupby(
    by=['roi_name', 'timeframe', 'buffer', 'dataset', 'scope']
).agg(
    cnt_measured=('pix_cnt_clean', 'sum')
).reset_index()

obs_info = total_pixels.merge(
    total_measured,
    how='inner',
    on=['roi_name', 'timeframe', 'buffer', 'dataset', 'scope']
)

obs_info['pixel_frac_measured'] = (
    obs_info['cnt_measured'] / obs_info['total_pix'] * 100
)

pixels_clean = pixels_clean.merge(
    obs_info,
    how='left',
    on=['roi_name', 'dataset', 'buffer', 'timeframe', 'scope']
)

del obs_info, total_measured, total_pixels, only_measured

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
                        
                        # Sum all the water/land pixels in the raster.
                        wtr_cnt = count_above(temp, threshold)
                        land_cnt = count_below(temp, threshold)

                        if wtr_cnt == 0 and land_cnt == 0:
                            print(f"""No data for: 
                                  {roi_name}, {dataset}, {timeframe}""")
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
                                'buffer': buffer,
                                'dataset': dataset,
                                'scope': scope,
                                'threshold': threshold
                            }
                            
                            data = pd.DataFrame(data)                          
                            results.append(data)


full_results = pd.concat(results, ignore_index=True)

del(buffer, measured_pix, data, dataset, land_cnt, results, roi_name, 
    scope, temp, threshold, timeframe, total_pix, wtr_cnt
)

# %% %% 5.0 Calculate differences across time periods

renamed = []

# Define the 'early' and 'late' timeframes for each dataset
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

        renamed.append(temp_s2)

    elif dataset in ['gswo', 'glad']:
        temp_gswo = temp.copy()
        temp_gswo['timeframe'] = temp_gswo['timeframe'].replace({
            'june': 'early',
            'aug': 'late'
        })

        renamed.append(temp_gswo)

# Concatonate the results
full_results = pd.concat(renamed, ignore_index=True)
# clean up the environment
del renamed, temp, temp_gswo, temp_s2

seasonal_results = pd.pivot_table(
    data=full_results,
    index=['roi_name', 'dataset', 'scope', 'buffer', 'threshold'],
    columns='timeframe',
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



# %%
def by_roi_sensitivity_viz(df, roi_name, y_var):
    
    temp = df.copy()
    temp.columns = temp.columns.droplevel(1)

    temp['threshold_int'] = temp['threshold'].astype(int)
    temp['buffer_int'] = temp['buffer'].astype(int)
    
    keep_cols = ['roi_name', 'dataset', 'scope', 'buffer_int', 'threshold_int', 'buffer', y_var] 
    drop_cols = [col for col in temp.columns if col not in keep_cols]
    temp.drop(columns=drop_cols, inplace=True)
    
    temp = temp[temp['roi_name'] == roi_name]
    
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
                ax.plot(buffer_subset['threshold_int'], buffer_subset[y_var],
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
    fig.text(0.04, 0.5, f'{y_var} %', va='center', rotation='vertical', fontweight='bold', fontsize=16)

    # Adjust layout
    #plt.tight_layout()
    plt.show()

y_vars = ['net_total_sn_frac', 'net_lake_sn_frac']
for roi_name in roi_names:
    for y_var in y_vars:
        by_roi_sensitivity_viz(seasonal_results, roi_name, y_var)

# %% 6.0 Create a function to plot the results
# %% Test new function with y_var argument

def side_by_side_bar_two_scopes(df, x_var, y_var):
    # Filter data by dataset
    df_matched_sentinel2 = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'sentinel2')]
    ds_matched_sentinel2 = df_matched_sentinel2.groupby(x_var)[y_var].mean()
    
    df_pld_sentinel2 = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'sentinel2')]
    ds_pld_sentinel2 = df_pld_sentinel2.groupby(x_var)[y_var].mean()
    
    df_matched_gswo = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'gswo')]
    ds_matched_gswo = df_matched_gswo.groupby(x_var)[y_var].mean()
    
    df_pld_gswo = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'gswo')]
    ds_pld_gswo = df_pld_gswo.groupby(x_var)[y_var].mean()
    
    df_matched_glad = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'glad')]
    ds_matched_glad = df_matched_glad.groupby(x_var)[y_var].mean()
    
    df_pld_glad = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'glad')]
    ds_pld_glad = df_pld_glad.groupby(x_var)[y_var].mean()
    
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
    ax.set_ylabel(f'{y_var} %')
    ax.legend()
    
    # Display the plot
    plt.show()
    
    # Calculate mean values
    mean_values_matched_is2 = (
        ds_matched_sentinel2.reindex(unique_xvar).fillna(0) + 
        ds_matched_gswo.reindex(unique_xvar).fillna(0) + 
        ds_matched_glad.reindex(unique_xvar).fillna(0)
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


def side_by_side_bar_one_scope(df, x_var, y_var, scope):
    # Filter data by dataset and the specified scope
    df_sentinel2 = df[(df['scope'] == scope) & (df['dataset'] == 'sentinel2')]
    ds_sentinel2 = df_sentinel2.groupby(x_var)[y_var].mean()
    
    df_gswo = df[(df['scope'] == scope) & (df['dataset'] == 'gswo')]
    ds_gswo = df_gswo.groupby(x_var)[y_var].mean()
    
    df_glad = df[(df['scope'] == scope) & (df['dataset'] == 'glad')]
    ds_glad = df_glad.groupby(x_var)[y_var].mean()
    
    # Prepare for plotting
    unique_xvar = ds_sentinel2.index.union(ds_gswo.index).union(ds_glad.index)
    unique_xvar = sorted(unique_xvar)  # Sort the x variables for consistency
    
    width = 0.2  # Bar width, adjusted for 3 bars
    ind = np.arange(len(unique_xvar))  # the x locations for the groups
    
    colors = {
        'Sentinel2': '#1f77b4',      # Blue
        'GSWO': '#ff7f0e',           # Orange
        'GLAD': '#2ca02c'            # Green
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotting bars for each dataset
    bars_sentinel2 = ax.bar(ind - width, 
                            ds_sentinel2.reindex(unique_xvar).fillna(0).values, 
                            width, 
                            color=colors['Sentinel2'],
                            edgecolor='black', 
                            label="Sentinel2")
    
    bars_gswo = ax.bar(ind, 
                       ds_gswo.reindex(unique_xvar).fillna(0).values, 
                       width, 
                       color=colors['GSWO'],
                       edgecolor='black', 
                       label="GSWO")
    
    bars_glad = ax.bar(ind + width, 
                       ds_glad.reindex(unique_xvar).fillna(0).values, 
                       width, 
                       color=colors['GLAD'],
                       edgecolor='black', 
                       label="GLAD")
    
    # Set x-axis labels and legend
    ax.set_xticks(ind)
    ax.set_xticklabels(unique_xvar, rotation=0)  # Rotate labels if they are lengthy
    ax.set_xlabel(x_var.capitalize(), fontweight='bold', fontsize=14)
    ax.set_ylabel(f'{y_var} %')
    ax.legend()
    
    # Display the plot
    plt.show()
    
    # Return the series for each dataset
    return unique_xvar, ds_sentinel2, ds_gswo, ds_glad

def interaction_plot_one_scope(df, x_var, y_var, scope):
    # Filter data by dataset and the specified scope
    df_sentinel2 = df[(df['scope'] == scope) & (df['dataset'] == 'sentinel2')]
    ds_sentinel2 = df_sentinel2.groupby(x_var)[y_var].mean()
    
    df_gswo = df[(df['scope'] == scope) & (df['dataset'] == 'gswo')]
    ds_gswo = df_gswo.groupby(x_var)[y_var].mean()
    
    df_glad = df[(df['scope'] == scope) & (df['dataset'] == 'glad')]
    ds_glad = df_glad.groupby(x_var)[y_var].mean()
    
    # Prepare for plotting
    unique_xvar = ds_sentinel2.index.union(ds_gswo.index).union(ds_glad.index)
    unique_xvar = sorted(unique_xvar)  # Sort the x variables for consistency
    
    colors = {
        'Sentinel2': '#1f77b4',      # Blue
        'GSWO': '#ff7f0e',           # Orange
        'GLAD': '#2ca02c'            # Green
    }
    
    markers = {
        'Sentinel2': 'o',            # Circle marker for Sentinel2
        'GSWO': 's',                 # Square marker for GSWO
        'GLAD': '^'                  # Triangle marker for GLAD
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotting points and lines for each dataset
    ax.plot(unique_xvar, ds_sentinel2.reindex(unique_xvar).fillna(0).values, 
            marker=markers['Sentinel2'], 
            color=colors['Sentinel2'], 
            label="Sentinel2", 
            linestyle='None', 
            markersize=10)
    
    ax.plot(unique_xvar, ds_gswo.reindex(unique_xvar).fillna(0).values, 
            marker=markers['GSWO'], 
            color=colors['GSWO'], 
            label="GSWO", 
            linestyle='None', 
            markersize=10)
    
    ax.plot(unique_xvar, ds_glad.reindex(unique_xvar).fillna(0).values, 
            marker=markers['GLAD'], 
            color=colors['GLAD'], 
            label="GLAD", 
            linestyle='None', 
            markersize=10)
    
    # Set x-axis labels and legend
    ax.set_xticks(np.arange(len(unique_xvar)))
    ax.set_xticklabels(unique_xvar, rotation=0)  # Rotate labels if needed
    ax.set_xlabel(x_var.capitalize(), fontweight='bold', fontsize=14)
    ax.set_ylabel(f'{y_var} %', fontweight='bold', fontsize=14)
    ax.legend()
    
    # Display the plot
    plt.show()
    
    # Return the series for each dataset
    return unique_xvar, ds_sentinel2, ds_gswo, ds_glad

def interaction_plot_one_scope_with_error(df, x_var, y_var, scope, threshold_low, threshold_high):
    # Filter data by dataset, scope, and threshold range
    df_sentinel2 = df[(df['scope'] == scope) & 
                      (df['dataset'] == 'sentinel2') & 
                      (df['threshold'] >= threshold_low) & (df['threshold'] <= threshold_high)]
    ds_sentinel2_mean = df_sentinel2.groupby(x_var)[y_var].mean()
    ds_sentinel2_min = df_sentinel2.groupby(x_var)[y_var].min()
    ds_sentinel2_max = df_sentinel2.groupby(x_var)[y_var].max()
    ds_sentinel2_lower_err = ds_sentinel2_mean - ds_sentinel2_min
    ds_sentinel2_upper_err = ds_sentinel2_max - ds_sentinel2_mean
    
    df_gswo = df[(df['scope'] == scope) & 
                 (df['dataset'] == 'gswo') & 
                 (df['threshold'] >= threshold_low) & (df['threshold'] <= threshold_high)]
    ds_gswo_mean = df_gswo.groupby(x_var)[y_var].mean()
    ds_gswo_min = df_gswo.groupby(x_var)[y_var].min()
    ds_gswo_max = df_gswo.groupby(x_var)[y_var].max()
    ds_gswo_lower_err = ds_gswo_mean - ds_gswo_min
    ds_gswo_upper_err = ds_gswo_max - ds_gswo_mean
    
    df_glad = df[(df['scope'] == scope) & 
                 (df['dataset'] == 'glad') & 
                 (df['threshold'] >= threshold_low) & (df['threshold'] <= threshold_high)]
    ds_glad_mean = df_glad.groupby(x_var)[y_var].mean()
    ds_glad_min = df_glad.groupby(x_var)[y_var].min()
    ds_glad_max = df_glad.groupby(x_var)[y_var].max()
    ds_glad_lower_err = ds_glad_mean - ds_glad_min
    ds_glad_upper_err = ds_glad_max - ds_glad_mean
    
    # Prepare for plotting
    unique_xvar = ds_sentinel2_mean.index.union(ds_gswo_mean.index).union(ds_glad_mean.index)
    unique_xvar = sorted(unique_xvar)  # Sort the x variables for consistency
    
    colors = {
        'Sentinel2': '#1f77b4',      # Blue
        'GSWO': '#ff7f0e',           # Orange
        'GLAD': '#2ca02c'            # Green
    }
    
    markers = {
        'Sentinel2': 'o',            # Circle marker for Sentinel2
        'GSWO': 's',                 # Square marker for GSWO
        'GLAD': '^'                  # Triangle marker for GLAD
    }
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotting points with error bars for each dataset
    ax.errorbar(unique_xvar, ds_sentinel2_mean.reindex(unique_xvar).fillna(0).values, 
                yerr=[
                    ds_sentinel2_lower_err.reindex(unique_xvar).fillna(0).values, 
                    ds_sentinel2_upper_err.reindex(unique_xvar).fillna(0).values
                ], 
                fmt=markers['Sentinel2'], 
                color=colors['Sentinel2'], 
                label="Sentinel2", 
                linestyle='None',   # No line connecting points
                markersize=10, 
                capsize=5)           # Error bar cap size
    
    ax.errorbar(unique_xvar, ds_gswo_mean.reindex(unique_xvar).fillna(0).values, 
                yerr=[
                    ds_gswo_lower_err.reindex(unique_xvar).fillna(0).values, 
                    ds_gswo_upper_err.reindex(unique_xvar).fillna(0).values
                ], 
                fmt=markers['GSWO'], 
                color=colors['GSWO'], 
                label="GSWO", 
                linestyle='None', 
                markersize=1, 
                capsize=5)
    
    ax.errorbar(unique_xvar, ds_glad_mean.reindex(unique_xvar).fillna(0).values, 
                yerr=[
                    ds_glad_lower_err.reindex(unique_xvar).fillna(0).values, 
                    ds_glad_upper_err.reindex(unique_xvar).fillna(0).values
                ], 
                fmt=markers['GLAD'], 
                color=colors['GLAD'], 
                label="GLAD", 
                linestyle='None', 
                markersize=10, 
                capsize=5)
    
    # Set x-axis labels and legend
    ax.set_xticks(np.arange(len(unique_xvar)))
    ax.set_xticklabels(unique_xvar, rotation=0)  # Rotate labels if needed
    ax.set_xlabel(x_var.capitalize(), fontweight='bold', fontsize=14)
    ax.set_ylabel(f'{y_var} %', fontweight='bold', fontsize=14)
    ax.legend()
    
    # Display the plot
    plt.show()
    
    # Return the series for each dataset
    return unique_xvar, ds_sentinel2_mean, ds_gswo_mean, ds_glad_mean



# %% Plot seasonal change for both scopes ('all_pld' and 'matched_is2')

df_analyze = seasonal_results.droplevel(level=1, axis=1)
keep_cols = ['roi_name', 'dataset', 'scope', 'buffer', 'threshold', 'net_total_sn_frac', 'net_lake_sn_frac']
drop_cols = [col for col in df_analyze.columns if col not in keep_cols]

df_analyze = df_analyze[
    (df_analyze['buffer'] == 120) & (df_analyze['roi_name'] != 'MRD_TUK_Anderson')
]

df_bar = df_analyze.copy()
df_bar = df_bar[df_bar['threshold'] == 80]

# Compare the variability of ICESat-2 lakes vs all PLD datasets
# Clustered bar plot
(returned_items_all_scopes) = side_by_side_bar_two_scopes(
    df_bar, 'roi_name', 'net_lake_sn_frac'
)

# Simple bar plot instead of interaction plot
side_by_side_bar_one_scope(
    df_bar, 'roi_name', 'net_total_sn_frac', 'all_pld'
)

# Interaction plot without error bars
interaction_plot_one_scope(
    df_analyze, 'roi_name', 'net_lake_sn_frac', 'all_pld'
)

"""
This is my favorite visualization thus far.
"""

# Interaction plots with error bars
(returned_items_lake_all_pld) = interaction_plot_one_scope_with_error(
    df_analyze, 'roi_name', 'net_lake_sn_frac', 'all_pld', 70, 86
)

(returned_items_total_all_pld) = interaction_plot_one_scope_with_error(
    df_analyze, 'roi_name', 'net_total_sn_frac', 'all_pld', 70, 86
)

(returned_items_lake_matched_is2) = interaction_plot_one_scope_with_error(
    df_analyze, 'roi_name', 'net_lake_sn_frac', 'matched_is2', 70, 86
)

(returned_items_total_matched_is2) = interaction_plot_one_scope_with_error(
    df_analyze, 'roi_name', 'net_total_sn_frac', 'matched_is2', 70, 86
)

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

is2_seasonality['seasonality'] = is2_seasonality[('mean_zdif', '8')] - is2_seasonality[('mean_zdif', '6')]

drop_cols = [
    ('lakes_cnt', '6'),
    ('lakes_cnt', '8'),
    ('mean_zdif', '6'),
    ('mean_zdif', '8'),
    ('std_zdif', '6'),
    ('std_zdif', '8')
]

is2_seasonality.drop(
    columns=drop_cols,
    inplace=True)

is2_seasonality.columns = is2_seasonality.columns.droplevel(1)

# %% Visualize just the ICESat-2 WSE (m) seasonality
is2_seasonality = is2_seasonality[is2_seasonality['roi_name'] != 'MRD_TUK_Anderson']

ordered_rois = returned_items_lake_matched_is2[0]

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

mean_values_df = returned_items_all_scopes[1].reset_index()
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
                    label='Area Change Mean of Optical Datasets',
                    color='grey', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('Lake Area Change % (June - August)')

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
ax1.legend(handles1, labels1, loc='lower left')

plt.title('Comparing seasonal draw-downs between optical datasets and ICESat-2')

# Display the plot
plt.show()
# %% Scatter plot probably works better for this comparison


from scipy import stats

scatter_df = merged_df.copy()
sentinel2 = returned_items_all_scopes[5].reset_index()
sentinel2.columns = ['roi_name', 'seasonality_s2']
scatter_df = pd.merge(scatter_df, sentinel2, on='roi_name', how='inner')

print(scatter_df)

# Your scatter plot data
x = merged_df['seasonality_is2']
y1 = merged_df['seasonality_mean']
y2 = scatter_df['seasonality_s2']

# Create the scatter plot
x = scatter_df['seasonality_is2']
y1 = scatter_df['seasonality_mean']
y2 = scatter_df['seasonality_s2']

# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the first data series
ax.scatter(x, y1, label='Optical Datasets Mean', color='grey', edgecolor='black', alpha=0.7, s=100, marker='o')

# Calculate the best-fit line for the first data series
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x, y1)
line1 = slope1 * x + intercept1

# Plot the best-fit line for the first data series
ax.plot(x, line1, color='red', linestyle='-', label=f'Best Fit Line 1 (R² = {r_value1**2:.2f})')

# Plot the second data series
ax.scatter(x, y2, label='Sentinel-2 Area Change', color='blue', edgecolor='black', alpha=0.7, s=100, marker='s')

# Calculate the best-fit line for the second data series
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x, y2)
line2 = slope2 * x + intercept2

# Plot the best-fit line for the second data series
ax.plot(x, line2, color='green', linestyle='--', label=f'Best Fit Line 2 (R² = {r_value2**2:.2f})')

# Add labels and title
ax.set_xlabel('WSE Change (meters)')
ax.set_ylabel('Lake Area Change % (June - August)')
ax.set_title('Comparison of Lake Area Change vs WSE Change')
ax.legend()
ax.grid(True)

# Display the plot
plt.show()


# %% Compare the matched Sentinel-2 lakes and ICESat-2
optical_values_df = returned_items_all_scopes[5].reset_index()
optical_values_df.columns = ['roi_name', 'seasonality_mean']
merged_df = pd.merge(is2_seasonality, optical_values_df, on='roi_name', how='inner')
merged_df.rename(columns={'seasonality': 'seasonality_is2'}, inplace=True)

merged_df['seasonality_is2_rescaled'] = merged_df['seasonality_is2'] * rescale_factor

# Set up the x-axis positions
x = np.arange(len(merged_df['roi_name']))
width = 0.35  # Width of the bars

fig, ax1 = plt.subplots(figsize=(14, 4))

# Plot the optical dataset bars on ax1
bars_mean = ax1.bar(x - width/2, merged_df['seasonality_mean'], width,
                    label='Sentinel-2 Area Change',
                    color='blue', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('Area Change % of Lake Area (June - August)')

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
ax1.legend(handles1, labels1, loc='lower left')

plt.title('Matched Sentinel-2 and IS2 Seasonality')

# Display the plot
plt.show()


# %% Compare the matched GSWO lakes and ICESat-2

optical_values_df = returned_items_all_scopes[6].reset_index()
optical_values_df.columns = ['roi_name', 'seasonality_mean']
merged_df = pd.merge(is2_seasonality, optical_values_df, on='roi_name', how='inner')
merged_df.rename(columns={'seasonality': 'seasonality_is2'}, inplace=True)

merged_df['seasonality_is2_rescaled'] = merged_df['seasonality_is2'] * rescale_factor

# Set up the x-axis positions
x = np.arange(len(merged_df['roi_name']))
width = 0.35  # Width of the bars

fig, ax1 = plt.subplots(figsize=(14, 4))

# Plot the optical dataset bars on ax1
bars_mean = ax1.bar(x - width/2, merged_df['seasonality_mean'], width,
                    label='GSWO Area Change',
                    color='orange', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('% Change of Lake Area (June - August)')

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
ax1.legend(handles1, labels1, loc='lower left')

plt.title('GSWO and IS2 Seasonality')

# Display the plot
plt.show()

# %% Compare the matched GLAD lakes and ICESat-2


optical_values_df = returned_items_all_scopes[7].reset_index()
optical_values_df.columns = ['roi_name', 'seasonality_mean']
merged_df = pd.merge(is2_seasonality, optical_values_df, on='roi_name', how='inner')
merged_df.rename(columns={'seasonality': 'seasonality_is2'}, inplace=True)

merged_df['seasonality_is2_rescaled'] = merged_df['seasonality_is2'] * rescale_factor

# Set up the x-axis positions
x = np.arange(len(merged_df['roi_name']))
width = 0.35  # Width of the bars

fig, ax1 = plt.subplots(figsize=(14, 4))

# Plot the optical dataset bars on ax1
bars_mean = ax1.bar(x - width/2, merged_df['seasonality_mean'], width,
                    label='GLAD Area Change',
                    color='green', edgecolor='black', alpha=0.7)

# Plot the rescaled IS2 data on ax1
bars_is2 = ax1.bar(x + width/2, merged_df['seasonality_is2_rescaled'], width,
                   label='WSE Change (meters)', color='pink', edgecolor='black')

# Set up the x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(merged_df['roi_name'], rotation=0)
ax1.set_ylabel('Lake Area Change % (June - August)')

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
ax1.legend(handles1, labels1, loc='lower left')

plt.title('GLAD and IS2 Seasonality')

# Display the plot
plt.show()

# %%
