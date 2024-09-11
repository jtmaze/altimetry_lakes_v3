# %% 1. Libraries and read data

import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3')

pixels_summary = pd.read_csv('./data/pixel_counts_test3.csv')
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
            values=['total_pix', 'wtr_frac_measured', 'land_frac_measured']
            ).reset_index()
        
        df['seasonality'] = df[('wtr_frac_measured', 'early')] - df[('wtr_frac_measured', 'late')]
        
        seasonal_diffs.append(df)
        
    elif dataset == 'gswo':
        temp_gswo = temp.copy()
        temp_gswo['timeframe'] = temp_gswo['timeframe'].replace({
            'june': 'early',
            'aug': 'late'
        })

        df = temp_gswo.pivot_table(
            index=['roi_name', 'buffer', 'dataset', 'threshold', 'scope'],
            columns=['timeframe'],
            values=['total_pix', 'wtr_frac_measured', 'land_frac_measured']
            ).reset_index()

        df['seasonality'] = df[('wtr_frac_measured', 'early')] - df[('wtr_frac_measured', 'late')]

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
            ax.set_title(f'ROI: {roi_name}, Dataset: {dataset}, Scope: {scope}')
            ax.set_xlabel('Water Threshold')
            ax.legend(title='Lake Buffer Value (m)')

    axes[0][0].set_ylabel('% Change (Early - Late)')

    # Adjust layout
    plt.tight_layout()
    plt.show()

for roi_name in roi_names:
    by_roi_sensitivity_viz(seasonal_results, roi_name)
# %%

# def side_by_side_bar(df, x_var):
#     # Filter data by dataset
#     df_matched_sentinel2 = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'sentinel2')]
#     ds_matched_sentinel2 = df_matched_sentinel2.groupby(x_var)['seasonality'].mean()
    
#     df_pld_sentinel2 = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'sentinel2')]
#     ds_pld_sentinel2 = df_pld_sentinel2.groupby(x_var)['seasonality'].mean()

#     df_matched_gswo = df[(df['scope'] == 'matched_is2') & (df['dataset'] == 'gswo')]
#     ds_matched_gswo = df_matched_gswo.groupby(x_var)['seasonality'].mean()

#     df_pld_gswo = df[(df['scope'] == 'all_pld') & (df['dataset'] == 'gswo')]
#     ds_matched_gswo = df_pld_gswo.groupby(x_var)['seasonality'].mean()
    
#     mean_seasonality = dsMatched.add(dsPLD, fill_value=0) / 2
#     mean_seasonality = mean_seasonality.sort_values(ascending=False)

#     width = 0.35  # Bar width
#     unique_xvar = mean_seasonality.index.tolist()
#     ind = np.arange(len(unique_xvar))  # the x locations for the groups

#     fig, ax = plt.subplots(figsize=(14, 4))
#     # Plotting bars for each dataset
#     # You need to aggregate or summarize the data if necessary, here assuming sum/mean should be calculated
#     barsIS2 = ax.bar(ind - width/2, 
#                      dsMatched.reindex(unique_xvar).fillna(0).values, 
#                      width, 
#                      edgecolor='black', 
#                      label="Matched IS2 Lakes")
#     barsPLD = ax.bar(ind + width/2, 
#                      dsPLD.reindex(unique_xvar).fillna(0).values, 
#                      width, 
#                      edgecolor='black',
#                      label="All PLD Lakes")


#     # Set x-axis labels and legend
#     ax.set_xticks(ind)
#     ax.set_xticklabels(unique_xvar, rotation=90)  # Rotate labels if they are lengthy
#     ax.set_xlabel(x_var)
#     ax.set_ylabel('Percent change (of lake mask) early to late summer')
#     ax.legend()

#     # Display the plot
#     plt.show()
    
#     return unique_xvar

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
    
    # Prepare for plotting
    unique_xvar = ds_matched_sentinel2.index.union(ds_pld_sentinel2.index).union(ds_matched_gswo.index).union(ds_pld_gswo.index)
    unique_xvar = sorted(unique_xvar)  # Sort the x variables for consistency

    width = 0.2  # Bar width, adjusted for 4 bars
    ind = np.arange(len(unique_xvar))  # the x locations for the groups

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plotting bars for each dataset
    bars_matched_sentinel2 = ax.bar(ind - 1.5*width, 
                                    ds_matched_sentinel2.reindex(unique_xvar).fillna(0).values, 
                                    width, 
                                    edgecolor='black', 
                                    label="Matched IS2 Sentinel2")
    
    bars_pld_sentinel2 = ax.bar(ind - 0.5*width, 
                                ds_pld_sentinel2.reindex(unique_xvar).fillna(0).values, 
                                width, 
                                edgecolor='black', 
                                label="All PLD Sentinel2")
    
    bars_matched_gswo = ax.bar(ind + 0.5*width, 
                               ds_matched_gswo.reindex(unique_xvar).fillna(0).values, 
                               width, 
                               edgecolor='black', 
                               label="Matched IS2 GSWO")
    
    bars_pld_gswo = ax.bar(ind + 1.5*width, 
                           ds_pld_gswo.reindex(unique_xvar).fillna(0).values, 
                           width, 
                           edgecolor='black', 
                           label="All PLD GSWO")
    
    # Set x-axis labels and legend
    ax.set_xticks(ind)
    ax.set_xticklabels(unique_xvar, rotation=90)  # Rotate labels if they are lengthy
    ax.set_xlabel(x_var)
    ax.set_ylabel('Percent change (of lake mask) early to late summer')
    ax.legend()

    # Display the plot
    plt.show()


# %%

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
    (df_analyze['buffer'] == 120) & (df_analyze['threshold'] == 78)
]

ordered_rois = side_by_side_bar(df_analyze, 'roi_name')

# %%
