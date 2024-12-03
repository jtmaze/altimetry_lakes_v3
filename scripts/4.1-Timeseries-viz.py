# %% 1.0 Visualize timeseries data for AGU

import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir('/Users/jmaze/Documents/projects/altimetry_lakes_v3/')

# %% 2.0 Load the data

lake_ts_path = './data/lake_timeseries/'
files_list = glob.glob(f'{lake_ts_path}/*.csv')

timeseries = []

for file in files_list:
	df = pd.read_csv(file)
	timeseries.append(df)

timeseries = pd.concat(timeseries)

timeseries['obs_date'] = pd.to_datetime(timeseries['obs_date'])
lake_ids = timeseries['lake_id'].unique()

# %% 3.0 Plot a random selection of lakes.

n_lakes = 5 
sampled_lake_ids = np.random.choice(lake_ids, n_lakes, replace=False)

plt.figure(figsize=(12, 8))

for lake_id in sampled_lake_ids:
	lake_data = timeseries[timeseries['lake_id'] == lake_id]
	lake_data = lake_data.sort_values(by='obs_date')
	plt.plot(
        lake_data['obs_date'], 
        lake_data['zdif_date'], 
		label=lake_id,
		marker='o',
		markersize=10,
		linestyle='-'
    )

plt.xlabel('Observation Date')
plt.ylabel('Median Elevation')
plt.title('Lake Elevation Over Time for Sampled Lakes')
plt.legend()
plt.show()

# %% 4.0 Plot illustrative timeseries for figure. 
plot_lakeids = [
    'AKCP_id_8130646682', 'YKdelta_id_8120590262', 'AKCP_id_8130481942', 
    'MRD_TUK_Anderson_id_8230226852', 'MRD_TUK_Anderson_id_8230360542',
    'AKCP_id_8130230822'
]

plt.figure(figsize=(12, 8))

for idx, lake_id in enumerate(plot_lakeids, start=1):
    lake_data = timeseries[timeseries['lake_id'] == lake_id]
    lake_data = lake_data.sort_values(by='obs_date')
    plt.plot(
        lake_data['obs_date'], 
        lake_data['zdif_date'], 
        label=f'Lake #{idx}',
        marker='o',
        markersize=10,
        linestyle='-'
    )

plt.xlabel('Observation Date')
plt.ylabel('Elevation Difference (m)')
plt.title('Elevation difference from lake 5-yr median WSE (no filtering)')
plt.legend()
plt.show()

# %% Plot timeseries filtered

plt.figure(figsize=(12, 8))

for idx, lake_id in enumerate(plot_lakeids, start=1):
    lake_data = timeseries[(timeseries['lake_id'] == lake_id) & (timeseries['zdif_date'] < 1)]
    lake_data = lake_data.sort_values(by='obs_date')
    plt.plot(
        lake_data['obs_date'], 
        lake_data['zdif_date'], 
        label=f'Lake #{idx}',
        marker='o',
        markersize=10,
        linestyle='-'
    )

plt.xlabel('Observation Date')
plt.ylabel('Elevation Difference (m)')
plt.title('Elevation difference from lake 5-yr median WSE (Filtered)')
plt.legend()
plt.show()



# %%
