#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd

# Data from the table in image 1
data = {
    'Filter Name': ['EKF', 'SIR', '(EKF + SIR)', 'UKF', 'UPF', ' (UKF + UPF)', 'EnKF', 'APF', ' (EnKF + APF)'],
    'Position RMSE (m)': [0.054460, 1.214203, 0.608204, 0.008835, 0.008709, 0.006362, 0.8313820, 0.000000, 0.8245531],
    'Rotation RMSE (m)': [0.004262, 0.004298, 0.004271, 0.092638, 0.350580, 0.091756, 0.0000000, 0.00000006, 0.00000004],
    'Latency (ms)': [0.0055, 73.00, 71.00, 0.123280, 19.44710, 0.056379, 0.179405, 0.090326, 1.290751]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the Position RMSE with increased font size
plt.figure(figsize=(14, 10))  # Increase figure size
colors = ['blue', 'orange', 'magenta', 'red', 'green', 'purple', 'brown', 'black', 'gray']
plt.bar(df['Filter Name'], df['Position RMSE (m)'], color=colors)
plt.xlabel('Filter Name', fontsize=20)
plt.ylabel('Position RMSE (m)', fontsize=20)
plt.title('Position RMSE of Kalman, Particle, and Fused Filters', fontsize=22)

# Adding values on top of the bars
for i, value in enumerate(df['Position RMSE (m)']):
    plt.text(i, value + 0.02, f'{value:.6f}', ha='center', va='bottom', fontsize=14)

plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)

# Increase grid size
#plt.grid(false, which='both', linestyle='--', linewidth=0.5)

# Save the figure in IEEE acceptable format
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/Position_RMSE.eps', format='eps', dpi=300)
plt.show()


# In[4]:


import matplotlib.pyplot as plt
import pandas as pd

# Data from the table in image 1
data = {
    'Filter Name': ['EKF', 'SIR', '(EKF + SIR)', 'UKF', 'UPF', ' (UKF + UPF)', 'EnKF', 'APF', ' (EnKF + APF)'],
    'Position RMSE (m)': [0.054460, 1.214203, 0.608204, 0.008835, 0.008709, 0.006362, 0.8313820, 0.000000, 0.8245531],
    'Rotation RMSE (m)': [0.004262, 0.004298, 0.004271, 0.092638, 0.350580, 0.091756, 0.0000000, 0.00000006, 0.00000004],
    'Latency (ms)': [0.0055, 73.00, 71.00, 0.123280, 19.44710, 0.056379, 0.179405, 0.090326, 1.290751]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the Rotation RMSE with adjusted scale and increased font size
plt.figure(figsize=(12, 8))
plt.bar(df['Filter Name'], df['Rotation RMSE (m)'], color=colors)
plt.xlabel('Filter Name', fontsize=18)
plt.ylabel('Rotation RMSE (m)', fontsize=18)
plt.title('Rotation RMSE of Kalman, Particle, and Fused Filters', fontsize=20)

# Adding values on top of the bars
for i, value in enumerate(df['Rotation RMSE (m)']):
    plt.text(i, value + 0.02, f'{value:.6f}', ha='center', va='bottom', fontsize=13)

# Adjust the scale of the y-axis
plt.ylim(0, 0.5)
plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)

# Increase grid size
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the figure in IEEE acceptable format
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/rotation_RMSE_papr.eps', format='eps', dpi=300)
plt.show()


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd

# Data from the table in image
data = {
    'Filter Name': ['EKF', 'SIR', '(EKF + SIR)', 'UKF', 'UPF', '(UKF + UPF)', 'EnKF', 'APF', '(EnKF + APF)'],
    'Latency (ms)': [0.0055, 73.00, 71.00, 0.123280, 19.44710, 0.056379, 0.179405, 0.090326, 1.290751]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the Latency with increased font size
plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'magenta', 'red', 'green', 'purple', 'brown', 'black', 'gray']
bars = plt.bar(df['Filter Name'], df['Latency (ms)'], color=colors)
plt.xlabel('Filter Name', fontsize=18)
plt.ylabel('Latency (ms)', fontsize=18)
plt.title('Latency of Kalman, Particle, and Fused Filters', fontsize=20)

# Adding values on top of the bars
for bar, value in zip(bars, df['Latency (ms)']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{value:.6f}', ha='center', va='bottom', fontsize=12)

plt.xticks(rotation=90, fontsize=18)
plt.yticks(fontsize=18)

# Increase grid size
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the figure in IEEE acceptable format
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/latency_filters.eps', format='eps', dpi=300)
plt.show()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the new dataset
new_file_path = r'C:\Users\Onyeka\Downloads\EKF_SIR.csv'  # Replace with your actual file path
new_data = pd.read_csv(new_file_path)

# Convert 'Timestamp' to datetime
new_data['Timestamp'] = pd.to_datetime(new_data['Timestamp'])

# Set 'Timestamp' as the index
new_data.set_index('Timestamp', inplace=True)

# Resample the data to 10Hz (every 100ms)
new_data_downsampled = new_data.resample('100L').mean().ffill()

# Calculate RMSE between true positions and filtered positions
true_positions_ds_new = new_data_downsampled[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
kf_positions_ds_new = new_data_downsampled[['KalmanFilteredPosX', 'KalmanFilteredPosY', 'KalmanFilteredPosZ']].values
pf_positions_ds_new = new_data_downsampled[['ParticleFilteredPosX', 'ParticleFilteredPosY', 'ParticleFilteredPosZ']].values
fused_positions_ds_new = new_data_downsampled[['FusedPosX', 'FusedPosY', 'FusedPosZ']].values

rmse_kf = np.sqrt(np.mean((true_positions_ds_new - kf_positions_ds_new) ** 2, axis=1))
rmse_pf = np.sqrt(np.mean((true_positions_ds_new - pf_positions_ds_new) ** 2, axis=1))
rmse_fused = np.sqrt(np.mean((true_positions_ds_new - fused_positions_ds_new) ** 2, axis=1))

# Calculate corrected RMSE (true position minus EKF with lowest RMSE)
min_rmse_idx = np.argmin([np.mean(rmse_kf), np.mean(rmse_pf), np.mean(rmse_fused)])
if min_rmse_idx == 0:
    lowest_rmse_positions = kf_positions_ds_new
elif min_rmse_idx == 1:
    lowest_rmse_positions = pf_positions_ds_new
else:
    lowest_rmse_positions = fused_positions_ds_new

true_positions_ds_new_corrected = true_positions_ds_new - lowest_rmse_positions
rmse_corrected = np.sqrt(np.mean(true_positions_ds_new_corrected ** 2, axis=1))

# Extract downsampled time
time_seconds_new = (new_data_downsampled.index - new_data_downsampled.index[0]).total_seconds()

# Plot RMSE over time
plt.figure(figsize=(10, 5))
plt.plot(time_seconds_new, rmse_kf, label='EKF Position RMSE', color='blue')
plt.plot(time_seconds_new, rmse_pf, label='SIR Position RMSE', color='orange')
plt.plot(time_seconds_new, rmse_fused, label='Fused Position RMSE', color='green')
plt.plot(time_seconds_new, rmse_corrected, label='True Position RMSE', color='red', linestyle='--')
plt.xlabel('Time (ms)', fontsize=15)
plt.ylabel('RMSE (meters)', fontsize=15)  # Assuming units are meters
plt.title('Position RMSE over Time (ms)', fontsize=18)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/Position RMSE_over_time_filters.pdf', format='pdf', dpi=300)
plt.show()


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the new dataset
new_file_path =  r'C:\Users\Onyeka\Downloads\EKF_SIR.csv'  # Replace with your actual file path
new_data = pd.read_csv(new_file_path)

# Convert 'Timestamp' to datetime
new_data['Timestamp'] = pd.to_datetime(new_data['Timestamp'])

# Set 'Timestamp' as the index
new_data.set_index('Timestamp', inplace=True)

# Resample the data to 10Hz (every 100ms)
new_data_downsampled = new_data.resample('100L').mean().ffill()

# Function to calculate RMSE
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2, axis=1))

# Extract rotations from the downsampled data
true_rotations_ds_new = new_data_downsampled[['HeadRotX', 'HeadRotY', 'HeadRotZ']].values
kf_rotations_ds_new = new_data_downsampled[['KalmanFilteredRotX', 'KalmanFilteredRotY', 'KalmanFilteredRotZ']].values
pf_rotations_ds_new = new_data_downsampled[['ParticleFilteredRotX', 'ParticleFilteredRotY', 'ParticleFilteredRotZ']].values
fused_rotations_ds_new = new_data_downsampled[['FusedRotX', 'FusedRotY', 'FusedRotZ']].values

# Calculate RMSE for rotations over time
rmse_kf_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, kf_rotations_ds_new)
rmse_pf_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, pf_rotations_ds_new)
rmse_fused_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, fused_rotations_ds_new)
rmse_true_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, kf_rotations_ds_new)  # This will be zero as it compares true values with themselves

# Extract downsampled time
time_seconds_new = (new_data_downsampled.index - new_data_downsampled.index[0]).total_seconds()

# Plot RMSE for rotations over time with adjusted time labels
plt.figure(figsize=(10, 5))
plt.plot(time_seconds_new, rmse_kf_rot_over_time_ds_new, label='EKF Rotation RMSE', color='blue')
plt.plot(time_seconds_new, rmse_pf_rot_over_time_ds_new, label='SIR Rotation RMSE', color='orange')
plt.plot(time_seconds_new, rmse_fused_rot_over_time_ds_new, label='Fused Rotation RMSE', color='green')
plt.plot(time_seconds_new, rmse_true_rot_over_time_ds_new, label='True Rotation RMSE', color='red', linestyle='dotted')
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.title('Rotation RMSE over Time (ms)', fontsize=18)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/Rotation RMSE_over_time_filters.pdf', format='pdf', dpi=300)
plt.show()


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\UKFUpdated.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Function to calculate RMSE
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2, axis=1))

# Downsample the data to 10Hz (every 100ms)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
data_downsampled = data.resample('100L').mean().ffill()

# Extract positions from the downsampled data
true_positions_ds = data_downsampled[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
ukf_positions_ds = data_downsampled[['UKFPosX', 'UKFPosY', 'UKFPosZ']].values
upf_positions_ds = data_downsampled[['UPFPosX', 'UPFPosY', 'UPFPosZ']].values
fused_positions_ds = data_downsampled[['FusedPosX', 'FusedPosY', 'FusedPosZ']].values

# Calculate RMSE for positions over time
rmse_ukf_pos_over_time_ds = calculate_rmse(true_positions_ds, ukf_positions_ds)
rmse_upf_pos_over_time_ds = calculate_rmse(true_positions_ds, upf_positions_ds)
rmse_fused_pos_over_time_ds = calculate_rmse(true_positions_ds, fused_positions_ds)
rmse_true_pos_over_time_ds = np.linalg.norm(true_positions_ds - fused_positions_ds, axis=1)

# Extract downsampled time
time_seconds = (data_downsampled.index - data_downsampled.index[0]).total_seconds()

# Plot RMSE for positions over time with adjusted time labels
plt.figure(figsize=(10, 5))
plt.plot(time_seconds, rmse_ukf_pos_over_time_ds, label='UKF Position RMSE', color='blue')
plt.plot(time_seconds, rmse_upf_pos_over_time_ds, label='UPF Position RMSE', color='orange')
plt.plot(time_seconds, rmse_fused_pos_over_time_ds, label='Fused Position RMSE', color='green')
plt.plot(time_seconds, rmse_true_pos_over_time_ds, label='True Position RMSE', color='red', linestyle='dotted')
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.title('Position RMSE over Time (ms)', fontsize=18)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/UKF_Position RMSE_over_time_filters.tiff', format='tiff', dpi=300)
plt.show()


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\UKFUpdated.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Function to calculate RMSE
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2, axis=1))

# Downsample the data to 10Hz (every 100ms)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
data_downsampled = data.resample('100L').mean().ffill()

# Extract rotations from the downsampled data
true_rotations_ds = data_downsampled[['HeadRotX', 'HeadRotY', 'HeadRotZ']].values
ukf_rotations_ds = data_downsampled[['UKFRotX', 'UKFRotY', 'UKFRotZ']].values
upf_rotations_ds = data_downsampled[['UPFRotX', 'UPFRotY', 'UPFRotZ']].values
fused_rotations_ds = data_downsampled[['FusedRotX', 'FusedRotY', 'FusedRotZ']].values

# Calculate RMSE for rotations over time
rmse_ukf_rot_over_time_ds = calculate_rmse(true_rotations_ds, ukf_rotations_ds)
rmse_upf_rot_over_time_ds = calculate_rmse(true_rotations_ds, upf_rotations_ds)
rmse_fused_rot_over_time_ds = calculate_rmse(true_rotations_ds, fused_rotations_ds)
rmse_true_rot_over_time_ds = np.linalg.norm(true_rotations_ds - fused_rotations_ds, axis=1)

# Extract downsampled time
time_seconds = (data_downsampled.index - data_downsampled.index[0]).total_seconds()

# Plot RMSE for rotations over time with adjusted time labels
plt.figure(figsize=(10, 5))
plt.plot(time_seconds, rmse_ukf_rot_over_time_ds, label='UKF Rotation RMSE', color='blue')
plt.plot(time_seconds, rmse_upf_rot_over_time_ds, label='UPF Rotation RMSE', color='orange')
plt.plot(time_seconds, rmse_fused_rot_over_time_ds, label='Fused Rotation RMSE', color='green')
plt.plot(time_seconds, rmse_true_rot_over_time_ds, label='True Rotation RMSE', color='red', linestyle='dotted')
plt.xlabel('Time (ms)', fontsize=15)
plt.ylabel('RMSE', fontsize=15)
plt.title('Rotation RMSE over Time (ms)', fontsize=20)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/UKF_Rotation RMSE_over_time_filters.tiff', format='tiff', dpi=300)
plt.show()


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the new dataset
new_file_path =  r'C:\Users\Onyeka\Downloads\EnKF_APF.csv'  # Replace with your actual file path
new_data = pd.read_csv(new_file_path)

# Function to convert custom timestamp format to datetime
def convert_to_datetime(timestamp):
    try:
        minutes, seconds = map(float, timestamp.split(':'))
        return datetime(2023, 1, 1) + timedelta(minutes=minutes, seconds=seconds)
    except ValueError as e:
        print(f"Error parsing timestamp {timestamp}: {e}")
        return datetime(2023, 1, 1)

# Convert 'Timestamp' to datetime
new_data['Timestamp'] = new_data['Timestamp'].apply(convert_to_datetime)

# Set 'Timestamp' as the index
new_data.set_index('Timestamp', inplace=True)

# Resample the data to 10Hz (every 100ms)
new_data_downsampled = new_data.resample('100L').mean().ffill()

# Function to calculate RMSE
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2, axis=1))

# Extract positions from the downsampled data
true_positions_ds_new = new_data_downsampled[['HeadPosX', 'HeadPosY', 'HeadPosZ']].values
kf_positions_ds_new = new_data_downsampled[['ENKFPosX', 'ENKFPosY', 'ENKFPosZ']].values
pf_positions_ds_new = new_data_downsampled[['APFPosX', 'APFPosY', 'APFPosZ']].values
fused_positions_ds_new = new_data_downsampled[['FusedPosX', 'FusedPosY', 'FusedPosZ']].values

# Calculate RMSE for positions over time
rmse_kf_pos_over_time_ds_new = calculate_rmse(true_positions_ds_new, kf_positions_ds_new)
rmse_pf_pos_over_time_ds_new = calculate_rmse(true_positions_ds_new, pf_positions_ds_new)
rmse_fused_pos_over_time_ds_new = calculate_rmse(true_positions_ds_new, fused_positions_ds_new)
rmse_true_pos_over_time_ds_new = calculate_rmse(true_positions_ds_new, pf_positions_ds_new)

# Extract downsampled time
time_seconds_new = (new_data_downsampled.index - new_data_downsampled.index[0]).total_seconds()

# Plot RMSE for positions over time with adjusted time labels
plt.figure(figsize=(10, 5))
plt.plot(time_seconds_new, rmse_kf_pos_over_time_ds_new, label='EnKF Position RMSE', color='blue')
plt.plot(time_seconds_new, rmse_pf_pos_over_time_ds_new, label='APF Position RMSE', color='orange')
plt.plot(time_seconds_new, rmse_fused_pos_over_time_ds_new, label='Fused Position RMSE', color='green')
plt.plot(time_seconds_new, rmse_true_pos_over_time_ds_new, label='True Position RMSE', color='red', linestyle='dotted')

plt.ylim([0, 2.2])  # Adjusted y-axis limit
plt.yticks(np.arange(0, 2.5, 0.5))
plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.title('Position RMSE over Time (ms)', fontsize=18)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/EnKF_APF_Position RMSE_over_time_filters.pdf', format='pdf', dpi=300)
plt.show()


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the new dataset
new_file_path = r'C:\Users\Onyeka\Downloads\EnKF_APF.csv'  # Replace with your actual file path
new_data = pd.read_csv(new_file_path)

# Function to convert custom timestamp format to datetime
def convert_to_datetime(timestamp):
    try:
        minutes, seconds = map(float, timestamp.split(':'))
        return datetime(2023, 1, 1) + timedelta(minutes=minutes, seconds=seconds)
    except ValueError as e:
        print(f"Error parsing timestamp {timestamp}: {e}")
        return datetime(2023, 1, 1)

# Convert 'Timestamp' to datetime
new_data['Timestamp'] = new_data['Timestamp'].apply(convert_to_datetime)

# Set 'Timestamp' as the index
new_data.set_index('Timestamp', inplace=True)

# Resample the data to 10Hz (every 100ms)
new_data_downsampled = new_data.resample('100L').mean().ffill()

# Function to calculate RMSE
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2, axis=1))

# Extract rotations from the downsampled data
true_rotations_ds_new = new_data_downsampled[['HeadRotX', 'HeadRotY', 'HeadRotZ']].values
kf_rotations_ds_new = new_data_downsampled[['ENKFRotX', 'ENKFRotY', 'ENKFRotZ']].values
pf_rotations_ds_new = new_data_downsampled[['APFRotX', 'APFRotY', 'APFRotZ']].values
fused_rotations_ds_new = new_data_downsampled[['FusedRotX', 'FusedRotY', 'FusedRotZ']].values

# Calculate RMSE for rotations over time
rmse_kf_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, kf_rotations_ds_new)
rmse_pf_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, pf_rotations_ds_new)
rmse_fused_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, fused_rotations_ds_new)
rmse_true_rot_over_time_ds_new = calculate_rmse(true_rotations_ds_new, pf_rotations_ds_new)

# Extract downsampled time
time_seconds_new = (new_data_downsampled.index - new_data_downsampled.index[0]).total_seconds()

# Plot RMSE for rotations over time with adjusted time labels
plt.figure(figsize=(10, 5))
plt.plot(time_seconds_new, rmse_kf_rot_over_time_ds_new, label='EnKF Rotation RMSE', color='blue', linewidth=1.5)
plt.plot(time_seconds_new, rmse_pf_rot_over_time_ds_new, label='APF Rotation RMSE', color='orange', linewidth=1.5)
plt.plot(time_seconds_new, rmse_fused_rot_over_time_ds_new, label='Fused Rotation RMSE', color='green', linewidth=1.5)
plt.plot(time_seconds_new, rmse_true_rot_over_time_ds_new, label='True Rotation RMSE', color='red', linestyle='dotted', linewidth=0.2)

plt.xlabel('Time (ms)', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.title('Rotation RMSE over Time (ms)', fontsize=18)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
plt.grid(False)
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/EnKF_APF_Rotation_RMSE_over_time_filters.pdf', format='pdf', dpi=300)
plt.show()


# In[13]:


#import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from datetime import datetime
#import plotly.graph_objects as go
#import plotly.express as px

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\EKF_SIR.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe and the column names

# Assuming the correct column names based on the CSV inspection
true_x = data['HeadPosX']
true_y = data['HeadPosY']
true_z = data['HeadPosZ']

ekf_x = data['KalmanFilteredPosX']
ekf_y = data['KalmanFilteredPosY']
ekf_z = data['KalmanFilteredPosZ']

sir_x = data['ParticleFilteredPosX']
sir_y = data['ParticleFilteredPosY']
sir_z = data['ParticleFilteredPosZ']

fused_x = data['FusedPosX']
fused_y = data['FusedPosY']
fused_z = data['FusedPosZ']

# Plotting the trajectory positions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# True positions
ax.plot(true_x, true_y, true_z, label='True Position', color='red', linestyle='dotted', linewidth=3)

# EKF positions
ax.plot(ekf_x, ekf_y, ekf_z, label='EKF Position', color='blue', linestyle='solid', linewidth=0.5)

# SIR positions
ax.plot(sir_x, sir_y, sir_z, label='SIR Position', color='orange', linestyle='dashed', linewidth=1.5)

# Fused positions
ax.plot(fused_x, fused_y, fused_z, label='Fused Position', color='green', linestyle='solid', linewidth=0.5)

ax.set_xlabel('Position X (meters)', fontsize=14)
ax.set_ylabel('Position Y (meters)', fontsize=14)
ax.set_zlabel('Position Z (meters)', fontsize=14)
ax.set_title('3D Trajectory Positions of True and Filtered Positions', fontsize=18)
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.2)

plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/EKF_SIR_Trajectory_Positions_3d.pdf', format='pdf', dpi=300)
plt.show()


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\EKF_SIR.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe and the column names

# Assuming the correct column names based on the CSV inspection
true_roll = data['HeadRotX']
true_pitch = data['HeadRotY']
true_yaw = data['HeadRotZ']

ekf_roll = data['KalmanFilteredRotX']
ekf_pitch = data['KalmanFilteredRotY']
ekf_yaw = data['KalmanFilteredRotZ']

sir_roll = data['ParticleFilteredRotX']
sir_pitch = data['ParticleFilteredRotY']
sir_yaw = data['ParticleFilteredRotZ']

fused_roll = data['FusedRotX']
fused_pitch = data['FusedRotY']
fused_yaw = data['FusedRotZ']

# Plotting the rotation trajectory
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# True rotations
ax.plot(true_roll, true_pitch, true_yaw, label='True Rotation', color='red', linestyle='dotted', linewidth=2)

# EKF rotations
ax.plot(ekf_roll, ekf_pitch, ekf_yaw, label='EKF Rotation', color='blue', linestyle='solid', linewidth=1.5)

# SIR rotations
ax.plot(sir_roll, sir_pitch, sir_yaw, label='SIR Rotation', color='orange', linestyle='dashed', linewidth=1.5)

# Fused rotations
ax.plot(fused_roll, fused_pitch, fused_yaw, label='Fused Rotation', color='green', linestyle='solid', linewidth=0.5)

ax.set_xlabel('Roll (degrees)', fontsize=14)
ax.set_ylabel('Pitch (degrees)', fontsize=14)
ax.set_zlabel('Yaw (degrees)', fontsize=14)
ax.set_title('3D Rotation Trajectory of True and Filtered Rotations', fontsize=18)
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/EKF_SIR_Rotation_Trajectory_3d.eps', format='eps', dpi=300)
plt.show()


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\UKFUpdated.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Assuming the correct column names based on the CSV inspection
true_x = data['HeadPosX']
true_y = data['HeadPosY']
true_z = data['HeadPosZ']

ukf_x = data['UKFPosX']
ukf_y = data['UKFPosY']
ukf_z = data['UKFPosZ']

upf_x = data['UPFPosX']
upf_y = data['UPFPosY']
upf_z = data['UPFPosZ']

fused_x = data['FusedPosX']
fused_y = data['FusedPosY']
fused_z = data['FusedPosZ']

# Plotting the trajectory positions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# True positions
ax.plot(true_x, true_y, true_z, label='True Position', color='red', linestyle='dotted', linewidth=1.9)

# UKF positions
ax.plot(ukf_x, ukf_y, ukf_z, label='UKF Position', color='blue', linestyle='solid', linewidth=0.5)

# UPF positions
ax.plot(upf_x, upf_y, upf_z, label='UPF Position', color='orange', linestyle='dashed', linewidth=1.5)

# Fused positions
ax.plot(fused_x, fused_y, fused_z, label='Fused Position', color='green', linestyle='solid', linewidth=0.5)

ax.set_xlabel('Position X (meters)', fontsize=14)
ax.set_ylabel('Position Y (meters)', fontsize=14)
ax.set_zlabel('Position Z (meters)', fontsize=14)
ax.set_title('3D Trajectory Positions of True and Filtered Positions', fontsize=18)
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.4)

plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/UKF_UPF_Trajectory_Positions_3d.eps', format='eps', dpi=300)
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\UKFUpdated.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Assuming the correct column names based on the CSV inspection for rotations
true_rot_x = data['HeadRotX']
true_rot_y = data['HeadRotY']
true_rot_z = data['HeadRotZ']

ukf_rot_x = data['UKFRotX']
ukf_rot_y = data['UKFRotY']
ukf_rot_z = data['UKFRotZ']

upf_rot_x = data['UPFRotX']
upf_rot_y = data['UPFRotY']
upf_rot_z = data['UPFRotZ']

fused_rot_x = data['FusedRotX']
fused_rot_y = data['FusedRotY']
fused_rot_z = data['FusedRotZ']

# Plotting the rotation trajectories
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# True rotations
ax.plot(true_rot_x, true_rot_y, true_rot_z, label='True Rotation', color='red', linestyle='dotted', linewidth=1.9)

# UKF rotations
ax.plot(ukf_rot_x, ukf_rot_y, ukf_rot_z, label='UKF Rotation', color='blue', linestyle='solid', linewidth=2)

# UPF rotations
ax.plot(upf_rot_x, upf_rot_y, upf_rot_z, label='UPF Rotation', color='orange', linestyle='dashed', linewidth=1.5)

# Fused rotations
ax.plot(fused_rot_x, fused_rot_y, fused_rot_z, label='Fused Rotation', color='green', linestyle='solid', linewidth=2)

ax.set_xlabel('Rotation X (degrees)', fontsize=14)
ax.set_ylabel('Rotation Y (degrees)', fontsize=14)
ax.set_zlabel('Rotation Z (degrees)', fontsize=14)
ax.set_title('3D Trajectory Rotations of True and Filtered Rotations', fontsize=18)
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.4)

plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/UKF_UPF_Trajectory_Rotations_3d.eps', format='eps', dpi=300)
plt.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\EnKF_APF.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Assuming the correct column names based on the CSV inspection
true_x = data['HeadPosX']
true_y = data['HeadPosY']
true_z = data['HeadPosZ']

enkf_x = data['ENKFPosX']
enkf_y = data['ENKFPosY']
enkf_z = data['ENKFPosZ']

apf_x = data['APFPosX']
apf_y = data['APFPosY']
apf_z = data['APFPosZ']

fused_x = data['FusedPosX']
fused_y = data['FusedPosY']
fused_z = data['FusedPosZ']

# Plotting the trajectory positions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# True positions
ax.plot(true_x, true_y, true_z, label='True Position', color='red', linestyle='dotted', linewidth=1.5)

# EnKF positions
ax.plot(enkf_x, enkf_y, enkf_z, label='EnKF Position', color='blue', linestyle='solid', linewidth=2)

# APF positions
ax.plot(apf_x, apf_y, apf_z, label='APF Position', color='orange', linestyle='dashed', linewidth=0.5)

# Fused positions
ax.plot(fused_x, fused_y, fused_z, label='Fused Position', color='green', linestyle='solid', linewidth=2)

ax.set_xlabel('Position X (meters)', fontsize=14)
ax.set_ylabel('Position Y (meters)', fontsize=14)
ax.set_zlabel('Position Z (meters)', fontsize=14)
ax.set_title('3D Trajectory Positions of True and Filtered Positions', fontsize=18)
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.4)


plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/EnKF_APF_Trajectory_Positions_3d.eps', format='eps', dpi=300)
plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
file_path = r'C:\Users\Onyeka\Downloads\EnKF_APF.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Assuming the correct column names based on the CSV inspection for rotations
true_rot_x = data['HeadRotX']
true_rot_y = data['HeadRotY']
true_rot_z = data['HeadRotZ']

enkf_rot_x = data['ENKFRotX']
enkf_rot_y = data['ENKFRotY']
enkf_rot_z = data['ENKFRotZ']

apf_rot_x = data['APFRotX']
apf_rot_y = data['APFRotY']
apf_rot_z = data['APFRotZ']

fused_rot_x = data['FusedRotX']
fused_rot_y = data['FusedRotY']
fused_rot_z = data['FusedRotZ']

# Plotting the rotation trajectories
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# True rotations
ax.plot(true_rot_x, true_rot_y, true_rot_z, label='True Rotation', color='red', linestyle='dotted', linewidth=1.9)

# EnKF rotations
ax.plot(enkf_rot_x, enkf_rot_y, enkf_rot_z, label='EnKF Rotation', color='blue', linestyle='solid', linewidth=0.5)

# APF rotations
ax.plot(apf_rot_x, apf_rot_y, apf_rot_z, label='APF Rotation', color='orange', linestyle='dashed', linewidth=1.5)

# Fused rotations
ax.plot(fused_rot_x, fused_rot_y, fused_rot_z, label='Fused Rotation', color='green', linestyle='solid', linewidth=0.5)

ax.set_xlabel('Rotation X (degrees)', fontsize=14)
ax.set_ylabel('Rotation Y (degrees)', fontsize=14)
ax.set_zlabel('Rotation Z (degrees)', fontsize=14)
ax.set_title('3D Trajectory Rotations of True and Filtered Rotations', fontsize=18)
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.4)

plt.tight_layout()
plt.savefig('C:/Users/Onyeka/Downloads/Doctoral School/EnKF_APF_Trajectory_Rotations_3d.eps', format='eps', dpi=300)
plt.show()


# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data from the table in the image
data = {
    'Filter Name': ['EKF', 'SIR', 'Fused (EKF + SIR)', 'UKF', 'UPF', 'Fused (UKF + UPF)', 'EnKF', 'APF', 'Fused (EnKF + APF)'],
    'Position RMSE (m)': [0.054460, 1.214203, 0.608204, 0.008835, 0.008709, 0.006362, 0.8313820, 0.000000, 0.8245531],
    'Rotation RMSE (m)': [0.004262, 0.004298, 0.004271, 0.092638, 0.350580, 0.091756, 0.0000000, 0.00000006, 0.00000004],
    'Latency (ms)': [0.0055, 73.00, 71.00, 0.123280, 19.44710, 0.056379, 0.179405, 0.090326, 1.290751]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a dictionary to store correlation matrices for each filter group
correlation_matrices = {}

# Define the filter groups
filter_groups = {
    
    'EKF_SIR_Fused': ['EKF', 'SIR', 'Fused (EKF + SIR)'],
    #'UKF_UPF_Fused': ['UKF', 'UPF', 'Fused (UKF + UPF)'],
    #'EnKF_APF_Fused': ['EnKF', 'APF', 'Fused (EnKF + APF)']
}

# Calculate the correlation matrix for each filter group and store in the dictionary
for group_name, filters in filter_groups.items():
    group_df = df[df['Filter Name'].isin(filters)]
    correlation_matrix = group_df[['Position RMSE (m)', 'Rotation RMSE (m)', 'Latency (ms)']].corr()
    correlation_matrices[group_name] = correlation_matrix

# Plot each correlation matrix using a heatmap
for group_name, matrix in correlation_matrices.items():
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix Heatmap - {group_name}')
    plt.savefig(r'C:/Users/Onyeka/Downloads/Doctoral School/EKF_SIR_FusedCorrelation_Matrix_{group_name}.eps', format='eps', dpi=300)
    plt.show()

    


# In[23]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data from the table in the image
data = {
    'Filter Name': ['EKF', 'SIR', 'Fused (EKF + SIR)', 'UKF', 'UPF', 'Fused (UKF + UPF)', 'EnKF', 'APF', 'Fused (EnKF + APF)'],
    'Position RMSE (m)': [0.054460, 1.214203, 0.608204, 0.008835, 0.008709, 0.006362, 0.8313820, 0.000000, 0.8245531],
    'Rotation RMSE (m)': [0.004262, 0.004298, 0.004271, 0.092638, 0.350580, 0.091756, 0.0000000, 0.00000006, 0.00000004],
    'Latency (ms)': [0.0055, 73.00, 71.00, 0.123280, 19.44710, 0.056379, 0.179405, 0.090326, 1.290751]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a dictionary to store correlation matrices for each filter group
correlation_matrices = {}

# Define the filter groups
filter_groups = {
    
    #'EKF_SIR_Fused': ['EKF', 'SIR', 'Fused (EKF + SIR)'],
    'UKF_UPF_Fused': ['UKF', 'UPF', 'Fused (UKF + UPF)'],
    #'EnKF_APF_Fused': ['EnKF', 'APF', 'Fused (EnKF + APF)']
}

# Calculate the correlation matrix for each filter group and store in the dictionary
for group_name, filters in filter_groups.items():
    group_df = df[df['Filter Name'].isin(filters)]
    correlation_matrix = group_df[['Position RMSE (m)', 'Rotation RMSE (m)', 'Latency (ms)']].corr()
    correlation_matrices[group_name] = correlation_matrix

# Plot each correlation matrix using a heatmap
for group_name, matrix in correlation_matrices.items():
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix Heatmap - {group_name}')
    plt.savefig(r'C:/Users/Onyeka/Downloads/Doctoral School/UKF_UPF_FusedCorrelation_Matrix_{group_name}.eps', format='eps', dpi=300)
    plt.show()

    


# In[24]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data from the table in the image
data = {
    'Filter Name': ['EKF', 'SIR', 'Fused (EKF + SIR)', 'UKF', 'UPF', 'Fused (UKF + UPF)', 'EnKF', 'APF', 'Fused (EnKF + APF)'],
    'Position RMSE (m)': [0.054460, 1.214203, 0.608204, 0.008835, 0.008709, 0.006362, 0.8313820, 0.000000, 0.8245531],
    'Rotation RMSE (m)': [0.004262, 0.004298, 0.004271, 0.092638, 0.350580, 0.091756, 0.0000000, 0.00000006, 0.00000004],
    'Latency (ms)': [0.0055, 73.00, 71.00, 0.123280, 19.44710, 0.056379, 0.179405, 0.090326, 1.290751]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a dictionary to store correlation matrices for each filter group
correlation_matrices = {}

# Define the filter groups
filter_groups = {
    
    #'EKF_SIR_Fused': ['EKF', 'SIR', 'Fused (EKF + SIR)'],
    #'UKF_UPF_Fused': ['UKF', 'UPF', 'Fused (UKF + UPF)'],
    'EnKF_APF_Fused': ['EnKF', 'APF', 'Fused (EnKF + APF)']
}

# Calculate the correlation matrix for each filter group and store in the dictionary
for group_name, filters in filter_groups.items():
    group_df = df[df['Filter Name'].isin(filters)]
    correlation_matrix = group_df[['Position RMSE (m)', 'Rotation RMSE (m)', 'Latency (ms)']].corr()
    correlation_matrices[group_name] = correlation_matrix

# Plot each correlation matrix using a heatmap
for group_name, matrix in correlation_matrices.items():
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Matrix Heatmap - {group_name}')
    plt.savefig(r'C:/Users/Onyeka/Downloads/Doctoral School/EnKF_APF_Fused_Correlation_Matrix_{group_name}.eps', format='eps', dpi=300)
    plt.show()

    


# In[25]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the image
img = mpimg.imread(r'C:\Users\Onyeka\Downloads\Doctoral School\performancepapr\CockPitHeadtrackingResult1.drawio.png')

# Display the image
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')  # No axes for better clarity
plt.show()


# In[ ]:




