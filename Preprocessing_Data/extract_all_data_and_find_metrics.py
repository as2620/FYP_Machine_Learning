import json
import heartpy as hp
import scipy.signal
import csv

def separate_shirt_data(data):
    timestamps = []
    chest_coil = []
    abdomen_coil = []
    gsr = []
    ppg_ir = []
    ppg_red = []

    for key, values in data.items():
        timestamps.append(str(values['timestamp']))
        chest_coil.append(int(values['chest_coil']))
        abdomen_coil.append(int(values['abdomen_coil']))
        gsr.append(int(values['gsr']))
        ppg_ir.append(int(values['ppg_ir']))
        ppg_red.append(int(values['ppg_red']))

    return {
        'timestamps': timestamps,
        'chest_coil': chest_coil,
        'abdomen_coil': abdomen_coil,
        'gsr': gsr,
        'ppg_ir': ppg_ir,
        'ppg_red': ppg_red
    }

def separate_mask_data(data):
    timestamps = []
    co2 = []
    voc = []

    for key, values in data.items():
        timestamps.append(str(values['timestamp']))
        co2.append(int(values['CO2']))
        voc.append(int(values['VOC']))

    return {
        'timestamps': timestamps,
        'co2': co2,
        'voc': voc
    }

# Define filepaths and user details
filepath = '../../Machine_Learning_Data/Stroop_Trial_2/stroop_trial_2_as.json'
output_csv = '../../Machine_Learning_Data/filtered_signals_data.csv'
user_id = '10'
classification = 'S2'

# Load the json data
with open(filepath, 'r') as file:
    json_data = json.load(file)

# For each filepath/data, split into signals.
data = json_data['UsersData']
user_data = data['GS5IERwOapdXlupUOqk5y52Vb5m1']

shirt_data = user_data['shirt_data']
shirt_data = separate_shirt_data(shirt_data)
shirt_timestamps = shirt_data['timestamps']

mask_data = user_data['mask_data']
mask_data = separate_mask_data(mask_data)
mask_timestamps = mask_data['timestamps']

# Filter each signal
chest_coil = scipy.signal.detrend(shirt_data['chest_coil'])
chest_coil = hp.filter_signal(chest_coil, cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

abdomen_coil = scipy.signal.detrend(shirt_data['abdomen_coil'])
abdomen_coil = hp.filter_signal(abdomen_coil, cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

gsr = hp.filter_signal(shirt_data['gsr'], cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

ppg_red = scipy.signal.detrend(shirt_data['ppg_red'])
ppg_red = hp.filter_signal(ppg_red, cutoff=[0.5, 4.5], sample_rate=20.0, filtertype='bandpass', return_top=False)

ppg_ir = scipy.signal.detrend(shirt_data['ppg_ir'])
ppg_ir = hp.filter_signal(ppg_ir, cutoff=[0.5, 4.5], sample_rate=20.0, filtertype='bandpass', return_top=False)

co2 = hp.filter_signal(mask_data['co2'], cutoff=1.5, sample_rate=20.0, filtertype='lowpass', return_top=False)

voc = hp.filter_signal(mask_data['voc'], cutoff=1.5, sample_rate=20.0, filtertype='lowpass', return_top=False)

# Align timestamps and merge data
aligned_data = []
shirt_index, mask_index = 0, 0

while shirt_index < len(shirt_timestamps) or mask_index < len(mask_timestamps):
    if shirt_index < len(shirt_timestamps):
        current_time = shirt_timestamps[shirt_index]
    else:
        current_time = mask_timestamps[mask_index]

    if mask_index < len(mask_timestamps):
        mask_time = mask_timestamps[mask_index]
    else:
        mask_time = shirt_timestamps[shirt_index]

    if current_time == mask_time:
        aligned_data.append({
            'timestamp': current_time,
            'user_id': user_id,
            'classification': classification,
            'chest_coil': chest_coil[shirt_index],
            'abdomen_coil': abdomen_coil[shirt_index],
            'gsr': gsr[shirt_index],
            'ppg_red': ppg_red[shirt_index],
            'ppg_ir': ppg_ir[shirt_index],
            'co2': co2[mask_index],
            'voc': voc[mask_index]
        })
        shirt_index += 1
        mask_index += 1
    elif shirt_index < len(shirt_timestamps) and (mask_index >= len(mask_timestamps) or shirt_timestamps[shirt_index] < mask_timestamps[mask_index]):
        aligned_data.append({
            'timestamp': shirt_timestamps[shirt_index],
            'user_id': user_id,
            'classification': classification,
            'chest_coil': chest_coil[shirt_index],
            'abdomen_coil': abdomen_coil[shirt_index],
            'gsr': gsr[shirt_index],
            'ppg_red': ppg_red[shirt_index],
            'ppg_ir': ppg_ir[shirt_index],
            'co2': co2[mask_index-1] if mask_index > 0 else '',
            'voc': voc[mask_index-1] if mask_index > 0 else ''
        })
        shirt_index += 1
    else:
        aligned_data.append({
            'timestamp': mask_timestamps[mask_index],
            'user_id': user_id,
            'classification': classification,
            'chest_coil': chest_coil[shirt_index-1] if shirt_index > 0 else '',
            'abdomen_coil': abdomen_coil[shirt_index-1] if shirt_index > 0 else '',
            'gsr': gsr[shirt_index-1] if shirt_index > 0 else '',
            'ppg_red': ppg_red[shirt_index-1] if shirt_index > 0 else '',
            'ppg_ir': ppg_ir[shirt_index-1] if shirt_index > 0 else '',
            'co2': co2[mask_index],
            'voc': voc[mask_index]
        })
        mask_index += 1

# Write aligned and filtered signals to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['timestamp', 'user_id', 'classification', 'chest_coil', 'abdomen_coil', 'gsr', 'ppg_red', 'ppg_ir', 'co2', 'voc']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in aligned_data:
        writer.writerow(row)

print("Filtered and aligned signals written to:", output_csv)
