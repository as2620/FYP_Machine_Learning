# This code will seperate the json data into its relevant signals and 
# find the metrics for each signals. The code will then save the metrics
# into a csv file and timestamped. 
# Since the slowest signal has a waveform every 20s, we will calculate the 
# metrics for each waveform every 20s. 

import json
import matplotlib.pyplot as plt
import heartpy as hp
import neurokit2 as nk
import scipy
from scipy.fft import rfft, rfftfreq
import numpy as np
import csv
import os

def plot_signal(signal, time, title):
    plt.figure(figsize=(8, 4)) 
    plt.plot(time, signal, label=title) 
    plt.title('Plot of a ' + title)  
    plt.xlabel('Time') 
    plt.ylabel('Amplitude') 
    plt.grid(True)  
    plt.show()  

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

def find_max_freq(data, sample_rate=20):
    # # Calculate N/2 to normalize the FFT output
    N = len(data)

    # Plot the actual spectrum of the signal
    freq = rfftfreq(N, d=1/sample_rate)
    amplitudes = 2*np.abs(rfft(data))/N

    rr_in_hz =  freq[np.argmax(amplitudes)]
    rr_in_bpm = rr_in_hz*60

    return rr_in_bpm

def find_spo2(ir_sig, red_sig):
    # Change data type to int32 to avoid overflow errors when squaring
    ir_sig = np.array(ir_sig, dtype=np.longfloat)
    red_sig = np.array(red_sig, dtype=np.longfloat)

    red_mean = np.mean(red_sig)
    ir_mean = np.mean(ir_sig)
    
    red_rms = np.sqrt(np.mean(np.array(red_sig)**2))
    ir_rms = np.sqrt(np.mean(np.array(ir_sig)**2))
   
    red = red_rms/red_mean
    ir = ir_rms/ir_mean

    R = red/ir
    
    spo2 =  ((-45.060*R*R)/10000) + ((30.354*R)/100) + 94.845 
    return spo2

def find_systolic_amplitude_and_hrv(sig):  
    peaks = nk.ppg_findpeaks(sig, sampling_rate=20)
    hrv = nk.hrv_time(peaks, sampling_rate=20)
    peaks = peaks["PPG_Peaks"]

    systolic_amplitudes = []
    peak_amp = []

    for peak in peaks:
        systolic_amplitudes.append(sig[peak])

    for i in range(len(sig)):
        if i in peaks: 
            peak_amp.append(sig[i])
        else:
            peak_amp.append(0)

    # Plot the signal and its peaks
    # plt.figure(figsize=(10, 5))
    # plt.plot(sig, label='Signal', color='blue')
    # plt.plot(peak_amp, color='red', label='Peaks')
    
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.title('Signal with Detected Peaks')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return np.mean(systolic_amplitudes), hrv["HRV_RMSSD"].to_numpy()[0]

def find_rsp_metrics(sig, rate=20):
    signals, _ = nk.rsp_process(sig, sampling_rate=rate)
    rsp_data = nk.rsp_intervalrelated(signals)
    return rsp_data["RSP_Rate_Mean"].to_numpy()[0], np.average(signals["RSP_RVT"]), np.average(signals["RSP_Symmetry_PeakTrough"]), np.average(signals["RSP_Symmetry_RiseDecay"]), rsp_data["RSP_Phase_Duration_Inspiration"].to_numpy()[0], rsp_data["RSP_Phase_Duration_Expiration"].to_numpy()[0], rsp_data["RSP_Phase_Duration_Ratio"].to_numpy()[0] 

def find_gsr_metrics(sig):
    # Process it
    signals, _ = nk.eda_process(sig, sampling_rate=20)
    gsr_data = nk.eda_intervalrelated(signals)
    return gsr_data["SCR_Peaks_N"].to_numpy()[0], gsr_data["SCR_Peaks_Amplitude_Mean"].to_numpy()[0], gsr_data["EDA_Tonic_SD"].to_numpy()[0]

def find_average_exhaled(sig): 
    return np.mean(sig)

# Define filepaths 
filepath = '../../Machine_Learning_Data/Stroop_Trial_2/stroop_trial_2_as.json'
classification = "S2"
particpant_id = "10"
stress_rating = "5"
reaction_time = "864"
score = "84"

#  Define window size and overlap
window_size = 40
overlap = 35

# Length of the data in minutes
minutes = 3

# Define arrays to store the metrics
heart_rates = []
spo2_levels = []
systolic_amplitudes = []
hrvs = []

chest_rates = []
chest_rvts = []
chest_symmetries_pt = []
chest_symmetries_rd = []
chest_inhale_times = []
chest_exhale_times = []
chest_ie_times = []

abdomen_rates = []
abdomen_rvts = []
abdomen_symmetries_pt = []
abdomen_symmetries_rd = []
abdomen_inhale_times = []
abdomen_exhale_times = []
abdomen_ie_times = []

num_sda_peaks = []
average_sda_amplitudes = []
eda_tonic_sds = []

average_co2s = []
average_vocs = []


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


# For each signal, filter. 
chest_coil = scipy.signal.detrend(shirt_data['chest_coil'])
chest_coil = hp.filter_signal(chest_coil, cutoff = 1, sample_rate = 20.0, filtertype='lowpass', return_top = False)

abdomen_coil = scipy.signal.detrend(shirt_data['abdomen_coil'])
abdomen_coil = hp.filter_signal(abdomen_coil, cutoff = 1, sample_rate = 20.0, filtertype='lowpass', return_top = False)

gsr = hp.filter_signal(shirt_data['gsr'], cutoff = 1, sample_rate = 20.0, filtertype='lowpass', return_top = False)

ppg_red = scipy.signal.detrend(shirt_data['ppg_red'])
ppg_red = hp.filter_signal(ppg_red, cutoff = [0.5, 4.5], sample_rate = 20.0, filtertype='bandpass', return_top = False)

ppg_ir = scipy.signal.detrend(shirt_data['ppg_ir'])
ppg_ir= hp.filter_signal(ppg_ir, cutoff = [0.5, 4.5], sample_rate = 20.0, filtertype='bandpass', return_top = False)

co2 = hp.filter_signal(mask_data['co2'], cutoff = 1.5, sample_rate = 20.0, filtertype='lowpass', return_top = False)
# plot_signal(co2, mask_timestamps, "co2")

voc = hp.filter_signal(mask_data['voc'], cutoff = 1.5, sample_rate = 20.0, filtertype='lowpass', return_top = False)

# Windowing
# Split data into the intervals. Make sure they consist of shirt and mask data
total_time = minutes * 60 

# Calculate sampling rate
shirt_rate = len(shirt_timestamps) / total_time
shirt_rate = int(shirt_rate)
mask_rate = len(mask_timestamps) / total_time
mask_rate = int(mask_rate)

# Calculate number of samples per window and overlap
shirt_samples_per_window = int(window_size * shirt_rate)
mask_samples_per_window = int(window_size * mask_rate)

shirt_overlap_per_window = int(overlap * shirt_rate)
mask_overlap_per_window = int(overlap * mask_rate)

# SHIRT DATA
## Iterate through the data to extract windows
start_index = 0
end_index = shirt_samples_per_window  # Initial end index for the first window
windows = []

while end_index <= len(shirt_timestamps):
    window = shirt_timestamps[start_index:end_index]  # Extract window
    windows.append(window)
    
    start_index += shirt_samples_per_window - shirt_overlap_per_window  # Move start index forward
    end_index += shirt_samples_per_window - shirt_overlap_per_window  # Move end index forward

print("Number of Windows:", len(windows))

for window in windows:
    _chest_coil = []
    _abdomen_coil = []
    _gsr = []
    _ppg_red = []
    _ppg_ir = []

    for timestamp in window:
        position = shirt_timestamps.index(timestamp)
        _chest_coil.append(chest_coil[position])
        _abdomen_coil.append(abdomen_coil[position])
        _gsr.append(gsr[position])
        _ppg_red.append(ppg_red[position])
        _ppg_ir.append(ppg_ir[position])
    
    # Find the metrics for each signal
    # PPG Metrics
    hr = find_max_freq(_ppg_ir)
    spo2 = find_spo2(_ppg_ir, _ppg_red)
    systolic_amp, hrv =  find_systolic_amplitude_and_hrv(np.array(_ppg_red))

    heart_rates.append(hr)
    spo2_levels.append(spo2)
    hrvs.append(hrv)
    systolic_amplitudes.append(systolic_amp)

    # K-RIP Metrics
    chest_rate, chest_rvt, chest_symmetry_pt, chest_symmetry_rd, chest_inhale_time, chest_exhale_time,  chest_ie_time = find_rsp_metrics(_chest_coil)
    abdomen_rate, abdomen_rvt, abdomen_symmetry_pt, abdomen_symmetry_rd, abdomen_inhale_time, abdomen_exhale_time,  abdomen_ie_time = find_rsp_metrics(_abdomen_coil)

    chest_rates.append(chest_rate)
    chest_rvts.append(chest_rvt)
    chest_symmetries_pt.append(chest_symmetry_pt)
    chest_symmetries_rd.append(chest_symmetry_rd)
    chest_inhale_times.append(chest_inhale_time)
    chest_exhale_times.append(chest_exhale_time)
    chest_ie_times.append(chest_ie_time)

    abdomen_rates.append(abdomen_rate)
    abdomen_rvts.append(abdomen_rvt)
    abdomen_symmetries_pt.append(abdomen_symmetry_pt)
    abdomen_symmetries_rd.append(abdomen_symmetry_rd)
    abdomen_inhale_times.append(abdomen_inhale_time)
    abdomen_exhale_times.append(abdomen_exhale_time)
    abdomen_ie_times.append(abdomen_ie_time)

    # GSR Metrics 
    num_peaks, average_amplitude, eda_tonic_sd = find_gsr_metrics(_gsr)

    num_sda_peaks.append(num_peaks)
    average_sda_amplitudes.append(average_amplitude)
    eda_tonic_sds.append(eda_tonic_sd)
    
# MASK DATA
## Iterate through the data to extract windows
start_index = 0
end_index = mask_samples_per_window  # Initial end index for the first window
windows = []

while end_index <= len(mask_timestamps):
    window = mask_timestamps[start_index:end_index]  # Extract window
    windows.append(window)
    
    start_index += mask_samples_per_window - mask_overlap_per_window  # Move start index forward
    end_index += mask_samples_per_window - mask_overlap_per_window  # Move end index forward

for window in windows:
    _co2 = []
    _voc = []

    for timestamp in window:
        position = mask_timestamps.index(timestamp)
        _co2.append(co2[position])
        _voc.append(voc[position])

    # CO2 Metrics 
    # We can get breathing rate from KRIP and we know KRIP and CO2 sensor are linearly correlated, therefore the only 
    # key value for us to get out is the avergae CO2 exhaled
    co2_exhaled = find_average_exhaled(_co2)

    # VOC Metrics
    # We know that VOC and CO2 are correlated and we have breathing rate from KRIP so therefore the only key value 
    # for us to get out is the average VOC exhaled. 
    voc_exhaled = find_average_exhaled(_voc)

    average_co2s.append(co2_exhaled)
    average_vocs.append(voc_exhaled)

# Append metrics and label to a csv file.
csv_filepath = filepath = '../../Machine_Learning_Data/total_data.csv'

print(particpant_id)
print(classification)
print(stress_rating)
print(reaction_time)
print(score)
print(np.shape(heart_rates))
print(np.shape(spo2_levels))
print(np.shape(systolic_amplitudes))
print(np.shape(hrvs))
print(np.shape(chest_rates))
print(np.shape(chest_rvts))
print(np.shape(chest_symmetries_pt))
print(np.shape(chest_symmetries_rd))
print(np.shape(chest_inhale_times))
print(np.shape(chest_exhale_times))
print(np.shape(chest_ie_times))
print(np.shape(abdomen_rates))
print(np.shape(abdomen_rvts))
print(np.shape(abdomen_symmetries_pt))
print(np.shape(abdomen_symmetries_rd))
print(np.shape(abdomen_inhale_times))
print(np.shape(abdomen_exhale_times))
print(np.shape(abdomen_ie_times))
print(np.shape(num_sda_peaks))
print(np.shape(average_sda_amplitudes))
print(np.shape(eda_tonic_sds))
print(np.shape(average_co2s))
print(np.shape(average_vocs))

try:
    with open(csv_filepath, 'a', newline='') as csvfile: 
        writer = csv.writer(csvfile)
        for i in range(len(heart_rates)):
            writer.writerow([particpant_id,
                            classification,
                            stress_rating,
                            reaction_time,
                            score,
                            heart_rates[i],
                            spo2_levels[i],
                            systolic_amplitudes[i],
                            hrvs[i],
                            chest_rates[i],
                            chest_rvts[i],
                            chest_symmetries_pt[i],
                            chest_symmetries_rd[i],
                            chest_inhale_times[i],
                            chest_exhale_times[i],
                            chest_ie_times[i],
                            abdomen_rates[i],
                            abdomen_rvts[i],
                            abdomen_symmetries_pt[i],
                            abdomen_symmetries_rd[i],
                            abdomen_inhale_times[i],
                            abdomen_exhale_times[i],
                            abdomen_ie_times[i],
                            num_sda_peaks[i],
                            average_sda_amplitudes[i],
                            eda_tonic_sds[i],
                            average_co2s[i],
                            average_vocs[i]])
        print("Data written to csv file")
except Exception as e:
    print("Error occurred while writing to the CSV file:", e)

print("CSV File Path:", csv_filepath)