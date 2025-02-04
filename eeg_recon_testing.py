import torch
from emg_graph_to_eeg_signal import EMGToEEGDecoder
from graph_data_preprocessing import dataloader, z_score_norm, inverse_z_score_norm
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from scipy.signal import welch
from graph_data_preprocessing import compute_adj_matrix_eeg_mat
from neural_correlations import get_corr_df, channel_names
import seaborn as sns
from scipy.signal import spectrogram


# Execute: python eeg_recon_testing.py

def color_eeg_by_bands(signal, sampling_rate=250, window_size_ms=100):
    """
    Plot EEG signal colored by dominant frequency bands
    
    Parameters:
    signal (array-like): The input EEG signal
    sampling_rate (float): Sampling rate in Hz
    window_size_ms (int): Window size in milliseconds for frequency analysis
    """
    # Define frequency bands and their colors
    bands = {
        'Delta': (0.5, 4, 'blue'),
        'Theta': (4, 8, 'green'),
        'Alpha': (8, 13, 'red'),
        'Beta': (13, 30, 'yellow'),
        'Gamma': (30, 100, 'purple')
    }
    
    # Calculate STFT
    window_size = int((window_size_ms/1000) * sampling_rate)
    f, t, Zxx = stft(signal, fs=sampling_rate, nperseg=window_size, noverlap=window_size-1)
    
    # Initialize arrays for plotting
    times = np.arange(len(signal)) / sampling_rate
    colors = np.zeros(len(signal), dtype='U10')
    
    # For each time window, determine dominant frequency band
    for i in range(len(t)-1):
        start_idx = int(t[i] * sampling_rate)
        end_idx = int(t[i+1] * sampling_rate) if i < len(t)-1 else len(signal)
        
        # Get power spectrum for this window
        power = np.abs(Zxx[:, i])
        
        # Find dominant frequency band
        band_powers = {}
        for band, (low, high, color) in bands.items():
            mask = (f >= low) & (f <= high)
            band_powers[band] = np.mean(power[mask])
                
        dominant_band = max(band_powers.items(), key=lambda x: x[1])[0]
        colors[start_idx:end_idx] = bands[dominant_band][2]
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot segments with different colors
    for i in range(len(signal)-1):
        plt.plot(times[i:i+2], signal[i:i+2], color=colors[i], linewidth=1)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=color, label=f'{band} ({low}-{high}Hz)')
                      for band, (low, high, color) in bands.items()]
    plt.legend(handles=legend_elements)
    
    plt.grid(True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('EEG Signal Colored by Dominant Frequency Bands')
    
    # Add frequency band percentages
    unique, counts = np.unique(colors, return_counts=True)
    percentages = {band: (counts[unique == color][0] / len(colors)) * 100 
                  for band, (_, _, color) in bands.items() 
                  if color in unique}
    
    plt.figtext(0.02, 0.02, '\n'.join([f'{band}: {perc:.1f}%' 
                                      for band, perc in percentages.items()]))


def analyze_eeg_frequency(eeg_signal, sampling_rate=500):
    """
    Analyze the frequency components of an EEG signal using Welch's method
    
    Parameters:
    eeg_signal (array-like): The input EEG signal
    sampling_rate (float): Sampling rate in Hz, default 250 Hz
    
    Returns:
    dict: Power in each frequency band
    """
    # Calculate power spectral density using Welch's method
    frequencies, psd = welch(eeg_signal, fs=sampling_rate, nperseg=sampling_rate*2)
    
    # Define frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    
    # Calculate power in each band
    band_powers = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(frequencies >= low, frequencies <= high)
        band_powers[band] = np.mean(psd[idx])
    
    # Plot the power spectrum
    plt.figure(figsize=(12, 6))
    plt.semilogy(frequencies, psd)
    
    # Add colored bands
    colors = {'Delta': 'blue', 'Theta': 'green', 'Alpha': 'red', 
              'Beta': 'yellow', 'Gamma': 'purple'}
    
    for band, (low, high) in bands.items():
        plt.axvspan(low, high, color=colors[band], alpha=0.3, label=band)
        print(high-low)
    
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (µV²/Hz)')
    plt.title('EEG Frequency Spectrum')
    plt.legend()
    plt.xlim(0, 50)  # Limit x-axis to 50 Hz for better visualization
    
    return band_powers

def get_avg_channel_signal(signal):
    mean_channels = []
    
    for channel in signal:
        avg_strength = np.mean(channel)
        mean_channels.append(avg_strength)
        
    return mean_channels


device = torch.device('cpu')
model = EMGToEEGDecoder().to(device)

model.load_state_dict(torch.load("models/eeg_recon_model_lstm_v3_epoch_19.pth", weights_only=True, map_location=device))
model.eval()

# Sample Data stuff: Get 5 inferences on random data
for i, (features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg) in enumerate(dataloader):
    eeg_true_mean = torch.mean(features_eeg, axis=1, keepdims=True)
    eeg_true_std_dev = torch.std(features_eeg, axis=1, keepdims=True)
    
    features_eeg = z_score_norm(features_eeg)
    features_emg = z_score_norm(features_emg)

    if i > 0:
        break
    emg_graph_data = Data(x=features_emg.squeeze(0), edge_attr=edge_weight_emg.squeeze(0), edge_index=edge_index_emg.squeeze(0))
    
    with torch.no_grad():
        eeg_decoded = inverse_z_score_norm(model(emg_graph_data), eeg_true_mean, eeg_true_std_dev)
        true_eeg = inverse_z_score_norm(features_eeg.squeeze(0), eeg_true_mean, eeg_true_std_dev)
        
        eeg_decoded = eeg_decoded.squeeze(0).numpy()
        true_eeg = true_eeg.squeeze().numpy()
    
        # print("Predicted EEG: ", eeg_decoded.shape)
        # print("True EEG: ", true_eeg.shape)
        
        # color_eeg_by_bands(eeg_decoded, 500)
        
        _, adj_mtx_pred = compute_adj_matrix_eeg_mat(eeg_decoded, sampling_frequency=500)
        _, adj_mtx_true = compute_adj_matrix_eeg_mat(true_eeg, sampling_frequency=500)
        
        corr_pred = get_corr_df(adj_mtx_pred)
        corr_true = get_corr_df(adj_mtx_true)
        
        corr_pred = corr_pred.to_csv("Predicted_Corrs.csv")
        corr_true = corr_true.to_csv("True_Corrs.csv")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(adj_mtx_pred, xticklabels=channel_names, yticklabels=channel_names, cmap="coolwarm", annot=False)
        plt.title("EEG Channel Pairwise Correlations PREDICTED")
        plt.show()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(adj_mtx_true, xticklabels=channel_names, yticklabels=channel_names, cmap="coolwarm", annot=False)
        plt.title("EEG Channel Pairwise Correlations TRUE")
        plt.show()
        
        single_channel_index = 2
        
        single_channel_data_pred = eeg_decoded[single_channel_index]
        single_channel_data_true = true_eeg[single_channel_index]
        
        print(analyze_eeg_frequency(single_channel_data_pred))

        plt.figure(figsize=(10, 4))
        
        plt.plot(single_channel_data_pred, label=f"Predicted EEG (Channel {single_channel_index})", color='blue', alpha=0.7)
        plt.plot(single_channel_data_true, label=f"True EEG (Channel {single_channel_index})", color='red', alpha=0.7)
        
        plt.title(f"Channel {single_channel_index}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.show()
        
        # mean_signal_pred = get_avg_channel_signal(eeg_decoded)
        # mean_signal_true = get_avg_channel_signal(true_eeg)
        
        # print("----------------")
        # print(mean_signal_pred)
        # print(mean_signal_true)
        
        # plt.figure(figsize=(12, 6))
        # channels = np.arange(1, 33)
        # plt.bar(channels, mean_signal_true, color='skyblue')

        # Customize plot
        # plt.title("Average Signal Strength of Each EEG Channel", fontsize=14)
        # plt.xlabel("Channel", fontsize=12)
        # plt.ylabel("Average Signal Strength", fontsize=12)
        # plt.xticks(channels) 
        # plt.grid(axis='y', linestyle='--', alpha=0.7)

        # # Show plot
        # plt.tight_layout()
        # plt.show()
    
    