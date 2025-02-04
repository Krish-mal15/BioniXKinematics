from dtaidistance import dtw
import numpy as np
from scipy.signal import correlate
import scipy.io

def get_mat_emg_data(file_path):

    mat_data = scipy.io.loadmat(file_path)
    # print(mat_data)
    emg_data = mat_data['emg']
    
    return emg_data  # NumPy array

num_emg_to_keep = 10000
emg_test_data = get_mat_emg_data('main_otto_nina_emg_data/S1_A1_E1.mat')
emg_test_data = emg_test_data[:num_emg_to_keep].T

def normalized_cross_correlation_matrix(signals):
    num_signals = signals.shape[0]
    adj_matrix = np.zeros((num_signals, num_signals))

    for i in range(num_signals):
        for j in range(num_signals):
            if i == j:
                adj_matrix[i, j] = 1  
            else:
                sig1 = (signals[i] - np.mean(signals[i])) / np.std(signals[i])
                sig2 = (signals[j] - np.mean(signals[j])) / np.std(signals[j])
                corr = np.max(correlate(sig1, sig2, mode='valid')) / len(sig1)
                adj_matrix[i, j] = corr
                print("Done")
    return adj_matrix

adj_matrix = normalized_cross_correlation_matrix(emg_test_data)
print(adj_matrix.shape)
