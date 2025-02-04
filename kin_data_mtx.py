from torch.utils.data import DataLoader, Dataset
from neural_data_graph_construction import compute_adj_matrix_emg_mat, compute_adj_matrix_eeg_mat
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from run_pred import pred_eeg

# Execute: python graph_data_preprocessing.py

def get_signals(filepath, segment_length):
    mat_data = loadmat(filepath)
    structure_var = mat_data['hs']

    emg_signal = structure_var['emg'][0][0][0][0]
    print(emg_signal)
    eeg_signal = structure_var['eeg'][0][0][0][0][1]
    
    # Indices to keep (human positions): [1, 2, 3, 5, 6, 7, 9, 10, 11, 18, 19, 20, 22, 23, 24, 26, 27, 28]
    human_pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 18, 19, 20, 22, 23, 24, 26, 27, 28]
    kin_angles = structure_var['kin'][0][0][0][0][1][:11000].T[human_pos].T
    # print(kin_angles.shape)

    keep_emg = emg_signal[:11000]
    keep_eeg = eeg_signal[:11000]

    eeg_segments = [keep_eeg[i:i + segment_length] for i in range(0, len(keep_eeg) - segment_length + 1, segment_length)]
    emg_segments = [keep_emg[i:i + segment_length] for i in range(0, len(keep_emg) - segment_length + 1, segment_length)]
    
    # Will use the start index of the 10 timestep kinematic data. This is so there is only one joint angle
    # for each of the joints per 10 timesteps since in real life they are probably not gonna be changing that fast 
    # with a sampling rate for EMG of 4370
    kin_segments = [kin_angles[i:i + segment_length] for i in range(0, len(kin_angles) - segment_length + 1, segment_length)]
    
    return np.array(eeg_segments), np.array(emg_segments), np.array(kin_segments)[:, [9], :]

# eeg, emg, kin = get_signals('way_eeg_gal_dataset/P3/HS_P3_S2.mat', 10)
# print(eeg.shape)
# print(emg.shape)
# print(kin.shape)

def get_data_matrices(main_file_path):
    eeg_dataset_matrix = []
    emg_dataset_matrix = []
    kin_dataset_matrix = []

    for participant, _, trials in os.walk(main_file_path):
        for trial in trials:
            if "HS" in trial:
                file_path = os.path.join(participant, trial)

                print(file_path)
                eeg_segments, emg_segments, kin_segments = get_signals(file_path, segment_length=10)
         
                eeg_dataset_matrix.extend(eeg_segments)
                emg_dataset_matrix.extend(emg_segments)
                kin_dataset_matrix.extend(kin_segments)

    return np.array(eeg_dataset_matrix), np.array(emg_dataset_matrix), np.array(kin_dataset_matrix)


# eeg_signals, emg_signals, kin_data = get_data_matrices('way_eeg_gal_dataset')

# print(eeg_signals.shape)
# print(emg_signals.shape)
# print(kin_data.shape)

# np.save("bionix_kin_data_n10/eeg_signals.npy", eeg_signals)
# np.save("bionix_kin_data_n10/emg_signals.npy", emg_signals)
# np.save("bionix_kin_data_n10/kin_data.npy", kin_data)


def normalize_min_max(x):

    x_min = x.min()
    x_max = x.max()

    return (x - x_min) / (x_max - x_min + 1e-8)


def l2_normalize(x):
    return x / (np.linalg.norm(x) + 1e-8)

def z_score_norm(x):
    mean = torch.mean(x, axis=1, keepdims=True)
    std = torch.std(x, axis=1, keepdims=True)
    
    return (x - mean) / std

# Assume x is z-score normalized
def inverse_z_score_norm(x, mean, std_dev):
    return x * std_dev + mean
    
    

class NeuroKinematicDataset(Dataset):
    def __init__(self, eeg_data_main, emg_data_main, kin_data_main):

        self.emg_signals = emg_data_main
        # self.eeg_signals = eeg_data_main
        self.kin_data_main = kin_data_main

    def __getitem__(self, idx):
        emg_signal = self.emg_signals[idx]
        # eeg_signal = self.eeg_signals[idx]
        kin_data = self.kin_data_main[idx]
        
        # eeg_signal = z_score_norm(torch.tensor(eeg_signal, dtype=torch.float))
        emg_signal = z_score_norm(torch.tensor(emg_signal, dtype=torch.float))
        kin_data = normalize_min_max(kin_data)

        # print("EEG: ", eeg_signal)
        # print("EMG: ", emg_signal)
        # print("Kin: ", kin_data)
        
        kin_data = torch.tensor(kin_data, dtype=torch.float)

        emg_signal, emg_adj_matrix = compute_adj_matrix_emg_mat(emg_signal, sampling_frequency=500)

        emg_adj_matrix = torch.tensor(emg_adj_matrix, dtype=torch.float)

        edge_index_emg, edge_weight_emg = dense_to_sparse(emg_adj_matrix)
        nodes_emg = torch.tensor(emg_signal, dtype=torch.float)

        # eeg_graph_data = Data(x=node_features_eeg, edge_index=edge_index_eeg, edge_attr=edge_weight_eeg)
        emg_graph_data = Data(x=emg_signal, edge_index=edge_index_emg, edge_attr=edge_weight_emg)
        
        eeg_signal = pred_eeg(emg_graph_data)
        eeg_signal, eeg_adj_matrix = compute_adj_matrix_eeg_mat(eeg_signal, sampling_frequency=500)
        eeg_adj_matrix = torch.tensor(eeg_adj_matrix, dtype=torch.float)
        edge_index_eeg, edge_weight_eeg = dense_to_sparse(eeg_adj_matrix)
        nodes_eeg = torch.tensor(eeg_signal, dtype=torch.float)

        # print(kin_data.shape)

        return nodes_emg, edge_index_emg, edge_weight_emg, nodes_eeg, edge_index_eeg, edge_weight_eeg, kin_data.permute(1, 0)

    def __len__(self):
        return len(self.emg_signals)


emg_main_array_dataset = np.load('bionix_kin_data_n10/emg_signals.npy')
eeg_main_array_dataset = np.load('bionix_kin_data_n10/eeg_signals.npy')
kin_main_array_dataset = np.load('bionix_kin_data_n10/kin_data.npy')

print(emg_main_array_dataset.shape)
print(eeg_main_array_dataset.shape)
print(kin_main_array_dataset.shape)

# [4032 data points x num_channels x seq_len]
emg_main_array_dataset = emg_main_array_dataset.reshape(69300, 5, 10)[:5000]
eeg_main_array_dataset = eeg_main_array_dataset.reshape(69300, 32, 10)[:5000]
kin_main_array_dataset = kin_main_array_dataset.reshape(69300, 18, 1)[:5000]

dataset = NeuroKinematicDataset(eeg_main_array_dataset, emg_main_array_dataset, kin_main_array_dataset)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg, kin_label in dataloader:
#     print(features_emg.shape)
#     print(edge_weight_emg.shape, "\n")

#     print(features_eeg.shape)
#     print(edge_weight_eeg.shape, "\n")
    
#     print(kin_label.shape)
#     # print(kin_label)
#     print('')


