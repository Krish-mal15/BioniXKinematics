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

# Execute: python graph_data_preprocessing.py

def get_signals(filepath, segment_length):
    mat_data = loadmat(filepath)
    structure_var = mat_data['hs']

    emg_signal = structure_var['emg'][0][0][0][0][0]
    eeg_signal = structure_var['eeg'][0][0][0][0][1]


    keep_emg = emg_signal[:110000]
    keep_eeg = eeg_signal[:110000]

    eeg_segments = [keep_eeg[i:i + segment_length] for i in range(0, len(keep_eeg) - segment_length + 1, segment_length)]
    emg_segments = [keep_emg[i:i + segment_length] for i in range(0, len(keep_emg) - segment_length + 1, segment_length)]

    return np.array(eeg_segments), np.array(emg_segments)

# eeg, emg = get_signals('data/way_eeg_gal_dataset/P3/HS_P3_S2.mat', 10)
# print(eeg.shape)
# print(emg.shape)

def get_data_matrices(main_file_path):
    eeg_dataset_matrix = []
    emg_dataset_matrix = []

    for participant, _, trials in os.walk(main_file_path):
        for trial in trials:
            if "HS" in trial:
                file_path = os.path.join(participant, trial)

                print(file_path)
                eeg_segments, emg_segments = get_signals(file_path, segment_length=10)
                

                eeg_dataset_matrix.extend(eeg_segments)
                emg_dataset_matrix.extend(emg_segments)
                

    return np.array(eeg_dataset_matrix), np.array(emg_dataset_matrix)


# eeg_signals, emg_signals = get_data_matrices('data/way_eeg_gal_dataset')

# print(eeg_signals.shape)
# print(emg_signals.shape)

# np.save("data/main_way_eeg_gal_dataset_matrices_seg_n10/eeg_signals.npy", eeg_signals)
# np.save("data/main_way_eeg_gal_dataset_matrices_seg_n10/emg_signals.npy", emg_signals)

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
    def __init__(self, eeg_data_main, emg_data_main):

        self.emg_signals = emg_data_main
        self.eeg_signals = eeg_data_main

    def __getitem__(self, idx):
        emg_signal = self.emg_signals[idx]
        eeg_signal = self.eeg_signals[idx]

        # eeg_signal = z_score_norm(eeg_signal)
        # emg_signal = z_score_norm(emg_signal)

        # print("EEG: ", eeg_signal)
        # print("EMG: ", emg_signal)

        emg_signal, emg_adj_matrix = compute_adj_matrix_emg_mat(emg_signal, sampling_frequency=500)
        eeg_signal, eeg_adj_matrix = compute_adj_matrix_eeg_mat(eeg_signal, sampling_frequency=500)

        emg_adj_matrix = torch.tensor(emg_adj_matrix, dtype=torch.float)
        eeg_adj_matrix = torch.tensor(eeg_adj_matrix, dtype=torch.float)

        edge_index_emg, edge_weight_emg = dense_to_sparse(emg_adj_matrix)
        edge_index_eeg, edge_weight_eeg = dense_to_sparse(eeg_adj_matrix)

        nodes_eeg = torch.tensor(eeg_signal, dtype=torch.float)
        nodes_emg = torch.tensor(emg_signal, dtype=torch.float)

        # eeg_graph_data = Data(x=node_features_eeg, edge_index=edge_index_eeg, edge_attr=edge_weight_eeg)
        # emg_graph_data = Data(x=node_features_emg, edge_index=edge_index_emg, edge_attr=edge_weight_emg)

        return nodes_emg, edge_index_emg, edge_weight_emg, nodes_eeg, edge_index_eeg, edge_weight_eeg

    def __len__(self):
        return len(self.emg_signals)


# emg_main_array_dataset = np.load('data/main_way_eeg_gal_dataset_matrices/emg_signals.npy')
# eeg_main_array_dataset = np.load('data/main_way_eeg_gal_dataset_matrices/eeg_signals.npy')

# print(emg_main_array_dataset.shape)
# print(eeg_main_array_dataset.shape)

# [4032 data points x num_channels x seq_len]
# emg_main_array_dataset = emg_main_array_dataset.reshape(4032, 5, 1200)
# eeg_main_array_dataset = eeg_main_array_dataset.reshape(4032, 32, 1200)

# dataset = NeuroKinematicDataset(eeg_main_array_dataset, emg_main_array_dataset)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg in dataloader:
#     print(features_emg.shape)
#     print(edge_weight_emg.shape, "\n")

#     print(features_eeg.shape)
#     print(edge_weight_eeg.shape, "\n")
    
#     g_data = Data(x=features_eeg.squeeze(0), edge_index=edge_index_eeg.squeeze(0), edge_attr=edge_weight_eeg.squeeze(0))
    
#     G = to_networkx(g_data, edge_attrs=['edge_attr'], node_attrs=['x'])
#     pos = nx.spring_layout(G)

#     # Extract edge attributes
#     edge_weights = nx.get_edge_attributes(G, 'edge_attr')
#     edge_colors = list(edge_weights.values())  # Use float values directly

#     # Create a figure and axis
#     fig, ax = plt.subplots()

#     # Draw the nodes
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)

#     # Draw the edges with variable thickness and color
#     edges = nx.draw_networkx_edges(
#         G, pos,
#         edge_color=edge_colors,  # Color edges based on weight
#         edge_cmap=plt.cm.viridis,  # Use a colormap for edge colors
#         edge_vmin=min(edge_colors),  # Minimum value for normalization
#         edge_vmax=max(edge_colors),  # Maximum value for normalization
#         ax=ax
#     )

#     # Draw node labels
#     nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", ax=ax)

#     # Add a color bar for edges
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
#     sm.set_array([])  # Required to make ScalarMappable compatible with colorbar
#     plt.colorbar(sm, ax=ax, label="Edge Value")

#     # Finalize the plot
#     plt.title("Graph Visualization")
#     plt.axis("off")
#     plt.show()

