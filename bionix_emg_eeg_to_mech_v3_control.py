import torch.nn as nn
from torch_geometric.nn import GIN, GAT
from torch_geometric.data import Data
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from neural_data_graph_construction import compute_adj_matrix_emg_mat, compute_adj_matrix_eeg_mat, create_neural_attn_adj
from run_pred import pred_eeg

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Uses a graph isomorphism network to emphasize variety of graph structures
# to differentiate kinematic signals with differnet EEG and EMG graphs
class EMGEEGFusionEncoderv3Control(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gin_featurizer_emg = GIN(
            in_channels=10,
            hidden_channels=64,
            out_channels=128,
            num_layers=2
        )
    
        
        self.node_attn_features = GAT(
            in_channels=128,
            hidden_channels=256,
            out_channels=128,
            num_layers=2
        )
        
        
    def forward(self, emg_graph):
        # Nodes: (5, 1) + (32, 1) -> (5, 128) + (32, 1) -> (5, 128) + (32, 128) -> (37, 128)
        emg_struct_feature_nodes = self.gin_featurizer_emg(emg_graph.x, emg_graph.edge_index, emg_graph.edge_attr) 
        attn_features = self.node_attn_features(emg_struct_feature_nodes, emg_graph.edge_index)  # (37, 128)
       
        return attn_features
        
class KinematicDecoderv3(nn.Module):
    def __init__(self, num_angles):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(5 * 128, 2056),
            nn.LeakyReLU(),
            nn.Linear(2056, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_angles),

        )
        
    def forward(self, fused_neural_graph_encoded):
        # print(fused_neural_graph_encoded.shape)
        fused_neural_graph_encoded = fused_neural_graph_encoded.reshape(1, fused_neural_graph_encoded.shape[0] * fused_neural_graph_encoded.shape[1])
        pred_kin = self.mlp(fused_neural_graph_encoded)
        
        return pred_kin
    
class BioniXDecoderV3Control(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = EMGEEGFusionEncoderv3Control()
        self.decoder = KinematicDecoderv3(num_angles=18)
        
    def forward(self, emg_graph):
        fused_neural_graph_pred = self.encoder(emg_graph)
        # print(fused_neural_graph_pred.shape)
        kin_pred = self.decoder(fused_neural_graph_pred)
        
        return kin_pred

# emg_data = np.load('emg_signals.npy')
# test_emg_nodes = torch.tensor(emg_data[1], dtype=torch.float).permute(1, 0)[:, :10]
# _, emg_adj_matrix = compute_adj_matrix_emg_mat(test_emg_nodes, sampling_frequency=500)
# emg_adj_matrix = torch.tensor(emg_adj_matrix, dtype=torch.float)
# test_edge_index_emg, test_edge_weight_emg = dense_to_sparse(emg_adj_matrix)
# emg_test_graph = Data(x=test_emg_nodes, edge_index=test_edge_index_emg, edge_attr=test_edge_weight_emg)
# print("EMG: ", emg_test_graph)

# eeg_pred_nt = pred_eeg(emg_test_graph)[:, :10]

# # print(eeg_pred_nt.shape)
# _, eeg_adj_matrix = compute_adj_matrix_eeg_mat(eeg_pred_nt, sampling_frequency=500)
# eeg_adj_matrix = torch.tensor(eeg_adj_matrix, dtype=torch.float)
# test_edge_index_eeg, test_edge_weight_eeg = dense_to_sparse(eeg_adj_matrix)
# eeg_test_graph = Data(x=eeg_pred_nt, edge_index=test_edge_index_eeg, edge_attr=test_edge_weight_eeg)
# print("EEG: ", eeg_test_graph)

# model = BioniXDecoderV3()
# model_out = model(emg_test_graph, eeg_test_graph)

# print(model_out.shape)