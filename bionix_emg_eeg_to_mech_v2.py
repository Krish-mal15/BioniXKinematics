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

# Uses a graph isomorphism network to emphasize variety of graph structures
# to differentiate kinematic signals with differnet EEG and EMG graphs
class EMGEEGFusionEncoderv2(nn.Module):
    def __init__(self, num_neural_timesteps, latent_dim):
        super().__init__()
        
        self.emg_gat = GAT(
            in_channels=num_neural_timesteps,
            hidden_channels=512,
            out_channels=latent_dim,
            num_layers=2
        )
        
        self.eeg_gat = GAT(
            in_channels=num_neural_timesteps,
            hidden_channels=512,
            out_channels=latent_dim,
            num_layers=2
        )
        
        self.emg_latent_space_proj = nn.Linear(latent_dim, latent_dim)
        self.eeg_latent_space_proj = nn.Linear(latent_dim, latent_dim)
        
        self.graph_struct = GIN(
            in_channels=128,
            hidden_channels=256,
            out_channels=128,
            num_layers=2
        )
        
        
    def forward(self, emg_graph, eeg_graph):
        
        # Graph-wiese attention (relaitonship based using edge weights)
        emg_feature_vector = self.emg_latent_space_proj(self.emg_gat(emg_graph.x, emg_graph.edge_index, emg_graph.edge_attr))  # (5, 128)
        eeg_feature_vector = self.eeg_latent_space_proj(self.eeg_gat(eeg_graph.x, eeg_graph.edge_index, eeg_graph.edge_attr))  # (32, 128)
        
        combined_neural_nodes = torch.cat((emg_feature_vector, eeg_feature_vector), dim=0)
        # Signal wise attention using node temporal properties
        fused_attn_adj = create_neural_attn_adj(combined_neural_nodes)
        
        edge_idx_neural, edge_attr_neural = dense_to_sparse(fused_attn_adj)
        
        new_fused_neural_graph = Data(x=combined_neural_nodes, edge_index=edge_idx_neural, edge_attr=edge_attr_neural)
        
        print(new_fused_neural_graph)

        fused_neural_graph_encoded = self.graph_struct(new_fused_neural_graph.x, new_fused_neural_graph.edge_index, new_fused_neural_graph.edge_attr)     
        return fused_neural_graph_encoded
        
class KinematicDecoderv2(nn.Module):
    def __init__(self, num_angles):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(37 * 128, 2056),
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
    
class BioniXDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = EMGEEGFusionEncoderv2(num_neural_timesteps=1200, latent_dim=128)
        self.decoder = KinematicDecoderv2(num_angles=8)
        
    def forward(self, emg_graph, eeg_graph):
        fused_neural_graph_pred = self.encoder(emg_graph, eeg_graph)
        # print(fused_neural_graph_pred.shape)
        kin_pred = self.decoder(fused_neural_graph_pred)
        
        return kin_pred

# emg_data = np.load('emg_signals.npy')
# test_emg_nodes = torch.tensor(emg_data[1], dtype=torch.float).permute(1, 0)
# _, emg_adj_matrix = compute_adj_matrix_emg_mat(test_emg_nodes, sampling_frequency=500)
# emg_adj_matrix = torch.tensor(emg_adj_matrix, dtype=torch.float)
# test_edge_index_emg, test_edge_weight_emg = dense_to_sparse(emg_adj_matrix)
# emg_test_graph = Data(x=test_emg_nodes, edge_index=test_edge_index_emg, edge_attr=test_edge_weight_emg)
# print("EMG: ", emg_test_graph)

# eeg_pred_nt = pred_eeg(emg_test_graph)
# # print(eeg_pred_nt.shape)
# _, eeg_adj_matrix = compute_adj_matrix_eeg_mat(eeg_pred_nt, sampling_frequency=500)
# eeg_adj_matrix = torch.tensor(eeg_adj_matrix, dtype=torch.float)
# test_edge_index_eeg, test_edge_weight_eeg = dense_to_sparse(eeg_adj_matrix)
# eeg_test_graph = Data(x=eeg_pred_nt, edge_index=test_edge_index_eeg, edge_attr=test_edge_weight_eeg)
# print("EEG: ", eeg_test_graph)

# model = BioniXDecoder()
# model_out = model(emg_test_graph, eeg_test_graph)

# print(model_out.shape)