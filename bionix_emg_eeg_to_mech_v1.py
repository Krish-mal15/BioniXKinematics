import torch.nn as nn
from torch_geometric.nn import GIN
from torch_geometric.data import Data
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import dense_to_sparse
from neural_data_graph_construction import compute_adj_matrix_emg_mat, compute_adj_matrix_eeg_mat
from run_pred import pred_eeg

# Uses a graph isomorphism network to emphasize variety of graph structures
# to differentiate kinematic signals with differnet EEG and EMG graphs
class EMGEEGFusionEncoder(nn.Module):
    def __init__(self, num_neural_timesteps, latent_dim):
        super().__init__()
        
        self.emg_gin = GIN(
            in_channels=num_neural_timesteps,
            hidden_channels=512,
            out_channels=latent_dim,
            num_layers=2
        )
        
        self.eeg_gin = GIN(
            in_channels=num_neural_timesteps,
            hidden_channels=512,
            out_channels=latent_dim,
            num_layers=2
        )
        
        self.emg_latent_space_proj = nn.Linear(latent_dim, latent_dim)
        self.eeg_latent_space_proj = nn.Linear(latent_dim, latent_dim)
        
        
    def forward(self, emg_graph, eeg_graph):
        emg_feature_vector = self.emg_gin(emg_graph.x, emg_graph.edge_index, emg_graph.edge_attr)
        eeg_feature_vector = self.eeg_gin(eeg_graph.x, eeg_graph.edge_index, eeg_graph.edge_attr)
        
        emg_proj = self.emg_latent_space_proj(emg_feature_vector)
        eeg_proj = self.eeg_latent_space_proj(eeg_feature_vector)
        
        fused_neural_graph_encoded = torch.cat((emg_proj, eeg_proj), dim=0)  # (37, latent_dim)
        
        return fused_neural_graph_encoded
        
class KinematicDecoder(nn.Module):
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
        print(fused_neural_graph_encoded.shape)
        fused_neural_graph_encoded = fused_neural_graph_encoded.reshape(1, fused_neural_graph_encoded.shape[0] * fused_neural_graph_encoded.shape[1])
        pred_kin = self.mlp(fused_neural_graph_encoded)
        
        return pred_kin
    
class BioniXDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = EMGEEGFusionEncoder(num_neural_timesteps=1200, latent_dim=128)
        self.decoder = KinematicDecoder(num_angles=8)
        
    def forward(self, emg_graph, eeg_graph):
        fused_neural_graph_pred = self.encoder(emg_graph, eeg_graph)
        kin_pred = self.decoder(fused_neural_graph_pred)
        
        return kin_pred

emg_data = np.load('emg_signals.npy')
test_emg_nodes = torch.tensor(emg_data[1], dtype=torch.float).permute(1, 0)
_, emg_adj_matrix = compute_adj_matrix_emg_mat(test_emg_nodes, sampling_frequency=500)
emg_adj_matrix = torch.tensor(emg_adj_matrix, dtype=torch.float)
test_edge_index_emg, test_edge_weight_emg = dense_to_sparse(emg_adj_matrix)
emg_test_graph = Data(x=test_emg_nodes, edge_index=test_edge_index_emg, edge_attr=test_edge_weight_emg)
print("EMG: ", emg_test_graph)

eeg_pred_nt = pred_eeg(emg_test_graph)
# print(eeg_pred_nt.shape)
_, eeg_adj_matrix = compute_adj_matrix_eeg_mat(eeg_pred_nt, sampling_frequency=500)
eeg_adj_matrix = torch.tensor(eeg_adj_matrix, dtype=torch.float)
test_edge_index_eeg, test_edge_weight_eeg = dense_to_sparse(eeg_adj_matrix)
eeg_test_graph = Data(x=eeg_pred_nt, edge_index=test_edge_index_eeg, edge_attr=test_edge_weight_eeg)
print("EEG: ", eeg_test_graph)

model = BioniXDecoder()
model_out = model(emg_test_graph, eeg_test_graph)

print(model_out.shape)