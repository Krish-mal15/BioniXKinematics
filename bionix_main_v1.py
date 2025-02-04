# This approach will use proven PLV assisted graph isomorphism to process EEG. To process EMG, it may be beneficial
# to just use a transformer (with spatial attention) because EMG directly correlates to movement unlike EEG

import torch.nn as nn
from torch_geometric.nn import GIN, GAT
import torch
import math
import numpy as np

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from neural_data_graph_construction import compute_adj_matrix_emg_mat, compute_adj_matrix_eeg_mat, create_neural_attn_adj
from run_pred import pred_eeg

class PositionalEmbedding(nn.Module):
    def __init__(self, emg_tsteps, d_model, dropout=0.1, max_len=256):
        super().__init__()
        self.embed = nn.Linear(emg_tsteps, d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure we are using the correct sequence length dimension of x
 
        x = self.embed(x)
    
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x).squeeze(0)

class EMGTransformerEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.pos_embedder = PositionalEmbedding(emg_tsteps=10, d_model=embed_dim)
        
        self.context_combination = nn.Linear(embed_dim * 2, embed_dim)

        
        # This graph isomorphic network will learn the structure of the anatomical relationships of EMG. Most likely, will just pass
        # in nodes (the signal), but if it doesnt have good results, then manually assign correlation neuroanatomically.
        self.spatial_emg_encoder = GIN(
            in_channels=10,
            hidden_channels=128,
            out_channels=embed_dim,
            num_layers=4
        )
        
        temp_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.temp_encoder = nn.TransformerEncoder(temp_encoder_layer, num_layers=4)
        self.ch_transform = nn.Linear(5, 1)
        
    def forward(self, emg_signal):
        directed_edge_index = torch.combinations(torch.arange(5), r=2, with_replacement=False).T
        edge_index = torch.cat([directed_edge_index, directed_edge_index.flip(0)], dim=1)
        
        emg_graph_struct_embeds = self.spatial_emg_encoder(emg_signal, edge_index)  # (5, 256)
        
        pos_embedded_emg_signal = self.pos_embedder(emg_signal)
        combined = torch.cat([pos_embedded_emg_signal, emg_graph_struct_embeds], dim=-1)  # (batch_size, d_model*2)
        
        fused_pred = self.context_combination(combined).unsqueeze(0)
        
        transformer_out = self.temp_encoder(fused_pred)  # (5, 256)
        transformer_out = self.ch_transform(transformer_out.permute(0, 2, 1))
        
        return transformer_out
    
class EEGGraphEncoder(nn.Module):
    def __init__(self, n_eeg_timesteps, graph_embed_dim):
        super().__init__()
        
        self.embed_dim = graph_embed_dim
        
        self.gin_encoder = GIN(
            in_channels=n_eeg_timesteps,
            hidden_channels=512,
            out_channels=graph_embed_dim,
            num_layers=2
        )
        
        self.gat_transform = GAT(
            in_channels=graph_embed_dim,
            hidden_channels=graph_embed_dim * 2,
            out_channels=graph_embed_dim,
            num_layers=1
        )
        
        self.lin_transform = nn.Linear(32 * self.embed_dim, graph_embed_dim)
        
    def forward(self, eeg_graph):
        eeg_features = self.gin_encoder(eeg_graph.x, eeg_graph.edge_index, eeg_graph.edge_attr) # The "edge_attr" parameter is for "edge_weight" in GIN forward pass
        eeg_features = self.gat_transform(eeg_features, eeg_graph.edge_index, eeg_graph.edge_attr)  # Edge attr isnt necessary fro this attention layer
        eeg_features = self.lin_transform(eeg_features.reshape(1, 32 * self.embed_dim))
        return eeg_features
    
class BioniXKinModel(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        
        self.emg_encoder = EMGTransformerEncoder(embed_dim=256)
        self.eeg_encoder = EEGGraphEncoder(n_eeg_timesteps=10, graph_embed_dim=256)
        
        self.biomech_decoder = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_joints),
        )
        
    def forward(self, emg_signal, eeg_graph):
        emg_features = self.emg_encoder(emg_signal)
        eeg_features = self.eeg_encoder(eeg_graph)
        
        # print(emg_features.shape)
        # print(eeg_features.shape)
      
        fused_neural_features = torch.cat((emg_features.squeeze(2), eeg_features), dim=1)  # (1, 512)
        pred_joint_angles = self.biomech_decoder(fused_neural_features)
        
        return pred_joint_angles
 
 
# model = BioniXKinModel(num_joints=18)

# emg_data = np.load('emg_signals.npy')
# test_emg_nodes = torch.tensor(emg_data[1], dtype=torch.float).permute(1, 0)[:, :10]
# _, emg_adj_matrix = compute_adj_matrix_emg_mat(test_emg_nodes, sampling_frequency=500)
# emg_adj_matrix = torch.tensor(emg_adj_matrix, dtype=torch.float)
# test_edge_index_emg, test_edge_weight_emg = dense_to_sparse(emg_adj_matrix)
# emg_test_graph = Data(x=test_emg_nodes, edge_index=test_edge_index_emg, edge_attr=test_edge_weight_emg)
# # print("EMG: ", emg_test_graph)

# eeg_pred_nt = pred_eeg(emg_test_graph)[:, :10]

# # print(eeg_pred_nt.shape)
# _, eeg_adj_matrix = compute_adj_matrix_eeg_mat(eeg_pred_nt, sampling_frequency=500)
# eeg_adj_matrix = torch.tensor(eeg_adj_matrix, dtype=torch.float)
# test_edge_index_eeg, test_edge_weight_eeg = dense_to_sparse(eeg_adj_matrix)
# eeg_test_graph = Data(x=eeg_pred_nt, edge_index=test_edge_index_eeg, edge_attr=test_edge_weight_eeg)
# # print("EEG: ", eeg_test_graph)

# print("EMG: ", test_emg_nodes.unsqueeze(0).shape)
# print("EEG: ", eeg_test_graph)

# model_out = model(test_emg_nodes.unsqueeze(0), eeg_test_graph)   
# print(model_out.shape)
