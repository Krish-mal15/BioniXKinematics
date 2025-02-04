import torch.nn as nn
from torch_geometric.nn import GIN, GAT
from torch_geometric.data import Data
import torch
import warnings
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
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
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x).squeeze(0)

# Uses a graph isomorphism network to emphasize variety of graph structures
# to differentiate kinematic signals with differnet EEG and EMG graphs
class EMGEEGFusionEncoderv3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.gin_featurizer_emg = GIN(
            in_channels=10,
            hidden_channels=64,
            out_channels=128,
            num_layers=2
        )
        
        self.gin_featurizer_eeg = GIN(
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
        
        
    def forward(self, emg_graph, eeg_graph):
        # Nodes: (5, 1) + (32, 1) -> (5, 128) + (32, 1) -> (5, 128) + (32, 128) -> (37, 128)
        emg_struct_feature_nodes = self.gin_featurizer_emg(emg_graph.x, emg_graph.edge_index, emg_graph.edge_attr)  # Add edge_attr for both so GIN can use this in node features
        eeg_struct_feature_nodes = self.gin_featurizer_eeg(eeg_graph.x, eeg_graph.edge_index, eeg_graph.edge_attr)

        combined_neural_nodes = torch.cat((emg_struct_feature_nodes, eeg_struct_feature_nodes), dim=0) 
        
        emg_edge_index = emg_graph.edge_index 
        eeg_edge_index = eeg_graph.edge_index + emg_struct_feature_nodes.size(0) # Shift Indices

        combined_edge_index = torch.cat([emg_edge_index, eeg_edge_index], dim=1)
        
        fused_graph = Data(x=combined_neural_nodes, edge_index=combined_edge_index)
        attn_features = self.node_attn_features(fused_graph.x, fused_graph.edge_index)  # (37, 128)
       
        return attn_features
        
class KinematicDecoderv3(nn.Module):
    def __init__(self, num_angles):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(d_model=128)
        t_enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.tranformer_encoder = nn.TransformerEncoder(t_enc_layer, 2)
    
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
    
    # "fused_neural_graph_encoded" are basically the nodes with graph learned embeddings
    def forward(self, fused_neural_graph_encoded):
        # print(fused_neural_graph_encoded.shape)
        
        pos_encoded_graph_features = self.pos_encoder(fused_neural_graph_encoded)
        transformer_encoded_graph_features = self.tranformer_encoder(pos_encoded_graph_features)
        
        transformer_encoded_graph_features = transformer_encoded_graph_features.reshape(1, transformer_encoded_graph_features.shape[0] * transformer_encoded_graph_features.shape[1])
        pred_kin = self.mlp(transformer_encoded_graph_features)
        
        return pred_kin
    
class BioniXDecoderV3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = EMGEEGFusionEncoderv3()
        self.decoder = KinematicDecoderv3(num_angles=18)
        
    def forward(self, emg_graph, eeg_graph):
        fused_neural_graph_pred = self.encoder(emg_graph, eeg_graph)
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