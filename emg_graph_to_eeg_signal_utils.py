from torch_geometric.nn import GAT
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import torch.nn as nn
import numpy as np
from neural_data_graph_construction import compute_adj_matrix_neural_mat
import torch
import math
# from graph_data_preprocessing import dataloader

#  python emg_graph_to_eeg_signal_utils.py

def corr_matrix_scaler(corr_mat):
    min_val = torch.min(corr_mat)
    max_val = torch.max(corr_mat)
    scaled_eeg_corr_matrix = (corr_mat - min_val) / (max_val - min_val)
    return scaled_eeg_corr_matrix

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
    

class EMGFeaturizer(nn.Module):
    def __init__(self, n_emg_graph_features, hidden_dim_factor, n_emg_channels, n_eeg_channels, n_eeg_samples):
        super().__init__()
        
        self.graph_featurizer = GAT(
            in_channels=n_eeg_samples,
            hidden_channels=n_eeg_samples * hidden_dim_factor,
            out_channels=n_emg_graph_features,
            num_layers=4,
            heads=4
        )
        
        self.proj_to_eeg_channels = nn.Linear(n_emg_channels, n_eeg_channels)
        
        temp_encoder_layer = nn.TransformerEncoderLayer(d_model=n_emg_graph_features, nhead=4, batch_first=True)
        self.temp_encoder = nn.TransformerEncoder(temp_encoder_layer, num_layers=4)
        
        self.pos_encoder = PositionalEncoding(n_emg_graph_features)
        
    def forward(self, graph_data):
        nodes_emg, edge_idx_emg, edge_correlation_emg = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        graph_feature_matrix_emg = self.graph_featurizer(nodes_emg, edge_idx_emg, edge_correlation_emg)
        # print(graph_feature_matrix_emg.shape)
        # output_feature_graph_vector = self.proj_to_seq(graph_feature_matrix_emg)
        poswise_graph_features = self.pos_encoder(graph_feature_matrix_emg)
        # print(poswise_graph_features.shape)
        spatiotemporal_features = self.temp_encoder(poswise_graph_features)
        # print(spatiotemporal_features.shape)
        spatiotemporal_features = spatiotemporal_features.permute(1, 0)
        eeg_encoded_vector = self.proj_to_eeg_channels(spatiotemporal_features)
        eeg_encoded_vector = eeg_encoded_vector.permute(1, 0)
        
        return eeg_encoded_vector 
    

class EEGDecoder(nn.Module):
    def __init__(self, feature_dim, n_eeg_timesteps):
        super().__init__()
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=4) 
        self.decoder = nn.TransformerDecoder(decoder_layer=transformer_decoder_layer, num_layers=4)
        
        self.feature_transform = nn.Linear(feature_dim, n_eeg_timesteps)
                
    def forward(self, tgt, memory_enc):
        
        # Make sure to set tgt to none bvecause we are not teacher forcing
        if tgt is None:
            tgt = torch.zeros_like(memory_enc)
        
        decoded_out = self.decoder(tgt, memory_enc)
        time_series_decoded = self.feature_transform(decoded_out)
        return time_series_decoded
    
    
class EEGDecoderV2(nn.Module):
    def __init__(self, feature_dim, n_eeg_timesteps, transformer_decode_first):
        super().__init__()
        
        self.transformer_decode_first = transformer_decode_first  # If transformer should run on [32x32], then linearly transformed into signal
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=4) 
        self.decoder = nn.TransformerDecoder(decoder_layer=transformer_decoder_layer, num_layers=4)
        
        self.temporal_transform_last = nn.Linear(feature_dim, n_eeg_timesteps)
        self.temporal_transform_initial = nn.Linear(32, feature_dim)
                
    def forward(self, tgt, memory_enc):
        eeg_corr_matrix = corr_matrix_scaler(torch.matmul(memory_enc, memory_enc.T))
        
        if self.transformer_decode_first:
            if tgt is None:
                tgt = torch.zeros_like(memory_enc)
                
            transformer_decoded_out = self.decoder(tgt, memory_enc)
            decoder_main_output = self.temporal_transform_last(transformer_decoded_out)    
        else:
            spt_corr_features = self.temporal_transform_initial(eeg_corr_matrix)
            if tgt is None:
                tgt = torch.zeros_like(spt_corr_features)
            decoder_main_output = self.temporal_transform_last(self.decoder(tgt, spt_corr_features))
        
        return decoder_main_output
    
    
class EEGDecoderV3(nn.Module):
    def __init__(self, n_eeg_timesteps, hidden_size):
        super().__init__()
        
        self.lstm = nn.LSTM(32, 32, num_layers=4, batch_first=False)
        self.n_eeg_timesteps = n_eeg_timesteps
        
                
    def forward(self, encoder_spatial_features):
        eeg_corr_matrix = corr_matrix_scaler(torch.matmul(encoder_spatial_features, encoder_spatial_features.T))
        lstm_out, _ = self.lstm(eeg_corr_matrix.unsqueeze(0))
        # print("LSTM MAIN OUT: ", lstm_out.shape)
        total_timesteps = []
        for timestep in range(self.n_eeg_timesteps):
            lstm_last_hidden = lstm_out[:, -1, :]
            pred_next_eeg = lstm_last_hidden
            total_timesteps.append(pred_next_eeg)

        signal_matrix = torch.stack(total_timesteps, dim=0)
        total_eeg_signal_tensor = signal_matrix.permute(1, 2, 0)
        return total_eeg_signal_tensor
        

# class EMGToEEGDecoder(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
        
#         self.graph_transformer_encoder = EMGFeaturizer(n_emg_graph_features=128,
#                                                   n_eeg_channels=32,
#                                                   n_eeg_samples=1200,
#                                                   n_emg_channels=5)
        
        
#         # If using DecoderV2, then set transformer_decode_first to false, as true does the same as V1
#         # self.decoder = EEGDecoderV3(n_eeg_timesteps=1200, hidden_size=512)
#         self.decoder = EEGDecoderV2(feature_dim=512, n_eeg_timesteps=1200, transformer_decode_first=False)
        
#     def forward(self, emg_graph):
#         emg_spt_features = self.graph_transformer_encoder(emg_graph)
#         # print(emg_spt_features.shape)
#         eeg_decoded = self.decoder(None, emg_spt_features)
        
#         return eeg_decoded
    

        
        
# emg_eeg_decoder = EMGToEEGDecoder()
# for i, (features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg) in enumerate(dataloader):
#     if i > 10:
#         break
#     emg_graph_data = Data(x=features_emg.squeeze(0), edge_attr=edge_weight_emg.squeeze(0), edge_index=edge_index_emg.squeeze(0))
#     # print(emg_graph_data)
#     eeg_decoded = emg_eeg_decoder(emg_graph_data)
#     print(eeg_decoded.shape)
#     # print(eeg_corr_matrix.flatten())

