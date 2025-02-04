# This model can take an EMG or EEG signal and predict something like Parkinsom's or athlete performance.
# An effective way to use this model is to use NeuroTransform to reconstruct the signal and obvious errors
# which can be identified by a graph attention mechanism in this model can be identified as neural pathway
# disruption, identifying motor control imprecision in sports or for Parkinson's diagnosis.

import torch.nn as nn
from torch_geometric.nn import GAT, GIN
import torch
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, neural_tsteps, d_model, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(neural_tsteps, d_model)
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(d_model, d_model)
        position = torch.arange(0, d_model, dtype=torch.float).unsqueeze(1)
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

class NeuralReconstructionModel(nn.Module):
    def __init__(self, n_tsteps, n_channels, should_graph_decode):
        super().__init__()
        
        self.should_graph_decode = should_graph_decode
        self.n_channels = n_channels
        
        self.pos_embedder = PositionalEmbedding(neural_tsteps=n_tsteps, d_model=1024)
        t_enc_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=4)
        self.tranformer_encoder = nn.TransformerEncoder(t_enc_layer, num_layers=4)
        
        self.graph_encoder = GIN(
            in_channels=1024,
            hidden_channels=512,
            out_channels=256,
            num_layers=4
        )
        t_enc_layer_recon = nn.TransformerEncoderLayer(d_model=256, nhead=4)
        self.transformer_decoder_recon = nn.TransformerEncoder(t_enc_layer_recon, num_layers=2)
        
        # OPTIONAL
        self.gat_decoder = GAT(
            in_channels=n_channels,
            hidden_channels=n_channels * 2,
            out_channels=n_tsteps,
            num_layers=2
        )
        
    def forward(self, neural_signal):
        pos_embedded_neural_sig = self.pos_embedder(neural_signal)
        transformer_encoded = self.tranformer_encoder(pos_embedded_neural_sig)
        
        edge_index = torch.cartesian_prod(torch.arange(self.n_channels), torch.arange(self.n_channels)).t()
        graph_spt_latent = self.graph_encoder(transformer_encoded, edge_index)
        
        if self.should_graph_decode:
            decoded_dot = torch.matmul(graph_spt_latent, graph_spt_latent.T)
            out_reconstructed = self.gat_decoder(decoded_dot)
        else:
            out_reconstructed = self.transformer_decoder_recon(graph_spt_latent)
            
        return out_reconstructed
            

class NeuralErrorClassifier(nn.Module):
    def __init__(self, neural_channels, timesteps):
        super().__init__()
        
        # Needs nodes as the transposed matrix of itself
        self.channel_error_attn_embeds = GAT(
            in_channels=neural_channels,
            hidden_channels=neural_channels * 2,
            out_channels=neural_channels,
            num_layers=4
        )
        
        self.struct_classifier = GIN(
            in_channels=timesteps,
            hidden_channels=timesteps // 2,
            out_channels=timesteps // 4,
            num_layers=2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(timesteps // 4, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )
    
    # Reconstruction signal error will have the same timesteps as the original signal
    def forward(self, recon_signal_error, original_idx):
        attn_based_pathways = self.channel_error_attn_embeds(recon_signal_error, original_idx)
        struct_based_features = self.struct_classifier(attn_based_pathways, original_idx)
        
        binary_pred = self.classifier(struct_based_features)
        return binary_pred
        
        
        
