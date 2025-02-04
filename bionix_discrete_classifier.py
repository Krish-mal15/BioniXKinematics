import torch.nn as nn
from torch_geometric.nn import GIN
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
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

class DiscreteEMGClassifier(nn.Module):
    def __init__(self, n_emg_timesteps, latent_dim):
        super().__init__()
        
        self.embedding = nn.Linear(n_emg_timesteps, latent_dim)
        self.pos_encoder = PositionalEncoding(d_model=latent_dim)
        
        t_enc_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(t_enc_layer, num_layers=2)
        
        self.graph_embedding = GIN(
            in_channels=latent_dim,
            hidden_channels=latent_dim * 2,
            out_channels=latent_dim,
            num_layers=2
        )
        
        self.graph_embedding_out = GIN(
            in_channels=latent_dim,
            hidden_channels=latent_dim // 2,
            out_channels=latent_dim // 4,
            num_layers=2
        )
        
        self.out_transform = nn.Linear(in_features=latent_dim // 4, out_features=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, emg_graph):
        temp_embeddings = self.transformer_encoder(emg_graph.x)
        graph_struct_embeds = self.graph_embedding(temp_embeddings, emg_graph.edge_index, emg_graph.edge_attr)
        graph_struct_embeds_out = self.graph_embedding_out(graph_struct_embeds, emg_graph.edge_index, emg_graph.edge_attr)
        
        out_pred_logits = self.out_transform(graph_struct_embeds_out)
        out_probs = self.softmax(out_pred_logits)
        
        return out_probs
        
        
        
        
