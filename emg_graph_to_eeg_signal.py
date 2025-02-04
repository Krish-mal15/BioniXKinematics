import torch.nn as nn
from emg_graph_to_eeg_signal_utils import EMGFeaturizer, EEGDecoderV2

class EMGToEEGDecoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.graph_transformer_encoder = EMGFeaturizer(n_emg_graph_features=128,
                                                       hidden_dim_factor=2,
                                                       n_eeg_channels=32,
                                                       n_eeg_samples=1200,
                                                       n_emg_channels=5)
        
        
        # If using DecoderV2, then set transformer_decode_first to false, as true does the same as V1
        # self.decoder = EEGDecoderV3(n_eeg_timesteps=1200, hidden_size=512)
        self.decoder = EEGDecoderV2(feature_dim=512, n_eeg_timesteps=1200, transformer_decode_first=False)
        
    def forward(self, emg_graph):
        emg_spt_features = self.graph_transformer_encoder(emg_graph)
        # print(emg_spt_features.shape)
        eeg_decoded = self.decoder(None, emg_spt_features)
        
        return eeg_decoded
    
    
class EMGToEEGDecoderSegmentsV2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.graph_transformer_encoder = EMGFeaturizer(n_emg_graph_features=128,
                                                       hidden_dim_factor=8,
                                                       n_eeg_channels=32,
                                                       n_eeg_samples=10,
                                                       n_emg_channels=5)
        
        
        # If using DecoderV2, then set transformer_decode_first to false, as true does the same as V1
        # self.decoder = EEGDecoderV3(n_eeg_timesteps=1200, hidden_size=512)
        self.decoder = EEGDecoderV2(feature_dim=512, n_eeg_timesteps=10, transformer_decode_first=False)
        
    def forward(self, emg_graph):
        emg_spt_features = self.graph_transformer_encoder(emg_graph)
        # print(emg_spt_features.shape)
        eeg_decoded = self.decoder(None, emg_spt_features)
        
        return eeg_decoded