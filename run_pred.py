import torch
from torch_geometric.data import Data
# from graph_data_preprocessing import dataloader
from emg_graph_to_eeg_signal import EMGToEEGDecoderSegmentsV2, EMGToEEGDecoder
from graph_data_preprocessing import inverse_z_score_norm

device = torch.device('cpu')
eeg_recon_model = EMGToEEGDecoderSegmentsV2().to(device)
eeg_recon_model.load_state_dict(torch.load("models/neurotransform_seg_n10_model_3.pth", weights_only=True, map_location=device))

def pred_eeg(emg_graph_data):
    eeg_recon_model.eval()
    with torch.no_grad():
        eeg_decoded_pred = eeg_recon_model(emg_graph_data) 
        inv_norm_signal = inverse_z_score_norm(eeg_decoded_pred, mean=0, std_dev=1)
        
        return inv_norm_signal
