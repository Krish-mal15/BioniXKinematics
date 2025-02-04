from emg_graph_to_eeg_signal import EMGToEEGDecoder
import torch
from torch_geometric.data import Data
from torch.optim import Adam
from graph_data_preprocessing import dataloader

# Execute in terminal: python eeg_emg_signal_graph_training.py

device = torch.device('cpu')

# Loss will include MAE [L1] (not MSE because MAE ignores noise/outliers) and Fourier transform to ensure
# similar frequency patterns as that is important to predict brain activation. Also, cosine
# similarity will be used to ensure signal "shapes" meanong oscillatory patterns are
# closely resembled as phase shifts are an important aspect that categorizes different motions

def eeg_signal_loss(pred_eeg, true_eeg, alpha=0.6, beta=0.2, gamma=0.2):
    mae_loss = torch.mean(torch.abs(pred_eeg - true_eeg))
    
    pred_freq = torch.fft.rfft(pred_eeg, dim=-1)
    target_freq = torch.fft.rfft(true_eeg, dim=-1)
    pred_amplitude = torch.abs(pred_freq) 
    target_amplitude = torch.abs(target_freq)
    fft_loss = torch.mean(torch.abs(pred_amplitude - target_amplitude))  # Not 1-fft because we are minimizing different of frequency amplitudes
    
    cos_sim = torch.nn.functional.cosine_similarity(pred_eeg, true_eeg, dim=-1)
    cos_sim_loss = 1 - torch.mean(cos_sim)
    
    return mae_loss * alpha + fft_loss * beta + cos_sim_loss * gamma  # Regularization function usage terms
            
                          
eeg_recon_model = EMGToEEGDecoder()
optimizer = Adam(eeg_recon_model.parameters(), lr=1e-4)

num_epochs = 50
for epoch in range(num_epochs):
    eeg_recon_model.train()
    for features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg in dataloader:
        features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg = features_emg.to(device), edge_index_emg.to(device), edge_weight_emg.to(device), features_eeg.to(device), edge_index_eeg.to(device), edge_weight_eeg.to(device)
        emg_graph_data = Data(x=features_emg.squeeze(0), edge_attr=edge_weight_emg.squeeze(0), edge_index=edge_index_emg.squeeze(0))
        
        optimizer.zero_grad()
        eeg_decoded_pred = eeg_recon_model(emg_graph_data)
        
        print("")
        # print(eeg_decoded_pred)
        # print(features_eeg)
     
        loss = eeg_signal_loss(eeg_decoded_pred, features_eeg)
        print(loss)
        print("")

        
        loss.backward()
        optimizer.step()

