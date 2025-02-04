from run_pred import pred_eeg
from bionix_main_v2 import BioniXKinModel
import torch
from kin_data_mtx import dataloader
import torch.nn as nn
from torch_geometric.data import Data
from run_pred import pred_eeg

device = torch.device('cpu')
model = BioniXKinModel(num_joints=18)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg, kin_label in dataloader:
        features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg, kin_label = features_emg.to(device), edge_index_emg.to(device), edge_weight_emg.to(device), features_eeg.to(device), edge_index_eeg.to(device), edge_weight_eeg.to(device), kin_label.to(device)
        
        emg_graph_data = Data(x=features_emg.squeeze(0), edge_attr=edge_weight_emg.squeeze(0), edge_index=edge_index_emg.squeeze(0))
        eeg_pred = pred_eeg(emg_graph_data)
        eeg_graph_data = Data(x=eeg_pred.squeeze(0), edge_attr=edge_weight_eeg.squeeze(0), edge_index=edge_index_eeg.squeeze(0))
        
        optimizer.zero_grad()
        
        print("EMG: ", features_emg.shape)
        print("EEG: ", eeg_graph_data)
        kin_pred = model(features_emg.squeeze(0), eeg_graph_data)
        print("Pred: ", kin_pred)

     
        loss = criterion(kin_pred, kin_label.squeeze(0))
        print(loss)
        print("")

        
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), "bionix_v5_n10_model_v5.pth")



