import torch
from bionix_emg_eeg_to_mech_v3_control import BioniXDecoderV3Control
from kin_data_mtx import dataloader
from torch_geometric.data import Data

model = BioniXDecoderV3Control()
model.load_state_dict(torch.load('bionix_v3_n10_model_control.pth', weights_only=True))
model.eval()

i = 0
sim_percents = []
for features_emg, edge_index_emg, edge_weight_emg, features_eeg, edge_index_eeg, edge_weight_eeg, kin_label in dataloader:
    if i > 1000:
        break
    emg_graph_data = Data(x=features_emg.squeeze(0), edge_attr=edge_weight_emg.squeeze(0), edge_index=edge_index_emg.squeeze(0))
    eeg_graph_data = Data(x=features_eeg.squeeze(0), edge_attr=edge_weight_eeg.squeeze(0), edge_index=edge_index_eeg.squeeze(0))
        
    with torch.no_grad():
        kin_pred = model(emg_graph_data)
        # print('-----------------------------')
        # print(kin_pred)
        # print(kin_label.squeeze(0))
        # print('-----------------------------')
        
        differences = torch.abs(kin_pred - kin_label)

        normalized_diff = differences / torch.max(kin_pred)

        average_diff = torch.mean(normalized_diff)

        similarity_percentage = (1 - average_diff)
        print(f"Similarity Percentage: {similarity_percentage:.2f}%")
        
        sim_percents.append(similarity_percentage)
        
        print(i)
        
    i += 1

avg_accuracy = sum(sim_percents) / len(sim_percents)
print("Average Accuracy: ", avg_accuracy)