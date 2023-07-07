import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import json
import torch 
from torch_geometric.data import Data
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import torch.nn as nn
import argparse
import os

from model import GNN
from utils import create_edge_attr_index, normalize_features


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-dir", type=str, help="the directory for map info")
    parser.add_argument("--data-path", type=str, help="the path to the data for evaluation")
    parser.add_argument("--label-path", type=str, help="the path to the label of the data for evaluation")
    parser.add_argument("--model-path", type=str, help="the path to the model")
    parser.add_argument("--output-dir", type=str, help="the directory for saving the model")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print(args)
    info_dir = args.info_dir
    data_path = args.data_path
    label_path = args.label_path
    model_path = args.model_path
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # abreviation to full name
    with open(os.path.join(info_dir, 'state_a2f.json'), 'r') as f:
        state_a2f_dict = json.load(f)

    # full name to abreviation
    state_f2a_dict = {}
    for key in state_a2f_dict:
        state_f2a_dict[state_a2f_dict[key]] = key

    # state full name to latitute and longitude
    with open(os.path.join(info_dir, 'state_f2latlon.json'), 'r') as f:
        state_f2latlon_list = json.load(f)

    # state abreviation to latitute and longitude
    state_a2latlon_dict = {}
    for d in state_f2latlon_list:
        state_a2latlon_dict[state_f2a_dict[d['state']]] = {'lat': d["latitude"], 'lon': d["longitude"]}

    raw_data = pd.read_csv(data_path)
    raw_label = pd.read_csv(label_path)
    node2idx = {}
    idx2node = {}
    for i, node in enumerate(raw_label['node']):
        node2idx[node] = i
        idx2node[i] = node
    
    # remove column from raw_data if it is not in raw_label
    all_nodes = set(raw_label['node'])
    raw_data = raw_data[raw_data['Origin'].isin(all_nodes) & raw_data['Destination'].isin(all_nodes)]


    label = raw_label[['node', 'import_resilience']]
    edge_attr, edge_index =create_edge_attr_index(raw_data)
    edge_attr = normalize_features(edge_attr)
    edge_index = edge_index.applymap(lambda node: node2idx[node]) 

    node_features = []
    for index in idx2node:
        node = idx2node[index]
        node_features.append([state_a2latlon_dict[node]['lat'], state_a2latlon_dict[node]['lon']])
    node_features = np.array(node_features)
    node_features = normalize_features(node_features)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    edge_attr_tensor = torch.tensor(edge_attr.values, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index.values.T, dtype=torch.long)
    label_tensor = torch.tensor(label['import_resilience'].values, dtype=torch.float)

    data = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=label_tensor)

    node_in_channels = data.num_node_features
    edge_in_channels = data.num_edge_features
    hidden_channels = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GNN(node_in_channels, edge_in_channels, hidden_channels).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Evaluate the model on the test set
    predict = model(data)
    predict = predict.detach().cpu().numpy()
    np.savetxt(os.path.join(output_dir, 'predict.csv'), predict, delimiter=",")



