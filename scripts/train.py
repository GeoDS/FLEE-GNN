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
    parser.add_argument("--data-dir", type=str, help="the path to the graph data")
    parser.add_argument("--label-dir", type=str, help="the directory for saving the label")
    parser.add_argument("--output-dir", type=str, help="the directory for saving the model")
    parser.add_argument("--size", type=int, help="the number of graph data")
    parser.add_argument("--epochs", type=int, help="the number of epochs")
    args = parser.parse_args()
    return args

# Train the model
def train(loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.view(-1), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss

# Evaluate the model
def test(loader, model, criterion, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        # print(out)
        loss = criterion(out.view(-1), batch.y)
        total_loss += loss.item() * len(batch)
    return total_loss

# Add a function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":

    args = parse_args()
    print(args)
    info_dir = args.info_dir
    data_dir = args.data_dir
    label_dir = args.label_dir
    output_dir = args.output_dir
    size = args.size

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

    dataset = []
    file_not_exist = []
    for index in range(1, size):
        data_path = os.path.join(data_dir, 'data_{}.csv'.format(index))
        label_path = os.path.join(label_dir, 'label_{}.csv'.format(index))
        
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            file_not_exist.append(index)
            continue
            
        raw_data = pd.read_csv(data_path)
        raw_label = pd.read_csv(label_path)
        node2idx = {}
        idx2node = {}
        for i, node in enumerate(raw_label['node']):
            node2idx[node] = i
            idx2node[i] = node
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
        label = label['import_resilience'].values
        label_tensor = torch.tensor(label, dtype=torch.float)
        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=label_tensor)
        dataset.append(data)

    print('file does not exist: ', file_not_exist)

    # Split the data into training and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    # Set up the GNN model, loss function, and optimizer
    node_in_channels = data.num_node_features
    edge_in_channels = data.num_edge_features
    hidden_channels = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN(node_in_channels, edge_in_channels, hidden_channels).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    # Initialize the best test loss to a high value
    best_test_loss = float("inf")

    # Training and evaluation loop
    num_epochs = args.epochs
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train(train_loader, model, criterion, optimizer, device)
        train_loss = train_loss / len(train_data)
        test_loss = test(test_loader, model, criterion, device)
        test_loss = test_loss / len(test_data)
        print(f'Epoch: {epoch+1:02d}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        # Check if the current test loss is better than the previous best
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model, os.path.join(output_dir, "best_model.pth"))
            # print(f"Best test loss improved to {best_test_loss:.4f}. Model saved.")

    # Plot training and test losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.savefig(os.path.join(output_dir, "running_loss.png"), dpi=300)

    