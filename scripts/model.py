import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import torch.nn.init as init

class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GNNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels + edge_dim, out_channels)
        init.kaiming_normal_(self.lin.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.edge_dim = edge_dim

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_index, edge_attr):
        node_edge_feature = torch.cat([x_j, edge_attr], dim=-1)
        return F.leaky_relu(self.lin(node_edge_feature), negative_slope=0.01)

class GNN(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GNNLayer(node_in_channels, hidden_channels, edge_in_channels)
        self.conv2 = GNNLayer(hidden_channels, hidden_channels * 2, edge_in_channels)
        self.conv3 = GNNLayer(hidden_channels * 2, hidden_channels * 4, edge_in_channels)
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.fc2 = nn.Linear(hidden_channels, 1)
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='linear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.fc2(x)
        x = self.sigmoid(x)   
        return x