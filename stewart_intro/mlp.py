import torch.nn as nn
import torch.nn.functional as F


default_n_hidden_nodes = 500


class MLP(nn.Module):

    def __init__(self, n_hidden_nodes=default_n_hidden_nodes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(71, n_hidden_nodes)
        self.fc2 = nn.Linear(n_hidden_nodes, 256)
        self.fc3 = nn.Linear(256, 68)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
