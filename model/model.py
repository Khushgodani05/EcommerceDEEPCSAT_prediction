import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0=nn.Linear(
            in_features=38,
            out_features=64,
        )
        self.fc1=nn.Linear(
            in_features=64,
            out_features=128,
        )
        self.fc2=nn.Linear(
            in_features=128,
            out_features=64,
        )
        self.fc3=nn.Linear(
            in_features=64,
            out_features=32,
        )
        self.fc4=nn.Linear(
            in_features=32,
            out_features=1,
        )
        
    def forward(self,x):
        x=nn.functional.relu(self.fc0(x))
        x=nn.functional.relu(self.fc1(x))
        x=nn.functional.relu(self.fc2(x))
        x=nn.functional.relu(self.fc3(x))
        x=self.fc4(x)
        return x 