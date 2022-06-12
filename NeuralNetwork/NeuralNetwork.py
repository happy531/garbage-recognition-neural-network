import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.convolutional_nn_1 = nn.Conv2d(3, 6, 5)
        self.convolutional_nn_2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.full_connection_layer_1 = nn.Linear(16 * 71 * 71, 120)
        self.full_connection_layer_2 = nn.Linear(120, 84)
        self.full_connection_layer_3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.convolutional_nn_1(x)))
        x = self.pool(F.relu(self.convolutional_nn_2(x)))

        x = x.view(x.size(0), 16 * 71 * 71)
        x = F.relu(self.full_connection_layer_1(x))
        x = F.relu(self.full_connection_layer_2(x))
        x = self.full_connection_layer_3(x)
        return x