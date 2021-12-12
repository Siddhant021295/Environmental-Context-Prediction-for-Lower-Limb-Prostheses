import torch.nn as nn
import torch.nn.functional as F
# Defining the CNN layers architecture
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv1d(6, 12, 5,stride = 1)
        self.conv2 = nn.Conv1d(12, 24, 5,stride = 1)
        self.conv3 = nn.Conv1d(24, 48, 5,stride = 1)
        self.fc1 = nn.Linear(48*48, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)
    def forward(self, x):
        x = x.float() 
        x = self.conv1(x)                                     
        x = F.relu(x)
        x = self.conv2(x)                                      
        x = F.relu(x)
        x = self.conv3(x)                                     
        x = F.relu(x)
        x = x.view(-1, 48*48)                                 
        x = self.fc1(x)                                       
        x = F.relu(x)
        x = self.fc2(x)                                       
        x = F.relu(x)
        x = self.fc3(x)                                       
            
        return x
