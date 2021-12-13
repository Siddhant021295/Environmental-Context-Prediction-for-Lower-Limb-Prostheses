import torch.nn as nn
import torch.nn.functional as F
# Defining the CNN with Batch Normalization layers architecture
class CNN_Norm(nn.Module):
    def __init__(self):
        super(CNN_Norm, self).__init__()
        self.conv1 = nn.Conv1d(6, 12, 5,stride = 1)
        self.norm1 = nn.BatchNorm1d(12)

        self.conv2 = nn.Conv1d(12, 24, 5,stride = 1)
        self.norm2 = nn.BatchNorm1d(24)

        self.conv3 = nn.Conv1d(24, 48, 5,stride = 1)
        self.norm3 = nn.BatchNorm1d(48)

        self.fc1 = nn.Linear(48*48, 256)
        self.norm4 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 64)
        self.norm5 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = x.float() 
        x = self.conv1(x)                                     
        x = self.norm1(x)
        x = F.relu(x)

        x = self.conv2(x)                                      
        x = self.norm2(x)
        x = F.relu(x)

        x = self.conv3(x)                                     
        x = self.norm3(x)
        x = F.relu(x)

        x = x.view(-1, 48*48)                                 
            
        x = self.fc1(x)                                       
        x = self.norm4(x)  
        x = F.relu(x)
            
        x = self.fc2(x)                                       
        x = self.norm5(x)
        x = F.relu(x)

        x = self.fc3(x)                                       
            
        return x