import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Batch_Normalization(nn.Module):
  def __init__(self,flag_cuda):
    
    super(CNN_LSTM_Batch_Normalization, self).__init__()
    self.flag_cuda = flag_cuda
    self.conv1 = nn.Conv1d(6, 12, 5,stride = 1,padding = 2)
    self.norm1 = nn.BatchNorm1d(12)
    self.conv2 = nn.Conv1d(12, 24, 5,stride = 1,padding = 2)
    self.norm2 = nn.BatchNorm1d(24)
    self.conv3 = nn.Conv1d(24, 48, 5,stride = 1,padding = 2)
    self.norm3 = nn.BatchNorm1d(48)
    self.lstm = nn.LSTM(48,128,2, batch_first=True)
    self.fc1 = nn.Linear(60*128, 256)
    self.norm4 = nn.BatchNorm1d(256)
    self.fc2 = nn.Linear(256, 64)
    self.norm5 = nn.BatchNorm1d(64)
    self.drop = nn.Dropout(p =0.1)
    self.fc3 = nn.Linear(64, 4)
  
  def forward(self, x):
    if self.flag_cuda:
        h0 = torch.zeros(2, x.size(0), 128).cuda()
        c0 = torch.zeros(2, x.size(0), 128).cuda()
    else:
        h0 = torch.zeros(2, x.size(0), 128)
        c0 = torch.zeros(2, x.size(0), 128)
    x = x.float() 
    x = self.conv1(x)                                     
    x = self.norm1(x) 
    x = F.relu(x)
    
    x = self.conv2(x)                                      
    x = self.norm2(x) 
    x = F.relu(x)
    x = self.drop(x)

    x = self.conv3(x)                                     
    x = self.norm3(x)
    x = F.relu(x)
    x = self.drop(x)
    x = x.reshape(-1,60,48)

    x, _ = self.lstm(x, (h0,c0))
    x = x.reshape(x.shape[0],-1)                              
    x = self.fc1(x) 

    x = self.norm4(x)
    x = F.relu(x)
    x = self.drop(x) 
      
    x = self.fc2(x)                                       
    x = self.norm5(x)   
    x = F.relu(x)
    
    x = self.drop(x)
    x = self.fc3(x)                                       
      
    return x
