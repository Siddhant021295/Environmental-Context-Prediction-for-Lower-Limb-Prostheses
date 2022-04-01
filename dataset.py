from torch.utils.data import Dataset
import torch
class CustomImageDataset(Dataset):
    def __init__(self, data,label,window_size,transforms = None):
        self.y_data_point = label
        self.x_data_point = data
        self.window_size = window_size
        self.transforms = transforms

    def __len__(self):
        return len(self.y_data_point)

    def __getitem__(self, idx):
        data = self.x_data_point[idx]
        label = self.y_data_point[idx]
        data = data.reshape(6,self.window_size)
        # if self.transforms is not None:
        #     data = self.transforms(data)
        # print(label,type(label))
        label = torch.tensor(label).long()
        return data, label

    
class CustomImageDataset_test(Dataset):
    def __init__(self, data,window_size):
        #self.y_data_point = label
        self.x_data_point = data
        self.window_size = window_size

    def __len__(self):
        return len(self.x_data_point)

    def __getitem__(self, idx):
        data = self.x_data_point[idx]
        #label = self.y_data_point[idx]
        data = data.reshape(6,self.window_size)
        return data