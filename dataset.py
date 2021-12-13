from torch.utils.data import Dataset

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
        return data, label

    
