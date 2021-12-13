from data_cleaning import data_cleaning
from data_processing import data_processing
from transforms import Normalize,Compose
from torchvision import transforms 
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import CustomImageDataset


if __name__ == '__main__':
    location = 'Data/TrainingData/'
    window_size,step_size=60,4
    min_count= [100000,100000,100000,100000]
    test_size = 0.1
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
    batch_size = 64
    
    
    
    data = data_cleaning(location)
    print(data.head())
    print(data.columns)
    print(data.dtypes)

    X_train, Y_train, Y_test, X_test, len_train, len_test, data_code = data_processing(data,window_size,step_size,test_size,min_count)
    
    # Size of the batch
    transform=Compose([norm_method, ToTensor()])
    train_data = CustomImageDataset(X_train,Y_train,window_size,transform)
    test_data = CustomImageDataset(X_test,Y_test,window_size,transform)

    # Selecting the training and test datasets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    train_features,train_labels  = next(iter(train_loader))

    # specify the image classes
    classes = ['Standing/Walking on Solid Ground','Up The Stairs','Down The Stairs','Walking on grass']
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

