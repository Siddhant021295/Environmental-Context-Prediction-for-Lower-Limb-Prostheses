import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
import random
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from scipy import stats
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def funition_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
funition_seed(42)


#Reading Data upscaled the data removed the 1st entry from the X values. 
dtype = {
    'subject_name'  : np.str_
    ,'subject_instance' : np.str_
    ,'timestamp'        : np.float64
    ,'x_accelerometer'  : np.float64
    ,'y_accelerometer'  : np.float64
    ,'z_accelerometer'  : np.float64
    ,'x_gyroscope'      : np.float64
    ,'y_gyroscope'      : np.float64
    ,'z_gyroscope'      : np.float64
    ,'labels'           : np.int64
    }
data_all=pd.read_csv('./data_all.csv',dtype=dtype)

data_all.head()

# Creating a New Column for identifing subject, name ,instance in the same column 
data_all['subject_name_instance'] = data_all['subject_name']+'_'+data_all['subject_instance']

# Finding the name of all the instances to interate over.
subject_name_instances=data_all['subject_name_instance'].unique()

#printing the first 5 columns
data_all.head()




# Changing the data to np.array 
data = np.array(data)
labels = np.array(labels)
print(data.shape)
print(labels.shape)

df=pd.DataFrame(zip(data,labels),columns=['data_point','labels'])
df['labels'].value_counts()

#train
X_train, X_test, Y_train, Y_test =train_test_split(df['data_point'],df['labels'], test_size=0.1, random_state=42, shuffle=True, stratify=df['labels'].values)
def stratified_with_same_number_of_samples(X_train,Y_train):
  min_count= [100000,100000,100000,100000]
  data_train=pd.concat([X_train,Y_train],axis = 1)


  df_majority_0 = data_train[data_train.iloc[:,-1]==0]
  df_minority_3 = data_train[data_train.iloc[:,-1]==3]
  df_minority_2 = data_train[data_train.iloc[:,-1]==2]
  df_minority_1 = data_train[data_train.iloc[:,-1]==1]
  

  df_majority_0_sample =   df_majority_0.sample(min_count[0])   
  df_majority_3_sample =   df_minority_3.sample(min_count[3],replace= True)   
  df_majority_2_sample =   df_minority_2.sample(min_count[2],replace= True)   
  df_majority_1_sample =   df_minority_1.sample(min_count[1],replace= True)


  data_all_sample = pd.concat([df_majority_0_sample, df_majority_1_sample,df_majority_2_sample,df_majority_3_sample], axis=0)
  X_train, Y_train=data_all_sample['data_point'].values, data_all_sample['labels'].values

  return X_train,Y_train

X_train, Y_train = stratified_with_same_number_of_samples(X_train,Y_train)

X_test,Y_test= X_test.values,Y_test.values

X_train = np.vstack(X_train)
X_test = np.vstack(X_test)

len_train = X_train.shape[0]
len_test = X_test.shape[0]
print(len_train,len_test)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(( 0.5), (0.5)) 
])

class CustomImageDataset(Dataset):
    def __init__(self, data,label,window_size):
        self.y_data_point = label
        self.x_data_point = data
        self.window_size = window_size

    def __len__(self):
        return len(self.y_data_point)

    def __getitem__(self, idx):
        data = self.x_data_point[idx]
        label = self.y_data_point[idx]
        data = data.reshape(6,window_size)
        return data, label

    
# Size of the batch
batch_size = 64
train_data = CustomImageDataset(X_train,Y_train,window_size)
test_data = CustomImageDataset(X_test,Y_test,window_size)

# Selecting the training and test datasets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

train_features,train_labels  = next(iter(train_loader))

# specify the image classes
classes = ['Standing/Walking on Solid Ground','Up The Stairs','Down The Stairs','Walking on grass']
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
