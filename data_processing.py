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

# Data Preparation function
def data_prep(window_size,step_size,data_all,subject_name_instances):
    train_labels = []
    data=[]
    for subject_name_instance in subject_name_instances:
        df_train = data_all[data_all['subject_name_instance']==subject_name_instance]
        for i in range(0, df_train.shape[0] - window_size, step_size):
            xs_acc = df_train['x_accelerometer'].values[i: i + window_size].reshape(1,window_size)
            ys_acc = df_train['y_accelerometer'].values[i: i + window_size].reshape(1,window_size)
            zs_acc = df_train['z_accelerometer'].values[i: i + window_size].reshape(1,window_size)
            xs_gyr = df_train['x_accelerometer'].values[i: i + window_size].reshape(1,window_size)
            ys_gyr = df_train['y_accelerometer'].values[i: i + window_size].reshape(1,window_size)
            zs_gyr = df_train['z_accelerometer'].values[i: i + window_size].reshape(1,window_size)
            label = stats.mode(df_train['labels'][i: i + window_size])[0][0]
            data_point=np.vstack((xs_acc,ys_acc,zs_acc,xs_gyr,ys_gyr,zs_gyr))
            train_labels.append(label)
            data.append(data_point.reshape(1,6,window_size))
    return data,train_labels

def stratified_with_same_number_of_samples(X_train,Y_train,min_count):
    data_train=pd.concat([X_train,Y_train],axis = 1)

    df_0 = data_train[data_train.iloc[:,-1]==0]
    df_1 = data_train[data_train.iloc[:,-1]==1]
    df_2 = data_train[data_train.iloc[:,-1]==2]
    df_3 = data_train[data_train.iloc[:,-1]==3]
    
    
    
    if df_0.shape[0] > min_count[0]:
        df_0_sample = df_0.sample(min_count[0])
    else:
        df_0_sample = df_0.sample(min_count[0],replace= True)
    
    if df_1.shape[0] > min_count[1]:
        df_1_sample = df_1.sample(min_count[1])
    else:
        df_1_sample =   df_1.sample(min_count[1],replace= True)
    
    if df_2.shape[0] > min_count[2]:
        df_2_sample =   df_2.sample(min_count[2])
    else:
        df_2_sample =   df_2.sample(min_count[2],replace= True)
    
    if df_3.shape[0] > min_count[3]:
        df_3_sample =   df_3.sample(min_count[3])
    else:
        df_3_sample =   df_3.sample(min_count[3],replace= True)
    
    

    data_all_sample = pd.concat([df_0_sample, df_1_sample,df_2_sample,df_3_sample], axis=0)
    X_train, Y_train=data_all_sample['data_point'].values, data_all_sample['labels'].values

    return X_train,Y_train


def data_processing(data_all,window_size,step_size,val_size,min_count,model_selection):
    # Creating a New Column for identifing subject, name ,instance in the same column 
    data_all['subject_name_instance'] = data_all['subject_name']+'_'+data_all['subject_instance']

    # Finding the name of all the instances to interate over.
    subject_name_instances=data_all['subject_name_instance'].unique()
    
    data,labels = data_prep(window_size,step_size,data_all,subject_name_instances)
    data_code = str(window_size)+'_'+str(step_size)

    # Changing the data to np.array 
    data = np.array(data)
    labels = np.array(labels)
    print(data.shape)
    print(labels.shape)

    df=pd.DataFrame(zip(data,labels),columns=['data_point','labels'])
    df['labels'].value_counts()

    #train
    if model_selection == True:
        X_train, X_test, Y_train, Y_test =train_test_split(df['data_point'],df['labels'], test_size=val_size, random_state=42, shuffle=True, stratify=df['labels'].values)
        
        X_train, Y_train = stratified_with_same_number_of_samples(X_train,Y_train,min_count)

        X_test,Y_test= X_test.values,Y_test.values

        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)

        len_train = X_train.shape[0]
        len_test = X_test.shape[0]
        return X_train, Y_train,Y_test,X_test, len_train,len_test,data_code
    else : 
        X , Y = stratified_with_same_number_of_samples(df['data_point'],df['labels'],min_count)

        return X,Y,data_code,X.shape[0]
