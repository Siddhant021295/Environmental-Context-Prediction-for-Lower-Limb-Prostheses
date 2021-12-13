
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
dtype = {'subject_name'  : np.str_
,'subject_instance' : np.str_
,'timestamp'        : np.float64
,'x_accelerometer'   : np.float64
,'y_accelerometer'  : np.float64
,'z_accelerometer'  : np.float64
,'x_gyroscope'      : np.float64
,'y_gyroscope'      : np.float64
,'z_gyroscope'      : np.float64
,'labels'           : np.int64}
data_all=pd.read_csv('./data_all.csv',dtype=dtype)

data_all.head()

# Creating a New Column for identifing subject, name ,instance in the same column 
data_all['subject_name_instance'] = data_all['subject_name']+'_'+data_all['subject_instance']

# Finding the name of all the instances to interate over.
subject_name_instances=data_all['subject_name_instance'].unique()

#printing the first 5 columns
data_all.head()

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
#Deciding the window and training the data
window_size,step_size=60,4
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
    # Converting RGB [0,255] to Tensor [0,1]
    transforms.ToTensor(),
    # Normalizes using specified mean and std per channel
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

# Defining the CNN with Batch Normalization layers architecture
class Net_CNN_Norm(nn.Module):
  def __init__(self):
    super(Net_CNN_Norm, self).__init__()
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

# Defining the CNN with Batch Normalization and Dropout layer after every layer architecture
class Net_CNN_Norm_Dropout_All(nn.Module):
  def __init__(self):
    super(Net_CNN_Norm_Dropout_All, self).__init__()
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
    self.drop = nn.Dropout(p =0.1)
    self.fc3 = nn.Linear(64, 4)
  def forward(self, x):
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

    x = x.view(-1, 48*48)                                 
      
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


# Defining the CNN with Batch Normalization(after every layer) and Dropout (only after fully connected layer) layers architecture
class Net_CNN_Norm_Dropout(nn.Module):
  def __init__(self):
    super(Net_CNN_Norm_Dropout, self).__init__()
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
    self.drop = nn.Dropout(p =0.1)
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
    x = self.drop(x)

      
    x = self.fc2(x)                                       
    x = self.norm5(x)   
    x = F.relu(x)
    x = self.drop(x)
    
    x = self.fc3(x)                                       
      
    return x

# Defining the CNN with Batch Normalization and Dropout layer after every layer architecture
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
class Net_CNN_LSTM_Norm_Dropout_All(nn.Module):
  def __init__(self):
    super(Net_CNN_LSTM_Norm_Dropout_All, self).__init__()
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
    if flag_cuda:
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



def validation(valid_losslist,model,test_loader):
  valid_loss = 0.0
  for data, target in test_loader:
      # Moving tensors to GPU if CUDA is available
      if flag_cuda:
          data, target = data.cuda(), target.cuda()
      output = model(data)
      loss = criterion(output, target)
      valid_loss += loss.item()*data.size(0)

  # Calculating average validation losses
  valid_loss = valid_loss/len_test
  valid_losslist.append(valid_loss)

  return valid_losslist,valid_loss,model

def training(train_losslist,model,train_loader):
  train_loss = 0.0
  for data, target in train_loader:
      # Moving tensors to GPU if CUDA is available
      if flag_cuda:
          data, target = data.cuda(), target.cuda()
      # Clearing the gradients of all optimized variables
      optimizer.zero_grad()
  
      output = model(data)
      # Calculating the batch loss
      loss = criterion(output, target)
      # Backward pass: compute gradient of loss with respect to parameters
      loss.backward()
      # Perform a single optimization step (parameter update)
      optimizer.step()
      # Update training loss
      train_loss += loss.item()*data.size(0)
  
  # Calculating average training losses
  train_loss = train_loss/len_train
  train_losslist.append(train_loss)
  
  return train_losslist,train_loss,model

import matplotlib.pyplot as plt

# Specifying the number of epochs
n_epochs = 20

def trainNet(model,criterion,n_epochs,flag_cuda,save_model_name,optimizer,train_loader,test_loader):
  
  # Unpacking the number of epochs to train the model
  epochs_list = [*range(1,n_epochs+1)]

  # List to store loss to visualize
  train_losslist = []
  valid_losslist = []
  valid_loss_min = np.Inf # track change in validation loss

  for epoch in epochs_list:
      # Change the mode of the model to training
      model.train()
      
      # Training
      train_losslist,train_loss,model = training(train_losslist,model,train_loader)
      
      # Change the mode of the model to evaluation
      model.eval()
      
      #Evaluation
      valid_losslist,valid_loss,model = validation(valid_losslist,model,test_loader)

      # Printing training/validation statistics 
      print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
      
      # Saving model if validation loss has decreased
      if valid_loss <= valid_loss_min:
          print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
          torch.save(model.state_dict(),'./Models_weight/'+save_model_name)
          valid_loss_min = valid_loss
        
  return epochs_list, train_losslist, valid_losslist, model

def assessNet(model,criterion,loader):
  # Tracking test loss and accuracy
  test_loss = 0.0
  class_correct = list(0. for i in range(len(classes)))
  class_total = list(0. for i in range(len(classes)))

  # Setting model to evaluate
  model.eval()

  # Iterating over batches of test data
  for data, target in loader:
      # Obtaining predictions and loss
      if flag_cuda:
          data, target = data.cuda(), target.cuda()
      output = model(data)
      loss = criterion(output, target)
      test_loss += loss.item()*data.size(0)

      # Converting output probabilities to predicted class
      _, pred = torch.max(output, 1)    
      # Comparing predictions to true label
      correct_tensor = pred.eq(target.data.view_as(pred))
      correct = np.squeeze(correct_tensor.numpy()) if not flag_cuda else np.squeeze(correct_tensor.cpu().numpy())
      # Calculating test accuracy for each object class
      for i in range(len(correct)):
          label = target.data[i]
          class_correct[label] += correct[i].item()
          class_total[label] += 1

  # Computing the average test loss
  test_loss = test_loss/len(test_loader.dataset)
  print('Loss: {:.6f}\n'.format(test_loss))

  # Computing the class accuracies
  for i in range(4):
      if class_total[i] > 0:
          print('Accuracy of %10s: %2d%% (%2d/%2d)' % (
              classes[i], 100 * class_correct[i] / class_total[i],
              np.sum(class_correct[i]), np.sum(class_total[i])))
      else:
          print('Accuracy of %10s: N/A (no training examples)' % (classes[i]))

  # Computing the overall accuracy
  print('\nAccuracy (Overall): %2d%% (%2d/%2d)' % (
      100. * np.sum(class_correct) / np.sum(class_total),
      np.sum(class_correct), np.sum(class_total)))
  
# Keep track of correct guesses in a confusion matrix
label_count=df['labels']
n_categories=len(df['labels'].unique())
confusion = torch.zeros(n_categories, n_categories)

def evaluate_confusion_matrix(model,test_loader):
  with torch.no_grad():
    for data, target in test_loader:
        # Moving tensors to GPU if CUDA is available
        if flag_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        _, preds = torch.max(output, 1)
        for t, p in zip(target.view(-1), preds.view(-1)):
          confusion[t.long(), p.long()] += 1
        #print(confusion)
    
    accuracy = 0
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
        accuracy += confusion[i][i]
    accuracy /= n_categories

    # Displaying the average accuracy
    print('Average Macro Accuracy = {:.2f}\n'.format(accuracy))
    return confusion

def test_plot(confusion, all_categories):
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

    
def models():
  model_CNN_Net = CNN_Net().float()
  model_Net_CNN_Norm = Net_CNN_Norm().float()
  model_Net_CNN_Norm_Dropout = Net_CNN_Norm_Dropout().float()
  model_Net_CNN_Norm_Dropout_All = Net_CNN_Norm_Dropout_All().float()
  model_Net_CNN_LSTM_Norm_Dropout_All = Net_CNN_LSTM_Norm_Dropout_All().float()
  return {'model_Net_CNN_LSTM_Norm_Dropout_All':model_Net_CNN_LSTM_Norm_Dropout_All,
          'model_CNN_Net':model_CNN_Net,
          'model_Net_CNN_Norm' : model_Net_CNN_Norm,
          'model_Net_CNN_Norm_Dropout' : model_Net_CNN_Norm_Dropout,
          'model_Net_CNN_Norm_Dropout_All': model_Net_CNN_Norm_Dropout_All,}

# Create a complete CNN
model_data={}

criterion = nn.CrossEntropyLoss()
flag_cuda = torch.cuda.is_available()

if not flag_cuda:
    print('Using CPU')
else:
    print('Using GPU')



models_all = models()
for model_name in models_all:
  model = models_all[model_name]
  if flag_cuda:
    model.cuda()
  
  # Specifying the loss function
  optimizer =optim.SGD(model.parameters(), lr=.001,momentum = 0.9)
  
  save_model_name = 'data_code_'+data_code+'_optimizer_SGD_Momentum'+model_name+'.pt'
  print('################################')
  print('Training ',save_model_name,'...')
  print('################################')

  epochs_list, train_losslist, valid_losslist, model = trainNet(model,criterion,n_epochs,flag_cuda,save_model_name,optimizer,train_loader,test_loader)
  
  model.load_state_dict(torch.load('./Models_weight/'+save_model_name))
  print('####################')
  print('Test')
  print('####################')
  assessNet(model,criterion,test_loader)
  print('\n####################')
  print('Train')
  print('####################')
  assessNet(model,criterion,train_loader)

  model_data[save_model_name] = {
      'epochs_list' : epochs_list, 
      'train_losslist': train_losslist, 
      'valid_losslist' : valid_losslist
    }

import matplotlib.ticker as ticker


models_all = models()
for model_name in models_all:
  model = models_all[model_name]
  if flag_cuda:
    model.cuda()
    
  save_model_name = 'data_code_'+data_code+'_optimizer_SGD_Momentum'+model_name+'.pt'
  model.load_state_dict(torch.load('./Models_weight/'+save_model_name))
  print('####################')
  print('Test')
  print('####################')
  assessNet(model,criterion,test_loader)
  print('\n####################')
  print('Train')
  print('####################')
  assessNet(model,criterion,train_loader)

  confusion = evaluate_confusion_matrix(model,test_loader)
  test_plot(confusion,['Standing/Walking','Up_Stairs','Down_Stairs', 'Walking on grass'])



plt.figure(figsize=(20,10))
legend = []
for data in  list(model_data.keys()):
  if data.find('SGD_Momentum') != -1 : 
    epochs_list= model_data[data]['epochs_list']
    train_losslist= model_data[data]['train_losslist']
    valid_losslist = model_data[data]['valid_losslist']
    plt.plot(epochs_list, train_losslist, epochs_list, valid_losslist)
    legend.append(data+' Training')
    legend.append(data+' Validation')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(legend,loc ='right')
plt.title("Performance of Models")
plt.show()

plt.figure(figsize=(20,10))
