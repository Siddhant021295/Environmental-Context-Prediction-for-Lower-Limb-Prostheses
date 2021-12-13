from data_cleaning import data_cleaning
from data_processing import data_processing
from transforms import Normalize,Compose
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
import torch
from util import *


if __name__ == '__main__':
    location = 'Data/TrainingData/'
    window_size,step_size=60,4
    min_count= [100000,100000,100000,100000]
    test_size = 0.1
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
    batch_size = 64
    n_epochs = 20
    optimizer = 'Adam'

    
    
    data = data_cleaning(location)
    print(data.head())
    print(data.columns)
    print(data.dtypes)

    X_train, Y_train, Y_test, X_test, len_train, len_test, data_code = data_processing(data,window_size,step_size,test_size,min_count)
    
    # Size of the batch
    transform = Compose([norm_method, ToTensor()])
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
    criterion = nn.CrossEntropyLoss()
    flag_cuda = torch.cuda.is_available()

    if not flag_cuda:
        print('Using CPU')
    else:
        print('Using GPU')



    label_count=df['labels']
    n_categories=len(df['labels'].unique())
    confusion = torch.zeros(n_categories, n_categories)
    model_data={}
    models_all = models()
    for model_name in models_all:
        model = models_all[model_name]
        if flag_cuda:
            model.cuda()
    
    # Specifying the loss function
        optimizer = 
        
        save_model_name = 'data_code_'+data_code+'_optimizer_Adam'+model_name+'.pt'
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




    models_all = models()
    for model_name in models_all:
        model = models_all[model_name]
        if flag_cuda:
            model.cuda()
        
        save_model_name = 'data_code_'+data_code+'_optimizer_Adam'+model_name+'.pt'
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


    # Plotting the learning curves
    legend = []
    plt.figure(figsize=(20,10))
    for data in  list(model_data.keys()):
        if data.find('Adam') != -1 : 
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
