import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Models import CNN_LSTM_Batch_Normalization,CNN_Norm_Dropout_All,CNN_Norm_Dropout,CNN_Norm,CNN
import numpy as np
from train import training
from validation import validation
import torch
from sklearn.metrics import classification_report
import pandas as pd


dtype = {
    'timestamp'        : np.float64
    ,'x_accelerometer'   : np.float64
    ,'y_accelerometer'  : np.float64
    ,'z_accelerometer'  : np.float64
    ,'x_gyroscope'      : np.float64
    ,'y_gyroscope'      : np.float64
    ,'z_gyroscope'      : np.float64
    }
def optimizers(optimizer_name,model):
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters())
    if optimizer_name == 'RMSprop':
        return  optim.RMSprop(model.parameters(), lr=.001,alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    if optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=.001,momentum = 0.9) 



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

def assessNet(model,criterion,loader,classes,flag_cuda):
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
    test_loss = test_loss/len(loader.dataset)
    print('Loss: {:.6f}\n'.format(test_loss))

    result_dic = {}
    # Computing the class accuracies
    for i in range(4):
        if class_total[i] > 0:
            print('Accuracy of %10s: %2d%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i],np.sum(class_correct[i]), np.sum(class_total[i])))
            result_dic['Accuracy of'+classes[i]] = 100 * class_correct[i] / class_total[i]
        else:
            print('Accuracy of %10s: N/A (no training examples)' % (classes[i]))

    # Computing the overall accuracy
    print('\nAccuracy (Overall): %2d%% (%2d/%2d)' % (100. * np.sum(class_correct) / np.sum(class_total),np.sum(class_correct), np.sum(class_total)))
    result_dic['Accuracy (Overall)'] = 100. * np.sum(class_correct) / np.sum(class_total)
    return result_dic


def trainNet(model,criterion,n_epochs,flag_cuda,save_model_name,optimizer,train_loader,test_loader,len_train,len_test):
      
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
        train_losslist,train_loss,model = training(train_losslist,model,train_loader,flag_cuda,criterion,optimizer,len_train)
        
        # Change the mode of the model to evaluation
        model.eval()
        
        #Evaluation
        valid_losslist,valid_loss,model = validation(valid_losslist,model,test_loader,flag_cuda,len_test,criterion)

        # Printing training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
        
        # Saving model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(),'./Models_weight/'+save_model_name)
            valid_loss_min = valid_loss
        
    return epochs_list, train_losslist, valid_losslist, model


def train_network(model,criterion,n_epochs,flag_cuda,save_model_name,optimizer,train_loader,len_train):
    epochs_list = [*range(1,n_epochs+1)]

    # List to store loss to visualize
    train_losslist = []
    
    for epoch in epochs_list:
        # Change the mode of the model to training
        model.train()
        
        # Training
        _,_,model = training(train_losslist,model,train_loader,flag_cuda,criterion,optimizer,len_train)
        
        torch.save(model.state_dict(),'./Models_weight/'+save_model_name)

    return model      


def evaluate_confusion_matrix(model,test_loader,flag_cuda,confusion,n_categories):
    prediction = []
    target_v =[]
    with torch.no_grad():
        for data, target in test_loader:
            # Moving tensors to GPU if CUDA is available
            if flag_cuda:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            _, preds = torch.max(output, 1)
            
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1
                prediction.append(int(t))
                target_v.append(int(p))
                #print(confusion)

        print(classification_report(target_v, prediction, target_names=['Standing/Walking','Up_Stairs','Down_Stairs', 'Walking on grass']))
    
        accuracy = 0
        for i in range(n_categories):
            confusion[i] = confusion[i] / confusion[i].sum()
            accuracy += confusion[i][i]
        accuracy /= n_categories
    return confusion


def data_prep_test(window_size,step_size,data_all):
    data=[]

    df_train = data_all
    for i in range(0, df_train.shape[0] - window_size, step_size):
        xs_acc = df_train['x_accelerometer'].values[i: i + window_size].reshape(1,window_size)
        ys_acc = df_train['y_accelerometer'].values[i: i + window_size].reshape(1,window_size)
        zs_acc = df_train['z_accelerometer'].values[i: i + window_size].reshape(1,window_size)
        xs_gyr = df_train['x_accelerometer'].values[i: i + window_size].reshape(1,window_size)
        ys_gyr = df_train['y_accelerometer'].values[i: i + window_size].reshape(1,window_size)
        zs_gyr = df_train['z_accelerometer'].values[i: i + window_size].reshape(1,window_size)
        #label = stats.mode(df_train['labels'][i: i + window_size])[0][0]
        data_point=np.vstack((xs_acc,ys_acc,zs_acc,xs_gyr,ys_gyr,zs_gyr))
        #train_labels.append(label)
        data.append(data_point.reshape(1,6,window_size))
    return data

def handle_file(path, filename, end_text):
    subject = int(filename.split(end_text)[0].split('subject_')[1].split('_')[0])
    instance = int(filename.split(end_text)[0].split('subject_')[1].split('_')[1])
    df = pd.read_csv(path+"/"+filename, header=None , dtype=dtype)
    df['subject_name'] = subject
    df['subject_instance'] = instance    
    return df


