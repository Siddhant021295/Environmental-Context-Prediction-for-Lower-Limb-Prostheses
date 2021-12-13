import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Models import CNN_LSTM_Batch_Normalization,CNN_Norm_Dropout_All,CNN_Norm_Dropout,CNN_Norm,CNN

def optimizer(optimizer_name,model):
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters())

def models():
    model_CNN_Net = CNN().float()
    model_Net_CNN_Norm = CNN_Norm().float()
    model_Net_CNN_Norm_Dropout = CNN_Norm_Dropout().float()
    model_Net_CNN_Norm_Dropout_All = CNN_Norm_Dropout_All().float()
    model_Net_CNN_LSTM_Norm_Dropout_All = CNN_LSTM_Batch_Normalization().float()
    return {'model_Net_CNN_LSTM_Norm_Dropout_All':model_Net_CNN_LSTM_Norm_Dropout_All,
          'model_CNN_Net':model_CNN_Net,
          'model_Net_CNN_Norm' : model_Net_CNN_Norm,
          'model_Net_CNN_Norm_Dropout' : model_Net_CNN_Norm_Dropout,
          'model_Net_CNN_Norm_Dropout_All': model_Net_CNN_Norm_Dropout_All,}


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

