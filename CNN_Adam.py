
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
  optimizer = optim.Adam(model.parameters())
  
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


import matplotlib.ticker as ticker


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
