from Models import CNN_LSTM_Batch_Normalization,CNN_Norm_Dropout_All,CNN_Norm_Dropout,CNN_Norm,CNN

def generate_model(opt):

    assert opt.model in ['model_CNN_Net','model_Net_CNN_LSTM_Norm_Dropout_All','model_Net_CNN_Norm','model_Net_CNN_Norm_Dropout','model_Net_CNN_Norm_Dropout_All']

    if opt.model == 'model_CNN_Net':
        model = CNN.CNN(opt.flag_cuda)
    elif opt.model == 'model_Net_CNN_LSTM_Norm_Dropout_All':
        model = CNN_Norm_Dropout_All.CNN_Norm_Dropout_All(opt.flag_cuda)
    elif opt.model == 'model_Net_CNN_Norm':
        model= CNN_Norm.CNN_Norm(opt.flag_cuda)
    elif opt.model == 'model_Net_CNN_Norm_Dropout':
        model = CNN_Norm_Dropout.CNN_Norm_Dropout(opt.flag_cuda)
    elif opt.model == 'model_Net_CNN_Norm_Dropout_All':
        model = CNN_LSTM_Batch_Normalization.CNN_LSTM_Batch_Normalization(opt.flag_cuda)
    return model
