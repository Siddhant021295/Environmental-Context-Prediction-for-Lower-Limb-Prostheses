import argparse
from transforms import Normalize

def parse_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs',default= 5,type=int,help='Number of total epochs to run')
    parser.add_argument('--min_count',default= [100000,100000,100000,100000],type=list,help='Number of total epochs to run')
    parser.add_argument('--batch_size',default= 64,type=int,help='Number of total epochs to run')
    parser.add_argument('--optimizer_name',default= 'Adam',type=str,help='Adam|RMSprop|SGD')
    parser.add_argument('--norm_method',default=Normalize([0, 0], [1, 1]))
    parser.add_argument('--model', default = 'model_CNN_Net', type =str, help = 'model_CNN_Net|model_Net_CNN_LSTM_Norm_Dropout_All|model_Net_CNN_Norm|model_Net_CNN_Norm_Dropout|model_Net_CNN_Norm_Dropout_All')
    parser.add_argument('--learning_rate', default=0.1, type=float, help= 'Initial learning rate (divided by 10 while training by lr scheduler)')    
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--flag_cuda',default= True,type =bool,help='While change if cuda not avaliable')
    parser.add_argument('--train',default= True,type=bool,help ='Training or testing')
    parser.add_argument('--alpha',default= 0.99, type=float,help='When using RSMProp we can change the value of alpha')
    parser.add_argument('--eps',default= 1e-08, type=float,help='When using RSMProp we can change the value of eps')
    parser.add_argument('--weight_decay',default= 1e-08, type=float,help='When using RSMProp we can change the value of weight_decay')
    parser.add_argument('--centered',default= 1e-08, type=float,help='When using RSMProp we can change the value of centered')
    parser.add_argument('--location',default='Data/TrainingData/',help = 'Locaition of the training Data')
    parser.add_argument('--window_size',default=60,type = int,help = 'Creating data parameter for the size of the window')
    parser.add_argument('--step_size',default=4,type = int,help = 'Creating data parameter')
    parser.add_argument('--model_selection',default=False,type = bool, help='While you are in the model selecttion process')
    parser.add_argument('--location_test',default='Data/TestData/',help = 'Locaition of the training Data')
    parser.add_argument('--classes', default= ['Standing/Walking on Solid Ground','Up The Stairs','Down The Stairs','Walking on grass'],help = 'the data labels')
    parser.add_argument('--val_size',default=0.01,type=float,help= 'size of the validation dateset')
    parser.add_argument('--prediction_path',default='./Prediction',type = str,help='Location of the Prediction files')
    parser.add_argument('--load_model',default='./Models_weight/data_code_60_4_optimizer_Adammodel_CNN_Net.pt')
    parser.add_argument('--testing',default=True)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    opt = parse_parameter()
    print(opt)