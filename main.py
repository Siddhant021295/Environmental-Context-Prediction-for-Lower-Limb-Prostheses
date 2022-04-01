from data_cleaning import data_cleaning
from data_processing import data_processing
from transforms import Normalize,Compose
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from dataset import CustomImageDataset,CustomImageDataset_test
import torch
from util import *
import torch.nn as nn
from option import parse_parameter
from model import generate_model
import json
import os
import pandas as pd



if __name__ == '__main__':
    opt = parse_parameter()

    criterion = nn.CrossEntropyLoss()
    
    if opt.flag_cuda == True:
        opt.flag_cuda = torch.cuda.is_available()

    data = data_cleaning(opt.location)
    print(data.head())
    print(data.columns)
    print(data.dtypes)

    if opt.model_selection == True:

        X_train, Y_train, Y_val, X_val, len_train, len_val, data_code = data_processing(data,
                                                                                        opt.window_size,
                                                                                        opt.step_size,
                                                                                        opt.val_size,
                                                                                        opt.min_count,
                                                                                        opt.model_selection)
    
        # Size of the batch
        train_data = CustomImageDataset(X_train,Y_train,opt.window_size,opt.norm_method)
        val_data = CustomImageDataset(X_val,Y_val,opt.window_size,opt.norm_method)

        # Selecting the training and test datasets
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
        val_loader= DataLoader(val_data, batch_size=opt.batch_size)
        train_features,train_labels  = next(iter(train_loader))

        # specify the classes
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")


        if not opt.flag_cuda:
            print('Using CPU')
        else:
            print('Using GPU')



        model_data={}
        model = generate_model(opt)
        if opt.flag_cuda:
            model.cuda()

        print(model)
        # Specifying the loss function
        optimizer = optimizers(opt.optimizer_name,model)
            
        save_model_name = 'data_code_'+data_code+'_'+opt.optimizer_name+'_'+opt.model+'.pt'
        print('################################')
        print('Training ',save_model_name,'...')
        print('################################')

        epochs_list, train_losslist, valid_losslist, model = trainNet(model,
                                                                    criterion,
                                                                    opt.n_epochs,
                                                                    opt.flag_cuda,
                                                                    save_model_name,
                                                                    optimizer,
                                                                    train_loader,
                                                                    val_loader,
                                                                    len_train,
                                                                    len_val)
            
        model.load_state_dict(torch.load('./Models_weight/'+save_model_name))
        print('\n####################')
        print('Train')
        print('####################')
        train_result=assessNet(model,criterion,train_loader,opt.classes,opt.flag_cuda)

        print('####################')
        print('Val')
        print('####################')
        val_result=assessNet(model,criterion,val_loader,opt.classes,opt.flag_cuda)

        
        model_data[save_model_name] = {
            'model_name' : save_model_name,
            'model':opt.model,
            'data_code':data_code,
            'window_size':opt.window_size,
            'step_size': opt.step_size,
            'epochs':epochs_list,
            'epochs_list' : epochs_list, 
            'train_losslist': train_losslist, 
            'valid_losslist' : valid_losslist,
            'train_result' : train_result,
            'val_result' : val_result
            }
        with open(save_model_name+'.txt', 'w') as convert_file:
            convert_file.write(json.dumps(model_data[save_model_name]))

    elif opt.testing == False and opt.model_selection == False : 
        
        X,Y,data_code,len_train = data_processing(data,opt.window_size,opt.step_size,opt.val_size,opt.min_count,opt.model_selection)
        
        train_data = CustomImageDataset(X,Y,opt.window_size,opt.norm_method)
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
        train_features,train_labels  = next(iter(train_loader))

        # specify the classes
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")


        if not opt.flag_cuda:
            print('Using CPU')
        else:
            print('Using GPU')



        model_data={}
        model = generate_model(opt)
        if opt.flag_cuda:
            model.cuda()

        print(model)
        # Specifying the loss function
        optimizer = optimizers(opt.optimizer_name,model)
            
        save_model_name = 'data_code_'+data_code+'_'+opt.optimizer_name+'_final_'+opt.model+'.pt'
        print('################################')
        print('Training ',save_model_name,'...')
        print('################################')

        model = train_network(  model,
                                criterion,
                                opt.n_epochs,
                                opt.flag_cuda,
                                save_model_name,
                                optimizer,
                                train_loader,
                                len_train)

if opt.testing == True and opt.model_selection == False:
    count_x = 0
    count_x_time = 0
    count_y = 0
    count_y_time = 0
    x_df_lst = []
    #x_time_df_lst = []
    y_df_lst = []
    #y_time_df_lst = []
    rows_x = 0
    rows_x_time = 0
    rows_y = 0
    rows_y_time = 0
    model = generate_model(opt)
    if opt.flag_cuda:
        model.to('cuda')
        model.load_state_dict(torch.load(opt.load_model,map_location=torch.device('cuda')))
    else:
        model.load_state_dict(torch.load(opt.load_model,map_location=torch.device('cpu')))

    # Change the mode of the model to evaluation
    model.eval()

    files = sorted(os.listdir(opt.location_test))
    print("#"*50)
    files_processed = []
    for i in files:
        if i not in files_processed:
            sub_inst  = i.split('__')[0]  
            ## x files
            j = sub_inst+"__"+"x.csv"
            df_temp = handle_file(opt.location_test , j , "x.csv")
            df_temp.columns = ['x_accelerometer' , 'y_accelerometer' , 'z_accelerometer',
                            'x_gyroscope' , 'y_gyroscope' , 'z_gyroscope', 'subject_name' , 'subject_instance'] 

            count_x = count_x + 1
            rows_x = rows_x + df_temp.shape[0]
            print("File "+j + " processed")
            files_processed.append(j)

        ######
        ## x files - timestamp    
            j = sub_inst+"__"+'x_time.csv'       
            df_temp1 = handle_file(opt.location_test , j , "x_time.csv")
            df_temp1.columns = ['timestamp' , 'subject_name' , 'subject_instance']
            count_x_time = count_x_time + 1
            rows_x_time = rows_x_time + df_temp.shape[0]
        ## merge
        #df_temp = pd.merge(left= df_temp, right=df_temp1, how='inner', left_on=['subject_name','subject_instance'], right_on=['subject_name','subject_instance'])
            if df_temp.shape[0] == df_temp1.shape[0] :
                #print("Rows are matching in : "+ str(i))
                df_temp['timestamp'] = df_temp1['timestamp']
                #x_df_lst.append(df_temp) 
                x_test_df = df_temp
                print("File "+j + " processed") 
                files_processed.append(j)   
            else:
                print("Rows NOT are matching in : "+ str(i))    

        ## y files - timestamp        
            j = sub_inst+"__"+'y_time.csv'            
            df_temp1 = handle_file(opt.location_test , j , "y_time.csv")
            df_temp1.columns = ['timestamp', 'subject_name', 'subject_instance'] 
            print("File "+j + " processed")
            files_processed.append(j)  
            y_test_df = df_temp1
            
            data = data_prep_test(opt.window_size,opt.step_size,df_temp)

        # Changing the data to np.array 
            data1 = np.array(data)
            Y_test = np.vstack(data1)
            test_data = CustomImageDataset_test(Y_test,opt.window_size)
            test_loader = DataLoader(test_data, shuffle=False)

            pred_y = []
            # Iterating over batches of test data
            for data in test_loader:
                # Obtaining predictions and loss
                if opt.flag_cuda:
                    data = data.cuda()
                output = model(data)

                # Converting output probabilities to predicted class
                _, pred = torch.max(output, 1)  
                pred_y.append(int(pred)) 

            first_15 = [pred_y[0] ] * ( y_test_df.shape[0] - len(pred_y) )
            final_pred_y = first_15 + pred_y 
            out_file_name = sub_inst +"__y.csv" 

            pred_y_df = pd.DataFrame(final_pred_y)
            pred_y_df.to_csv(opt.prediction_path + '/' + out_file_name , header=False , index = False)

            print("pred done")


        print("#"*50)




    