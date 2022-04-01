import pandas as pd 
import numpy as np
import os


def data_cleaning(location):
    # Reading all the data
    file_x_time=[location+x   for x in os.listdir(location) if x.find('x_time.csv') != -1] 
    file_y_time=[location+x   for x in os.listdir(location) if x.find('y_time.csv') != -1 ]
    file_x=[location+x  for x in os.listdir(location)  if x.find('x.csv') != -1 ]
    file_y=[location+x  for x in os.listdir(location) if x.find('y.csv') != -1 ]

    file_x_merge = {}
    file_y_merge = {}

    for x_time in file_x_time:
        x=file_x[file_x.index(x_time.split('_time.')[0] +'.'+ x_time.split('_time.')[1])]
        file_name = x_time.split('_time.')[0]+'_merge.csv'
        x_time_data = pd.read_csv(x_time,skiprows = 1 ,header =None,names=['time_x'])
        x_data = pd.read_csv(x,skiprows = 1 ,header=None,names=['x_accelerometer' , 'y_accelerometer' , 'z_accelerometer', 'x_gyroscope' , 'y_gyroscope' , 'z_gyroscope'])
        x_data['time_x'] = x_time_data['time_x']
        file_x_merge[file_name] = x_data

    for y_time in file_y_time:
        y=file_y[file_y.index(y_time.split('_time.')[0] +'.'+ y_time.split('_time.')[1])]
        file_name = y_time.split('_time.')[0]+'_merge.csv'
        y_time_data = pd.read_csv(y_time,header =None,names=['time_y'])
        y_data = pd.read_csv(y,header=None,names=['labels'])
        y_data['time_y'] = y_time_data['time_y']
        file_y_merge[file_name] = y_data 

    file_data = {}                                   
    for y,x in zip(sorted(list(file_y_merge.keys())),sorted(list(file_x_merge.keys()))):
        file_name = y.replace('__y_','__data_')
        # Number of lines to insert
        line_ins = 3  
        res_dict = {col: [y for val in file_y_merge[y][col] for y in [val] + [np.nan]*line_ins][:-line_ins] for col in file_y_merge[y].columns}
        file_y_merge[y] = pd.DataFrame(res_dict)
        file_y_merge[y]['time_y'] = file_y_merge[y]['time_y'].interpolate(method='pad')
        file_y_merge[y]['labels'] = file_y_merge[y]['labels'].interpolate(method='pad')
        file_x_merge[x] = file_x_merge[x].reset_index()
        file_x_merge[x]=file_x_merge[x].drop(['index'], axis=1)
        file_data[file_name] = file_x_merge[x]
        file_data[file_name]['time_y'] =file_y_merge[y]['time_y']
        file_data[file_name]['labels'] =file_y_merge[y]['labels']
        subject_name=  str(file_name.split('_')[1])
        subject_instance = str(file_name.split('_')[2])
        file_data[file_name]['subject_name'],file_data[file_name]['subject_instance'] = subject_name,subject_instance
        file_data[file_name] = file_data[file_name].dropna()

    data_complete = file_data[list(file_data.keys())[0]]
    for x in list(file_data.keys())[1:]:
        data_complete=pd.concat([data_complete, file_data[x]], axis=0)

    data_complete=data_complete.dropna()

    return data_complete

if __name__ == '__main__':
    location = 'Data/TrainingData/'
    print(data_cleaning(location).head())
    
    # data_complete.to_csv('Data/CleanData/data_all.csv',index= False)




