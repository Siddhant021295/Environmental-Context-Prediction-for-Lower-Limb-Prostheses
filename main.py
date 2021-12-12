from data_cleaning import data_cleaning
from data_processing import data_processing
if __name__ == '__main__':
    location = 'Data/TrainingData/'
    data = data_cleaning(location)
    print(data.head())
    print(data.columns)
    print(data.dtypes)
    window_size,step_size=60,4
    min_count= [100000,100000,100000,100000]

    X_train, Y_train,len_train,len_test,data_code = data_processing(data,window_size,step_size,min_count)
    
    
