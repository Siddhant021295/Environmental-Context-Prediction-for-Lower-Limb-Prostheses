from data_cleaning import data_cleaning
if __name__ == '__main__':
    location = 'Data/TrainingData/'
    data = data_cleaning(location)
    print(data.head(),data.columns)