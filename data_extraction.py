
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

#get the data
def get_data(ending_index):
    data = pd.read_csv("train.csv")
    #getting just only the shortened data instead of all of it
    data = data.head(ending_index)
    #deleting rows that have empty values
    data.dropna(subset = ["pressure", "u_in", "u_out"], inplace=True)
    print(data.head(ending_index))
    #extracting the y values from the table
    y_vals = data.loc[:, 'pressure']
    data = data.drop(columns = ["pressure"])
    return data, y_vals


#delete the unnecessary columns
def delete_columns(data, column_names):
    data = data.drop(columns = column_names)
    return data

#getting the non numerical columns
def find_non_numerical_columns(data):
    non_numerical = data.select_dtypes(include =['object'])
    return non_numerical

#using one hot encoder method to encode the non numerical data
def encoding_data(data, non_numerical_data):
    #convert the non numerical into array
    array = non_numerical_data.values.tolist()
    #one hot encode the columns
    encoded_data = one_hot_encoder(array)
    #convert array into a numpy array
    array = np.array(array)
    #add it to pandas
    for column_index in range(len(array)):
        data[array[column_index, 0]] = encoded_data[column_index, 1:len(encoded_data)]
    return data

def one_hot_encoder(non_numerical_data):
    encoder = OneHotEncoder()
    encoder.fit(non_numerical_data)
    encoded_data = np.array(encoder.transform(non_numerical_data))
    return encoded_data

#converting the dataframe into a list so that it can be understood by pca
def getArray(data, y_vals):
    data_array = np.array(data.values.tolist())
    y_vals = np.array(y_vals.values.tolist())
    #removing the first element in each row that contains the column name
    for row_index in range(len(data_array)):
        for col_index in range(len(data_array[row_index])):
            np.delete(data_array, np.where(row_index == row_index and col_index == 0))
    #converting it into a list
    return data_array, y_vals

#splitting the data into train and test
def split_data(data, y_vals, split_index, end_index):
    train = np.array(data[0:split_index])
    y_train = y_vals[0:split_index]
    y_test = y_vals[split_index: end_index]
    test = np.array(data[split_index: end_index])
    return train, test, y_train, y_test

#scaling the data
def scale_data(x_data, y_data):
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data, y_data)
    return x_data
    
#run all of these methods so it is just one big method
def extract_data(end_index, split_index):
    data, y_vals = get_data(ending_index=end_index)
    data, y_vals = getArray(data, y_vals)
    data = scale_data(data, y_vals)
    train, test, y_train, y_test = split_data(data, y_vals, split_index, end_index)
    return train, test, y_train, y_test

