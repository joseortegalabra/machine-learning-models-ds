from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")


def get_dataset_house(path_raw_data):
    
    # read data
    data = pd.read_csv(path_raw_data)


    # there 2 features (priori very important features) that have a lot of nulls, simple solution, I delete this columns
    columns_to_delete = ['BuildingArea', 'YearBuilt']
    data = data[list(set(list(data.columns)) - set(columns_to_delete))]


    # there are some null values, so I delete the rows that have nulls
    data.dropna(inplace = True)
    data.reset_index(inplace = True)


    # delete features that I consider NO important: 
    # - date is no important, maybe the older sold is cheaper but it need more feature engineer, so, I delete that
    # - method
    # - Suburb: there are a lot, it need to be grouped. Need more feature engineer
    # - Address: already exist longitud and latitud
    # - SellerG: a lot of people. Need more feature engineer
    # - CouncilArea: a lot. Need more feature engineer
    features_no_important = ['Date', 'Method', 'Suburb', 'Address', 'SellerG', 'CouncilArea']
    data = data[list(set(list(data.columns)) - set(features_no_important))]


    # there 2 feature that are categorical -> one hot label encoded
    list_categorical_features = ['Regionname', 'Type']
    data_numeric = data[list(set(list(data.columns)) - set(list_categorical_features))]
    data_categorical = data[list_categorical_features]
    enc = OneHotEncoder()
    data_categorical_enc = pd.DataFrame(enc.fit_transform(data_categorical).toarray())
    data = pd.merge(data_numeric, data_categorical_enc, left_index=True, right_index=True)


    # separate in X and y
    target = 'Price'
    X = data[list(set(list(data.columns)) - set([target]))]
    y = data[[target]]


    # print shape
    print('Shape X: ', X.shape)
    print('Shape y: ', y.shape)

    return X, y


def main():
    print('Generating Data Example 4...')
    
    """ PARAMETERS """
    # parameters save file
    path_data_folder = 'data'
    path_data_example = 'example4'

    """ GENERATE DATA X, y """
    # This dataset is different because is downloaded from Kaggle, so in the folder of example 4 are the raw data downloaded direcly from kaggle.
    # Source: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
    path = path_data_folder + '/' + path_data_example
    path_raw_data = path + '/melb_data.csv'
    data_X, data_y = get_dataset_house(path_raw_data)

    """ SPLIT TRAIN AND TEST """
    X_train, X_test, y_train, y_test = train_test_split(
        data_X, data_y, test_size = 0.2, random_state = 42, shuffle = True
    )


    """ SAVE DATA """
    # Path folder where to save
    path_save = path_data_folder + '/' + path_data_example

    # save 
    X_train.to_csv(path_save + '/' + 'X_train.csv')
    y_train.to_csv(path_save + '/' + 'y_train.csv')
    X_test.to_csv(path_save + '/' + 'X_test.csv')
    y_test.to_csv(path_save + '/' + 'y_test.csv')