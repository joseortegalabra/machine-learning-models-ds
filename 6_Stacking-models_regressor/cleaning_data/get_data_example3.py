from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

import warnings
warnings.filterwarnings("ignore")


def get_dataset_diamonds():

    # load data
    dataset_diamonds_tf = tfds.load('diamonds', split = 'train')
    dataset_diamonds_tf = tfds.as_dataframe(dataset_diamonds_tf)
    
    # define list features and target
    target = 'price'
    features = list(set(list(dataset_diamonds_tf.columns)) - set([target]))
    
    # get data X and y
    X = dataset_diamonds_tf[features]
    y = dataset_diamonds_tf[[target]]

    # print shape
    print('Shape X: ', X.shape)
    print('Shape y: ', y.shape)
    
    return X, y


def main():
    print('Generating Data Example 3...')
    
    """ PARAMETERS """
    # parameters save file
    path_data_folder = 'data'
    path_data_example = 'example3'

    """ GENERATE DATA X, y """
    data_X, data_y = get_dataset_diamonds()

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