from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split



def get_dataset_regression(n_samples, n_features, n_informative, noise, random_state):
    
    # generate data
    X, y = make_regression(n_samples=n_samples, 
                           n_features=n_features, 
                           n_informative=n_informative, 
                           noise=noise, 
                           random_state=random_state)
    
    # save in a dataframe X
    columns_name = ['feature_' + str(x + 1) for x in range(n_features)]
    X = pd.DataFrame(X, columns = columns_name)
    
    # save in a dataframe y
    target_name = ['target']
    y = pd.DataFrame(y, columns = target_name)
    
    return X, y


def main():
    print('Generating Data Example 2...')
    
    """ PARAMETERS """
    # parameters my example
    N_SAMPLES = 1000
    N_FEATURES = 25
    N_INFORMATIVE = 5
    N_REDUNDANT = 5 # for classification
    NOISE = 10 # for regression
    RANDOM_STATE = 42

    # parameters save file
    path_data_folder = 'data'
    path_data_example = 'example2'

    """ GENERATE DATA X, y """
    data_X, data_y = get_dataset_regression(n_samples = N_SAMPLES, 
                                            n_features = N_FEATURES, 
                                            n_informative = N_INFORMATIVE, 
                                            noise = NOISE, 
                                            random_state = RANDOM_STATE)

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