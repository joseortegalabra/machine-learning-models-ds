import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR # It take a long time to train, so I delete svm
import xgboost
import lightgbm as ltb
from sklearn import datasets, linear_model
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


### -------------------------
### LOAD AUXILIAR FUNCTIONS NOTEBOOK 1

def evaluate_metrics(model, X, y_true):
    '''
    Evaluate rmse, mae, mean of rmse+mae.
    
    OBS: 
     - Inside the functions is called the method model.predict()
     - Calculate the metrics (using y_true vs y_predict) and save it in a dataframe
    '''
    # predict
    prediction = model.predict(X)
    
    # create a dataframe where the metrics are saved
    df_metrics = pd.DataFrame([], columns = ['metric_value'])
    
    
    #### rmse ####
    # calculate
    rmse = mean_squared_error(y_true = y_true,
                  y_pred = prediction,
                  squared = False)
    # save df
    df_metrics = df_metrics.append(pd.DataFrame([rmse], index = ['RMSE'], columns = ['metric_value']))
    print('RMSE: ', rmse)
    
    
    
    #### mae ####
    # calculate
    mae = mean_absolute_error(y_true = y_true,
                         y_pred = prediction)
    
    # save df
    df_metrics = df_metrics.append(pd.DataFrame([mae], index = ['MAE'], columns = ['metric_value']))
    print('MAE: ', mae)
    
    
    
    #### (rmse + mae) / 2 ####
    # calculate
    rmse_mae = (rmse + mae) / 2
    
    # save df
    df_metrics = df_metrics.append(pd.DataFrame([rmse_mae], index = ['RMSE_MAE'], columns = ['metric_value']))
    print('MEAN_RMSE_MAE: ', rmse_mae)
    
    
    return df_metrics


def evaluate_quality_models(model, X, y_true, df_metrics):
    '''
    Make 3 plots evaluating the quality of the models:
    1) Scatter plot between true and predicted values
    2) Histogram of errors (calcualing y_true - y_pred)
    3) Metrics of the models. (the metrics are saved in a dataframe)
    
    
    - df_metrics needs to have this structure (getting using the function evaluate_metrics)
                metric_value
    RMSE	    124.345780
    MAE	        100.801996
    RMSE_MAE	112.573888
    
    '''
    
    #y_pred = model.predict(X)  # some models return (N_observations, 1) and other models return (N_observations, )
    y_pred = model.predict(X).reshape(X.shape[0], 1)

    fig, axs = plt.subplots(1, 3, figsize = (20, 5))

    # plot scatter y_true vs y_pred
    axs[0].scatter(x = y_pred, y = y_true)
    axs[0].set_xlabel('y_pred', fontsize = 15)
    axs[0].set_ylabel('y_true', fontsize = 15)
    axs[0].set_title('y_true vs y_pred', fontsize = 20)


    # plot histogram errors, only the difference between real and predicted (y_true - y_pred)
    axs[1].hist(y_true - y_pred)
    axs[1].set_xlabel('Freq', fontsize = 15)
    axs[1].set_ylabel('Errors', fontsize = 15)
    axs[1].set_title('Histogram', fontsize = 20)


    # plot bar with the values of the metrics using the dataframe
    axs[2].grid()
    axs[2].bar(x = df_metrics.index, height = df_metrics['metric_value'])
    axs[2].set_xlabel('Metrics', fontsize = 15)
    axs[2].set_ylabel('Value', fontsize = 15)
    axs[2].set_title('Evaluate Metrics', fontsize = 20)


    plt.plot()
#------------------------------

def load_data(path_data):
    '''
    Given the path of the folder where are located the data, load it.
    OBS: the data must to be a csv file and separed into X_train, y_train, X_test, y_test
    '''
    
    X_train = pd.read_csv(path_data + '/' + 'X_train.csv', index_col = 0)
    y_train = pd.read_csv(path_data + '/' + 'y_train.csv', index_col = 0)

    X_test = pd.read_csv(path_data + '/' + 'X_test.csv', index_col = 0)
    y_test = pd.read_csv(path_data + '/' + 'y_test.csv', index_col = 0)

    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    
    return X_train, y_train, X_test, y_test


def ensemble_model_3():
    # define the base models
    level0 = []
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor(random_state = 42)))
    #level0.append(('svm', SVR())) # It take a long time to train, so I delete svm

    # define meta learner model
    level1 = LinearRegression()

    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model

def ensemble_model_4():
    # define the base models
    level0 = []
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor(random_state = 42)))
    #level0.append(('svm', SVR()))
    level0.append(('rf', RandomForestRegressor(random_state = 42))) # append model

    # define meta learner model
    level1 = LinearRegression()

    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model

def ensemble_model_5():
    # define the base models
    level0 = []
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor(random_state = 42)))
    #level0.append(('svm', SVR()))
    level0.append(('rf', RandomForestRegressor(random_state = 42)))
    level0.append(('xgb', xgboost.XGBRegressor(random_state = 42))) # append model

    # define meta learner model
    level1 = LinearRegression()

    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model

def ensemble_model_6():
    # define the base models
    level0 = []
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor(random_state = 42)))
    #level0.append(('svm', SVR()))
    level0.append(('rf', RandomForestRegressor(random_state = 42)))
    level0.append(('xgb', xgboost.XGBRegressor(random_state = 42)))
    level0.append(('ltb', ltb.LGBMRegressor(random_state = 42))) # append model

    # define meta learner model
    level1 = LinearRegression()

    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model


def get_models():
    '''
    Create a function that have ALL THE MODELS THAT WILL trained
    '''
    # create a dictionary where are saved the different models that will be trained
    models = dict()

    # complete dictionary with models
    models['lr'] = LinearRegression()
    models['cart'] = DecisionTreeRegressor(random_state = 42)
    models['rf'] = RandomForestRegressor(random_state = 42)
    models['knn'] = KNeighborsRegressor()
    #models['svr'] = SVR()
    models['xgb'] = xgboost.XGBRegressor(random_state = 42)
    models['ltb'] = ltb.LGBMRegressor(random_state = 42)
    models['ensemble3'] = ensemble_model_3()
    models['ensemble4'] = ensemble_model_4()
    models['ensemble5'] = ensemble_model_5()
    models['ensemble6'] = ensemble_model_6()
    
    # return dictionary with the models
    return models


def evaluate_model(model, X, y):
    '''
    Given a model and a dataset train (X, y).
    Generate the splitter and then train the model with the differents folds and return the metrics getting in 
    the training of each folder
    '''
    # defining a CV SPLITTER (repeated K folds)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)
    
    # getting the scores for each FOLD
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    scores = -scores
    
    return scores


def evaluate_quality_of_models(X, y, dicc_models):
    '''
    Evaluate the quality of the models
    
    Given the dataframe of Train and a dictionary with all the models that will be evaluated using repated cross validation
    '''
    
    # initialize
    results_metrics_model, name_model = [], []


    # run
    for name, model in dicc_models.items():
        print('\nEvaluating... ', name)

        # get metrics of each model
        scores = evaluate_model(model, X, y)
        results_metrics_model.append(scores)
        name_model.append(name)

        #print
        print(f'Mean: {round(np.mean(scores), 3)} // std: {round(np.std(scores), 3)}')
        
    
    return results_metrics_model, name_model


def main(path_data_folder, path_data_example):
    '''
    Given the path of folder with data (in the structure X_train, y_train, X_test, y_test), train differents models (individual and ensemble models) and evaluate them
    '''
    print('evaluating model...')

    # load data
    path_data = path_data_folder + '/' + path_data_example
    X_train, y_train, X_test, y_test = load_data(path_data)


    # Evaluate quality of the models
    models = get_models()

    results_metrics_model, name_model = evaluate_quality_of_models(X = X_train, 
                                                               y = y_train, 
                                                               dicc_models = models)


    # Make boxplot with the distribution of the metrics of each model
    plt.figure(figsize = (20, 10))
    plt.boxplot(results_metrics_model, labels = name_model, showmeans=True)
    plt.savefig(fname = f'reports/results_{path_data_example}')

    print('evaluating models done')
