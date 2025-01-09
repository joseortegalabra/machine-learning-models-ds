## GENERAL ##

In this repo you will see a simple example of Esembles Methods (in some parts you see that like Stackings Methods) using sklearn library. You only need to execute the script run.py to get data and train different models (individual and ensembles models) and get a boxplot with the distribution of metrics of each model trained to see which model is better. Also there are a folder discovery when you can see a jupyter Notebook with the same codes that are runned in the script main.py

If you search the Kaggle Competitions (or any competitions of Machine Learning) you will see the top results are stacking/esembles models. This ensembles methods consists in stacking multiple and diferent models to get a final model with better performance 

Stacking the solutions of differents models for example linear regression, random forest, XGBoost and Neuronal Netwook in a first level, called most of time like "Base-Models" where are all the models used to predict and a second level called "Meta-Models" that generate a final predicction considering the predictions of all the models in the First Level


## Discleimer ##
In this repo you only see a example of Esemble methods of data where each observation are independient each other, the data is for a "regression problems".


## Content of this Repo ##
- In this Repo are used 4 datasets:
  - The first is a duplicated of the post used as a base of this example (https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
  - The second one has different parameters 
  - The third is the tensorflow dataset, diamonds price (https://www.tensorflow.org/datasets/catalog/diamonds)
  - The last one is a Kaggle simple dataset of Houses price (https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)

- In the first step is I use Juputer notebooks to generate the data of first dataset and evaluate different models. All of the process is in the folder discovery
- In the second step I used the script run.py to run the generalization of the codes present in the folder discovery to train the same ensemble architecture with different EXAMPLES datasets.
- You can decide to study the codes in jupyter notebooks (folder discovery) or study the script run.py to do the same process present in the notebooks with the difference that are running scripts
- The only difference between Notebooks and scripts are that the Notebooks are running with only the first dataset and scripts run the 4 datasets present in this repo 


## Order Folder ##
In this repo you can see the following folder:
- data: folder where are located all the datasets used in the examples of this repo
- discovery: folder where are located all the Notebooks used to discovery and study ensembles methods (with dataset 1)
- cleaning_data: folder with scripts to generate/cleaning the 4 datasets
- model_train: folder with script to train the models (individual and ensemble). A generalized script that recibe any dataset and evaluate different models with it
- reports: folder with results of training individual models and ensembles methods with different datasets


## Source ##

This repo contains information that I learn studying other posts. I replicated it to learn how to do this things.

Source 1: https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
