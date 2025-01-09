# Identify-Data-Drift
 Repo with differents techniques to identify Data Dritfts

## INCOMPLETE - WORK IN PROGESS

## Content
This repo contains differents codes with techniques to Identify Data Dritfs. Given two differents data sets, named "data A" and "data B", the objective is identify how differents are these datasets. 

The list of techniques are the following:

- **Adversarial Validation**: train a model (carefully with train complex models and overfitting) using "data A" and evaluate it with "data B". The ojetive is evaluate the model with "data B" and have bad metrics (acc = 0.5, etc). So the model hasn't the ability to identify between these two datasets, and there are no data drift

- **Euclidian Distance**: in progress

- **Statistical Test (KS)**: in progress