from model_train import train
from cleaning_data import get_data_example1, get_data_example2, get_data_example3, get_data_example4

### PARAMETERS
NUMBER_EXAMPLES = 4   # I load 4 datasets to compare ensemble/stacking methods
path_data_folder = 'data'


# GENERATE DATA AND TRAIN DIFFERENT MODELS TO EVALUATE WHICH OF THEM IS BETTER (repetated cross validation and NO hp tunning)
for example_number in range(1, NUMBER_EXAMPLES+1):
    print(example_number)


    # --- GENERATE DATA AND SAVE IT ---
    # generate data because this is an example with all the datasets are "toys" datasets where all of them are generated or are datasets presents 
    # in the librarys to test models
    eval(f'get_data_example{example_number}.main()')



    # ---- EVALUATE MODELS ------
    train.main(path_data_folder = path_data_folder, 
                path_data_example = f'example{example_number}'
                )