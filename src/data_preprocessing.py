import pandas as pd
from utils import remove_outlier_data, standardize_train_set, standardize_test_set
from settings import Y_COLUMN_NAME, X_COLUMNS_NAMES, TEST_SIZE, RANDOM_SEED
from sklearn.model_selection import train_test_split

def preprocess_data(data: pd.DataFrame):
    data = remove_outlier_data(data)  # function to remove outliers
    data_subsets = dict()
    x_train, x_test, y_train, y_test = train_test_split(data[X_COLUMNS_NAMES], data[Y_COLUMN_NAME], test_size=TEST_SIZE,
                                                        random_state=RANDOM_SEED)
    data_subsets['x_train'] = x_train
    data_subsets['y_train'] = y_train
    data_subsets['x_test'] = x_test
    data_subsets['y_test'] = y_test
    data_subsets['x_train_norm'] = standardize_train_set(x_train, X_COLUMNS_NAMES)
    data_subsets['y_train_norm'] = standardize_train_set(y_train.to_frame(), [Y_COLUMN_NAME])
    data_subsets['x_test_norm'] = standardize_test_set(x_train, x_test, X_COLUMNS_NAMES)

    return data_subsets

