import pandas as pd
from utils import remove_outlier_data, standardize_train_set, standardize_test_set
from settings import Y_COLUMN_NAME, X_COLUMNS_NAMES, TEST_SIZE, RANDOM_SEED
from sklearn.model_selection import train_test_split

def preprocess_data(data: pd.DataFrame):
    data = remove_outlier_data(data)  # function to remove outliers
    x_train, x_test, y_train, y_test = train_test_split(data[X_COLUMNS_NAMES], data[Y_COLUMN_NAME], test_size=TEST_SIZE,
                                                        random_state=RANDOM_SEED)
    x_train_norm = standardize_train_set(x_train)
    y_train_norm = standardize_train_set(y_train.to_frame())
    x_test_norm = standardize_test_set(x_train, x_test)
    return y_test, y_train, x_train_norm, y_train_norm, x_test_norm

