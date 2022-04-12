import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def nan_data_counter(data: pd.DataFrame) -> None:
    nulls_summary = pd.DataFrame(data.isnull().any(), columns=['Nulls'])
    nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(data.isnull().sum())
    nulls_summary['Num_of_nulls [%]'] = round((data.isnull().mean() * 100), 2)
    print(nulls_summary)


def independent_and_dependent_variables_generator(data: pd.DataFrame, dependent_variable_col_num: int) -> Tuple[int, str, str]:
    x = list(range(0, data.shape[1]))
    x.remove(dependent_variable_col_num)
    independent_variables_col_num = x
    dependent_variable = data.columns[dependent_variable_col_num]
    independent_variables = data.columns[independent_variables_col_num]
    return independent_variables_col_num, dependent_variable, independent_variables


def data_outliers_remover(data: pd.DataFrame) -> pd.DataFrame:
    data_without_outliers = data.copy()
    outliers_15iqr = boudaries(data)
    for row in outliers_15iqr.iterrows():
        data_without_outliers = data_without_outliers[(data_without_outliers[row[0]] >= row[1]['lower_boundary']) & (
                data_without_outliers[row[0]] <= row[1]['upper_boundary'])]
    return data_without_outliers


def boudaries(data: pd.DataFrame) -> pd.DataFrame:
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    low_boundary = (q1 - 1.5 * iqr)
    upp_boundary = (q3 + 1.5 * iqr)
    num_of_outliers_L = (data[iqr.index] < low_boundary).sum()
    num_of_outliers_U = (data[iqr.index] > upp_boundary).sum()
    outliers_15iqr = pd.DataFrame(
        {'lower_boundary': low_boundary, 'upper_boundary': upp_boundary, 'num_of_outliers_L': num_of_outliers_L,
         'num_of_outliers_U': num_of_outliers_U})
    return outliers_15iqr


def data_standarization(data_train: pd.DataFrame, data_test: pd.DataFrame, independent_variable_col_num: int, dependent_variable_col_num: int) -> \
Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train_norm = scaler.transform(data_train)
    data_test_norm = scaler.transform(data_test)
    x_train_norm = data_train_norm[:, independent_variable_col_num]
    x_test_norm = data_test_norm[:, independent_variable_col_num]
    y_train_norm = data_train_norm[:, dependent_variable_col_num]
    return x_train_norm, x_test_norm, y_train_norm


def file_safer(filename: str, file: str) -> None:
    with open(filename, mode='a', encoding='utf-8') as wf:
        wf.write(f"{file}\n")
