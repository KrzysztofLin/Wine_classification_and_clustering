import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List


def summarize_nan_values(data: pd.DataFrame) -> None:
    nulls_summary = pd.DataFrame(data.isnull().any(), columns=['Nulls'])
    nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(data.isnull().sum())
    nulls_summary['Num_of_nulls [%]'] = round((data.isnull().mean() * 100), 2)
    print(nulls_summary)


def remove_outlier_data(data: pd.DataFrame) -> pd.DataFrame:
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
    outliers_15iqr = pd.DataFrame({'lower_boundary': low_boundary, 'upper_boundary': upp_boundary,
                                   'num_of_outliers_L': num_of_outliers_L, 'num_of_outliers_U': num_of_outliers_U})
    return outliers_15iqr


def standardize_train_set(data_train: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    scaler = MinMaxScaler().fit(data_train.values)
    data_train_norm = scaler.transform(data_train.values)
    data_train_norm = pd.DataFrame(data_train_norm, columns = column_names)
    return data_train_norm


def standardize_test_set(x_train: pd.DataFrame, data_test: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    scaler = MinMaxScaler().fit(x_train.values)
    data_test_norm = scaler.transform(data_test.values)
    data_test_norm = pd.DataFrame(data_test_norm, columns=column_names)
    return data_test_norm


def save_file(filename: str, file: str) -> None:
    with open(filename, mode='a', encoding='utf-8') as wf:
        wf.write(f"{file}\n")
