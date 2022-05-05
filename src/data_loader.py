import pandas as pd

from settings import PATH
from utils import summarize_nan_values


def load_data() -> pd.DataFrame:
    data = pd.read_csv(PATH, sep=";")
    return data


def check_data_correctness(data: pd.DataFrame):
    pd.set_option("display.max_columns", 12)
    data.head()
    summarize_nan_values(data)
