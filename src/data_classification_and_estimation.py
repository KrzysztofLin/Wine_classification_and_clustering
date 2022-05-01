from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from settings import (CLASSIFICATION_ALGORITHMS_AND_PARAMETERS,
                      ESTIMATION_ALGORITHMS_AND_PARAMETERS)
from abstract import EvaluateModelAbstract

### tutaj musze uproscic zrobic np model_enavulate

def classify_data(data_subsets):
    for algorithm, parameters in CLASSIFICATION_ALGORITHMS_AND_PARAMETERS.items():
        algorithm_with_best_parameters = FindBestHyperparameters(
            data_subsets
        ).crossvalidate(algorithm, parameters)
        Classification(data_subsets).evaluate(algorithm_with_best_parameters)


def estimate_data(data_subsets):
    for algorithm, parameters in ESTIMATION_ALGORITHMS_AND_PARAMETERS.items():
        algorithm_with_best_parameters = FindBestHyperparameters(
            data_subsets
        ).crossvalidate(algorithm, parameters)
        Estimation(data_subsets).evaluate(algorithm_with_best_parameters)


class FindBestHyperparameters():
    def __init__(self, data_subsets):
        self.x_train_norm = data_subsets["x_train_norm"]
        self.y_train = data_subsets["y_train"]

    def crossvalidate(self, algorithm, parameters) -> None:
        classifier = GridSearchCV(algorithm, parameters, cv=4, n_jobs=-1)
        classifier.fit(self.x_train_norm, self.y_train)
        print(classifier.best_params_)
        return classifier.best_estimator_


class Classification(EvaluateModelAbstract):
    def __init__(self, data_subsets):
        self.x_train_norm = data_subsets["x_train_norm"]
        self.x_test_norm = data_subsets["x_test_norm"]
        self.y_test = data_subsets["y_test"]
        self.y_train = data_subsets["y_train"]

    def evaluate(self, algorithm_with_best_parameters) -> None:
        classifier = algorithm_with_best_parameters
        classifier.fit(self.x_train_norm, self.y_train)
        y_predicted_train = classifier.predict(self.x_train_norm)
        calculate_metrics(self.y_train, y_predicted_train)
        y_predicted_test = classifier.predict(self.x_test_norm)
        calculate_metrics(self.y_test, y_predicted_test)


class Estimation(EvaluateModelAbstract):
    def __init__(self, data_subsets):
        self.y_train_norm = data_subsets["y_train_norm"]
        self.x_train_norm = data_subsets["x_train_norm"]
        self.x_test_norm = data_subsets["x_test_norm"]
        self.y_test = data_subsets["y_test"]

    def evaluate(self, algorithm_with_best_parameters) -> None:
        estimator = algorithm_with_best_parameters
        estimator.fit(self.x_train_norm, self.y_train_norm)
        y_predicted_test = estimator.predict(self.x_test_norm)
        y_predicted_denormalized_test = denormalization(y_predicted_test, self.y_test)
        print(y_predicted_denormalized_test)
        y_predicted_denormalized_rounded = []
        [
            y_predicted_denormalized_rounded.append(int(round(i, 0)))
            for i in y_predicted_denormalized_test
        ]
        calculate_metrics(self.y_test, y_predicted_denormalized_rounded)
        print(self.y_test, y_predicted_denormalized_rounded)

def denormalization(y_predicted: np.ndarray, y_actual: pd.Series) -> np.ndarray:
    y_predicted_denormalized = np.zeros(y_predicted.shape[0])
    for i in range(len(y_predicted.shape[0])):
        y_predicted_denormalized[i] = (
            y_predicted[i] * (max(y_actual) - min(y_actual))
        ) + min(y_actual)
    return y_predicted_denormalized


def calculate_metrics(y_actual, y_predicted):
    def check_MSE_accuracy(y_actual: pd.Series, y_predicted: List[float]) -> None:
        print(f"MAE for set {round(mean_absolute_error(y_actual, y_predicted), 4)}")

    def accuracy_calculator(y_actual: pd.Series, y_predicted: List[float]) -> None:
        res = y_actual - y_predicted
        count = 0
        count_with_extended_interval = 0
        # ukryc ponizej te funkcje, stworzyc nowa lub z podkreslinikem
        for i in res:
            if i == 0:
                count += 1
                count_with_extended_interval += 1
            elif i == 1 or i == -1:
                count_with_extended_interval += 1
        print(
            f"accuracy for set is equal: {round(count / len(res), 4)}",
            f"accuracy with extended interval (-1 or 1): {round(count_with_extended_interval / len(res), 4)}",
        )


    check_MSE_accuracy(y_actual, y_predicted)
    accuracy_calculator(y_actual, y_predicted)
