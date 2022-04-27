import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from data_visualization_v2 import plot_actual_vs_predicted_values_graph
import abc

class FindBestHyperparametersAbstract(abc.ABC):
    @abc.abstractmethod
    def crossvalidation(self, y_train, x_train_norm, y_train_norm, algorithm, parameters):
        pass


class EvaluateModelAbstract(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, data, model):
        pass


class FindBestHyperparameters(FindBestHyperparametersAbstract):
    def __init__(self, y_train: pd.Series, x_train_norm: np.ndarray, y_train_norm: np.ndarray):
        self.y_train = y_train
        self.y_train_norm = y_train_norm
        self.x_train_norm = x_train_norm

    def crossvalidation(self, algorithm, parameters) -> None:
        classifier = GridSearchCV(algorithm, parameters, cv=5, n_jobs=-1)
        classifier.fit(self.x_train_norm, self.y_train)
        print(classifier.best_params_)
        return classifier.best_estimator_


class Classification(EvaluateModelAbstract):
    def __init__(self, y_test: pd.Series, y_train: pd.Series, x_train_norm: np.ndarray, x_test_norm: np.ndarray, y_train_norm: np.ndarray):
        self.x_train_norm = x_train_norm
        self.x_test_norm = x_test_norm
        self.y_test = y_test
        self.y_train = y_train

    def evaluate(self, algorithm_with_best_parameters) -> None:
        classifier = algorithm_with_best_parameters
        classifier.fit(self.x_train_norm, self.y_train)
        y_predicted_train = classifier.predict(self.x_train_norm)
        calculate_metrics(self.y_train, y_predicted_train)
        y_predicted_test = classifier.predict(self.x_test_norm)
        calculate_metrics(self.y_test, y_predicted_test)


class Estimation(EvaluateModelAbstract):
    def __init__(self, y_test: pd.Series, y_train: pd.Series, x_train_norm: np.ndarray, x_test_norm: np.ndarray, y_train_norm: np.ndarray):
        self.y_test = y_test
        self.y_train = y_train
        self.y_train_norm = y_train_norm
        self.x_train_norm = x_train_norm
        self.x_test_norm = x_test_norm


    def evaluate(self, algorithm_with_best_parameters) -> None:
        estimator = algorithm_with_best_parameters
        estimator.fit(self.x_train_norm, self.y_train_norm)
        y_predicted_test = estimator.predict(self.x_test_norm)
        y_predicted_denormalized_test = denormalization(y_predicted_test, self.y_test)
        y_predicted_denormalized_rounded = []
        [y_predicted_denormalized_rounded.append(int(round(i, 0))) for i in y_predicted_denormalized_test]
        calculate_metrics(self.y_test, y_predicted_denormalized_rounded)


def denormalization(y_predicted: np.ndarray, y_actual: pd.Series) -> np.ndarray:
    y_predicted_denormalized = np.zeros(y_predicted.shape[0])
    for i in range(len(y_predicted.shape[0])):
        y_predicted_denormalized[i] = (y_predicted[i] * (max(y_actual) - min(y_actual))) + min(y_actual)
    return y_predicted_denormalized


def calculate_metrics(y_actual, y_predicted):
    def check_MSE_accuracy(y_actual: pd.Series, y_predicted: List[float]) -> None:
        print(f"MAE for set {round(mean_absolute_error(y_actual, y_predicted), 4)}")

    def accuracy_calculator(y_actual: pd.Series, y_predicted: List[float]) -> None:
        res = y_actual - y_predicted
        count = 0
        count_with_extended_interval = 0
        for i in res:
            if i == 0:
                count += 1
                count_with_extended_interval += 1
            elif i == 1 or i == -1:
                count_with_extended_interval += 1
        print(f"accuracy for set is equal: {round(count / len(res), 4)}",
              f"accuracy with extended interval (-1 or 1): {round(count_with_extended_interval  / len(res), 4)}")

    check_MSE_accuracy(y_actual, y_predicted)
    accuracy_calculator(y_actual, y_predicted)