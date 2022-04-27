import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from data_visualization_v2 import final_plot
import abc

class PipelineAbstract(abc.ABC):
    def load_data(self, dataLoader):
        pass
    def preprocess_data(self):
        pass
    def train_model(self, model, Trainer):
        pass
    def evaluate_data(self, model, data):
        pass

Modeltrainer.train_model(data, model= KNN, model_trainer= Classificator )

Modeltrainer.train_model(data, model =  MLP, model_trainer= Estymator )

class TrainModelAbstract(abc.ABC):
    @abc.abstractmethod
    def fit(self, data, model):
        pass

class EvaluateModelAbstract(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, data, model):
        pass

class ModelTrain(TrainModelAbstract):
    def train_model(self, data, model_trainer = KNeighborsClassifier(n_neighbors=2, weights="distance"), params = ):
       model_trainer.fit(data[1], data[2])

class Classificator(EvaluateModelAbstract):
    pass
class Estimator(TrainModelAbstract):
    pass
class ClassificationAndEstimation:
    def __init__(self, y_test: pd.Series, y_train: pd.Series, x_train_norm: np.ndarray, x_test_norm: np.ndarray, y_train_norm: np.ndarray):
        self.y_test = y_test
        self.y_train = y_train
        self.y_train_norm = y_train_norm
        self.x_train_norm = x_train_norm
        self.x_test_norm = x_test_norm

    def knn_classification_crossvalidation(self) -> None:
        print("\nKNN - classification crossvalidation")
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(2, 40)}
        knn_gscv = GridSearchCV(knn, param_grid, cv=5)
        knn_gscv.fit(self.x_train_norm, self.y_train)
        best_n_neighbors = knn_gscv.best_params_.values()

        knn_best = KNeighborsClassifier(n_neighbors=list(best_n_neighbors)[0], weights="distance")
        knn_best.fit(self.x_train_norm, self.y_train)

        y_predicted_train = knn_best.predict(self.x_train_norm)
        y_predicted_test = knn_best.predict(self.x_test_norm)
        accuracy(self.y_train, y_predicted_train")
        accuracy(self.y_test, y_predicted_test )
        final_plot( name="knn_classifier", y_test, y_predicted_test)

    def knn_classification(self) -> None:
        print("KNN - classification")
        knn_best = KNeighborsClassifier(n_neighbors=35, weights="distance")
        knn_best.fit(self.x_train_norm, self.y_train)
        y_predicted_train = knn_best.predict(self.x_train_norm)
        y_predicted_test = knn_best.predict(self.x_test_norm)
        accuracy_on_train_and_test_set(self.y_train, y_predicted_train, self.y_test, y_predicted_test, name="knn_classifier")

    def mlp_classification(self, random_seed: int, perceptor_num: int) -> None:
        print("\nMLP classification")
        mlp = MLPClassifier(hidden_layer_sizes=(perceptor_num), activation='relu', solver='adam', alpha=0.0001,
                                  max_iter=25000, random_state=random_seed)
        mlp.fit(self.x_train_norm, self.y_train)
        y_predicted_train = mlp.predict(self.x_train_norm)
        y_predicted_test = mlp.predict(self.x_test_norm)
        accuracy_on_train_and_test_set(self.y_train, y_predicted_train, self.y_test, y_predicted_test, name="MLP classifier")

    def mlp_estimation(self, random_seed: int, perceptor_num: int) -> None:
        print("\nMLP estimation")
        mlp = MLPRegressor(hidden_layer_sizes=(perceptor_num), activation='relu', solver='adam', alpha=0.0001,
                                 max_iter=25000, random_state=random_seed)
        mlp.fit(self.x_train_norm, self.y_train_norm)

        y_predicted_train = mlp.predict(self.x_train_norm)
        y_predicted_test = mlp.predict(self.x_test_norm)

        y_predicted_denor_train = denormalization(y_predicted_train, self.y_train)
        y_predicted_denor_test = denormalization(y_predicted_test, self.y_train)
        y_predicted_denor_train_rounded = []
        y_predicted_denor_test_rounded = []
        [y_predicted_denor_train_rounded.append(int(round(i, 0))) for i in y_predicted_denor_train]
        [y_predicted_denor_test_rounded.append(int(round(i, 0))) for i in y_predicted_denor_test]

        accuracy_on_train_and_test_set(self.y_train, y_predicted_denor_train_rounded, self.y_test, y_predicted_denor_test_rounded,
                                      name="MLP_estimator")


def denormalization(y_predicted: np.ndarray, y_actual: pd.Series) -> np.ndarray:
    y_predicted_denor = np.zeros(y_predicted.shape[0])
    i = 0
    while i <= (y_predicted.shape[0] - 1):
        y_predicted_denor[i] = (y_predicted[i] * (max(y_actual) - min(y_actual))) + min(y_actual)
        i += 1
    return y_predicted_denor


def accuracy(y_train: pd.Series, y_predicted_train:List[float]) -> None:
    print(type(y_train), type(y_predicted_train))
    print(f"MAE for set {round(mean_absolute_error(y_train, y_predicted_train), 4)}")
    accuracy_calculator(y_train, y_predicted_train)


def accuracy_calculator(y_actual: pd.Series, y_predicted: np.ndarray) -> None:
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
