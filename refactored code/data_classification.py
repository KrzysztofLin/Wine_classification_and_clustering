import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from data_visualization_v2 import final_plot

class ClassificationAndEstimation:
    def __init__(self, y_test, y_train, x_train_norm, x_test_norm, y_train_norm):
        self.y_test = y_test
        self.y_train = y_train
        self.y_train_norm = y_train_norm
        self.x_train_norm = x_train_norm
        self.x_test_norm = x_test_norm

    def knn_classification_crossvalidation(self):
        print("KNN - classification crossvalidation")
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(5, 40)}
        knn_gscv = GridSearchCV(knn, param_grid, cv=5)
        knn_gscv.fit(self.x_train_norm, self.y_train)
        best_n_neighbors = knn_gscv.best_params_.values()

        knn_best = KNeighborsClassifier(n_neighbors=list(best_n_neighbors)[0], weights="distance")
        knn_best.fit(self.x_train_norm, self.y_train)

        y_predicted_train = knn_best.predict(self.y_train_norm)
        y_predicted_test = knn_best.predict(self.x_test_norm)
        accuracy_on_train_and_test_set(self.y_train, y_predicted_train, self.y_test, y_predicted_test, name="knn_classifier")

    def knn_classification(self):
        print("KNN - classification")
        knn_best = KNeighborsClassifier(n_neighbors=35, weights="distance")
        knn_best.fit(self.x_train_norm, self.y_train)

        y_predicted_train = knn_best.predict(self.x_train_norm)
        y_predicted_test = knn_best.predict(self.x_test_norm)

        accuracy_on_train_and_test_set(self.y_train, y_predicted_train, self.y_test, y_predicted_test, name="knn_classifier")
        '''
        acc_train, acc_train1, acc_test, acc_test1 = acuracy_on_train_and_test_set(self.y_train, y_predicted_train, self.y_test, y_predicted_test)
        file = (n_neigh, acc_train, acc_train1, acc_test, acc_test1)
        fileSafer("knn_pomiary.txt", str(file))
        '''

    def mlp_classification(self, random_seed: int, perceptor_num: int):
        print("\nMLP classification")
        mlp = MLPClassifier(hidden_layer_sizes=(perceptor_num), activation='relu', solver='adam', alpha=0.0001,
                                  max_iter=25000, random_state=random_seed)
        mlp.fit(self.x_train_norm, self.y_train)

        y_predicted_train = mlp.predict(self.x_train_norm)
        y_predicted_test = mlp.predict(self.x_test_norm)
        accuracy_on_train_and_test_set(self.y_train, y_predicted_train, self.y_test, y_predicted_test, name="MLP classifier")
        '''
        acc_train, acc_train1, acc_test, acc_test1 = acuracy_on_train_and_test_set(self.y_train, y_predicted_train, se;f.y_test, y_predicted_test)
        file = (perceptor_num, acc_train, acc_train1, acc_test, acc_test1)
        fileSafer("MLP_klasyfikacja_pomiary.txt", str(file))
        '''

    def mlp_estimation(self, random_seed: int, perceptor_num: int):
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
        '''
        acc_train, acc_train1, acc_test, acc_test1 = acuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test)
        file = (perceptor_num, acc_train, acc_train1, acc_test, acc_test1)
        fileSafer("MLP_szacowanie_pomiary.txt", str(file))
        '''


def denormalization(y_predicted, y_actual):
    y_predicted_denor = np.zeros(y_predicted.shape[0])
    i = 0
    while i <= (y_predicted.shape[0] - 1):
        y_predicted_denor[i] = (y_predicted[i] * (max(y_actual) - min(y_actual))) + min(y_actual)
        i += 1
    return y_predicted_denor


def accuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test, name):
    print(f"MAE for train set {round(mean_absolute_error(y_train, y_predicted_train), 4)}")
    print(f"MAE for test set {round(mean_absolute_error(y_test, y_predicted_test), 4)}")
    accuracy_calculator(y_train, y_predicted_train)
    accuracy_calculator(y_test, y_predicted_test)
    final_plot(name, y_test, y_predicted_test)


def accuracy_calculator(y_actual, y_predicted) -> None:
    res = y_actual - y_predicted
    count = 0
    count_with_extended_interval = 0
    for i in res:
        if i == 0:
            count += 1
            count_with_extended_interval  += 1
        elif i == 1 or i == -1:
            count_with_extended_interval  += 1

    print(f"accuracy for set is equal: {round(count / len(res), 4)}",
          f"accuracy with extended interval (-1 or 1): {round(count_with_extended_interval  / len(res), 4)}")
