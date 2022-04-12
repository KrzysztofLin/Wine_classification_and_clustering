import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier


def knn_classification_crossvalidation(y_test, y_train, X_train_norm, X_test_norm):
    print("Klasyfikacja knn")
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': np.arange(5, 40)}
    knn_gscv = GridSearchCV(knn, param_grid, cv=5)
    knn_gscv.fit(X_train_norm, y_train)
    best_n_neighbors = knn_gscv.best_params_.values()

    knn_best = KNeighborsClassifier(n_neighbors=list(best_n_neighbors)[0], weights="distance")
    print(list(best_n_neighbors)[0])
    knn_best.fit(X_train_norm, y_train)

    y_predicted_train = knn_best.predict(X_train_norm)
    y_predicted_test = knn_best.predict(X_test_norm)

    print(acuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test, name = "knn_classifier"))


def knn_classification(y_test, y_train, X_train_norm, X_test_norm):
    print("Klasyfikacja knn")

    knn_best = KNeighborsClassifier(n_neighbors=35, weights="distance")
    knn_best.fit(X_train_norm, y_train)

    y_predicted_train = knn_best.predict(X_train_norm)
    y_predicted_test = knn_best.predict(X_test_norm)

    acuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test, name = "knn_classifier")
    '''
    acc_train, acc_train1, acc_test, acc_test1 = acuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test)
    file = (n_neigh, acc_train, acc_train1, acc_test, acc_test1)
    fileSafer("knn_pomiary.txt", str(file))
    '''


def MLP_classification(y_test, y_train, X_train_norm, X_test_norm, random_seed, perceptor_num):
    print("\n KLASYFIKACJA MLP \n")
    siec_neur = MLPClassifier(hidden_layer_sizes=(perceptor_num), activation='relu', solver='adam', alpha=0.0001, max_iter=25000, random_state=random_seed)
    siec_neur.fit(X_train_norm, y_train)

    y_predicted_train = siec_neur.predict(X_train_norm)
    y_predicted_test = siec_neur.predict(X_test_norm)

    acuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test, name = "MLP classifier")
    '''
    acc_train, acc_train1, acc_test, acc_test1 = acuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test)
    file = (perceptor_num, acc_train, acc_train1, acc_test, acc_test1)
    fileSafer("MLP_klasyfikacja_pomiary.txt", str(file))
    '''


def MLP_estimation(y_test, y_train, X_train_norm, X_test_norm, y_train_norm, random_seed, perceptor_num):
    #print("\n SZACOWANIE \n")

    siec_neur = MLPRegressor(hidden_layer_sizes=(perceptor_num), activation='relu', solver='adam', alpha=0.0001, max_iter=25000, random_state=random_seed)
    siec_neur.fit(X_train_norm, y_train_norm)

    y_predicted_train = siec_neur.predict(X_train_norm)
    y_predicted_test = siec_neur.predict(X_test_norm)

    y_predicted_denor_train = denormalization(y_predicted_train, y_train)
    y_predicted_denor_test = denormalization(y_predicted_test, y_train)

    y_predicted_denor_train_rounded = []
    y_predicted_denor_test_rounded = []

    for i in y_predicted_denor_train:
        y_predicted_denor_train_rounded.append(int(round(i,0)))

    for i in y_predicted_denor_test:
        y_predicted_denor_test_rounded.append(int(round(i,0)))

    acuracy_on_train_and_test_set(y_train, y_predicted_denor_train_rounded, y_test, y_predicted_denor_test_rounded, name = "MLP_estimator")
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


def acuracy_on_train_and_test_set(y_train, y_predicted_train, y_test, y_predicted_test, name):
    MAE_train = MAE(y_train, y_predicted_train)
    print(f"MAE dla zbioru uczacego {MAE_train}")
    MAE_test = MAE(y_test, y_predicted_test)
    print(f"MAE dla zbioru testowego {MAE_test}")
    accuracy_calculator(y_train, y_predicted_train)
    accuracy_calculator(y_test, y_predicted_test)
    final_plot(name, y_test, y_predicted_test)


def MAE(y_actual, y_predicted):
    return round(mean_absolute_error(y_actual, y_predicted),4)


def accuracy_calculator(y_actual, y_predicted):
    res = y_actual - y_predicted
    count_with_ones = 0
    count = 0
    for i in res:
        if i == 0:
            count += 1
            count_with_ones += 1
        elif i == 1 or i == -1:
            count_with_ones += 1

    print(f"trafnosc dla zbioru wynosi {round(count / len(res), 4)}",
          f"trafnosc z -1 lub 1 {round(count_with_ones / len(res), 4)}")

