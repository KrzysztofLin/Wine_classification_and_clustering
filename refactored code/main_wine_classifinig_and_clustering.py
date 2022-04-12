import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

def main():
    data = pd.read_csv("C:\\Users\\Krzychiu\\Documents\\Analiza_danych_studia\\ED\\KL_projekt_zaliczeniowy\\winequality-white.csv", sep=';')
    random_seed = 316733
    pd.set_option('display.max_columns', 12)
    #fileSafer("tabela_nr_1.txt",data.describe())
    #lack_of_data_checker(data) #- sprawdzenie czy istnieją brakujące dane

    #Wybór zmniennej celu, w przypadku wybranych danych zmniennej quality (11 kolumna) oraz predyktorów
    dependend_variable_col_num = 11
    independend_variables_col_num, dependend_variable, independend_variables = independend_and_dependend_variables_generator(data, dependend_variable_col_num)

    correlation_graph_generator(data, independend_variables, dependend_variable, filename_cor = "Macierz_korelacji.png")
    dispersion_graph_generator(data, independend_variables, dependend_variable, filename_disp="Macierz_rozrzutu.png")
    histogram_generator(data)


    #Ponowny wybór danych i generacja wykresów rozrzutu oraz macierzy korelacji
    data = data[data.columns[[0, 1, 2, 3, 4, 6, 8, 9, 10, 11]]]
    dependend_variable_col_num = 9
    independend_variables_col_num, dependend_variable, independend_variables = independend_and_dependend_variables_generator(data, dependend_variable_col_num)
    correlation_graph_generator(data, independend_variables, dependend_variable, filename_cor="Macierz_korelacji_zmodyfikowana.png")

    #data = riding_off_outliered_data(data) #funkcja do usunięcia obserwacji odstających

    # Podział danych na zbiór uczący i testowy
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=random_seed)
    y_test = data_test[dependend_variable]
    y_train = data_train[dependend_variable]


    #standaryzacja danych
    X_train_norm, X_test_norm, y_train_norm = data_standarization(data_train, data_test, independend_variables_col_num, dependend_variable_col_num)

    #klasyfikacja danych
    #knn_classification_crossvalidation(y_test, y_train, X_train_norm, X_test_norm) - niewykorzystywane gdyż gorzej działą

    knn_classification(y_test, y_train, X_train_norm, X_test_norm)
    perceptor_num = 200
    MLP_classification(y_test, y_train, X_train_norm, X_test_norm, random_seed, perceptor_num=perceptor_num)

    #estymacja
    MLP_estimation(y_test, y_train, X_train_norm, X_test_norm, y_train_norm, random_seed, perceptor_num = perceptor_num)

    # grupowanie
    clustering_data_using_kmeans(data_train, data_test, y_test, independend_variables, random_seed)


def lack_of_data_checker(data):
    nulls_summary = pd.DataFrame(data.isnull().any(), columns=['Nulls'])
    nulls_summary['Num_of_nulls [qty]'] = pd.DataFrame(data.isnull().sum())
    nulls_summary['Num_of_nulls [%]'] = round((data.isnull().mean() * 100), 2)
    print(nulls_summary)


def independend_and_dependend_variables_generator(data, dependend_variable_col_num):
    x = list(range(0, data.shape[1]))
    x.remove(dependend_variable_col_num)
    independend_variables_col_num = x
    dependend_variable = data.columns[dependend_variable_col_num]
    independend_variables = data.columns[independend_variables_col_num]
    return independend_variables_col_num, dependend_variable, independend_variables


def correlation_graph_generator(data, independend_variables, dependend_variable, filename_cor):
    names = data.columns
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 10))
    sns.heatmap(np.corrcoef(data.values.T), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13}, yticklabels=names, xticklabels=names)
    plt.savefig(filename_cor)


def dispersion_graph_generator(data, independend_variables, dependend_variable, filename_disp):
    sns.pairplot(data, x_vars=independend_variables, y_vars=dependend_variable, diag_kind=None)
    plt.tight_layout()
    plt.savefig(filename_disp)


def histogram_generator(data):
    data.hist(column='fixed acidity')
    data.hist(column='volatile acidity')
    data.hist(column='residual sugar')
    data.hist(column='citric acid')
    data.hist(column='chlorides')
    data.hist(column='free sulfur dioxide')
    data.hist(column='total sulfur dioxide')
    data.hist(column='density')
    data.hist(column='pH')
    data.hist(column='sulphates')
    data.hist(column='alcohol')
    data.hist(column='quality')


def riding_off_outliered_data(data):
    data_without_outliers = data.copy()
    outliers_15iqr = boudaries(data)
    for row in outliers_15iqr.iterrows():
        data_without_outliers = data_without_outliers[(data_without_outliers[row[0]] >= row[1]['lower_boundary']) & (
                    data_without_outliers[row[0]] <= row[1]['upper_boundary'])]
    return data_without_outliers


def boudaries(data):
    q1 = data.quantile(0.05)  # wartości zmiennej na granicy pierwszego i drugiego kwartyla
    q3 = data.quantile(0.95)  # wartości zmiennej na granicy trzeciego i czwartego kwartyla
    iqr = q3 - q1
    low_boundary = (q1 - 1.5 * iqr)
    upp_boundary = (q3 + 1.5 * iqr)
    num_of_outliers_L = (data[iqr.index] < low_boundary).sum()
    num_of_outliers_U = (data[iqr.index] > upp_boundary).sum()
    outliers_15iqr = pd.DataFrame(
        {'lower_boundary': low_boundary, 'upper_boundary': upp_boundary, 'num_of_outliers_L': num_of_outliers_L,
         'num_of_outliers_U': num_of_outliers_U})
    return outliers_15iqr


def data_standarization(data_train, data_test, independend_variable_col_num ,dependend_variable_col_num):
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_train_norm = scaler.transform(data_train)
    data_test_norm = scaler.transform(data_test)
    X_train_norm = data_train_norm[:, independend_variable_col_num]
    X_test_norm = data_test_norm[:, independend_variable_col_num]
    y_train_norm = data_train_norm[:, dependend_variable_col_num]
    return X_train_norm, X_test_norm, y_train_norm


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


def final_plot(name, y_actual, y_predicted_denor):
    fig = plt.figure()
    a1 = fig.add_axes([0,0,1,1])
    x = range(len(y_actual))
    a1.plot(x,y_actual, 'ro')
    a1.set_ylabel('Actual')
    a2 = a1.twinx()
    a2.plot(x, y_predicted_denor,'o')
    a2.set_ylabel('Predicted')
    fig.legend(labels = ('Actual','Predicted'),loc='upper left')
    plt.savefig(f"{name} Wykres wartości przewidywanych względem obserwowanych.png")


def clustering_data_using_kmeans(data_train, data_test, y_test, independend_variables, random_seed):
    predictors = stats.zscore(np.log(data_train[independend_variables] + 1))
    predictors_test = stats.zscore(np.log(data_test[independend_variables] + 1))

    #analiza wykresów rozrzutu
    dispersion_graph_generator(data_train, independend_variables, 'alcohol', filename_disp = "rozrzuty_grupowanie.png")
    dispersion_graph_generator(data_train, independend_variables, 'sulphates', filename_disp="rozrzuty_grupowanie_sulphates.png")
    dispersion_graph_generator(data_train, independend_variables, 'chlorides',filename_disp="rozrzuty_grupowanie_chlorides.png")
    dispersion_graph_generator(data_train, independend_variables, 'total sulfur dioxide', filename_disp="rozrzuty_grupowanie_total_sulfur.png")
    dispersion_graph_generator(data_train, independend_variables, 'citric acid',filename_disp="rozrzuty_grupowanie_citric acid.png")

    # Trójwymiarowy wykres rozrzutu
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(predictors['citric acid'])
    y = np.array(predictors['total sulfur dioxide'])
    z = np.array(predictors['chlorides'])
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()

    #algorytm do sprawdzenia liczby klastrów
    print("Grupowanie - algorytm do sprawdzenia wartości Silhoutta, liczby klastrów do wykorzystanai")
    silhouette_score(data_train)

    #dla 2 grup
    km = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=random_seed)
    kmeans = km.fit(predictors)
    cluster = kmeans.labels_

    Cluster0 = predictors.loc[cluster == 0]
    Cluster1 = predictors.loc[cluster == 1]

    opis0 = Cluster0.describe()
    opis1 = Cluster1.describe()

    lis_op = (opis0, opis1)
    fileSafer("data_train_grupowanie_2_grup.txt", lis_op)

    # dla 6 grup uczacych
    km6 = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=random_seed)
    kmeans = km6.fit(predictors)
    cluster = kmeans.labels_

    Cluster0 = predictors.loc[cluster == 0]
    Cluster1 = predictors.loc[cluster == 1]
    Cluster2 = predictors.loc[cluster == 2]
    Cluster3 = predictors.loc[cluster == 3]
    Cluster4 = predictors.loc[cluster == 4]
    Cluster5 = predictors.loc[cluster == 5]

    opis0 = Cluster0.describe()
    opis1 = Cluster1.describe()
    opis2 = Cluster2.describe()
    opis3 = Cluster3.describe()
    opis4 = Cluster4.describe()
    opis5 = Cluster5.describe()

    lis_op = (opis0, opis1, opis2, opis3, opis4, opis5)
    fileSafer("data_train_grupowanie_6_grup.txt", lis_op)

    km_test = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=random_seed)
    kmeans = km_test.fit(predictors_test)
    cluster_test = kmeans.labels_

    Cluster0 = predictors_test.loc[cluster_test == 0]
    Cluster1 = predictors_test.loc[cluster_test == 1]
    Cluster2 = predictors_test.loc[cluster_test == 2]
    Cluster3 = predictors_test.loc[cluster_test == 3]
    Cluster4 = predictors_test.loc[cluster_test == 4]
    Cluster5 = predictors_test.loc[cluster_test == 5]

    opis0 = Cluster0.describe()
    opis1 = Cluster1.describe()
    opis2 = Cluster2.describe()
    opis3 = Cluster3.describe()
    opis4 = Cluster4.describe()
    opis5 = Cluster5.describe()


    lis_op = (opis0, opis1, opis2, opis3, opis4, opis5)
    fileSafer("data_test_grupowanie_6_grup.txt", lis_op)
    graph_for_kmeans_result(cluster_test, y_test.array)



def silhouette_score(data_train):
    # potencjalna liczba grup
    n_clusters = [2, 3, 4 ,5, 6, 7, 8, 9]
    parameter_grid = ParameterGrid({'n_clusters': n_clusters})
    best_score = -1
    kmeans_model = KMeans()
    silhouette_scores = []
    # ewaluacja oparta o wynik silhouette
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # ustawienie obecnego parametru
        kmeans_model.fit(data_train)  # incjilalizacja działania algorytmu na danych testowych
        ss = metrics.silhouette_score(data_train, kmeans_model.labels_)  # policzenie wyniku silhouette
        silhouette_scores += [ss]  # zmienna do przechowania wszystkich rezultatów
        print('Parameter:', p, 'Score', ss)
        # sprawdzenie które p ma najlepszy wynik
        if ss > best_score:
            best_score = ss
            best_grid = p
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(n_clusters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()


def fileSafer(filename, file):
    with open(filename, mode='a', encoding='utf-8') as wf:
        wf.write(f"{file}\n")


def graph_for_kmeans_result(clusters, y):
    sns.set()  # make the plots look pretty
    df = pd.DataFrame({'Grupa do której należą wina': clusters+1, 'Jakość wina': y})
    df['dummy'] = 1
    ag = df.groupby(['Grupa do której należą wina', 'Jakość wina']).sum().unstack()
    ag.columns = ag.columns.droplevel()

    ag.plot(kind='bar', colormap=cm.Accent, width=1)
    plt.legend(loc='upper center')

    plt.savefig("Porównanie jakości z grupowaniem")



main()