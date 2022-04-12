import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from scipy import stats
from utils import file_safer
from data_visualization_v2 import graph_for_kmeans_result, dispersion_graph_generator

def clustering_data_using_kmeans(data_train, data_test, y_test, independend_variables, random_seed):
    predictors = stats.zscore(np.log(data_train[independend_variables] + 1))
    predictors_test = stats.zscore(np.log(data_test[independend_variables] + 1))

    '''# zrefactorować!'''''
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
    file_safer("data_train_grupowanie_2_grup.txt", lis_op)

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
    file_safer("data_train_grupowanie_6_grup.txt", lis_op)

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
    file_safer("data_test_grupowanie_6_grup.txt", lis_op)
    graph_for_kmeans_result(cluster_test, y_test.array)



def silhouette_score(data_train):
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 9] #potencial group number
    parameter_grid = ParameterGrid({'n_clusters': n_clusters})
    kmeans_model = KMeans()
    silhouette_scores = []
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # ustawienie obecnego parametru
        kmeans_model.fit(data_train)  # incjilalizacja działania algorytmu na danych testowych
        ss = metrics.silhouette_score(data_train, kmeans_model.labels_)  # policzenie wyniku silhouette
        silhouette_scores += [ss]  # zmienna do przechowania wszystkich rezultatów
        print('Parameter:', p, 'Score', ss)
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(n_clusters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()