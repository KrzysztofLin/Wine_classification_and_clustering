import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from scipy import stats
from utils import file_safer
from data_visualization_v2 import graph_for_kmeans_result, dispersion_graph_generator, dispersion_graph_3D


def clustering_data_using_kmeans(data_train: pd.DataFrame, y_train: pd.Series, independent_variables: pd.Index, random_seed: int) -> None:
    predictors = stats.zscore(np.log(data_train[independent_variables] + 1))
    # dispersion graphs used in analysis
    variables_to_dispersion_graph = ['alcohol', 'sulphates', 'chlorides', 'total sulfur dioxide', 'citric acid']
    for variable in variables_to_dispersion_graph:
        dispersion_graph_generator(data_train, independent_variables, variable, filename_disp=f"dispersion_graph_clusterin+{variable}")

    dispersion_graph_3D(predictors, column_name_1 = 'citric acid', column_name_2 = 'total sulfur dioxide', column_name_3 = 'chlorides')

    # for 2 clusters
    kmeans_cluster_description(predictors, n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=random_seed)

    # for 6 clusters
    cluster_6 = kmeans_cluster_description(predictors, n_clusters=6, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=random_seed)
    graph_for_kmeans_result(cluster_6, y_train.array)


def silhouette_score(data_train: pd.DataFrame) -> plt.figure:
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
    parameter_grid = ParameterGrid({'n_clusters': n_clusters})
    kmeans_model = KMeans()
    silhouette_scores = []
    for p in parameter_grid:
        kmeans_model.set_params(**p)
        kmeans_model.fit(data_train)
        ss = metrics.silhouette_score(data_train, kmeans_model.labels_)
        silhouette_scores += [ss]
        print('Parameter:', p, 'Score', ss)
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(n_clusters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()


def kmeans_cluster_description(predictors: pd.DataFrame, n_clusters: int, init, n_init: str, max_iter: int, tol: float, random_state:int):
    kmeans_algorithm = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
    cluster = kmeans_algorithm.fit(predictors).labels_
    cluster_description = []
    for cluster_number in range(n_clusters):
        cluster_description.append(predictors.loc[cluster == cluster_number].describe())
    file_safer("data_train_clustering_2_groups.txt", cluster_description)
    return cluster