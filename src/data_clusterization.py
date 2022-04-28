import abc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid

from data_visualization_v2 import (plot_3D_dispersion_graph,
                                   plot_dispersion_graphs,
                                   plot_kmeans_result_graph)
from settings import (CLUSERIZATOR_WITH_PARAMETERS,
                      COLUMN_NAMES_CLUSTERIZATION_3D_DISPERSION_GRAPH,
                      COLUMN_NAMES_CLUSTERIZATION_DISPERSION_GRAPH,
                      X_COLUMNS_NAMES)
from utils import save_file


def explore_data_clusterization(data_subsets):
    data_exploration = DataExplorationForCluseriazation(data_subsets)
    data_exploration.check_dispersion_graphs()
    # Score to check optimal number of cluster for kmeans (only)
    data_exploration.check_silhouette_score()


def cluster_data(data_subset):
    data_clusterization = Clusterization(data_subset)
    cluster = data_clusterization.cluster(CLUSERIZATOR_WITH_PARAMETERS)
    data_clusterization.save_results_for_cluster_analysis(cluster)


class DataExplorationForClusterizationAbstract(abc.ABC):
    @abc.abstractmethod
    def check_dispersion_graphs(self):
        pass

    def check_silhouette_score(self):
        pass


class ClusterizationAbstract(abc.ABC):
    @abc.abstractmethod
    def cluster(self, algorithm_with_best_parameters):
        pass

    def save_result_for_cluster_analysis(self, cluster):
        pass


class DataExplorationForCluseriazation(DataExplorationForClusterizationAbstract):
    def __init__(self, data_subsets: pd.DataFrame):
        self.x_train = data_subsets["x_train"]
        self.predictors = stats.zscore(np.log(self.x_train + 1))

    def check_dispersion_graphs(self):
        for column_name in COLUMN_NAMES_CLUSTERIZATION_DISPERSION_GRAPH:
            plot_dispersion_graphs(
                self.x_train,
                X_COLUMNS_NAMES,
                column_name,
                filename_disp=f"dispersion_graph_clustering+{column_name}",
            )
        plot_3D_dispersion_graph(
            self.predictors, COLUMN_NAMES_CLUSTERIZATION_3D_DISPERSION_GRAPH
        )

    def check_silhouette_score(self) -> plt.figure:
        n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
        parameter_grid = ParameterGrid({"n_clusters": n_clusters})
        kmeans_model = KMeans()
        silhouette_scores = []
        for p in parameter_grid:
            kmeans_model.set_params(**p)
            kmeans_model.fit(self.x_train)
            ss = metrics.silhouette_score(self.x_train, kmeans_model.labels_)
            silhouette_scores += [ss]
            print("Parameter:", p, "Score", ss)
        plt.bar(
            range(len(silhouette_scores)),
            list(silhouette_scores),
            align="center",
            width=0.5,
        )
        plt.xticks(range(len(silhouette_scores)), list(n_clusters))
        plt.title("Silhouette Score", fontweight="bold")
        plt.xlabel("Number of Clusters")
        plt.show()


class Clusterization(ClusterizationAbstract):
    def __init__(self, data_subsets: pd.DataFrame):
        self.x_train = data_subsets["x_train"]
        self.predictors = stats.zscore(np.log(self.x_train + 1))
        self.y_train = data_subsets["y_train"]

    def cluster(self, algorithm_with_parameters) -> None:
        cluster = algorithm_with_parameters.fit(self.predictors).labels_
        plot_kmeans_result_graph(cluster, self.y_train.array)
        return cluster

    def save_results_for_cluster_analysis(self, cluster):
        cluster_description = []
        for cluster_number in range(max(cluster) + 1):
            cluster_description.append(
                self.predictors.loc[cluster == cluster_number].describe()
            )
        save_file("data_train_clustering_6_groups.txt", cluster_description)
