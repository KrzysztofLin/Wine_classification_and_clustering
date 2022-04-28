import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
from typing import List


def plot_correlation_graph(data: pd.DataFrame, filename_cor: str) -> plt.figure:
    names = data.columns
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 10))
    sns.heatmap(np.corrcoef(data.values.T), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13},
                yticklabels=names, xticklabels=names)
    plt.savefig(filename_cor)


def plot_dispersion_graphs(data: pd.DataFrame, x_column_names: List[str], y_column_names: str, filename_disp: str) -> plt.figure:
    sns.pairplot(data, x_vars=x_column_names, y_vars=y_column_names, diag_kind=None)
    plt.tight_layout()
    plt.savefig(filename_disp)


def plot_histograms(data: pd.DataFrame):
    [data.hist(column) for column in data.columns]


def plot_actual_vs_predicted_values_graph(name: str, y_actual: pd.Series, y_predicted_denor: pd.Series) -> plt.figure:
    fig = plt.figure()
    a1 = fig.add_axes([0, 0, 1, 1])
    x = range(len(y_predicted_denor))
    a1.plot(x, y_actual, 'ro')
    a1.set_ylabel('Actual')
    a2 = a1.twinx()
    a2.plot(x, y_predicted_denor, 'o')
    a2.set_ylabel('Predicted')
    fig.legend(labels=('Actual', 'Predicted'), loc='upper left')
    plt.savefig(f"{name} Graph of actual vs predicted values.png")


def plot_3D_dispersion_graph(predictors, columns_names) -> plt.figure:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(predictors[columns_names[0]])
    y = np.array(predictors[columns_names[1]])
    z = np.array(predictors[columns_names[2]])
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()


def plot_kmeans_result_graph(clusters, y):
    sns.set()
    df = pd.DataFrame({'Cluster to which belong wine': clusters + 1, 'Wine quality': y})
    df['dummy'] = 1
    ag = df.groupby(['Cluster to which belong wine', 'Wine quality']).sum().unstack()
    ag.columns = ag.columns.droplevel()
    ag.plot(kind='bar', colormap=cm.Accent, width=1)
    plt.legend(loc='upper center')
    plt.savefig("Comparation of clustering quality.png")

