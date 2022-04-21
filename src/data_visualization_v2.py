import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm


def correlation_graph_generator(data: pd.DataFrame, filename_cor: str) -> plt.figure:
    names = data.columns
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 10))
    sns.heatmap(np.corrcoef(data.values.T), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 13},
                yticklabels=names, xticklabels=names)
    plt.savefig(filename_cor)


def dispersion_graph_generator(data: pd.DataFrame, independent_variables: str, dependent_variable: str, filename_disp: str) -> plt.figure:
    sns.pairplot(data, x_vars=independent_variables, y_vars=dependent_variable, diag_kind=None)
    plt.tight_layout()
    plt.savefig(filename_disp)


def histogram_generator(data: pd.DataFrame):
    [data.hist(column) for column in data.columns]


def final_plot(name: str, y_actual: pd.Series, y_predicted_denor: pd.Series) -> plt.figure:
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


def dispersion_graph_3D(predictors, column_name_1: str, column_name_2, column_name_3) -> plt.figure:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(predictors[column_name_1])
    y = np.array(predictors[column_name_2])
    z = np.array(predictors[column_name_3])
    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()


def graph_for_kmeans_result(clusters, y):
    sns.set()
    df = pd.DataFrame({'Cluster to which belong wine': clusters + 1, 'Wine quality': y})
    df['dummy'] = 1
    ag = df.groupby(['Cluster to which belong wine', 'Wine quality']).sum().unstack()
    ag.columns = ag.columns.droplevel()
    ag.plot(kind='bar', colormap=cm.Accent, width=1)
    plt.legend(loc='upper center')
    plt.savefig("Comparation of clustering quality")

