import pandas as pd
from sklearn.model_selection import train_test_split
from utils import independent_and_dependent_variables_generator, data_standarization, data_outliers_remover, nan_data_counter
from data_visualization_v2 import correlation_graph_generator, dispersion_graph_generator, histogram_generator
from data_classification import ClassificationAndEstimation
from data_clusterization import clustering_data_using_kmeans, silhouette_score
PATH = "C:\\Users\\Krzychiu\\Documents\\Analiza_danych_studia\\1 semestr\\ED\\KL_projekt_zaliczeniowy\\refactored code"


def main():
    ''' DATA ANALYSIS AND PREPARATION  '''
    data = pd.read_csv(f"{PATH}\\winequality-white.csv", sep=';')
    random_seed = 123
    pd.set_option('display.max_columns', 12)
    # nan_data_counter(data) # check number of NaN

    # Choice of independent and dependent variables, in case of wine-data set, the dependent value is quality (11 column)
    dependent_variable_col_num = 11
    independent_variables_col_num, dependent_variable, independent_variables = independent_and_dependent_variables_generator(
        data, dependent_variable_col_num)

    # Generation of the plots
#    correlation_graph_generator(data, filename_cor="Correlation_graph.png")
#    dispersion_graph_generator(data, independent_variables, dependent_variable, filename_disp="Dispersion_graph.png")
#    histogram_generator(data)

    # Second choice of independent and dependent variables, generation of correlation matrix
    data = data[data.columns[[0, 1, 2, 3, 4, 6, 8, 9, 10, 11]]]
    dependent_variable_col_num = 9
    independent_variables_col_num, dependent_variable, independent_variables = independent_and_dependent_variables_generator(
        data, dependent_variable_col_num)
    data = data_outliers_remover(data)  # function to remove outliers

#    correlation_graph_generator(data, filename_cor="Modified_correlation_graph.png")
#    histogram_generator(data)

    # train test split and data standardization
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=random_seed)
    y_test = data_test[dependent_variable]
    y_train = data_train[dependent_variable]
    x_train_norm, x_test_norm, y_train_norm = data_standarization(data_train, data_test, independent_variables_col_num,
                                                                  dependent_variable_col_num)

    ''' DATA CLASSIFICATION, ESTIMATION AND CLUSTERIZATION '''
    classification_and_estimation = ClassificationAndEstimation(y_test, y_train, x_train_norm, x_test_norm,
                                                                y_train_norm)

    # DATA CLASSIFICATION
    classification_and_estimation.knn_classification_crossvalidation()
    classification_and_estimation.knn_classification()

    perceptor_num = 200
    classification_and_estimation.mlp_classification(random_seed, perceptor_num=perceptor_num)

    # DATA ESTIMATION
    classification_and_estimation.mlp_estimation(random_seed, perceptor_num=perceptor_num)

    # DATA CLUSTERIZATION

    print("Silhouette score used to check optimal number of cluster")
    silhouette_score(data_train)
    clustering_data_using_kmeans(data_train, y_train, independent_variables, random_seed)


if __name__ == "__main__":
    main()
