'''
import pandas as pd
from utils import independent_and_dependent_variables_generator, data_standarization, data_outliers_remover, nan_data_counter
from data_visualization_v2 import correlation_graph_generator, dispersion_graph_generator, histogram_generator
from data_classification import ClassificationAndEstimation
from data_clusterization import clustering_data_using_kmeans, silhouette_score
from settings import RANDOM_SEED, TEST_SIZE, PERCEPTOR_NUMBER_CLASSIFICATION, PERCEPTOR_NUMBER_ESTIMATION
'''


from data_loader import load_data
from explorative_data_analysis import explore_data
from data_preprocessing import preprocess_data
from data_classification import FindBestHyperparameters, Classification, Estimation
import abc
from settings import ALGORGITHMS_AND_PARAMETERS

# #    histogram_generator(data)
    # load data
    # 1. Load data from file
    # 2. Select data
    # 3. Split data
    # 4. Preprocess data
    # train test split and data standardization

'''
    data_loader()
    explorative_data_analysis()
    data_preprocessing()
    data_classification()
    data_estimation()
    data_grouping()
    '''

class PipelineAbstract(abc.ABC):
    def load_data(self, dataLoader):
        pass
    def preprocess_data(self):
        pass
    def train_model(self, model, Trainer):
        pass
    def evaluate_data(self, model, data):
        pass


def main(*args, **kwargs):
    # ''' DATA ANALYSIS AND PREPARATION  '''

    data = load_data()
    data = explore_data(data)
    y_test, y_train, x_train_norm, y_train_norm, x_test_norm = preprocess_data(data)
    for algorithm, parameters in ALGORGITHMS_AND_PARAMETERS.items():
        algorithm_with_best_parameters = FindBestHyperparameters(y_train, x_train_norm, y_train_norm).crossvalidation(algorithm, parameters)
        print(algorithm_with_best_parameters)
        Classification(y_test, y_train, x_train_norm, x_test_norm, y_train_norm).evaluate(algorithm_with_best_parameters)




    '''
    ''' # DATA CLASSIFICATION, ESTIMATION AND CLUSTERIZATION
    '''
    classification_and_estimation = ClassificationAndEstimation(y_test, y_train, x_train_norm, x_test_norm,
                                                                y_train_norm)

    # DATA CLASSIFICATION
    classification_and_estimation.knn_classification_crossvalidation()
    classification_and_estimation.knn_classification()

    classification_and_estimation.mlp_classification(RANDOM_SEED, perceptor_num=PERCEPTOR_NUMBER_CLASSIFICATION

    # DATA ESTIMATION
    classification_and_estimation.mlp_estimation(RANDOM_SEED, perceptor_num=PERCEPTOR_NUMBER_ESTIMATION)

    # DATA CLUSTERIZATION

    print("Silhouette score used to check optimal number of cluster")
    silhouette_score(data_train)
    clustering_data_using_kmeans(data_train, y_train, independent_variables, RANDOM_SEED)

    '''
if __name__ == "__main__":
    main()
