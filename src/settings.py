from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np

X_COLUMNS_NAMES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                   'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']

Y_COLUMN_NAME = 'quality'
RANDOM_SEED = 123
TEST_SIZE = 0.25
CROSSVALIDATION_FOLDS = 4

CLASSIFICATION_ALGORITHMS_AND_PARAMETERS = {
    KNeighborsClassifier(): {'n_neighbors': np.arange(2, 26), 'algorithm': ['auto'],
                             'weights': ('uniform', 'distance')},
    MLPClassifier(): {'hidden_layer_sizes': [(80, ), (100, ), (120, ), (140, ), (160,), (200,)], 'activation': ['relu'], 'solver': ['adam'],
                      'alpha': [0.005, 0.0001], 'max_iter': [1000], 'random_state': [RANDOM_SEED]}}

ESTIMATION_ALGORITHMS_AND_PARAMETERS = {
    MLPRegressor(): {'hidden_layer_sizes': [(80, ), (100, ), (120, ), (140, ), (160,), (200,)],
                     'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], 'alpha': [0.0001, 0.005],
                     'max_iter': [1000], 'random_state': [RANDOM_SEED]}}


CLUSERIZATOR_WITH_PARAMETERS = KMeans(n_clusters=6, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=RANDOM_SEED)

COLUMN_NAMES_FOR_CLUSTERIZATION_DISPERSION_GRAPH = ['alcohol', 'sulphates', 'chlorides', 'total sulfur dioxide', 'citric acid']
COLUMN_NAMES_FOR_CLUSTERIZATION_3D_DISPERSION_GRAPH = ['citric acid', 'total sulfur dioxide', 'chlorides']