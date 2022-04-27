from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy

X_COLUMNS_NAMES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                   'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']
Y_COLUMN_NAME = 'quality'
RANDOM_SEED = 123
TEST_SIZE = 0.25
PERCEPTOR_NUMBER_CLASSIFICATION = 200
PERCEPTOR_NUMBER_ESTIMATION = 200

ALGORGITHMS_AND_PARAMETERS = {KNeighborsClassifier(): {'n_neighbors': np.arange(2, 26), 'algorithm': ['auto'], 'weights': ('uniform', 'distance')},
MLPClassifier(): {'hidden_layer_sizes': [(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,),(12,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], 'alpha': [0.0001, 0.005], 'max_iter': [1000], 'random_state': [RANDOM_SEED]}}

#MLPRegressor() : {'hidden_layer_sizes': (''), activation: ('relu', ''), solver: ('adam', 'lfgbs'), alpha=0.0001, max_iter=25000, random_state = RANDOM_SEED}}
