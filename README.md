# Wine_classification_and_clustering
Classification, estimation and clustering the quality of White Wine using Machine Learning Algorithms. 

#The goal
The aim of the programme was to test (and select) various classification, estimation and clustering methods in order to predict, on the basis of the predictors, the quality of wine with the highest possible accuracy. 

#About dataset
The file winequality-white.csv contains data on 4898 samples of Portuguese white wine. Each sample is described by 12 parameters, of which parameters 1-11 are assessed by objective physico-chemical tests and parameter 12 is a subjective assessment of wine quality made by experts.
The collection was compiled as the result of measurements carried out from May 2004 to February 2007 on samples approved by the official certification association - CVRVV (association aiming to improve quality and promote Portuguese vino verde).

**How to use program?**
To run the program use: main_wine_file.py. The main program calls the functions defined in the other files, contains all methods needed to experimate on date and print some statistics with graphs (histograms, dispersion graph, correleation matrix). 
The main file can be modified depending on your needs. The program is suited for white_wine_dataset.csv, however it can be modified and used in diffrent datasets (only the independent variables and dependent variable should be pointed).

**The program structure**
The application is created of 5 files:
- main_wine_file.py: main script, used to conduct data analysis, clustering, classification and estimation,
- utils.py: file contating methods used to "work" with data, "to clean" them. Mainly used to prepare data for further analysis,
- data_classification.py: file containg methods used to classificate and estimate data. In the script were applied estimators: K-Nearest Neighbour, K-Nearest Neighbour with Crossvalidation, Multi Layer Perceptor used for classification and estimation,
- data_clusterization.py: file containg methods used to cluster data using k-means algorithm and method to create silhouette score,
- data_visualization_v2.py: file contating all methods needed to create useful graphs visualizing analized data. Some graphs are tailroed espesially for some methodes. With this script the dispersion, 3D-dsicpersion, correlation, accuracy, dependency between clusters and dependent value, etc can be visualised.

**Used technologies**
- Python
- Pandas
- Sklearn
- Matplotlib
- Seaborn
- Typing
