import abc

from data_classification_and_estimation import classify_data, estimate_data
from data_clusterization import cluster_data, explore_data_clusterization
from data_loader import check_data_correctness, load_data
from data_preprocessing import preprocess_data
from explorative_data_analysis import explore_data


def main():
    data = load_data()
    # check_data_correctness(data)
    data = explore_data(data, show_plots = False)
    data_subsets = preprocess_data(data)
    # classify_data(data_subsets)
    estimate_data(data_subsets)
    # explore_data_clusterization(data_subsets)
    # cluster_data(data_subsets)


if __name__ == "__main__":
    main()