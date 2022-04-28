from data_loader import load_data
from explorative_data_analysis import explore_data
from data_preprocessing import preprocess_data
from data_classification import classify_data, estimate_data
from data_clusterization import explore_data_clusterization, cluster_data
import abc


def main(*args, **kwargs):
    data = load_data()
    data = explore_data(data)
    data_subsets = preprocess_data(data)
    #classify_data(data_subsets)
    #estimate_data(data_subsets)
    #explore_data_clusterization(data_subsets)
    cluster_data(data_subsets)


if __name__ == "__main__":
    main()


class PipelineAbstract(abc.ABC):
    def load_data(self, dataLoader):
        pass
    def preprocess_data(self):
        pass
    def train_model(self, model, Trainer):
        pass
    def evaluate_data(self, model, data):
        pass