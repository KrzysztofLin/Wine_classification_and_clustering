import abc

class EvaluateModelAbstract(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, algorithm_with_best_parameters):
        pass