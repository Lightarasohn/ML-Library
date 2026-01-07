from machine_learning import Machine_Learning as ml
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    _ml_: ml
    _optimizer_: str
    _data_: np.ndarray
    _labels_: np.ndarray
    _print_range_: int
    _test_data_: np.ndarray
    _test_labels: np.ndarray
    
    def __init__(self, data, labels, initialize_parameters_method, num_of_hidden_layers, output_size, activation_type, task, learning_rate, optimizer, iterations, print_range, test_data, test_labels):
        self._print_range_ = print_range
        self._data_ = data
        self._labels_ = labels
        self._optimizer_ = optimizer
        self._test_data_ = test_data
        self._test_labels = test_labels
        self._ml_ = ml(
            num_of_hidden_layers,
            data.shape[0],
            output_size,
            activation_type,
            task,
            learning_rate,
            initialize_parameters_method
        )
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass