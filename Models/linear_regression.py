from model import Model
import numpy as np
from matplotlib import pyplot as plt
from helper_classes import Helper_Classes

class Linear_Regression(Model):
    _epochs_: int
    _batch_size_: int
    
    def __init__(self, data, labels, initialize_parameters_method, num_of_hidden_layers, output_size, learning_rate, optimizer, iterations, print_range, test_data, test_labels, epochs, batch_size):
        super().__init__(data, labels, initialize_parameters_method, num_of_hidden_layers, output_size, Helper_Classes.Activation_Type.RELU, Helper_Classes.Task.REGRESSION, learning_rate, optimizer, iterations, print_range, test_data, test_labels)
        self._epochs_ = epochs
        self._batch_size_ = batch_size
        
    def normalize_data(self):
        X = self._data_.astype(np.float64)

        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        
        X_scaled = (X - X_min) / np.where(X_max - X_min == 0, 1, X_max - X_min)
        
        self._data_ = X_scaled
        if self._test_data_ is not None:
            X_test = self._test_data_.astype(np.float64)
            X_test_scaled = (X_test - X_min) / np.where(X_max - X_min == 0, 1, X_max - X_min)
            self._test_data_ = X_test_scaled

        
    def gradient_descent(self):
        self._ml_.select_initialize_parameters_method()
        loss_history = []

        for i in range(self._iterations_):
            A, forward_layers = self._ml_.forward_propagation(self._data_)
            backward_layers = self._ml_.backward_propagation(A, self._labels_, forward_layers)
            self._ml_.update_parameters(backward_layers)
            loss = self._ml_.loss(A, self._labels_)
            loss_history.append(loss)
            if(i % self._print_range_ == 0):
                print(f"Iteration: {i}/{self._iterations_} | Loss: {loss}")
                
        plt.figure()
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()    
        
    def mini_batch_gradient_descent(self):
        m = self._data_.shape[1]
        self._ml_.select_initialize_parameters_method()

        loss_history = []
        
        for epoch in range(self._epochs_):

            permutation = np.random.permutation(m)
            X_shuffled = self._data_[:, permutation]
            y_shuffled = self._labels_[:, permutation]  # <- burada axis düzeltilmeli

            num_batches = int(np.ceil(m / self._batch_size_))

            for batch_id in range(num_batches):
                start = batch_id * self._batch_size_
                end = min(start + self._batch_size_, m)

                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[:, start:end]  # <- burası da sütun bazlı

                A, forward_layers = self._ml_.forward_propagation(X_batch)
                backward_layers = self._ml_.backward_propagation(A, y_batch, forward_layers)
                self._ml_.update_parameters(backward_layers)

            A_full, _ = self._ml_.forward_propagation(self._data_)
            loss = self._ml_.loss(A_full, self._labels_)

            loss_history.append(loss)

            print(f"Epoch {epoch + 1}/{self._epochs_} | Loss: {loss:.4f}")
            
        plt.figure()
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    
    def train(self):
        if(self._optimizer_.lower() == "gradient_descent"):
            self.gradient_descent()
        elif(self._optimizer_.lower() == "mini_batch_gradient_descent"):
            self.mini_batch_gradient_descent()
    
    def test(self):
        if self._test_data_ is None or self._test_labels is None:
            raise ValueError("Test data or test labels not provided")

        A_out, _ = self._ml_.forward_propagation(self._test_data_)

        loss = self._ml_.loss(A_out, self._test_labels)
        print(f"Test Loss (MSE): {loss}")
        
        return A_out