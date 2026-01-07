from model import Model
from helper_classes import Helper_Classes
from matplotlib import pyplot as plt
import numpy as np

class Logistic_Regression(Model):
    _iterations_: int
    _epochs_: int
    _batch_size_: int
    
    def __init__(self, data, labels, initialize_parameters_method, num_of_hidden_layers, output_size, learning_rate, optimizer, print_range, test_data, test_labels, iterations=0, epochs=0, batch_size=0):
        super().__init__(data, labels, initialize_parameters_method, num_of_hidden_layers, output_size, Helper_Classes.Activation_Type.SIGMOID, Helper_Classes.Task.BINARY, learning_rate, optimizer, iterations, print_range, test_data, test_labels)
        self._iterations_ = iterations
        self._epochs_ = epochs
        self._batch_size_ = batch_size
        
    def gradient_descent(self):
        if(self._iterations_ == 0):
            print("ERROR: In order to use gradient descent method, define iterations first.")
            return
        
        self._ml_.select_initialize_parameters_method()
        loss_history = []
        accuracy_history = []

        for i in range(self._iterations_):
            A, forward_layers = self._ml_.forward_propagation(self._data_)
            backward_layers = self._ml_.backward_propagation(A, self._labels_, forward_layers)
            self._ml_.update_parameters(backward_layers)
            loss = self._ml_.loss(A, self._labels_)
            loss_history.append(loss)
            accuracy = self._ml_.get_accuracy(self._data_, self._labels_)
            accuracy_history.append(accuracy)
            if(i % self._print_range_ == 0 or i == self._print_range_):
                print(f"Iteration: {i}/{self._iterations_} | Accuracy: {accuracy} | Loss: {loss}")
        A, forward_layers = self._ml_.forward_propagation(self._data_)
        loss = self._ml_.loss(A, self._labels_)
        loss_history.append(loss)
        accuracy = self._ml_.get_accuracy(self._data_, self._labels_)
        accuracy_history.append(accuracy)
        print(f"Iteration: {self._iterations_}/{self._iterations_} | Accuracy: {accuracy} | Loss: {loss}")    
                
        plt.figure()
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        
        plt.figure()
        plt.plot(accuracy_history)
        plt.title("Training Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.show()
    
    def normalize_data(self, division_normal:np.float64=255.0):
        if(division_normal == 0):
            print("Error: Division normal could not be zero. Zero division error could occur")
        
        X_train = self._data_.astype(np.float64)
        X_test = self._test_data_.astype(np.float64)
        
        self._test_data_ = X_test / division_normal
        self._data_ = X_train / division_normal
        
    def mini_batch_gradient_descent(self):
        if(self._epochs_ == 0 or self._batch_size_ == 0):
            print("ERROR: In order to use mini batch gradient descent method, define epochs and batch size first.")
        
        m = self._data_.shape[1]
        self._ml_.select_initialize_parameters_method()

        loss_history = []
        accuracy_history = []
        
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
            accuracy = self._ml_.get_accuracy()
            accuracy_history.append(accuracy)
            if(self._print_range_ % epoch == 0 | self._print_range_ == epoch):
                print(f"Epoch {epoch + 1}/{self._epochs_} | Accuracy: {accuracy} | Loss: {loss:.4f}")
            
        plt.figure()
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.figure()
        plt.plot(accuracy_history)
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
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
        accuracy = self._ml_.get_accuracy(self._test_data_, self._test_labels)
        print(f"Test Accuracy: {accuracy} | Test Loss (MSE): {loss}")
        
        return A_out
