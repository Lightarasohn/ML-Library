from model import Model
import numpy as np
from matplotlib import pyplot as plt
from helper_classes import Helper_Classes

class Linear_Regression_Classification(Model):
    _epochs_: int
    _batch_size_: int
    _iterations_: int
    
    def __init__(self, data, labels, initialize_parameters_method, num_of_hidden_layers, output_size, learning_rate, optimizer, print_range, test_data, test_labels, iterations=0, epochs=0, batch_size=0):
        super().__init__(data, labels, initialize_parameters_method, num_of_hidden_layers, output_size, Helper_Classes.Activation_Type.RELU, Helper_Classes.Task.MULTICLASS, learning_rate, optimizer, print_range, test_data, test_labels)
        self._epochs_ = epochs
        self._batch_size_ = batch_size
        self._iterations_ = iterations
    
    def gradient_descent(self):
        self._ml_.select_initialize_parameters_method()
        
        loss_history = []
        accuracy_history = []
        
        for i in range(self._iterations_):
            A, forward_layers = self._ml_.forward_propagation(self._data_)
            backward_layers = self._ml_.backward_propagation(A, self._labels_, forward_layers)
            self._ml_.update_parameters(backward_layers)
            
            loss = self._ml_.loss(A,self._labels_)
            loss_history.append(loss)
            accuracy = self._ml_.get_accuracy(self._data_, self._labels_)
            accuracy_history.append(accuracy)
            if(i % self._print_range_ == 0):
                print(f"Iteration: {self._iterations_} | Accuracy: {accuracy} | Loss: {loss}")
                
        loss = self._ml_.loss(A,self._labels_)
        loss_history.append(loss)
        
        accuracy = self._ml_.get_accuracy(self._data_, self._labels_)
        accuracy_history.append(accuracy)
        print(f"Iteration: {self._iterations_} | Accuracy: {accuracy} | Loss: {loss}")
        
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
        accuracy_history = []

        for epoch in range(self._epochs_):

            permutation = np.random.permutation(m)
            X_shuffled = self._data_[:, permutation]
            y_shuffled = self._labels_[permutation]

            num_batches = int(np.ceil(m / self._batch_size_))

            for batch_id in range(num_batches):
                start = batch_id * self._batch_size_
                end = min(start + self._batch_size_, m)

                X_batch = X_shuffled[:, start:end]
                y_batch = y_shuffled[start:end]

                A, forward_layers = self._ml_.forward_propagation(X_batch)
                backward_layers = self._ml_.backward_propagation(A, y_batch, forward_layers)
                self._ml_.update_parameters(backward_layers)

            # Epoch metrikleri (full train set)
            A_full, _ = self._ml_.forward_propagation(self._data_)
            loss = self._ml_.loss(A_full, self._labels_)
            acc = self._ml_.get_accuracy(self._data_, self._labels_)

            loss_history.append(loss)
            accuracy_history.append(acc)

            print(f"Epoch {epoch + 1}/{self._epochs_} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

        A_full, _ = self._ml_.forward_propagation(self._data_)
        loss = self._ml_.loss(A_full, self._labels_)
        acc = self._ml_.get_accuracy(self._data_, self._labels_)

        loss_history.append(loss)
        accuracy_history.append(acc)

        print(f"Epoch {self._epochs_}/{self._epochs_} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
        
        # Grafikler
        plt.figure()
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

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

        accuracy = self._ml_.get_accuracy(self._test_data_, self._test_labels)
        loss = self._ml_.loss(A_out, self._test_labels)

        print(f"Test Accuracy: {accuracy}")
        print(f"Test Loss: {loss}")

        predictions = self._ml_.get_predictions(self._test_data_)
        return predictions
