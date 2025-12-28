from model import Model
import numpy as np
from matplotlib import pyplot as plt

class Lineer_Regression_Classification(Model):
    _epochs_: int
    _batch_size_: int
    
    def __init__(self, epochs, batch_size, data, labels, num_of_hidden_layers, output_size, activation_type, task, learning_rate, optimizer, iterations, print_range, test_data, test_labels):
        super().__init__(data, labels, num_of_hidden_layers, output_size, activation_type, task, learning_rate, optimizer, iterations, print_range, test_data, test_labels)
        self._epochs_= epochs
        self._batch_size_ = batch_size
    
    def gradient_descent(self):
        self._ml_.initialize_parameters_with_classic_method()
        
        for i in range(self._iterations_):
            A, forward_layers = self._ml_.forward_propagation(self._data_)
            backward_layers = self._ml_.backward_propagation(A, self._labels_, forward_layers)
            self._ml_.update_parameters(backward_layers)
            if(i % self._print_range_ == 0):
                print(f"Iteration: {i} | Accuracy: {self._ml_.get_accuracy(self._data_, self._labels_)}")
    
    def mini_batch_gradient_descent(self):
        m = self._data_.shape[1]
        self._ml_.initialize_parameters_with_classic_method()

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

        task = self._ml_._task_.lower()

        if task == "regression":
            loss = self._ml_.loss(A_out, self._test_labels)
            print(f"Test Loss (MSE): {loss}")
            return A_out

        else:
            accuracy = self._ml_.get_accuracy(self._test_data_, self._test_labels)
            loss = self._ml_.loss(A_out, self._test_labels)

            print(f"Test Accuracy: {accuracy}")
            print(f"Test Loss: {loss}")

            predictions = self._ml_.get_predictions(self._test_data_)
            return predictions
