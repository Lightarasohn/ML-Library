import numpy as np
from tools import Tools as tl

class Machine_Learning:
    _num_of_hidden_layers_: int
    _input_size_: int
    _output_size_: int
    _weights_: dict
    _biases_: dict
    _activation_type_: str
    _task_: str
    _learning_rate_: np.float64
    _initialize_parameters_method_: str
    
    def __init__(self, num_of_hidden_layers, input_size, output_size, activation_type, task, learning_rate, initialize_parameters_method):
        self._num_of_hidden_layers_ = num_of_hidden_layers
        self._input_size_ = input_size
        self._output_size_ = output_size
        self._weights_ = {}
        self._biases_ = {}
        self._activation_type_ = activation_type
        self._task_ = task
        self._learning_rate_ = learning_rate
        self._initialize_parameters_method_ = initialize_parameters_method
    
    def initialize_parameters_with_classic_method(self):
        Nhidden = (self._input_size_ + self._output_size_) * 2 / 3
        Nhidden = int(Nhidden)
        self.initialize_parameters(Nhidden)
        
    def initialize_parameters_with_li_chow_and_yu_method(self):
        Nhidden = (np.sqrt(1 + 8 * self._input_size_) - 1) / 2
        Nhidden = int(Nhidden)
        self.initialize_parameters(Nhidden)
        
    def initialize_parameters_with_shibata_and_ikeda_method(self):
        Nhidden = np.sqrt(self._input_size_ * self._output_size_)
        Nhidden = int(Nhidden)
        self.initialize_parameters(Nhidden)
        
    def initialize_parameters_with_tamura_and_tateishi_method(self):
        Nhidden = self._input_size_ - 1
        Nhidden = max(1, int(Nhidden))
        self.initialize_parameters(Nhidden)
        
    def initialize_parameters_with_sheela_and_deepa_method(self):
        if self._input_size_**2 <= 8:
            raise ValueError("Sheela-Deepa method invalid for small input size")
        Nhidden = (4 * self._input_size_ * self._input_size_ + 3) / (self._input_size_ * self._input_size_ - 8)
        Nhidden = int(Nhidden)
        self.initialize_parameters(Nhidden)
    
    def select_initialize_parameters_method(self):
        if(self._initialize_parameters_method_.lower() == "classic_method"):
            self.initialize_parameters_with_classic_method()
        elif(self._initialize_parameters_method_.lower() == "li_chow_and_yu_method"):
            self.initialize_parameters_with_li_chow_and_yu_method()
        elif(self._initialize_parameters_method_.lower() == "shibata_and_ikeda_method"):
            self.initialize_parameters_with_shibata_and_ikeda_method()
        elif(self._initialize_parameters_method_.lower() == "tamura_and_tateishi_method"):
            self.initialize_parameters_with_tamura_and_tateishi_method()
        elif(self._initialize_parameters_method_.lower() == "sheela_and_deepa_method"):
            self.initialize_parameters_with_sheela_and_deepa_method()
        else:
            print(f"Warning: No initialize parameter named {self._initialize_parameters_method_}\nUsing classic_method instead")
            self.initialize_parameters_with_classic_method()
    
    def _scale(self, number_of_neurons, is_output = False):
        if(is_output == True):
            return np.sqrt(1 / number_of_neurons)
        if self._activation_type_.lower() == "relu":
            return np.sqrt(2 / number_of_neurons)
        return np.sqrt(1 / number_of_neurons)
        
    def initialize_parameters(self, hidden_neurons):
        Nhidden = hidden_neurons
        max_layer = self._num_of_hidden_layers_ + 1
        for layer in range(1, max_layer + 1):
            if(layer == 1):
                number_of_neurons = self._input_size_
                weight = np.random.randn(Nhidden, number_of_neurons) * self._scale(number_of_neurons)
                bias = np.zeros((Nhidden, 1))
            elif(layer == max_layer):
                number_of_neurons = Nhidden
                weight = np.random.randn(self._output_size_, number_of_neurons) * self._scale(number_of_neurons, True)
                bias = np.zeros((self._output_size_, 1))
            else:
                number_of_neurons = Nhidden
                weight = np.random.randn(Nhidden, number_of_neurons) * self._scale(number_of_neurons)
                bias = np.zeros((Nhidden, 1))
            self._weights_[f"w{layer}"] = weight
            self._biases_[f"b{layer}"] = bias
    
    def activation_function(self, z):
        if(self._activation_type_.lower() == "relu"):
            return tl.ReLU(z)
        elif(self._activation_type_.lower() == "sigmoid"):
            return tl.Sigmoid(z)
        else:
            return tl.Tanh(z)

    def output_activation_function(self, z):
        if(self._task_.lower() == "binary" or self._task_.lower() == "multilabel"):
            return tl.Sigmoid(z)
        elif(self._task_.lower() == "multiclass"):
            return tl.Softmax(z)
        elif(self._task_.lower() == "regression"):
            return z
        else:
            raise Exception("output_activation_function error: invalid task")
        
    def forward_propagation(self, train_data):
        forward_layers = {}
        
        A = train_data
        
        forward_layers["a0"] = A 
        
        for hidden_layer in range(1, self._num_of_hidden_layers_ + 1):
            W = self._weights_[f"w{hidden_layer}"]
            b = self._biases_[f"b{hidden_layer}"]
            
            Z = tl.Matrix_Multiplication(W, A) + b
            A = self.activation_function(Z)
            
            forward_layers[f"z{hidden_layer}"] = Z
            forward_layers[f"a{hidden_layer}"] = A
            
        output_layer = self._num_of_hidden_layers_ + 1
        W = self._weights_[f"w{output_layer}"]
        b = self._biases_[f"b{output_layer}"]

        Z = tl.Matrix_Multiplication(W, A) + b
        A = self.output_activation_function(Z)

        forward_layers[f"z{output_layer}"] = Z
        forward_layers[f"a{output_layer}"] = A
        
        return A, forward_layers
    
    def loss(self, outputs, labels):
        if(self._task_.lower() == "binary" or self._task_.lower() == "multilabel"):
            return tl.Binary_Cross_Entropy_Loss(outputs, labels)
        elif(self._task_.lower() == "multiclass"):
            return tl.Softmax_Cross_Entropy_Loss(outputs, labels)
        elif(self._task_.lower() == "regression"):
            return tl.Mean_Squared_Error_Loss(outputs, labels)
        else:
            raise Exception("output_activation_function error: invalid task")
    
    def compute_dZ_last(self, A_out: np.ndarray, labels: np.ndarray):
    
        task = self._task_.lower()
        
        if task == "multiclass":
            Y = tl.One_Hot(labels)
            dZ = A_out - Y
            return dZ

        elif task == "binary" or task == "multilabel":
            Y = labels.reshape(A_out.shape)
            dZ = A_out - Y
            return dZ

        elif task == "regression":
            dZ = 2 * (A_out - labels)
            return dZ

        else:
            raise Exception("compute_dZ_last error: invalid task")

    def activation_derivative(self, Z):
        act = self._activation_type_.lower()
        
        if act == "relu":
            return tl.Derivitive_ReLU(Z)
        elif act == "sigmoid":
            A = tl.Sigmoid(Z)
            return tl.Derivative_Sigmoid_from_A(A)
        else:
            return tl.Derivative_Tanh(Z)

    
    def backward_propagation(self, A_out: np.ndarray, labels: np.ndarray, forward_layers: dict):
        backward_layers = {}
        m = labels.shape[1] if labels.ndim > 1 else labels.size

        
        num_of_layers = self._num_of_hidden_layers_ + 1
        
        dZ = self.compute_dZ_last(A_out, labels)
        dW = (1 / m) * tl.Matrix_Multiplication(dZ, forward_layers[f"a{num_of_layers - 1}"].T)
        dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        backward_layers[f"dZ{num_of_layers}"] = dZ
        backward_layers[f"dW{num_of_layers}"] = dW
        backward_layers[f"dB{num_of_layers}"] = dB
        
        for layer in range(num_of_layers - 1, 0, -1):
            dZ = self.activation_derivative(forward_layers[f"z{layer}"]) * tl.Matrix_Multiplication(self._weights_[f"w{layer + 1}"].T, backward_layers[f"dZ{layer + 1}"])
            dW = (1 / m) * tl.Matrix_Multiplication(dZ, forward_layers[f"a{layer - 1}"].T)
            dB = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            
            backward_layers[f"dZ{layer}"] = dZ
            backward_layers[f"dW{layer}"] = dW
            backward_layers[f"dB{layer}"] = dB
        
        return backward_layers
    
    def update_parameters(self, backward_layers):
        num_of_layers = self._num_of_hidden_layers_ + 1
        for layer in range(num_of_layers, 0, -1):
            self._weights_[f"w{layer}"] -= self._learning_rate_ * backward_layers[f"dW{layer}"]
            self._biases_[f"b{layer}"] -= self._learning_rate_ * backward_layers[f"dB{layer}"]
        
    def get_predictions(self, X):
        A_out, _ = self.forward_propagation(X)
        task = self._task_.lower()
        
        if task == "binary":
            return (A_out >= 0.5).astype(int)

        elif task == "multilabel":
            return (A_out >= 0.5).astype(int)

        elif task == "multiclass":
            return np.argmax(A_out, axis=0)

        elif task == "regression":
            return A_out

        else:
            raise Exception("get_predictions error: invalid task")
        
    def get_accuracy(self, X, y):
        predictions = self.get_predictions(X)
        task = self._task_.lower()

        if task == "binary":
            y = y.reshape(predictions.shape)
            return np.mean(predictions == y)

        elif task == "multiclass":
            return np.mean(predictions == y)

        elif task == "multilabel":
            y = y.reshape(predictions.shape)
            return np.mean(np.all(predictions == y, axis=0))

        elif task == "regression":
            raise Exception("Accuracy is not defined for regression")

        else:
            raise Exception("get_accuracy error: invalid task")


        