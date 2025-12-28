import numpy as np

class Tools:
    
    @staticmethod
    def Matrix_Multiplication(x: np.ndarray, y: np.ndarray):
        return x @ y
    
    @staticmethod
    def ReLU(z: np.ndarray):
        return np.maximum(0,z)
    
    @staticmethod
    def Derivitive_ReLU(z: np.ndarray):
        return (z > 0).astype(float)

    
    @staticmethod
    def Sigmoid(z: np.ndarray):
        z = np.clip(z, -500, 500) # 50’den sonra sigmoid zaten “tam 1”
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def Derivative_Sigmoid_from_A(a):
        return a * (1 - a)

    @staticmethod
    def Tanh(z: np.ndarray):
        return np.tanh(z)
    
    @staticmethod
    def Derivative_Tanh(z: np.ndarray):
        return 1 - np.tanh(z) ** 2
        
    @staticmethod
    def Softmax(z: np.ndarray):
        z_shift = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    @staticmethod
    def One_Hot(labels: np.ndarray):
        one_hot_labels = np.zeros((labels.size, np.max(labels) + 1))
        one_hot_labels[np.arange(labels.size) , labels] = 1
        one_hot_labels = one_hot_labels.T
        return one_hot_labels
    
    @staticmethod
    def Binary_Cross_Entropy_Loss(outputs, labels):
        epsilon = 1e-10
        outputs = np.clip(outputs, epsilon, 1 - epsilon)
        labels = labels.reshape(outputs.shape)
        loss = -np.mean(labels * np.log(outputs) + (1 - labels) * np.log(1 - outputs))
        return loss
    
    @staticmethod
    def Softmax_Cross_Entropy_Loss(outputs: np.ndarray, labels: np.ndarray):
        m = labels.shape[0] 
        one_hot_Y = Tools.One_Hot(labels)
        
        # log(0) hatasını önlemek için küçük epsilon
        epsilon = 1e-10
        loss = -np.sum(one_hot_Y * np.log(outputs + epsilon)) / m
        
        return loss
    
    @staticmethod
    def Mean_Squared_Error_Loss(outputs: np.ndarray, labels: np.ndarray):
        m = labels.size
        
        loss = np.sum((outputs - labels) ** 2) / m
        
        return loss
        
