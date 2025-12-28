class Helper_Classes:
    class Task:
        BINARY = "binary"
        MULTICLASS = "multiclass"
        REGRESSION = "regression"
        MULTILABEL = "multilabel"

    class Activation_Type:
        RELU = "relu"
        SIGMOID = "sigmoid"
        TANH = "tanh"
        OTHER = "other(uses tanh)"

    class Optimizer:
        GRADIENT_DESCENT = "gradient_descent"
        MINI_BATCH_GRADIENT_DESCENT = "mini_batch_gradient_descent"
