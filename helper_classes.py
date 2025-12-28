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
        
    class Initialize_Method:
        CLASSIC_METHOD = "classic_method"
        LI_CHOW_AND_YU_METHOD = "li_chow_and_yu_method"
        SHIBATA_AND_IKEDA_METHOD = "shibata_and_ikeda_method"
        TAMURA_AND_TATEISHI_METHOD = "tamura_and_tateishi_method"
        SHEELA_AND_DEEPA_METHOD = "sheela_and_deepa_method"
