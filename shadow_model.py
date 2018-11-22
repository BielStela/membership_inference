import numpy as np
from sklearn.model_selection import train_test_split


class ShadowModels:
    """
    Creates a swarm of shadow models and trains them with a split
    of the synthetic data.

    TODO:
        - Split the synth data in n_models parts
        - The splited data is splited again in train and test (1/2)
        - Instance n models (must be similar to the target model)
        - Train each model on training splited data
        - Run prediction on both training and test splited data
        - Label the resulting prediction vector with "in"/"out" 
          if it was train or test
        - drop mic
    """

    def __init__(self, n_models: int, data: np.ndarray, target_classes: int):
        self.n_models = n_models
        self.data = data
        self.target_classes = target_classes
