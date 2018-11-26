from typing import List, Tuple, Dict

from tqdm import tqdm_notebook

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore")


class AttackModels:
    def __init__(self, target_classes, attack_learner):
        """
        Attacker models to learn class membership from shadow data.


        Parameters
        ----------
        target_classes: int
            number of classes that the target model can predict

        attack_learning: learner
            trainable learner to model memebership from shadow data.
            The learner its cloned into n models, one for each target class,
            and each model is trained on a class subset of the shadow data.


        Returns
        -------
        AttackModels class instance
        """
        self.target_classes = target_classes
        self.attack_learner = attack_learner

        self.attack_models = [clone(self.attack_learner)
                              for _ in range(target_classes)]

        self._fited = False

    def fit(self, shadow_data) -> None:
        """
        Trains `attack_models` with `shadow_data`. Each model is trained with
        with a subset of the same class of `shadow_data`.


        Parameters
        ----------
        shadow_data: np.ndarray
            Shadow data. Results from `ShadowModels`.
            Last column (`[:,-1]`) must be the membership label of the shadow
            prediction, where 1 means that the record was present in the 
            shadow training set ('in') and 0 if the recored was in the test
            set ('out').
            Second last column (`[:,-2]`) must be the data class. this will
            be used as grouper to split the data for each attack model.
            The rest of the columns are the class probability vector
            predicted by the shadow model.


        Returns
        -------
        None
        """
        # split data into subsets, n == target_classes
        data = shadow_data[:, :-2]
        membership_label = shadow_data[:, -1]
        class_label = shadow_data[:, -2]
        for i, model in enumerate(self.attack_models):
            X = data[class_label == i]
            y = membership_label[class_label == i]

            # train model
            model.fit(X, y)

        self._fited = True

    def predict(self, X, y, batch=False):
        """
        Predicts if `X` is real member of `y` in the attacked
        private training set.

        Parameters
        ----------
        X: np.ndarray
            Probability vector result from target model

        y: int, np.ndarray
            estimated class of the data record used to get `X`
        """
        if not self._fited:
            print('Must run .fit() first!')
            return

        if not batch:
            model_class = y
            model = self.attack_models[model_class]
            prob_vec = model.predict_proba(X)

            if y == np.argmax(prob_vec):
                print(y)
                print('YES! Looks like this record of class'
                      f' {y} is a real member of the training'
                      ' private dataset!')

            else:
                print(y)
                print('No luck this time :(')

        elif batch:
            raise NotImplementedError
