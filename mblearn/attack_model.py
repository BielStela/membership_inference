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
        # 1 model for each class
        self.attack_models = [clone(self.attack_learner) for _ in range(target_classes)]

        self._fited = False

    @staticmethod
    def _update_learner_params(learner, **learner_params) -> None:
        # safety check if dict is well formed
        for k in learner_params.keys():
            if not hasattr(learner, k):
                raise AttributeError(
                    f"Learner parameter {k} is not an attribute of {learner.__class__}"
                )

        # update learner params
        learner.__dict__.update(**learner_params)

    def fit(self, shadow_data, **learner_kwargs) -> None:
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

        TODO
        ----
            Tweak model params with something like **learner_kwargs
            cross-validate
            grid search?
        """
        # split data into subsets, n == target_classes
        membership_label = shadow_data[:, -1]
        class_label = shadow_data[:, -2]
        data = shadow_data[:, :-2]
        for i, model in enumerate(self.attack_models):
            X = data[class_label == i]
            y = membership_label[class_label == i]

            # update model params
            self._update_learner_params(model, **learner_kwargs)
            # train model
            model.fit(X, y)

        self._fited = True

    def predict(self, X, y, batch=False) -> np.ndarray:
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
            print("Must run `fit` method first")
            return

        if not batch:
            model_cls = y
            model = self.attack_models[model_cls]
            prob_vec = model.predict_proba(X)

            if y == np.argmax(prob_vec) and np.argmax(prob_vec) == 1:
                return 1

            else:
                return 0

        elif batch:

            model_classes = np.unique(y)
            res = []
            for model_cls in model_classes:
                X_cls = X[y == model_cls]
                model = self.attack_models[model_cls]
                attack_res = model.predict_proba(X_cls)
                res.append(attack_res)

            return np.concatenate(res)
