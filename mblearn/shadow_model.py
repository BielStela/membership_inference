from typing import List, Tuple, Dict
from copy import copy

from tqdm import tqdm_notebook

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import clone, BaseEstimator

import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


class ShadowModels:
    """
    Creates a swarm of shadow models and trains them with a split
    of the synthetic data.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 n_models: int, target_classes: int,
                 learner, **fit_kwargs):
        """
        Creates a swarm of shadow models and trains them with a split
        of the synthetic data.

        Parameters
        ----------

        n_models: int
            number of shadow models to build. Higher number returns
            better results but is limited by the number of records 
            in the input data.

        data: np.ndarray
            input data with the target label as last column

        target_classes: int
            number of classes of the target model or lenght of the
            prediction array of the target model.

        learner: learner? #fix type
            learner to use as shadow model. It must be as similar as 
            possible to the target model. It must have `predict_proba` 
            method. Now only sklearn learners are implemented.

        Returns
        -------
        ShadowModels object
        """

        self.n_models = n_models
        self.X = X
        if self.X.ndim > 1:
            self.X = self.X.reshape(self.X.shape[0],-1)  # flatten images or matrices inside 1rst axis

        self.y = y
        self.target_classes = target_classes
        self.splits = self._split_data(self.X, self.y, self.n_models, self.target_classes)
        self.learner = learner
        self.models = self._make_model_list(self.learner, self.n_models)

        # train models
        self.results = self.train_predict_shadows(**fit_kwargs)

    @staticmethod
    def _split_data(X: np.ndarray,
                    y: np.ndarray,
                    n_splits: int,
                    n_classes: int) -> List[np.ndarray]:
        """
        Split manually into n datasets maintaining class proportions
        """
        # data = np.hstack((data[0], data[1].reshape(-1, 1)))
        # X = data
        # y = data[:, -1]
        classes = range(n_classes)
        class_partitions = []
        # Split by class
        for clss in classes:

            X_cls = X[y == clss]
            y_cls = y[y == clss]
            batch_size = len(X_cls)//n_splits
            splits = []
            for i in range(n_splits):
                split_X = X_cls[i*batch_size:(i+1)*batch_size, :]
                split_y = y_cls[i*batch_size:(i+1)*batch_size]
                splits.append((split_X, split_y))
            class_partitions.append(splits)

        # -------------------
        # consolidate splits into ndarrays
        # -------------------

        grouped = []
        for split in range(n_splits):
            parts = []
            for part in class_partitions:
                parts.append(part[split])
            grouped.append(parts)

        splits = []
        for group in grouped:
            splits.append(np.vstack(group))

        return splits

    @staticmethod
    def _make_model_list(learner, n) -> List:
        """
        Intances n shadow models, copies of the input parameter learner
        """
        if isinstance(learner, tf.keras.models.Model):
            models = [copy(learner) for _ in range(n)]

        elif isinstance(learner, BaseEstimator):
            models = [clone(learner) for _ in range(n)]

        return models

    def train_predict_shadows(self, **fit_kwargs):
        """
        "in" : 1
        "out" : 0
        """

        # TRAIN and predict
        results = []
        for model, data_subset in tqdm_notebook(zip(self.models, self.splits)):
            X = data_subset[:, :-1]
            y = data_subset[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=0.5)

            model.fit(X_train, y_train, **fit_kwargs)
            # data IN training
            y_train = y_train.reshape(-1, 1)
            predict_in = model.predict_proba(X_train)
            res_in = np.hstack((predict_in, y_train, np.ones_like(y_train)))

            # data OUT training
            y_test = y_test.reshape(-1, 1)
            predict_out = model.predict_proba(X_test)
            res_out = np.hstack((predict_out, y_test, np.zeros_like(y_test)))

            # concat in single array
            model_results = np.vstack((res_in, res_out))
            results.append(model_results)

        results = np.vstack(results)
        return results

    def __repr__(self):
        rep = (f"Shadow models: {self.n_models}, {self.learner.__class__}\n"
               f"lengths of data splits : {[len(s) for s in self.splits]}")
        return rep
