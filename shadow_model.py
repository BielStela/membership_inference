from typing import List, Tuple, Dict

from tqdm import tqdm_notebook
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.base import clone
import pandas as pd


class ShadowModels:
    """
    Creates a swarm of shadow models and trains them with a split
    of the synthetic data.

    TODO:
        - Run prediction on both training and test splited data
        - Label the resulting prediction vector with "in"/"out"
          if it was train or test
        - drop mic
    """

    def __init__(self, n_models: int, data: np.ndarray,
                 target_classes: int, learner):
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
        self.data = data
        self.target_classes = target_classes
        self.splits = self._split_data(self.data, self.n_models)
        self.learner = learner
        self.models = self._make_model_list(self.learner, self.n_models)

        # train models
        self.results = self.train_predict_shadows()

        def _split_data(data, n_splits) -> List[np.ndarray]:
        """
        Split manually into n datasets maintaining class proportions

        Suposes class label is at data[:,-1]
        """
        # data = np.hstack((data[0], data[1].reshape(-1, 1)))
        X = data
        y = data[:, -1]
        classes = np.unique(y)
        n_classes = len(classes)

        cls_partitions = []
        # Split by class
        for cl in classes:

            X_cls = X[y == cl, :]
            # y_cls = y[y == cl]
            batch_size = len(X_cls)//n_splits
            splits = []
            for i in range(n_splits):
                split_X = X_cls[i*batch_size:(i+1)*batch_size, :]
                # split_y = y_cls[i*batch_size:(i+1)*batch_size]
                splits.append(split_X)
            cls_partitions.append(splits)

        # -------------------
        # consolidate splits
        # -------------------
        grouped = []
        for split in range(n_splits):
            parts = []
            for part in cls_partitions:
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
        models = [clone(learner) for _ in range(n)]
        return models

    def train_predict_shadows(self):
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

            model.fit(X_train, y_train)
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
