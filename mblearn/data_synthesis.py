# Implementation of the data synthesis algorithm proposet by Shokri et al.
import itertools

import numpy as np
from tqdm import tqdm_notebook


def features_generator(n_features: int, dtype: str, rang: tuple = (0, 1)) -> np.ndarray:
    """
    Creates a n features vector with uniform features
    sampled from a given range.

    Parameters
    ----------
    n_features: int
        number of features or length of the vector

    dtype: str
        type of the features. All the features must will have the same type.

    rang: tuple(int, int)
        range of the random uniform population from
        where to drawn samples

    Returns
    -------
    np.ndarray
        features vector
    """
    # D-fence params
    if dtype not in ("bool", "int", "float"):
        raise ValueError("Parameter `dtype` must be 'bool', 'int' or 'float'")

    if dtype == "bool":
        x = np.random.randint(0, 2, n_features)
    if dtype == "int":
        x = np.random.randint(rang[0], rang[1], n_features)
    if dtype == "float":
        x = np.random.uniform(rang[0], rang[1], n_features)
    return x.reshape((1, -1))


def feature_randomizer(x: np.ndarray, k: int, dtype: str, rang: tuple) -> np.ndarray:
    """
    Randomizes k features from feature vector x

    Parameters
    ----------
    x: np.ndarray
        input array that will be modified

    k: int
        number of features to modify

    dtype: str
        type of the features. It only accepts uniform dtype.

    rang: tuple(int, int)
        range of the random uniform population from 
        where to drawn samples

    Returns
    -------
    x: np.ndarray
        input vector with k modified features

    """
    idx_to_change = np.random.randint(0, x.shape[1], size=k)

    new_feats = features_generator(k, dtype, rang)

    x[0, idx_to_change] = new_feats
    return x


def synthesize(
    target_model, fixed_cls: int, k_max: int, dtype: str, n_features: int = None
) -> np.ndarray:
    """
    Generates synthetic records that are classified
    by the target model with high confidence.

    Parameters
    ----------
    target_model: estimator
        Estimator that returns a class probability vector
        from an input features vector. Implemented for
        sklearn.base.BaseEstimator with `predict_proba()`
        method.

    fixed_cls: int
        target model class to create data point from

    k_max: int
        max "radius" of feature perturbation
    
    dtype: str
        dtype of the features (float, int, bool)

    n_features: int
        number of features per input vector
    
    Returns
    -------
    np.ndarray
        synthetic feature vector

    False
        If failed to synthesize vector.
        This may be becaus number of iters exceded
    """

    if not hasattr(target_model, "predict_proba"):
        raise AttributeError("target_model must have predict_proba() method")

    if not hasattr(target_model, "n_features_") and n_features is None:
        raise ValueError("please specify the number of features in `n_features`")
    else:
        n_features = target_model.n_features_

    x = features_generator(n_features, dtype=dtype)  # random record

    y_c_current = 0  # target model’s probability of fixed class
    n_rejects = 0  # consecutives rejections counter
    k = k_max
    k_min = 1
    max_iter = 1000
    conf_min = 0.8  # min prob cutoff to consider a record member of the class
    rej_max = 5  # max number of consecutive rejections
    for _ in range(max_iter):
        y = target_model.predict_proba(x)  # query target model
        y_c = y.flat[fixed_cls]
        if y_c >= y_c_current:
            if (y_c > conf_min) and (fixed_cls == np.argmax(y)):
                return x
            # reset vars
            x_new = x
            y_c_current = y_c
            n_rejects = 0
        else:
            n_rejects += 1
            if n_rejects > rej_max:
                k = max(k_min, int(np.ceil(k / 2)))
                n_rejects = 0

        x = feature_randomizer(x_new, k, dtype=dtype, rang=(0, 1))

    return False


def synthesize_batch(target_model, fixed_cls, n_records, k_max):
    """
    Synthesize a batch of records
    """
    n_features = target_model.n_features_
    x_synth = np.zeros((n_records, n_features))

    for i in tqdm_notebook(range(n_records)):
        while True:  # repeat until synth finds record
            x_vec = synthesize(target_model, fixed_cls, k_max, dtype)
            if isinstance(x_vec, np.ndarray):
                break
        x_synth[i, :] = x_vec

    return x_synth
