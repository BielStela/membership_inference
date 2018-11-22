# Implementation of the data synthesis algorithm proposet by Shokri et al.
import itertools

import numpy as np
from tqdm import tqdm_notebook


def features_generator(n_features: int, types: str,
                       rang: tuple = (0, 1)) -> np.ndarray:
    """
    Creates a n features vector with uniform features
    sampled from a given range.

    Parameters
    ----------
    n_features: int
        number of features or length of the vector

    types: str
        type of the features. It only accepts uniform types.

    rang: tuple(int, int)
        range of the random uniform population from 
        where to drawn samples

    Returns
    -------
    x: np.ndarray
        features vector
    """
    # D-fence params
    if types not in ('binary', 'int', 'float'):
        raise ValueError(
            "Parameter `types` must be 'binary', 'int' or 'float'")

    if types == 'binary':
        x = np.random.randint(0, 2, n_features)
    if types == 'int':
        x = np.random.randint(rang[0], rang[1], n_features)
    if types == 'float':
        x = np.random.uniform(rang[0], rang[1], n_features)
    return x.reshape((1, -1))


def feature_randomizer(x: np.ndarray, k: int,
                       types: str, rang: tuple) -> np.ndarray:
    """
    Randomizes k features from feature vector x

    Parameters
    ----------
    x: np.ndarray
        input array that will be modified

    k: int
        number of features to modify

    types: str
        type of the features. It only accepts uniform types.

    rang: tuple(int, int)
        range of the random uniform population from 
        where to drawn samples

    Returns
    -------
    x: np.ndarray
        input vector with k modified features

    """
    idx_to_change = np.random.randint(0, x.shape[1], size=k)

    new_feats = features_generator(k, types, rang)

    x[0, idx_to_change] = new_feats
    return x


def synthesize(target_model, fixed_class: int,  k_max: int):
    """
    Generates synthetic records that are classified
    by the target model with high confidence.

    Parameters
    ----------
    class_: int
        fixed class which attacker wants to drawn samples from

    n_features: int
        number of features per input vector

    target_model: estimator
        Estimator that returns a class probability vector
        from an input features vector. Implemented for
        sklearn.base.BaseEstimator with `predict_proba()`
        method.

    k_max: int
        max "radius" of feature perturbation

    Returns
    -------
    x: np.ndarray
        synthetic feature vector
    """

    if not hasattr(target_model, 'predict_proba'):
        raise AttributeError('tarjet_model must have predict_proba() method')

    n_features = target_model.n_features_
    x = features_generator(n_features, types='float')  # random record

    y_c_current = 0  # target model’s probability of fixed class
    n_rejects = 0  # consecutives rejections counter
    k = k_max
    k_min = 1
    max_iter = 1000
    conf_min = 0.8  # min prob cutoff to consider a record member of the class
    rej_max = 10  # max number of consecutive rejections
    for _ in range(max_iter):
        y = target_model.predict_proba(x)  # query target model
        y_c = y.flat[fixed_class]
        if y_c >= y_c_current:
            if (y_c > conf_min) and (fixed_class == np.argmax(y)):
                return x
            # reset vars
            x_new = x
            y_c_current = y_c
            n_rejects = 0
        else:
            n_rejects += 1
            if n_rejects > rej_max:
                k = max(k_min, int(np.ceil(k/2)))
                n_rejects = 0

        x = feature_randomizer(x_new, k, types='float', rang=(0, 1))

    return "Failed to synthesize"


def synthesize_batch(tarjet_model, fixed_class, n_records):
    """
    Synthesize a batch of records
    """
    n_features = tarjet_model.n_features_
    x_synth = np.zeros((n_records, n_features))

    for i in tqdm_notebook(range(x_synth.shape[0])):
        x_vec = synthesize(tarjet_model, fixed_class, k_max=3)
        if isinstance(x_vec, str):
            x_synth[i, :] = np.nan
        else:
            x_synth[i, :] = x_vec

    return x_synth
