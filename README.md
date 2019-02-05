[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BielStela/membership_inference/master)

# Membership Inference Attacks
Python package to create adversarial agents for membership inference attacks against machine learning models using Scikit-learn learners.

Implementation of the work done by Shokri _et al_ ([paper](https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf))

# Examples
Find some examples in `notebooks/`

The main classes and functions are:

### Data Synthetiser

To synthesize data only using a black-box like model `target_model` and predictions using the algorithm proposed by Shokri _et al_

```python 
from mblearn import synthetize

x = synthesize(target_model, fixed_class, k_max)
```

### Shadow models
Train $n$ shadow models on synthetic data with a given learner. The learner must be a scikit-learn estimator with the `predict_proba` method.


```python
from mblearn import ShadowModels

shadows = ShadowModels(n_models, data, target_classes, learner)

shadow_data = shadows.results
```

### Attacker models

Using the data generated with the shadow models, trains a attack models
on each label of the shadow dataset.

```python
from mblearn import AttackModels

attacker = AttackModels(target_classes, attack_learner)

# train the attacker with the shadow data
attacker.fit(shadow_data)

# query the target model and get the predicted class prob vector
X = target_model.predict_proba(test_data)

# especulate about the class this test_data belongs to
y = 0

# get the prediction:
# True if `test_data` is classified as a member of
# the private model training set for the given class
# False otherwise
attacker.predict(X, y)
```
## Bibliography
 R. Shokri, M. Stronati, and V. Shmatikov. Membership inference attacks against machine learning models. _Security and Privacy (SP), 2017 IEEE Symposium_
, IEEE, 2017.

Y. Long, V. Bindschaedler, L Wang, D. Bu, _et al_. Understanding Membership Inferences on Well-Generalized Learning Models. _arXiv preprint [arXiv:1802.04889](https://arxiv.org/pdf/1802.04889.pdf)_, 2018.

S. Truex, L. Liu, M. E. Gursoy, L. Yu, W. Wei. Towards Demystifying Membership Inference Attacks. _arXiv preprint [arXiv:1807.09173](https://arxiv.org/pdf/1807.09173.pdf)_, 2018.

## Warning

The maturity of the package is far from alpha. This is just a proof of concept and all the interface and inner wheels will change in the next few months.
