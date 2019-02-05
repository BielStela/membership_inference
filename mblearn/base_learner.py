from abc import ABCMeta, abstractmethod


class Learner(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict_proba(self):
        pass
