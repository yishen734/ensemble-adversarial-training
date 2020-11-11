import abc
from torch import nn


class DefenderBase(nn.Module, abc.ABC):
    """
    Abstract Base Class for defenders
    """
    def __init__(self):
        super(DefenderBase, self).__init__()

    @abc.abstractmethod
    def train_adversarial_samples(self, model, dataset, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def eval_adversarial_samples(self, model, dataset):
        raise NotImplementedError

