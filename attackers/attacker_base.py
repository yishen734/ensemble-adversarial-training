# encoding: utf-8
"""
@author: Jingrong Feng
@contact: flo.fjr@gmail.com
@version: 0.1
@file: attacker_base.py
@time: 24/10/2020
"""
import abc
from torch import nn


class AttackerBase(nn.Module, abc.ABC):
    """
    Abstract Base Class for attackers
    """
    def __init__(self):
        super(AttackerBase, self).__init__()

    @abc.abstractmethod
    def train_adversarial_samples(self, model, dataset, **kwargs):
        """
        Train adversarial examples
        Args:
            model: the pre-trained model which will be attacked
            dataset: the CIFAR10 dataset
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval_adversarial_samples(self, model, dataset):
        """
        Evaluate the effects of adversarial examples
        Args:
            model: the pre-trained model which will be evaluated on
            dataset: the CIFAR10 dataset
        """
        raise NotImplementedError

