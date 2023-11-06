from abc import ABCMeta, abstractmethod

class Trainer(metaclass=ABCMeta):
    """Base class for Trainers, defining the interface for MLTrainer and DLTrainer."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self):
        pass
