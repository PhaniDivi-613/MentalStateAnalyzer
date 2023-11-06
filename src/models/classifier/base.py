import torch.nn as nn

from config import ClassifierConfig


class Classifier(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super(Classifier, self).__init__()
        self.config = config

    def forward(self, batch):
        raise NotImplementedError