from trainer import DLTrainer  # Circular import
from models.classifier import (
    # CNNClassifier,
    # LinearClassifier,
    RNNClassifier
    # RCNNClassifier,
    # DRNNClassifier,
    # DPCNNClassifier,
    # TransformerClassifier
)
from models.embedding_layer import (
    EmbeddingLayerConfig,
    StaticEmbeddingLayer,
    BertEmbeddingLayer
)
from models.dl_model import DLModel
# from tester import DLTester
from trainer.criterion import CrossEntropyLoss, BinaryCrossEntropyLoss
from config.optimizer import OptimizerConfig
from config.classifier import ClassifierConfig
from config.criterion import CriterionConfig
from config.scheduler import SchedulerConfig


CONFIG_TO_CLASS = {
    "DLTrainerConfig": DLTrainer,
    # "CNNClassifierConfig": CNNClassifier,
    "RNNClassifierConfig": RNNClassifier,
    # "RCNNClassifierConfig": RCNNClassifier,
    # "DRNNClassifierConfig": DRNNClassifier,
    # "DPCNNClassifierConfig": DPCNNClassifier,
    "StaticEmbeddingLayerConfig": StaticEmbeddingLayer,
    "BertEmbeddingLayerConfig": BertEmbeddingLayer,
    "DLModelConfig": DLModel,
    # "DLTesterConfig": DLTester,
    "CrossEntropyLossConfig": CrossEntropyLoss,
    "BinaryCrossEntropyLossConfig": BinaryCrossEntropyLoss
    # "FocalLossConfig": FocalLoss,
    # "TransformerClassifierConfig": TransformerClassifier
}

CONFIG_CHOICES = {
    OptimizerConfig: OptimizerConfig.__subclasses__(),
    ClassifierConfig: ClassifierConfig.__subclasses__(),
    CriterionConfig: CriterionConfig.__subclasses__(),
    SchedulerConfig: SchedulerConfig.__subclasses__(),
    EmbeddingLayerConfig: EmbeddingLayerConfig.__subclasses__(),
}
