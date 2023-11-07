from typing import Optional, List

from .base import ConfigBase

class CriterionConfig(ConfigBase):
    # Whether to use GPU, to be determined at runtime
    use_cuda: Optional[bool] = None

class CrossEntropyLossConfig(CriterionConfig):
    # A manual rescaling weight given to each class. If given, it has to be a Tensor of size C.
    weight: Optional[List[float]] = None

    # Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    # 'none': no reduction will be applied,
    # 'mean': the sum of the output will be divided by the number of elements in the output,
    # 'sum': the output will be summed.
    # Default: 'mean'
    reduction: str = 'mean'

    # Whether to apply label smoothing
    label_smooth_eps: Optional[float] = None

class BinaryCrossEntropyLossConfig(CriterionConfig):
    # A manual rescaling weight given to each class. If given, it has to be a Tensor of size C.
    weight: Optional[List[float]] = None

    # Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    # 'none': no reduction will be applied,
    # 'mean': the sum of the output will be divided by the number of elements in the output,
    # 'sum': the output will be summed.
    # Default: 'mean'
    reduction: str = 'mean'

    # Whether to apply label smoothing
    label_smooth_eps: Optional[float] = None
