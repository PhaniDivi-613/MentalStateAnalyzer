from typing import Tuple

from .base import ConfigBase

"""Reference: https://pytorch.org/docs/stable/optim.html"""

class OptimizerConfig(ConfigBase):
    # Learning rate for the embedding layer
    embedding_lr: float = 1e-3
    
    # Learning rate for other layers
    lr: float = 1e-3

    # The embedding layer will not be trained before the static_epoch
    # After the static_epoch, the embedding will be trained with embedding_lr as the learning rate
    static_epoch: int = 0

class AdamConfig(OptimizerConfig):
    """
    Adam algorithm.
    It has been proposed in "Adam: A Method for Stochastic Optimization."
    """

    # Coefficients used for computing running averages of the gradient
    # and its square (default: (0.9, 0.999))
    betas: Tuple[float, float] = (0.9, 0.999)

    # Term added to the denominator to improve numerical stability (default: 1e-8)
    eps: float = 1e-8

    # Weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0.

    # Whether to use the AMSGrad variant of this algorithm as described in
    # the paper "On the Convergence of Adam and Beyond"
    amsgrad: bool = False

class AdadeltaConfig(OptimizerConfig):
    """
    Adadelta algorithm.
    It has been proposed in "ADADELTA: An Adaptive Learning Rate Method."
    """

    # Coefficient used for computing a running average of squared gradients (default: 0.9)
    rho: float = 0.9

    # Term added to the denominator to improve numerical stability (default: 1e-6)
    eps: float = 1e-6

    # Weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0.

class AdagradConfig(OptimizerConfig):
    """
    Adagrad algorithm.
    It has been proposed in "Adaptive Subgradient Methods for
        Online Learning and Stochastic Optimization."
    """

    # Learning rate decay (default: 0)
    lr_decay: float = 0.0

    # Weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0.0

    # Term added to the denominator to improve numerical stability (default: 1e-10)
    eps: float = 1e-10

class AdamWConfig(OptimizerConfig):
    """
    AdamW algorithm.
    The original Adam algorithm was proposed in "Adam: A Method for Stochastic Optimization."
    The AdamW variant was proposed in "Decoupled Weight Decay Regularization."
    """

    # Coefficients used for computing running averages of the gradient and its square (default: (0.9, 0.999))
    betas: Tuple[float, float] = (0.9, 0.999)

    # Term added to the denominator to improve numerical stability (default: 1e-8)
    eps: float = 1e-8

    # Weight decay coefficient (default: 1e-2)
    weight_decay: float = 1e-2

    # Whether to use the AMSGrad variant of this algorithm as described in
    # the paper "On the Convergence of Adam and Beyond" (default: False)
    amsgrad: bool = False

class AdamaxConfig(OptimizerConfig):
    """
    Adamax algorithm (a variant of Adam based on infinity norm).
    It has been proposed in "Adam: A Method for Stochastic Optimization."
    """

    # Coefficients used for computing running averages of the gradient and its square (default: (0.9, 0.999))
    betas: Tuple[float, float] = (0.9, 0.999)

    # Term added to the denominator to improve numerical stability (default: 1e-8)
    eps: float = 1e-8

    # Weight decay coefficient (default: 1e-2)
    weight_decay: float = 1e-2

class ASGDConfig(OptimizerConfig):
    """
    Averaged Stochastic Gradient Descent.
    It has been proposed in "Acceleration of stochastic approximation by averaging."
    """

    # Decay term (default: 1e-4)
    lambd: float = 1e-4

    # Power for eta update (default: 0.75)
    alpha: float = 0.75

    # Point at which to start averaging (default: 1e6)
    t0: float = 1e6

    # Weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0

class RMSpropConfig(OptimizerConfig):
    """
    RMSprop algorithm.
    Proposed by G. Hinton in his course.
    """

    # Momentum factor (default: 0)
    momentum: float = 0.0

    # Smoothing constant (default: 0.99)
    alpha: float = 0.99

    # Term added to the denominator to improve numerical stability (default: 1e-8)
    eps: float = 1e-8

    # If True, compute the centered RMSProp, where the gradient is normalized by an estimation of its variance
    centered: bool = False

    # Weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0.

class RpropConfig(OptimizerConfig):
    """
    The resilient backpropagation algorithm.
    """

    # Pair of (etaminus, etaplis), which are multiplicative increase and decrease factors (default: (0.5, 1.2))
    etas: Tuple[float, float] = (0.5, 1.2)

    # A pair of minimal and maximal allowed step sizes (default: (1e-6, 50))
    step_sizes: Tuple[float, float] = (1e-6, 50)

class SGDConfig(OptimizerConfig):
    """
    Stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from "On the importance of initialization and momentum in deep learning."
    """

    # Momentum factor (default: 0)
    momentum: float = 0.

    # Weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0.

    # Dampening for momentum (default: 0)
    dampening: float = 0.

    # Enables Nesterov momentum (default: False)
    nesterov: bool = False
