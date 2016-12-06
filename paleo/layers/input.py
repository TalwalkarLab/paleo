"""The module estimates 2D convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paleo.layers import base


class Input(base.BaseLayer):
    """Estimator for 2D Convolutional layers. """

    def __init__(self, name, inputs):
        """Initialize estimator. """
        super(Input, self).__init__(name, 'input')
        self._inputs = inputs
        self._outputs = list(inputs)
