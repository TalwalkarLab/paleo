"""The module estimates 2D convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from paleo.layers import base


class Softmax(base.BaseLayer):
    """Estimator for 2D Max Pooling layers. """

    def __init__(self, name, inputs, num_classes):
        """Initialize estimator. """
        super(Softmax, self).__init__(name, 'softmax')
        self._inputs = inputs
        self._num_classes = num_classes
        self._outputs = [self._inputs[0], self._num_classes]

    @property
    def outputs(self):
        return self._outputs

    @property
    def num_classes(self):
        return self._num_classes

    def additional_summary(self):
        return "Classes: %d" % self._num_classes

    def memory_in_bytes():
        return 0
