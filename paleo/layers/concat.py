"""The module estimates 2D convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from paleo.layers import base


class Concatenate(base.BaseLayer):
    """Estimator for Dropout layers. """

    def __init__(self, name, inputs, dim):
        """Initialize estimator. """
        super(Concatenate, self).__init__(name, 'concat')
        self._inputs = inputs
        self._outputs = list(self._inputs[0])
        self._outputs[dim] = 0
        for inp in self._inputs:
            self._outputs[dim] += inp[dim]
            # Assert equal for other dimensions.
            for d in xrange(len(inp)):
                if d != dim:
                    assert inp[d] == self._outputs[d]
        self._dim = dim

    @property
    def batch_size(self):
        for i in range(len(self.inputs)):
            assert self._inputs[i][0] == self._outputs[0]
        return self._inputs[0][0]

    @batch_size.setter
    def batch_size(self, batch_size):
        for i in range(len(self.inputs)):
            self._inputs[i][0] = batch_size
        self._outputs[0] = batch_size

    @property
    def dim(self):
        return self._dim

    def memory_in_bytes():
        """Returns weights."""
        return 0
