from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from paleo.layers import base


class Concatenate(base.BaseLayer):
    """Estimator for Concatenate layers. """

    def __init__(self, name, inputs, dim):
        """Initialize estimator. """
        super(Concatenate, self).__init__(name, 'concat')
        self._inputs = inputs
        self._outputs = list(self._inputs[0])
        self._outputs[dim] = 0
        for inp in self._inputs:
            self._outputs[dim] += inp[dim]
            # Assert equal for other dimensions.
            for d in six.moves.range(len(inp)):
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


class Reshape(base.BaseLayer):
    def __init__(self, name, inputs, output_shape):
        super(Reshape, self).__init__(name, 'reshape')
        self._inputs = list(inputs)
        self._outputs = list(output_shape)
        if self._outputs[0] == -1:
            self._outputs[0] = self._inputs[0]
        elif self._outputs[-1] == -1:
            self._outputs[-1] = np.prod(self._inputs) // self._outputs[0]


class Elementwise(base.BaseLayer):
    """Estimator for Elementwise layers. """

    def __init__(self, name, inputs):
        """Initialize estimator. """
        super(Elementwise, self).__init__(name, 'elementwise')
        for inp in inputs:
            assert inp == inputs[0], (
                'Elementwise op must have the same shape '
                '%s != %s' % (inp, inputs[0]))
        self._inputs = list(inputs)
        self._outputs = list(self._inputs[0])

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

    def memory_in_bytes():
        """Returns weights."""
        return 0


class Dropout(base.BaseLayer):
    """Estimator for Dropout layers. """

    def __init__(self, name, inputs, keep_prob):
        """Initialize estimator. """
        super(Dropout, self).__init__(name, 'dropout')
        self._inputs = inputs
        self._keep_prob = keep_prob
        self._outputs = self._inputs

    @property
    def keep_prob(self):
        return self._keep_prob

    def additional_summary(self):
        return "Keep prob: %f" % self._keep_prob

    def memory_in_bytes():
        """Returns weights."""
        return 0


class Softmax(base.BaseLayer):
    """Estimator for Softmax layers. """

    def __init__(self, name, inputs, num_classes=None):
        """Initialize estimator. """
        super(Softmax, self).__init__(name, 'softmax')
        self._inputs = inputs
        if num_classes is None:
            self._num_classes = self._inputs[-1]
            self._outputs = [self._inputs[0], self._inputs[-1]]
        else:
            self._num_classes = num_classes
            self._outputs = [self._inputs[0], num_classes]

    @property
    def outputs(self):
        return self._outputs

    @property
    def num_classes(self):
        return self._num_classes

    def additional_summary(self):
        return "Classes: %d" % self.num_classes

    def memory_in_bytes():
        return 0


class Sigmoid(base.BaseLayer):
    def __init__(self, name, inputs):
        super(Sigmoid, self).__init__(name, 'sigmoid')
        self._inputs = inputs
        self._outputs = [self._inputs[0], 1]


class InnerProduct(base.BaseLayer):
    """InnerProduct layers."""

    def __init__(self, name, inputs, num_outputs=None):
        super(InnerProduct, self).__init__(name, 'innerproduct')
        self._inputs = list(inputs)
        if len(self._inputs) != 2:
            # Auto flatten into (N, activations)
            self._inputs = [self._inputs[0], np.prod(self._inputs[1:])]
        self._num_outputs = num_outputs
        self._outputs = [self._inputs[0], self._num_outputs]
        self._weights = [self._inputs[1], self._num_outputs]

    @property
    def num_outputs(self):
        return self._num_outputs

    @property
    def weights(self):
        return self._weights

    @property
    def weights_in_bytes(self):
        _BYTES_FLOAT = 4
        weights_in_bytes = np.prod(self._weights) * _BYTES_FLOAT
        bias_in_bytes = self._num_outputs * _BYTES_FLOAT
        return weights_in_bytes + bias_in_bytes

    @property
    def num_params(self):
        weights = np.prod(self._weights)
        bias = self._num_outputs
        return weights + bias

    def additional_summary(self):
        return "Outputs: {} Params: {:,}".format(self.num_outputs,
                                                 self.num_params)
