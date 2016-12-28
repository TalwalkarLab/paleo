from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

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


class Elementwise(base.BaseLayer):
    """Estimator for Elementwise layers. """

    def __init__(self, name, inputs):
        """Initialize estimator. """
        super(Elementwise, self).__init__(name, 'elementwise')
        for inp in inputs:
            assert inp == inputs[0], (
                'Elementwise op must have the same shape '
                '%s != %s' % (inp, inputs[0]))
        self._inputs = inputs
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
    """Estimator for 2D Max Pooling layers. """

    def __init__(self, name, inputs, num_classes=None):
        """Initialize estimator. """
        super(Softmax, self).__init__(name, 'softmax')
        self._inputs = inputs
        if num_classes is None:
            self._num_classes = self._inputs[-1]
            self._outputs = [self._inputs[0], self._inputs[-1]]
        else:
            self._outputs = [self._inputs[0], num_classes]

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
