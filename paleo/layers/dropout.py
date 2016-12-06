"""The module estimates 2D convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from paleo.layers import base


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
