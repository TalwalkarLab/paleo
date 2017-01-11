"""Definitions of RNN Cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paleo.layers import base

import numpy as np


class RNNLayer(base.BaseLayer):
    def __init__(self,
                 name,
                 inputs,
                 output_dim,
                 length=10,
                 depth=1,
                 cell_type='simple'):
        super(RNNCell, self).__init__(name, 'rnn_{}'.format(cell_type))
        self._inputs = inputs
        self._outputs = [self._inputs[0], output_dim]
        self._length = length
        self._depth = depth
        if cell_type == 'simple':
            self.cell = SimpleRNNCell(self._inputs, self._outputs)

    @property
    def weights_in_bytes(self):
        _BYTES_FLOAT = 4
        return self.num_params * _BYTES_FLOAT

    @property
    def num_params(self):
        return self.cell.num_params


class RNNCell(object):
    def __init__(self, inputs, outputs, cell_type):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.input_dim = inputs[-1]
        self.output_dim = outputs[-1]
        self.cell_type = cell_type

    @property
    def num_params(self):
        return 0


class SimpleRNNCell(RNNCell):
    """A simple RNN Cell:
        y_t = f(x_t*W + h_{t-1}*U + b) """

    def __init__(self, name, inputs, outputs):
        super(SimpleRNNCell, self).__init__(name, 'simple')
        # Hidden states
        self.h = [self.inputs[0], self.output_dim]

        self.W = [self.input_dim, self.output_dim]
        self.U = [self.output_dim, self.output_dim]
        self.b = [self.output_dim]

    @property
    def num_params(self):
        weights = np.prod(self.W) + np.prod(self.U) + np.prod(self.b)
        return weights


class GRUCell(RNNCell):
    """A gated recurrent unit/GRU Cell.

    Reference:
      https://github.com/datalogai/recurrentshop/blob/master/recurrentshop/cells.py#L62
    """

    def __init__(self, name, inputs, outputs):
        super(GRUCell, self).__init__(name, 'gru')
        # Hidden states
        self.h = [self.inputs[0], self.output_dim]

        self.W = [self.input_dim, 3 * self.output_dim]
        self.U = [self.output_dim, 3 * self.output_dim]
        self.b = [3 * self.output_dim]

    @property
    def num_params(self):
        weights = np.prod(self.W) + np.prod(self.U) + np.prod(self.b)
        return weights
