"""The module estimates 2D convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from paleo.layers import base


class Pool2d(base.BaseLayer):
    """Estimator for 2D Max Pooling layers. """

    def __init__(self, name, inputs, kernel, strides, padding,
                 pool_type='max'):
        """Initialize estimator. """
        super(Pool2d, self).__init__(name, 'pool2d')
        self._inputs = inputs
        self._kernel = kernel
        self._strides = strides
        self._padding = padding
        self._outputs = None
        assert pool_type in ('max', 'avg'), ('Pooling type %s invalid' %
                                             pool_type)
        self._pool_type = pool_type

    @property
    def outputs(self):
        """Returns the output shape."""
        # Lazy calculation.
        if self._outputs is None:
            self._outputs = self._calculate_output_shape()
        return self._outputs

    @property
    def pool_type(self):
        return self._pool_type

    @property
    def kernel(self):
        return self._kernel

    @property
    def strides(self):
        return self._strides

    @property
    def padding(self):
        return self._padding

    def additional_summary(self):
        return "Kernel: %s  Stride: %s" % (self._kernel[1:3],
                                           self._strides[1:3])

    def _calculate_output_shape(self):
        """Returns the output tensor shape."""
        n, h, w, c = self._inputs
        _, kernel_h, kernel_w, _ = self._kernel
        _, stride_h, stride_w, _ = self._strides
        if self._padding == 'VALID':
            out_height = int(
                math.ceil(float(h - kernel_h + 1) / float(stride_h)))
            out_width = int(
                math.ceil(float(w - kernel_w + 1) / float(stride_w)))
            self._pad_h = 0
            self._pad_w = 0
        elif self._padding == 'SAME':
            out_height = int(math.ceil(float(h) / float(stride_h)))
            out_width = int(math.ceil(float(w) / float(stride_w)))

            pad_along_height = (h - 1) * stride_h + kernel_h - h
            pad_along_width = (w - 1) * stride_w + kernel_w - w
            self._pad_h = pad_along_height // 2
            self._pad_w = pad_along_width // 2

        return [n, out_height, out_width, c]

    def memory_in_bytes():
        return 0


class UpSampling2D(base.BaseLayer):
    def __init__(self, name, inputs, kernel):
        super(UpSampling2D, self).__init__(name, 'upsampling2d')
        self._inputs = inputs
        self._kernel = kernel
        self._outputs = list(inputs)
        for dim in [1, 2]:
            self._outputs[dim] = self._outputs[dim] * self._kernel[dim]
