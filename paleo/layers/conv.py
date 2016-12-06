"""The module estimates 2D convolution layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from paleo.layers import base


class Conv2d(base.BaseLayer):
    """Estimator for 2D Convolutional layers. """

    def __init__(self,
                 name,
                 inputs,
                 filters,
                 strides,
                 padding,
                 use_cudnn=False,
                 backprop=True,
                 activation_fn='relu',
                 percent_holes=0.0,
                 splits=None):
        """Initialize estimator. """
        super(Conv2d, self).__init__(name, 'conv2d')
        self._inputs = list(inputs)
        self._filters = list(filters)
        self._strides = list(strides)
        self._padding = padding
        if splits is not None:
            self.split_model(splits)
        self._outputs = self._calculate_output_shape()

        self._use_cudnn = use_cudnn
        self._backprop = backprop
        self._activation_fn = activation_fn
        # Percent of holes in astrous convolution.
        self._percent_holes = percent_holes

    @property
    def percent_holes(self):
        return self._percent_holes

    @property
    def activation_fn(self):
        return self._activation_fn

    @property
    def bias(self):
        return self._filters[-1]

    @property
    def filters(self):
        return self._filters

    @property
    def backprop(self):
        return self._backprop

    @property
    def strides(self):
        return self._strides

    @property
    def padding(self):
        return self._padding

    def split_model(self, num_splits):
        """Split in model parallel fashion."""
        self._filters[3] = self._filters[3] // num_splits

    def additional_summary(self):
        return "Filters: {}  Pad: {} ({}, {}) Stride: {}, {} Params: {:,}".format(
            self._filters, self._padding, self._pad_h, self._pad_w,
            self.strides[1], self.strides[2], self.num_params)

    def _calculate_output_shape(self):
        """Returns the output tensor shape."""
        n, h, w, c = self._inputs
        kernel_h, kernel_w, in_channel, out_channel = self._filters
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
        elif isinstance(self._padding, list):
            self._pad_h, self._pad_w = self._padding
            out_height = (h + 2 * self._pad_h - kernel_h) // stride_h + 1
            out_width = (w + 2 * self._pad_w - kernel_w) // stride_w + 1

        assert in_channel == c, (
            "Input channel shall match. Layer %s: %d != %d" %
            (self.name, in_channel, c))

        #out_h = (h + 2 * self._pad_h - kernel_h) // stride_h + 1
        #out_w = (w + 2 * self._pad_w - kernel_w) // stride_w + 1

        return [n, out_height, out_width, out_channel]

    @property
    def weights_in_bytes(self):
        """Returns weights."""
        _BYTES_FLOAT = 4
        kernel_h, kernel_w, in_channel, out_channel = self._filters
        filters_in_bytes = (kernel_h * kernel_w * in_channel * out_channel *
                            _BYTES_FLOAT)
        bias_in_bytes = out_channel * _BYTES_FLOAT
        return filters_in_bytes + bias_in_bytes

    @property
    def num_params(self):
        weights = reduce(lambda x, y: x * y, self._filters, 1)
        bias = self._filters[-1]
        return weights + bias

    def gradients(self, wrt='inputs'):
        """Returns a conv layer that is equivalent to calculating the gradient
        on this layer.

        Args:
            wrt: inputs or filters
        """

        layer = self

        def _compute_padding(layer):
            # Reference: TensorFlow ConvBackpropExtractAndVerifyDimension()
            # Convolution of inputs with padded output grads and filters.
            expanded_output_h = (layer.outputs[1] - 1) * layer.strides[1] + 1
            expanded_output_w = (layer.outputs[2] - 1) * layer.strides[2] + 1

            padded_out_h = layer.inputs[1] + layer.filters[0] - 1
            padded_out_w = layer.inputs[2] + layer.filters[1] - 1

            # Number of padding elements to be added before/after this
            # dimension of input when computing Conv2DBackpropInput.
            pad_before_h = layer.filters[0] - 1 - layer._pad_h
            pad_before_w = layer.filters[1] - 1 - layer._pad_w
            pad_after_h = padded_out_h - expanded_output_h - pad_before_h
            pad_after_w = padded_out_w - expanded_output_w - pad_before_w

            # Add one when padding is odd.
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_grad_filter_ops.cc#L471
            if pad_before_h == pad_after_h - 1:
                expanded_output_h += 1
            if pad_before_w == pad_after_w - 1:
                expanded_output_w += 1
            return (expanded_output_h, expanded_output_w, pad_before_h,
                    pad_before_w)

        expanded_output_h, expanded_output_w, pad_h, pad_w = _compute_padding(
            layer)

        holes = (expanded_output_h * expanded_output_w - self.outputs[1] *
                 self.outputs[2])
        percent_holes = (holes / expanded_output_h / expanded_output_w)

        if wrt == 'inputs':
            dummy_layer = Conv2d(
                name="dummy_layer",
                inputs=[layer.outputs[0], expanded_output_h, expanded_output_w,
                        layer.outputs[3]],
                filters=[layer.filters[0], layer.filters[1], layer.filters[3],
                         layer.filters[2]],
                strides=[1, 1, 1, 1],
                padding=[pad_h, pad_w])

        elif wrt == 'filters':
            # Convolution of inputs with inputs and output grads.
            dummy_layer = Conv2d(
                name="dummy_layer",
                inputs=[layer.inputs[3], layer.inputs[1], layer.inputs[2],
                        layer.inputs[0]],
                filters=[expanded_output_h, expanded_output_w,
                         layer.outputs[0], layer.outputs[3]],
                strides=[1, 1, 1, 1],
                padding=[layer._pad_h, layer._pad_w],
                percent_holes=percent_holes)

        return dummy_layer
