"""A time estimator by running TensorFlow operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import tensorflow as tf
import numpy as np
from six.moves import range

from paleo.profilers.base import BaseProfiler, TimeMeasure


class TensorFlowProfiler(BaseProfiler):
    def __init__(self, options, device='/gpu:0'):
        super(TensorFlowProfiler, self).__init__('TensorFlowProfiler', options)
        self._device = device
        self._logger.info('TensorFlow version: %s' % tf.__version__)

    def profile(self, layer):
        graph = tf.Graph()
        ops, bwd_ops = None, None
        if layer.layertype == 'conv2d':
            ops, bwd_ops = self._ops_conv2d(layer, graph)
        elif layer.layertype == 'innerproduct':
            ops, bwd_ops = self._ops_innerproduct(layer, graph)
        elif layer.layertype == 'pool2d':
            ops, bwd_ops = self._ops_pool2d(layer, graph)
        elif layer.layertype == 'dropout':
            ops, bwd_ops = self._ops_dropout(layer, graph)
        elif layer.layertype == 'concat':
            ops, bwd_ops = self._ops_concat(layer, graph)
        elif layer.layertype == 'reshape':
            ops, bwd_ops = self._ops_reshape(layer, graph)
        else:
            self._logger.warning('Unimplemented \'%s\'' % layer.layertype)

        return self._execute(ops, bwd_ops, graph)

    def profile_full_pass(self, layers):
        graph, end_points, variables = self._compose_full_graph(layers)

        # Forward pass.
        if layers[-1].layertype in ['softmax', 'sigmoid']:
            last_op = end_points[layers[-2].name]
            loss_op = end_points[layers[-1].name]
        else:
            last_op = end_points[layers[-1].name]
            loss_op = None
        forward_time = self._execute(last_op, None, graph)

        # Backward pass.
        softmax_time = TimeMeasure()
        backward_time = TimeMeasure()
        if loss_op is not None:
            softmax_time = self._execute(loss_op, None, graph)

            with graph.as_default():
                grad_op = tf.gradients(loss_op, variables)
            backward_time = self._execute(grad_op, None, graph)

            backward_time = backward_time - softmax_time
            softmax_time = softmax_time - forward_time
        return forward_time, softmax_time, backward_time

    def _compose_full_graph(self, layers):
        graph = tf.Graph()
        end_points = dict()  # collects out tensors for each layer
        variables = [None]  # collects trainable variables
        for layer in layers:
            if layer.layertype == 'conv2d':
                ops, _ = self._ops_conv2d(layer, graph, end_points, variables)
            elif layer.layertype == 'deconv2d':
                ops, _ = self._ops_deconv2d(layer, graph, end_points,
                                            variables)
            elif layer.layertype == 'innerproduct':
                ops, _ = self._ops_innerproduct(layer, graph, end_points,
                                                variables)
            elif layer.layertype == 'pool2d':
                ops, _ = self._ops_pool2d(layer, graph, end_points)
            elif layer.layertype == 'upsampling2d':
                ops, _ = self._ops_upsampling2d(layer, graph, end_points)
            elif layer.layertype == 'dropout':
                ops, _ = self._ops_dropout(layer, graph, end_points)
            elif layer.layertype == 'concat':
                ops, _ = self._ops_concat(layer, graph, end_points)
            elif layer.layertype == 'reshape':
                ops, _ = self._ops_reshape(layer, graph, end_points)
            elif layer.layertype == 'softmax':
                ops, _ = self._ops_softmax(layer, graph, end_points)
            elif layer.layertype == 'sigmoid':
                ops, _ = self._ops_sigmoid(layer, graph, end_points)
            elif layer.layertype == 'input':
                # skip data/input layer.
                continue
            else:
                raise NotImplementedError('Cannot create ops for layer %s [%s]'
                                          % (layer.name, layer.layertype))
            end_points[layer.name] = ops

        return graph, end_points, variables[1:]

    def _get_inputs(self, layer, end_points=None):
        if end_points is None or layer.parents[0] == 'data':
            # Isolation mode: inputs for the layer are random constants.
            inputs = tf.constant(
                2 * np.random.random_sample(layer.inputs) - 1,
                dtype=tf.float32,
                name="fake_inputs")
            return inputs
        else:
            # Chain mode: get inputs from parent layer outputs.
            inputs = [end_points[p] for p in layer.parents]
            if len(inputs) == 1:
                return inputs[0]
            return inputs

    def _get_variable(self, shape, name='constant'):
        return tf.Variable(
            tf.truncated_normal(
                shape, dtype=tf.float32, stddev=1e-1),
            name='rand_{}'.format(name))

    def _get_fake_targets(self, batch_size, num_classes):
        labels = np.random.randint(0, num_classes, batch_size)
        return tf.constant(labels, dtype=tf.int32, name='fake_targets')

    def _ops_conv2d(self, layer, graph, end_points=None, variables=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                filters = self._get_variable(layer.filters, name='filters')

                if variables:
                    variables.append(filters)

                conv = None
                if self.options.direction == 'forward':
                    conv = tf.nn.conv2d(
                        inputs, filters, layer.strides, padding=layer.padding)

                bwd_inputs_op, bwd_filter_op = None, None
                if self.options.direction == 'backward':
                    if self.options.gradient_wrt == 'data' and layer.backprop:
                        bwd_inputs_op = tf.nn.conv2d_backprop_input(
                            layer.inputs,
                            filters,
                            self._get_variable(
                                layer.outputs, name='outputs'),
                            layer.strides,
                            layer.padding)
                    elif self.options.gradient_wrt == 'filter':
                        bwd_filter_op = tf.nn.conv2d_backprop_filter(
                            inputs, layer.filters,
                            self._get_variable(layer.outputs, 'outputs'),
                            layer.strides, layer.padding)
        return conv, [bwd_inputs_op, bwd_filter_op]

    def _ops_deconv2d(self, layer, graph, end_points=None, variables=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                filters = self._get_variable(layer.filters, name='filters')

                if variables:
                    variables.append(filters)

                deconv = tf.nn.conv2d_transpose(
                    inputs,
                    filters,
                    output_shape=layer.outputs,
                    strides=layer.strides)
        return deconv, None

    def _ops_innerproduct(self, layer, graph, end_points=None, variables=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                weights = self._get_variable(layer.weights, name='weights')

                if variables:
                    variables.append(weights)

                innerprod = tf.matmul(inputs, weights)
        return innerprod, None

    def _ops_pool2d(self, layer, graph, end_points=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                if layer.pool_type == 'max':
                    pool_op = tf.nn.max_pool
                elif layer.pool_type == 'avg':
                    pool_op = tf.nn.avg_pool
                else:
                    raise NotImplementedError('Invalid pool type: %s' %
                                              layer.pool_type)
                pool = pool_op(
                    inputs, layer.kernel, layer.strides, padding=layer.padding)
        return pool, None

    def _ops_upsampling2d(self, layer, graph, end_points=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                upsampling = tf.image.resize_nearest_neighbor(
                    inputs, layer.outputs[1:3])
        return upsampling, None

    def _ops_dropout(self, layer, graph, end_points=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                dropout = tf.nn.dropout(inputs, layer.keep_prob)
        return dropout, None

    def _ops_concat(self, layer, graph, end_points=None):
        with graph.as_default():
            with tf.device(self._device):
                if end_points:
                    inputs = self._get_inputs(layer, end_points)
                else:
                    inputs = [tf.Variable(tf.random_normal(inp))
                              for inp in layer.inputs]
                concat = tf.concat(layer.dim, inputs)
        return concat, None

    def _ops_reshape(self, layer, graph, end_points=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                reshape = tf.reshape(inputs, layer.outputs)
        return reshape, None

    def _ops_softmax(self, layer, graph, end_points=None):
        # For simplicity, here combine softmax and loss
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        tf.squeeze(inputs), self._get_fake_targets(
                            layer.outputs[0], layer.outputs[1])))
        return loss, None

    def _ops_sigmoid(self, layer, graph, end_points=None):
        with graph.as_default():
            with tf.device(self._device):
                inputs = self._get_inputs(layer, end_points)
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(inputs, tf.zeros(
                        layer.outputs)))
        return loss, None

    def _execute(self, layer_ops, bwd_ops, graph):
        with graph.as_default():
            with tf.device(self._device):
                config = tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=(
                        self._logger.getEffectiveLevel() == logging.DEBUG),
                    graph_options=tf.GraphOptions(
                        optimizer_options=tf.OptimizerOptions(
                            opt_level=tf.OptimizerOptions.L0)))

                ops_to_run = None
                if self.options.direction == 'forward':
                    if layer_ops is None:
                        return TimeMeasure()

                    if isinstance(layer_ops, list):
                        target_fwd_op = [tf.group(op) for op in layer_ops]
                    else:
                        shape = tf.shape(layer_ops)
                        target_fwd_op = tf.group(shape)
                    ops_to_run = target_fwd_op
                elif self.options.direction == 'backward':
                    if bwd_ops is None:
                        return TimeMeasure()
                    else:
                        if self.options.gradient_wrt == 'data':
                            target = bwd_ops[0]
                        elif self.options.gradient_wrt == 'filter':
                            target = bwd_ops[1]
                        else:
                            self._logger.warning(
                                'TensorFlowProfiler cannot run two'
                                'backward ops for now.')
                            return TimeMeasure()
                    if target is None:
                        return TimeMeasure()
                    target_bwd_op = tf.group(tf.shape(target))
                    ops_to_run = target_bwd_op

                init = tf.initialize_all_variables()

                # Create a session and initialize variables.
                with tf.Session(config=config) as sess:

                    # writer = tf.train.SummaryWriter('logs/', sess.graph)
                    sess.run(init)

                    # Run the ops.
                    durations = []
                    for i in range(self.options.num_warmup +
                                   self.options.num_iter):
                        start_time = time.time()
                        sess.run(ops_to_run)
                        duration = time.time() - start_time

                        if i >= self.options.num_warmup:
                            # Mesure time in milliseconds.
                            durations.append(duration * (10**3))

                mean_time = np.mean(durations)
        tf.reset_default_graph()
        return TimeMeasure(total_time=mean_time)
