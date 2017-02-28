"""Graph Representation of Deep Neural Network Architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging

from paleo import layers

logger = logging.getLogger("paleo")


class LayerSpec(object):
    """The specification of layers/operations in the DNN."""

    def __init__(self, layer_name, layer_params):
        super(LayerSpec, self).__init__()
        self.name = layer_name
        self.params = dict(layer_params)
        self.layer_op = None
        self.parents = []  # TODO: deprecate this property.
        self.inbounds = []
        self.outbounds = []

    def attach_op(self, layer_op):
        self.layer_op = layer_op

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __getitem__(self, key):
        return self.params[key]

    @property
    def device_id(self):
        if '@' in self.name:
            return self.name.split('@')[1]
        else:
            # By default, the layer is run on device 0.
            return 0

    def get(self, key, default):
        return self.params.get(key, default)


class OperationGraph(object):
    """The dependency graph of operations in the DNN."""

    def __init__(self, filename=None, attach_ops=True):
        super(OperationGraph, self).__init__()
        self.nested_list = None
        self._topological_order = None
        self.attach_ops = attach_ops
        if filename:
            self.load(filename)

    def load(self, filename):
        """Load the neural net architecture from JSON format.

            Convert the neural net into a list.
            Arrange the layers into multiple groups,
            such that the next group depend on the previous group.
              [(a), (b, c), (d)]
              [(a), ([b, c], [d, e]), (f)]

            Each node is LayerSpec object.
        """
        with open(filename, 'r') as f:
            net = json.load(f)
        self._create_graph(net)

    def load_from_string(self, string):
        net = json.loads(string)
        self._create_graph(net)

    @property
    def topology_order(self):
        return self._topological_order

    def _create_topology_order(self):
        """
        Expand the network to a plain list.
        The list is in topology order.
        """
        logger.debug('Creating topological order of layers '
                     'based on the nested list representation.')

        def flatten(layer):
            if isinstance(layer, (tuple, list)):
                _layer_list = []
                for l in layer:
                    _layer_list.extend(flatten(l))
                return _layer_list
            else:
                return [layer]

        if self._topological_order is None:
            self._topological_order = []
            for layer in self.nested_list:
                self._topological_order.extend(flatten(layer))

    def _attach_layer_op(self):
        """Flatten the list in topology order."""
        names_to_specs = dict()
        for layer_spec in self.topology_order:
            if len(layer_spec['parents']) == 1:
                parent_name = layer_spec['parents'][0]
                inputs = names_to_specs[parent_name].layer_op.outputs
            else:
                inputs = []
                try:
                    for parent_name in layer_spec['parents']:
                        inputs.append(
                            names_to_specs[parent_name].layer_op.outputs)
                except KeyError:
                    raise KeyError('Cannot find parent "%s" of "%s"' %
                                   (parent_name, layer_spec.name))

            try:
                layer = None
                if layer_spec['type'] == 'Input':
                    layer = layers.Input(layer_spec.name, layer_spec['tensor'])
                elif layer_spec['type'] == 'Convolution':
                    layer = layers.Conv2d(
                        layer_spec.name,
                        inputs,
                        layer_spec['filter'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        backprop=('data' not in layer_spec['parents']),
                        activation_fn=layer_spec.get('activation_fn', 'relu'),
                        splits=layer_spec.get('splits', None))
                elif layer_spec['type'] == 'Deconvolution':
                    layer = layers.Deconv2D(
                        layer_spec.name,
                        inputs,
                        layer_spec['filter'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        layer_spec['output_shape'],
                        backprop=('data' not in layer_spec['parents']),
                        activation_fn=layer_spec.get('activation_fn', 'relu'))
                elif layer_spec['type'] == 'Pooling':
                    layer = layers.Pool2d(
                        layer_spec.name,
                        inputs,
                        layer_spec['ksize'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        pool_type='max')
                elif layer_spec['type'] == 'UpSampling2D':
                    layer = layers.UpSampling2D(layer_spec.name, inputs,
                                                layer_spec['ksize'])
                elif layer_spec['type'] == 'AvgPool':
                    layer = layers.Pool2d(
                        layer_spec.name,
                        inputs,
                        layer_spec['ksize'],
                        layer_spec['strides'],
                        layer_spec['padding'],
                        pool_type='avg')
                elif layer_spec['type'] == 'Dropout':
                    layer = layers.Dropout(layer_spec.name, inputs,
                                           layer_spec['dropout_keep_prob'])
                elif layer_spec['type'] == 'Concatenate':
                    layer = layers.Concatenate(layer_spec.name, inputs,
                                               layer_spec['dim'])
                elif layer_spec['type'] == 'Reshape':
                    layer = layers.Reshape(layer_spec.name, inputs,
                                           layer_spec['output_shape'])
                elif layer_spec['type'] == 'Elementwise':
                    layer = layers.Elementwise(layer_spec.name, inputs)
                elif layer_spec['type'] == 'Softmax':
                    layer = layers.Softmax(layer_spec.name, inputs,
                                           layer_spec.get('num_classes', None))
                elif layer_spec['type'] == 'Sigmoid':
                    layer = layers.Sigmoid(layer_spec.name, inputs)
                elif layer_spec['type'] == 'InnerProduct':
                    layer = layers.InnerProduct(layer_spec.name, inputs,
                                                layer_spec['num_outputs'])
                else:
                    layer = layers.Generic(layer_spec.name, inputs,
                                           layer_spec['type'])
                #  else:
                #      raise ValueError('Cannot create layer object for %s,'
                #                       '%s is an unknown layer type.' %
                #                       (layer_spec.name, layer_spec['type']))
            except Exception as e:
                logger.error('Error when attaching ops for layer %s' %
                             layer_spec.name)
                logger.exception(e)
                exit()

            if layer:
                logger.debug('Attach layer op: %s inputs: %s  ouputs: %s' %
                             (layer.name, layer.inputs, layer.outputs))
                layer_spec.parents.extend(
                    [names_to_specs[p] for p in layer_spec['parents']])
                layer.parents = layer_spec['parents']
                layer_spec.attach_op(layer)
                names_to_specs[layer_spec.name] = layer_spec

    def _create_graph(self, net):
        names_to_specs = dict()  # layer_name -> LayerSpec object
        # Shortcuts, allow use block_name as parent.
        block_endpoints = dict()

        # This dictionary records how many splits each end_point name has.
        layernames_to_splits = dict()

        def _parents(parents, current_split=None):
            # Replace with endpoint if parent is a block.
            transformed_parents = []
            for parent_name in parents:
                # We don't support pointing to specific split.
                if '@all' in parent_name:
                    parent_name = parent_name.replace('@all', '')
                    splits = layernames_to_splits[parent_name]
                    for s in range(splits):
                        transformed_parents.append(
                            block_endpoints.get(parent_name, parent_name) + (
                                '@%d' % s))
                elif '@self' in parent_name:
                    parent_name = parent_name.replace('@self', '')
                    assert parent_name in layernames_to_splits, (
                        'Parent %s is not split.')
                    transformed_parents.append(
                        block_endpoints.get(parent_name, parent_name) + '@%d' %
                        current_split)
                else:
                    transformed_parents.append(
                        block_endpoints.get(parent_name, parent_name))
            return transformed_parents

        # First count split.
        for layer_name, layer_params in net['layers'].items():
            if layer_params.get('type', None) == 'ModelParallel':
                block_name = layer_name
                num_splits = layer_params.get('splits', 1)
                for sublayer_name in layer_params['layers']:
                    layernames_to_splits['%s/%s' % (
                        block_name, sublayer_name)] = num_splits

        # Transform all specs into LayerSpec objects.
        for layer_name, layer_params in net['layers'].items():
            if layer_params.get('type', None) in ['Block', 'ModelParallel']:
                is_model_parallel = (layer_params['type'] == 'ModelParallel')
                block_name = layer_name
                block_parents = _parents(layer_params['parents'])

                # For model parallel, the specified layers are repeated.
                num_splits = layer_params.get('splits', 1)

                for s in range(num_splits):
                    for sublayer_name, sublayer_params in layer_params[
                            'layers'].items():
                        sublayer_name = '%s/%s' % (block_name, sublayer_name)
                        if is_model_parallel:
                            sublayer_name = sublayer_name + ("@%d" % s)
                            sublayer_params['splits'] = num_splits

                        sublayer = LayerSpec(sublayer_name, sublayer_params)

                        # Update parents
                        if len(sublayer_params['parents']) == 0:
                            # Use the parent of the block.
                            sublayer_parents = block_parents
                        else:
                            # Add blockname to the parent names.
                            sublayer_parents = map(
                                lambda n: '%s/%s' % (block_name, n),
                                sublayer_params['parents'])
                            sublayer_parents = _parents(sublayer_parents, s)

                        sublayer.params['parents'] = sublayer_parents

                        assert sublayer_name not in names_to_specs, (
                            'Duplicate %s' % sublayer_name)
                        names_to_specs[sublayer_name] = sublayer

                # If block provides an endpoint, subsequent layers can
                # refer to the block name as parent.
                if 'endpoint' in layer_params:
                    block_endpoints[block_name] = '%s/%s' % (
                        block_name, layer_params['endpoint'])
            else:
                layer_params['parents'] = _parents(layer_params['parents'])
                layer = LayerSpec(layer_name, layer_params)
                assert layer_name not in names_to_specs, ('Duplicate %s' %
                                                          layer_name)
                names_to_specs[layer_name] = layer

        # Add edges.
        for layer_name, layer_spec in names_to_specs.items():
            for parent_name in _parents(layer_spec['parents']):
                assert parent_name in names_to_specs, (
                    'Parent layer %s of %s '
                    'does not have a LayerSpec object.' % (parent_name,
                                                           layer_name))
                names_to_specs[parent_name].outbounds.append(layer_spec)
                layer_spec.inbounds.append(names_to_specs[parent_name])

        graphwalker = GraphWalker(names_to_specs)
        self.nested_list = graphwalker.start(names_to_specs['data'])
        self._create_topology_order()
        logger.debug(self.nested_list)
        logger.debug(self._topological_order)
        if self.attach_ops:
            self._attach_layer_op()


class GraphWalker(object):
    def __init__(self, names_to_nodes):
        self.names_to_nodes = names_to_nodes
        self.indegrees = dict()
        self.joints = set()
        for node in names_to_nodes.values():
            self.indegrees[node] = len(node.inbounds)
            if self.indegrees[node] > 1:
                self.joints.add(node)

    def start(self, starting_node):
        nested_list, _ = self.nested_list_till_joints([], [starting_node])
        return nested_list

    def nested_list_till_joints(self, history, frontiers):
        """Returns current history and encountered joint"""
        if len(frontiers) == 0:
            return history, []

        # If all frontiers are joints, stop DFS.
        # The returned path does not include the joints.
        all_joints = True
        for node in frontiers:
            all_joints = all_joints and (node in self.joints)
        if all_joints:
            return history, frontiers

        if len(frontiers) == 1:
            # Simply chain the layer and continue
            history.append(node)
            for x in node.outbounds:
                assert self.indegrees[x] > 0
                self.indegrees[x] -= 1
            frontiers = node.outbounds
            return self.nested_list_till_joints(history, frontiers)
        else:
            # When the frontier has more than one nodes, we explore from each
            # frontier individually until they reach a common joint.
            supernode = []
            merged_joints = set()
            for node in frontiers:
                # The search will stop at joints.
                sub_list, joints = self.nested_list_till_joints([], [node])
                if len(sub_list) > 0:
                    supernode.append(sub_list)
                for joint in joints:
                    if self.indegrees[joint] == 0:
                        merged_joints.add(joint)
            if len(supernode) > 0:
                history.append(tuple(supernode))

            # Reached joints becomes the new frontiers.
            for joint in merged_joints:
                self.joints.remove(joint)
            frontiers = list(merged_joints)
            return self.nested_list_till_joints(history, frontiers)
