"""Graph Representation of Deep Neural Network Architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
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
        self.parents = []

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

    def __init__(self, filename=None):
        super(OperationGraph, self).__init__()
        self.in_degree = defaultdict(int)
        self.adj_list = defaultdict(list)
        self.nested_list = None
        self._topolopy_order = None
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
        return self._topolopy_order

    def _create_toplogy_order(self):
        """
        Expand the network to a plain list.
        The list is in topology order.
        """

        def flatten(layer):
            if isinstance(layer, (tuple, list)):
                _layer_list = []
                for l in layer:
                    _layer_list.extend(flatten(l))
                return _layer_list
            else:
                return [layer]

        if self._topolopy_order is None:
            self._topolopy_order = []
            for layer in self.nested_list:
                self._topolopy_order.extend(flatten(layer))

    def _attach_layer_op(self):
        """Flatten the list in topology order."""
        names_to_layers = dict()
        for layer_spec in self.topology_order:
            if len(layer_spec['parents']) == 1:
                parent_name = layer_spec['parents'][0]
                inputs = names_to_layers[parent_name].layer_op.outputs
            else:
                inputs = []
                for parent_name in layer_spec['parents']:
                    inputs.append(names_to_layers[
                        parent_name].layer_op.outputs)

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
            elif layer_spec['type'] == 'Pooling':
                layer = layers.Pool2d(
                    layer_spec.name,
                    inputs,
                    layer_spec['ksize'],
                    layer_spec['strides'],
                    layer_spec['padding'],
                    pool_type='max')
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
            elif layer_spec['type'] == 'Softmax':
                layer = layers.Softmax(layer_spec.name, inputs,
                                       layer_spec['num_classes'])
            else:
                raise ValueError('Cannot create layer object for %s,'
                                 '%s is an unknown layer type.' %
                                 (layer_spec.name, layer_spec['type']))
            if layer:
                logger.debug('%s inputs: %s  ouputs: %s' %
                             (layer.name, layer.inputs, layer.outputs))
                layer_spec.parents.extend([names_to_layers[p]
                                           for p in layer_spec['parents']])
                layer.parents = layer_spec['parents']
                layer_spec.attach_op(layer)
                names_to_layers[layer_spec.name] = layer_spec

    def _create_graph(self, net):
        # Convert JSON into a graph
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)

        nodes = dict()  # layer_name -> LayerSpec object
        # Shortcuts, allow use block_name as parent.
        # endpoint will be used instead.
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
                        'Parent %s is not splited.')
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

        # Transform all layers into a LayerSpec object.
        for layer_name, layer_params in net['layers'].items():
            if layer_params.get('type', None) in ['Block', 'ModelParallel']:
                is_model_parallel = (layer_params['type'] == 'ModelParallel')
                block_name = layer_name
                block_parents = _parents(layer_params['parents'])

                # For model paralle, we repeat the specified layers K times.
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
                            sublayer_parents = block_parents
                        else:
                            sublayer_parents = map(
                                lambda n: '%s/%s' % (block_name, n),
                                sublayer_params['parents'])
                            sublayer_parents = _parents(sublayer_parents, s)

                        sublayer.params['parents'] = sublayer_parents

                        assert sublayer_name not in nodes, ('Duplicate %s' %
                                                            sublayer_name)
                        nodes[sublayer_name] = sublayer

                # If block provides an endpoint, subsequent layers can
                # refer to the block name as parent.
                if 'endpoint' in layer_params:
                    block_endpoints[block_name] = '%s/%s' % (
                        block_name, layer_params['endpoint'])
            else:
                layer_params['parents'] = _parents(layer_params['parents'])
                layer = LayerSpec(layer_name, layer_params)
                assert layer_name not in nodes, ('Duplicate %s' % layer_name)
                nodes[layer_name] = layer

        # Add edges.
        for layer_name, layer_spec in nodes.items():
            # add edges to the adjacent list.
            for parent_name in layer_spec['parents']:
                adj_list[nodes[parent_name]].append(nodes[layer_name])
                in_degree[nodes[layer_name]] += 1

        self.in_degree = in_degree
        self.adj_list = adj_list
        self.nested_list = self.net_graph(nodes['data'], in_degree, adj_list)
        self._create_toplogy_order()
        self._attach_layer_op()

    def net_graph(self, starting_node, in_degree, adj_list):
        self.in_degree = in_degree
        self.adj_list = adj_list

        # Find all bottleneck nodes, i.e in degree >= 2.
        bottlenecks = set([node for node in self.in_degree
                           if self.in_degree[node] >= 2])

        # Building the graph from the starting nodes
        layer_list, end = self.net_graph_dfs([], [starting_node], bottlenecks)
        return layer_list

    def net_graph_dfs(self, history, frontiers, bottlenecks):
        if len(frontiers) == 0:
            # Done when where are no nodes to explore
            return history, []

        if len(frontiers) == 1:
            layer = frontiers[0]
            # Stop DFS if hit a bottleneck.
            if layer in bottlenecks:
                return history, [layer]

            # simply chain the layer
            history.append(layer)

            # continue
            frontiers = [child for child in self.adj_list[layer]]
            return self.net_graph_dfs(history, frontiers, bottlenecks)

        encounter_bottlenecks, bns = False, []
        for node in frontiers:
            if node in bottlenecks:
                bns.append(node)
                encounter_bottlenecks = True
        if encounter_bottlenecks:
            return history, bns

        supernode = []
        encountered_bottlenecks = []
        for node in frontiers:
            # the search will stop at bottlenecks
            # Now we assume they stop at the same bottleneck
            sub_chain, bns = self.net_graph_dfs([], [node], bottlenecks)
            supernode.append(sub_chain)
            encountered_bottlenecks.extend(bns)
        history.append(tuple(supernode))

        # Encountered bottlenecks becomes the new frontiers.
        encountered_bottlenecks = list(set(encountered_bottlenecks))

        # Remove bottlenecks
        for bn in encountered_bottlenecks:
            if bn in bottlenecks:
                bottlenecks.remove(bn)

        frontiers = encountered_bottlenecks
        return self.net_graph_dfs(history, frontiers, bottlenecks)
