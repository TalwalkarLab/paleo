"""Model spec convertors.

Converts model spec from other platforms to the Paleo JSON format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging

FORMAT = "%(levelname)s %(pathname)s:%(lineno)d] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("paleo:convertor")
logger.setLevel(logging.INFO)

# NCHW -> NHWC
NCHW_2_NHWC = (0, 3, 1, 2)


class ModelSpec(object):
    def __init__(self, name):
        self.model_spec = dict()
        self.model_spec['name'] = name
        self.model_spec['layers'] = dict()

    def add_layer(self, layer_name, layer_params):
        assert 'parents' in layer_params, ('"parents" not found for layer %s' %
                                           layer_name)
        assert layer_name not in self.model_spec['layers'], (
            'layer %s already exist' % layer_name)
        self.model_spec['layers'][layer_name] = layer_params

    def as_dict(self):
        """Returns the model spec as a dictionary."""
        return self.model_spec

    def save_json(self, filename):
        """Save the model spec as a json file."""
        with open(filename, 'w') as f:
            json.dump(self.model_spec, f)


class CaffeConvertor(object):
    def __init__(self):
        self._LAYER_TYPE_MAPPING = {'Eltwise': 'Elementwise',
                                    'Data': 'Input',
                                    'Concat': 'Concatenate',
                                    'SoftmaxWithLoss': 'Softmax'}
        pass

    def convert(self, prototxt_filename):
        from paleo.third_party import caffe_pb2
        from google.protobuf.text_format import Merge

        net = caffe_pb2.NetParameter()
        with open(prototxt_filename, 'r') as f:
            Merge(str(f.read()), net)

        paleo_net = ModelSpec(net.name)
        caffe_data_layer_name = 'data'

        if len(net.input_dim) > 0:
            # Read input layer spec from root level fields.
            layer_name = 'data'
            layer_params = dict()
            layer_params['type'] = 'Input'

            batch, channel, dim0, dim1 = net.input_dim
            layer_params['tensor'] = [batch, dim0, dim1, channel]  # to NHWC
            layer_params['parents'] = []
            paleo_net.add_layer(layer_name, layer_params)

        for layer in net.layer:
            layer_name = layer.name
            layer_params = dict()
            layer_params['type'] = self._LAYER_TYPE_MAPPING.get(layer.type,
                                                                layer.type)
            layer_params['parents'] = []
            if layer_params['type'] == 'Input':
                layer_name = 'data'
                caffe_data_layer_name = layer.name
                # FIXME: the input size is not known if not provided.

            if len(layer.bottom) > 0:
                for name in layer.bottom:
                    layer_params['parents'].append(
                        str(name) if name != caffe_data_layer_name else 'data')

            # For layers spec:
            #   http://caffe.berkeleyvision.org/tutorial/layers.html

            # Skip test layers
            if layer.include:
                if len(layer.include) > 0:
                    if layer.include[0].phase == 1:  # test
                        continue

            if layer.type == 'Input':
                batch, channel, dim0, dim1 = layer.input_param.shape[0].dim
                layer_params['tensor'] = [batch, dim0, dim1, channel]  # NHWC
                layer_params['parents'] = []
            elif layer.type == 'Convolution':
                param = layer.convolution_param
                c_in = -1  # Input channel to be derived automatically
                c_out = param.num_output

                if len(param.kernel_size) > 0:
                    kh = param.kernel_size[0]
                    kw = kh
                else:
                    if param.HasField('kernel_h'):
                        kh = param.kernel_h
                    if param.HasField('stride_w'):
                        kw = param.kernel_w

                sh, sw = 1, 1
                if len(param.stride) > 0:
                    sh = param.stride[0]
                    sw = sh
                else:
                    if param.HasField('stride_h'):
                        sh = param.stride_h
                    if param.HasField('stride_w'):
                        sw = param.stride_w

                padding = 'VALID'
                if len(param.pad) > 0:
                    pad = param.pad[0]
                    if pad != 0:
                        padding = 'SAME'
                else:
                    pad_h = param.pad_h
                    pad_w = param.pad_w
                    if pad_h != 0 or pad_w != 0:
                        padding = 'SAME'

                layer_params['filter'] = [kh, kw, c_in, c_out]
                layer_params['strides'] = [1, sh, sw, 1]
                layer_params['padding'] = padding

            elif layer.type == 'InnerProduct':
                c_out = layer.inner_product_param.num_output
                layer_params['num_outputs'] = c_out

            elif layer.type == 'SoftmaxWithLoss':
                # Only use the first parent to support DenseNet spec.
                layer_params['parents'] = [layer_params['parents'][0]]

            elif layer.type == 'Accuracy':
                # Only use the first parent to support DenseNet spec.
                layer_params['parents'] = [layer_params['parents'][0]]

            elif layer.type == 'Pooling':
                param = layer.pooling_param
                if param.HasField('kernel_size'):
                    kh = param.kernel_size
                    kw = kh

                sh, sw = 1, 1
                if param.HasField('stride'):
                    sh = param.stride
                    sw = sh

                padding = 'VALID'
                if param.HasField('pad'):
                    pad = param.pad
                    if pad != 0:
                        padding = 'SAME'

                if param.HasField('pool'):
                    pool_type = param.pool
                    if pool_type == 'AVE':
                        layer_params['type'] = 'AvgPool'

                layer_params['ksize'] = [1, kh, kw, 1]
                layer_params['strides'] = [1, sh, sw, 1]
                layer_params['padding'] = padding
            elif layer.type == 'Dropout':
                param = layer.dropout_param
                layer_params['dropout_keep_prob'] = param.dropout_ratio
            elif layer.type == 'Concat':
                param = layer.concat_param
                layer_params['dim'] = NCHW_2_NHWC[param.axis]
            else:
                if (len(layer.bottom) == 1 and len(layer.top) == 1 and
                        layer.bottom[0] == layer.top[0]):
                    # FIXME: layer activation ops.
                    continue
                logging.warning('Dummy layer {}, {}'.format(layer.name,
                                                            layer.type))

            paleo_net.add_layer(layer_name, layer_params)
        return paleo_net
