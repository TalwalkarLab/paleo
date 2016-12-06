"""Save conv layer"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json


def save_conv_layer(filename, layer):
    """Save the layer into a individual file that can be run in isolation."""
    output_layer = dict()
    output_layer['name'] = layer.name
    output_layer['layers'] = {
        "data": {
            "parents": [],
            "type": "Input",
            "tensor": layer._inputs
        },
        ("%s" % layer.name): {
            "parents": ["data"],
            "type": "Convolution",
            "filter": layer._filters,
            "padding": layer._padding,
            "strides": layer._strides,
            "bias": layer._filters[-1]
        }
    }
    with open(filename, 'w') as f:
        f.write(json.dumps(output_layer, indent=4))
