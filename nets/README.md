# Network Architecture Specification

Paleo uses a JSON format for specifying network architectures.
There are only two root level objects


* `name`: The name of network
* `layers`: A dictionary specifiying layers
            `{"layer name": layer parameters}`

The following is an example skeleton:

    {
        "name": "AlexNet",
        "layers": {
            "data": {
                "parents": [],
                "type": "Input",
                "tensor": [128, 224, 224, 3]
            },
            ...
        }
    }

## Layers

Each layer specification must have the structure:

    "layer_name": {
        "parents": [],        // list of names of parent layers
        "type": "LayerType",  // type of the layer
        ...                   // layer parameters
    }

Additional parameters depend on the type of the layer. We list several types
of layers and their parameters as follows. Note that the following is not an
exhaustive list. At this point, Paleo mainly focuses on commonly used
structural layers and computationaly intensive layers. More types of layers
are to be supported gradually (issues and PRs are welcome).

### Input

    "data": {
        "type": "Input"
        "parents": [],            // parents must be empty
        "tensor": [N, H, W, C]    // shape of the input tensor
    }

### Convolution

    "conv1": {
        "type": "Convolution",
        "filter": [3, 3, 3, 32],       // [H, W, channel_in, channel_out]
        "padding": "VALID",            // "VALID" or "SAME"
        "strides": [1, 2, 2, 1],       // [1, stride_h, stride_w, 1]
        "activation_fn": "relu",
        "normalizer_fn": "batch_norm"
    }

### Pooling

    "pool1": {
        "type": "Pooling",         // "Pooling" or "AvgPool"
        "ksize": [1, 2, 2, 1],     // [1, kernel_h, kernel_w, 1]
        "strides": [1, 2, 2, 1],   // [1, stride_h, stride_w, 1]
        "padding": "VALID"         // "VALID" or "SAME"
    }

### InnerProduct

    "fc1": {
        "type": "InnerProduct",
        "num_outputs": 1024
    }

### Concatenate

    "concat1": {
        "type": "Concatenate",
        "parents": [
            "Branch_0/Conv2d_0a_1x1",  // layers whose outputs are to be concatenated
            "Branch_1/Conv2d_0b_5x5",
            "Branch_2/Conv2d_0c_3x3",
            "Branch_3/Conv2d_0b_1x1"
        ],
        "dim": 3                       // dimension to concatenate
    }

## Blocks

For some complex model, sometimes it's easier to specify the network
architecture with blocks.

Each block provides a namespace for the layers inside it: while the
underlying ids are `block_name/layer_name`, within the block they can be
referred to as `layer_name` as a shortcut.

The following snippet is from `inception_v3.json`:

	"Mixed_5d": {
        "type": "Block",
        "parents": ["Mixed_5c/concat"],
        "endpoint": "concat",   // the endpoint layer determines the output shape of the block
        "layers": {
            "Branch_0/Conv2d_0a_1x1": {
                "parents": [],    // inherit parents from the block
                "type": "Convolution",
                "filter": [1, 1, 288, 64],
                "padding": "SAME",
                "strides": [1, 1, 1, 1],
                "activation_fn": "relu",
                "normalizer_fn": "batch_norm"
            },
            ...
        }
    }

## Model Splits

With model parallelism, the model is partitioned into a number of splits.
Each worker holds one split of the model and communicates with other workers
for sending/fetching activations.

To specify model split, Paleo specification format employees a special type of
Block called _ModelParallel_. In addition to the namespacing in ordinary
blocks, ModelParallel blocks has several new properties:

* `splits`: specifies the number of filter splits (along dim 3)
* `parents`: refer to parent layers in the same worker or other workers
    - `layer_name@self`: use outputs on the same worker
    - `layer_name@all`: fetch outputs from all other workers

The following snippets is from `alex_v2_4gpu.json` where the `fc6` layer
contains 4096 filters split across 4 workers. The `dropout6` layer depends on
feature maps on the same worker, whereas `concat6` layer operates on feature
maps from all 4 workers.

	"parallel_fc67": {
        "type": "ModelParallel",
        "splits": 4,
        "parents": ["pool5"],
        "layers": {
            "fc6": {
                "parents": [],
                "type": "Convolution",
                "filter": [5, 5, 256, 4096],
                "padding": "VALID",
                "strides": [1, 1, 1, 1],
                "activation_fn": "relu"
            },
            "dropout6": {
                "parents": ["fc6@self"],
                "type": "Dropout",
                "dropout_keep_prob": 0.5
            },
            "concat6": {
                "parents": ["dropout6@all"],
                "type": "Concatenate",
                "dim": 3
            },
            ...
        }
    }
