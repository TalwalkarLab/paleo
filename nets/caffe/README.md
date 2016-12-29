# Caffe Models

Paleo supports porting network architectures from Caffe's prototxt files.

Model specs under this directory are ported from the following sources.

| Model     | Source                                |
| --------- | ------------------------------------- |
| ResNet-50 | [KaimingHe/deep-residual-networks][1] |
| DenseNet  | [liuzhuang13/DenseNetCaffe][2]        |

[1]: https://github.com/KaimingHe/deep-residual-networks
[2]: https://github.com/liuzhuang13/DenseNetCaffe

Note that the prototxt file shall specify input data shape in the root fields.
For example:

    name: "ResNet-50"
    input: "data"
    input_dim: 1     # batch size
    input_dim: 3     # input channel
    input_dim: 224   # input image height
    input_dim: 224   # input image width

    layer {
        ...
    }

To convert prototxt to Paleo's json format:

    python paleo/utils/convert.py nets/caffe/resnet-50.prototxt > nets/resnet50.json


