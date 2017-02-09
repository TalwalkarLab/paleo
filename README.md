# Paleo

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Paleo is an analytical model to estimate the scalability and performance of deep learning systems.
It can be used to
  - efficiently explore the space of scalable deep learning systems,
  - quickly diagnose their eﬀectiveness for a given problem instance.

[Live demo](https://talwalkarlab.github.io/paleo/)

## Getting Started

### Installation

Paleo uses the following dependencies:

- numpy
- click
- six (Not fully compatible with Python 3 yet)
- cuDNN (Optional. Use `--use_only_gemm` to disable cuDNN heuristics)
- TensorFlow (Optional. For empirical comparison only.)

Tested with Python 2.7, cuDNN v4, and TensorFlow 0.9 on Ubuntu 14.04.

To install Paleo, run the following command in the cloned directory:

    python setup.py install

### Usage

Paleo provides the following commands.


- `summary` prints static characteristics of the given model architecture.
- `simulate` simulates the scalability of the given model architecture.
- `profile` performs layer-wise profiling of the given model architecture.
- `fullpass` runs full-pass profiler with TensorFlow.

To get help on arguments to each command:

    paleo [command] --help

To reproduce experiments presented in our paper submission:

    ./scripts/<exp_script>.sh

## Definitions

**Model Architectures**

Paleo uses a [special json format](nets/README.md) to for model architecture
specification. Predefined architectures can be found under the [nets/](nets/)
directory. Paleo also provides a convertor for Caffe prototxt format
(see [nets/caffe/](nets/caffe/) for details).

- AlexNet v2
- Inception v3
- NiN
- Overfeat
- VGG-16
- ResNet-50 (from Caffe spec)
- DenseNet (from Caffe spec)



**Hardware**

Predefined hardware specificiations are in `paleo/device.py`.

## Reference Paper

Hang Qi, Evan R. Sparks, and Ameet Talwalkar.
[Paleo: A Performance Model for Deep Neural Networks][1].
ICLR 2017.

[1]: https://openreview.net/pdf?id=SyVVJ85lg

## License

Apache 2.0
