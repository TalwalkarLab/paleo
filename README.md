# Paleo

Paleo is an analytical model to estimate the scalability and performance of deep learning systems.
It can be used to
  - efficiently explore the space of scalable deep learning systems,
  - quickly diagnose their eﬀectiveness for a given problem instance.

[Live demo](https://talwalkarlab.github.io/paleo/)

## Dependencies

- Python 2.7 and packages in `requirements.txt`
- cuDNN 4
- TensorFlow 0.9 (optional)

Tested on Ubuntu 14.04 workstations.

## Usage

Paleo provides the following commands.


- `summary` prints static characteristics of the given model architecture.
- `simulate` simulates the scalability of the given model architecture.
- `profile` performs layer-wise profiling of the given model architecture.
- `fullpass` runs full-pass profiler with TensorFlow.

To get help on arguments to each command:

    ./paleo.sh [command] --help

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
Under review for ICLR 2017.

[1]: https://openreview.net/forum?id=SyVVJ85lg

## License

Apache 2.0
