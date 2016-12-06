# Paleo

Paleo is an analytical performance modeling tool for deep neural networks. It
can be used to
  - efficiently explore the space of scalable deep learning systems,
  - quickly diagnose their eﬀectiveness for a given problem instance.

## Dependencies

- Python 2.7 and packages in `requirements.txt`
- cuDNN 4
- TensorFlow 0.9<sup>[1](#footnote1)</sup>

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

Paleo uses a special json format to for model architecture specification.
Predefined architectures can be found under `nets/` directory.

| Model Architecture  | Files               |
| ------------------- | ------------------  |
| AlexNet v2          | `alex_v2.json`      |
| Inception v3        | `inception_v3.json` |
| NiN                 | `nin.json`          |
| Overfeat            | `overfeat.json`     |
| VGG-16              | `vgg16.json`        |

**Hardware**

Predefined hardware specificiations are in `paleo/device.py`.

## Reference Paper

Hang Qi, Evan R. Sparks, and Ameet Talwalkar.
[Paleo: A Performance Model for Deep Neural Networks][1].
Under review for ICLR 2017.

[1]: https://openreview.net/forum?id=SyVVJ85lg

## License

Apache 2.0

<small><a name='#footnote1'>1</a> The current version depends on TensorFlow for
gathering emperical ground  truth. We will make this optional in later releases.
</small>
