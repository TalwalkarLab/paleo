# Paleo

[![Build Status](https://travis-ci.org/TalwalkarLab/paleo.svg?branch=master)](https://travis-ci.org/TalwalkarLab/paleo)
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
- six
- cuDNN (Optional. Use `--use_only_gemm` to disable cuDNN heuristics)

Tested with Python 2.7, cuDNN v4 on Ubuntu 14.04.

To install Paleo, run the following command in the cloned directory:

    python setup.py install

### Usage

Paleo provides programmatic APIs to retrieve runtime estimations.

The following is an example of estimating SGD executions under strong scaling.

```python
from paleo.profilers import BaseProfiler

class SGDProfiler(BaseProfiler):
    def __init__(self, filename):
        super(SGDProfiler, self).__init__(filename)

    def simulate(self, workers, batch_size=128):
        fwd_time, params_in_bytes = self.estimate_forward(batch_size //
                                                          workers)
        bwd_time = self.estimate_backward(batch_size // workers)
        update_time = self.estimate_update(params_in_bytes)

        t_comp = fwd_time + bwd_time + update_time
        t_comm = self.estimate_comm(workers, params_in_bytes)
        return t_comp + t_comm
```

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
International Conference on Learning Representations (ICLR), 2017.

    @inproceedings{qi17paleo,
      author={Hang Qi and Evan R. Sparks and Ameet Talwalkar},
      booktitle={Proceedings of the International Conference on Learning Representations},
      title={Paleo: A Performance Model for Deep Neural Networks},
      year={2017}
    }

[1]: https://openreview.net/pdf?id=SyVVJ85lg

## License

Apache 2.0
