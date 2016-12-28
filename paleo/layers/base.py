"""The base class of estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod, abstractproperty


class BaseLayer(object):
    """Base class for estimator. """

    def __init__(self, name, layertype):
        self._name = name
        self._layertype = layertype
        self._inputs = None
        self._outputs = None
        self._parents = None

    def __repr__(self):
        return '%s\t%s  %s' % (self.name, self.outputs,
                               self.additional_summary())

    @property
    def parents(self):
        return self._parents

    @parents.setter
    def parents(self, val):
        self._parents = val

    @property
    def batch_size(self):
        assert self._inputs[0] == self._outputs[0]
        return self._inputs[0]

    @batch_size.setter
    def batch_size(self, batch_size):
        self._inputs[0] = batch_size
        self._outputs[0] = batch_size

    @abstractproperty
    def name(self):
        """The name of this layer."""
        return self._name

    @abstractproperty
    def layertype(self):
        """The type of this layer."""
        return self._layertype

    @abstractmethod
    def additional_summary(self):
        """Returns the additional summary when print the layer as string."""
        return ""

    @abstractproperty
    def inputs(self):
        """Returns the shape of input tensor of this layer."""
        return self._inputs

    @abstractproperty
    def outputs(self):
        """Returns the shape of output tensor for this layer."""
        return self._outputs

    @abstractproperty
    def weights_in_bytes(self):
        """Returns the size of weights in this layer in bytes."""
        return 0

    @abstractproperty
    def num_params(self):
        """Returns the number of trainable parameters in this layer."""
        return 0


class Generic(BaseLayer):
    """Estimator for Generic layers. """

    def __init__(self, name, inputs, type):
        """Initialize estimator. """
        super(Generic, self).__init__(name, 'generic_{}'.format(type))
        self._inputs = inputs
        self._outputs = list(self._inputs)

    def additional_summary(self):
        return 'Generic layer: %s' % self._layertype

    def memory_in_bytes(self):
        """Returns weights."""
        return 0
