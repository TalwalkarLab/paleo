"""Model for communication schemes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


class CommunicationBase(object):
    def __init__(self, name, num_nodes, device, ppp_comm):
        self._name = name
        self._num_nodes = num_nodes
        self._device = device
        self._ppp_comm = ppp_comm

    @property
    def name(self):
        return self._name

    @property
    def tree_height(self):
        return math.ceil(math.log(self._num_nodes) / math.log(2))

    def _time_in_communication(self, data_in_bytes):
        """Returns the minimal time required under the given bandwidth in
        milliseconds."""
        bytes_per_seconds = self._device.bandwidth / 8 * 10 ** 9
        bytes_per_seconds = bytes_per_seconds
        return data_in_bytes / bytes_per_seconds * 1000 / self._ppp_comm

    def all_reduce(self, data_in_bytes):
        """
        Returns the minimal time required for all reduce for one iteration.
        """
        pass


class OneToAll(CommunicationBase):
    def __init__(self, num_nodes, device, ppp_comm):
        super(OneToAll, self).__init__('OneToAll', num_nodes, device, ppp_comm)

    def all_reduce(self, data_in_bytes):
        """Bottlenecked by master bandwidth."""
        return 2 * self._time_in_communication(data_in_bytes *
                                               (self._num_nodes - 1))


class TreeAllReduce(CommunicationBase):
    def __init__(self, num_nodes, device, ppp_comm):
        super(TreeAllReduce, self).__init__('TreeAllReduce', num_nodes, device,
                                            ppp_comm)

    def all_reduce(self, data_in_bytes):
        """It takes log2(n) steps for aggregation and log2(n) steps for
        broadcasting."""
        one_link_time = self._time_in_communication(data_in_bytes)
        return 2 * self.tree_height * one_link_time


class ButterflyAllReduce(CommunicationBase):
    def __init__(self, num_nodes, device, ppp_comm):
        super(ButterflyAllReduce, self).__init__('ButterflyAllReduce',
                                                 num_nodes, device, ppp_comm)

    def all_reduce(self, data_in_bytes):
        """It takes log2(n) steps to communicate."""
        one_link_time = self._time_in_communication(data_in_bytes)
        return self.tree_height * one_link_time


class ButterflyMixing(CommunicationBase):
    def __init__(self, num_nodes, device, ppp_comm):
        super(ButterflyMixing, self).__init__('ButterflyMixing', num_nodes,
                                              device, ppp_comm)

    def all_reduce(self, data_in_bytes):
        """It allows partial reduction."""
        one_link_time = self._time_in_communication(data_in_bytes)
        return one_link_time


def get_all_comm_schemes(num_nodes, device, ppp_comm):
    return [
        OneToAll(num_nodes, device, ppp_comm),
        TreeAllReduce(num_nodes, device, ppp_comm),
        ButterflyAllReduce(num_nodes, device, ppp_comm),
        ButterflyMixing(num_nodes, device, ppp_comm)
    ]


def get_comm_scheme(name, num_nodes, device, ppp_comm):
    if name == 'TreeAllReduce':
        return TreeAllReduce(num_nodes, device, ppp_comm)
    elif name == 'OneToAll':
        return OneToAll(num_nodes, device, ppp_comm)
    elif name == 'ButterflyAllReduce':
        return ButterflyAllReduce(num_nodes, device, ppp_comm)
    elif name == 'ButterflyMixing':
        return ButterflyMixing(num_nodes, device, ppp_comm)
