"""The base of estimator"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import logging


class TimeMeasure(object):
    def __init__(self, comp_time=0, comm_time=0, total_time=None):
        self._comp_time = comp_time
        self._comp_time_std = 0
        self._comm_time = comm_time
        self._comm_time_std = 0
        if not total_time:
            total_time = comp_time + comm_time
        self._total_time = total_time
        self._total_time_std = 0

    def __repr__(self):
        return '%f (comp=%f, comm=%f)' % (self.total_time, self.comp_time,
                                          self.comm_time)

    @property
    def comp_time(self):
        return self._comp_time

    @comp_time.setter
    def comp_time(self, val):
        self._comp_time = val
        self._total_time = self._comp_time + self._comm_time

    @property
    def comm_time(self):
        return self._comm_time

    @comm_time.setter
    def comm_time(self, val):
        self._comm_time = val
        self._total_time = self._comp_time + self._comm_time

    @property
    def total_time(self):
        return self._total_time

    @property
    def lowerbound(self):
        """A lowerbound under perfect pipelining."""
        return max(self._comp_time, self._comm_time)

    @total_time.setter
    def total_time(self, val):
        self._total_time = val

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, int):
            return TimeMeasure(total_time=self.total_time + other)
        res = TimeMeasure(
            comp_time=self.comp_time + other.comp_time,
            comm_time=self.comm_time + other.comm_time,
            total_time=self.total_time + other.total_time)
        # Sometimes the time measure may not have detailed items,
        # in this case we set total time directly.
        # expected_total_time = self.total_time + other.total_time
        # if res.total_time != expected_total_time:
        #     res.comm_time = 0
        #     res.comp_time = 0
        #     res.total_time = expected_total_time
        return res

    def __sub__(self, other):
        res = TimeMeasure(
            comp_time=self.comp_time - other.comp_time,
            comm_time=self.comm_time - other.comm_time,
            total_time=self.total_time - other.total_time)
        return res


class ProfilerOptions(object):
    """The options for profilers"""

    def __init__(self):
        self.direction = 'forward'  # forward, backward
        self.gradient_wrt = 'data'  # data, filter, None
        self.num_warmup = 10
        self.num_iter = 50
        self.use_cudnn_heuristics = True

        # By default we don't include bias and activation.
        # this will make layer-wise comparison easier.
        self.include_bias_and_activation = False

        # Platform percent of peek.
        self.ppp_comp = 1.0
        self.ppp_comm = 1.0


class BaseProfiler(object):
    """The base class of profilers. """

    def __init__(self, name, options):
        self._name = name
        self._logger = logging.getLogger('paleo.profilers.' + self._name)
        self._msg = ''
        self._options = options

    @property
    def message(self):
        return self._msg

    @message.setter
    def message(self, msg):
        self._msg = msg

    def clear_msg(self):
        self._msg = ''

    @property
    def options(self):
        return self._options

    @property
    def name(self):
        if self._name:
            return self._name
        return "BaseProfiler"

    @abstractmethod
    def profile(self, layer, num_iter=50, num_warmup=10, direction='forward'):
        """Profiles the given layer and returns the time and std across num_iter
        trails."""
        return TimeMeasure()
