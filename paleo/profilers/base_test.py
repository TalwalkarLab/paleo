from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from paleo.profilers.base import TimeMeasure


class TimeMeasureTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_add(self):
        # Time with sub items.
        t1 = TimeMeasure(comp_time=1, comm_time=3)
        t2 = TimeMeasure(comp_time=2, comm_time=2)
        t_sum = t1 + t2
        self.assertEqual(t_sum.comp_time, t1.comp_time + t2.comp_time)
        self.assertEqual(t_sum.comm_time, t1.comm_time + t2.comm_time)
        self.assertEqual(t_sum.total_time, t1.total_time + t2.total_time)

        # Time without sub items.
        t1 = TimeMeasure(total_time=10)
        t2 = TimeMeasure(comp_time=1, comm_time=4)
        t_sum = t1 + t2
        self.assertEqual(t_sum.total_time, t1.total_time + t2.total_time)

    def test_iadd(self):
        t1 = TimeMeasure(comp_time=1, comm_time=3)
        t2 = TimeMeasure(comp_time=2, comm_time=2)
        t1 += t2
        self.assertEqual(t1.comp_time, 1 + t2.comp_time)
        self.assertEqual(t1.comm_time, 3 + t2.comm_time)
        self.assertEqual(t1.total_time, 4 + t2.total_time)

    def test_sum(self):
        t1 = TimeMeasure(comp_time=1, comm_time=2)
        t2 = TimeMeasure(comp_time=3, comm_time=4)
        t3 = TimeMeasure(comp_time=5, comm_time=6)
        sum_times = sum([t1, t2, t3])
        self.assertEqual(sum_times.comp_time, 9)
        self.assertEqual(sum_times.comm_time, 12)
        self.assertEqual(sum_times.total_time, 21)
        print(sum_times)


if __name__ == '__main__':
    unittest.main(verbosity=2)
