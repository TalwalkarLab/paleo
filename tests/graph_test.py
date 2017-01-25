"""Tests for Graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import six

from paleo import graph


class GraphTest(unittest.TestCase):
    def setUp(self):
        self.graph = graph.OperationGraph(attach_ops=False)

    def _to_strings(self, l):
        """Convert all nested objects in l to string."""
        if isinstance(l, list):
            return [self._to_strings(e) for e in l]
        elif isinstance(l, tuple):
            return tuple([self._to_strings(e) for e in l])
        else:
            return str(l)

    def tearDown(self):
        self.graph = None
        pass

    def test_graph_single_tower(self):
        """Shall be able to parse and order a single tower network."""
        single_tower = """{
            "name" : "test",
            "layers" : {
                "data": {
                    "parents": []
                },
                "conv1": {
                    "parents": ["data"]
                },
                "conv2": {
                    "parents": ["conv1"]
                }
            }
        }
        """
        self.graph.load_from_string(single_tower)
        #self.assertEqual(str(self.graph.nested_list), "[data, conv1, conv2]")
        self.assertListEqual(
            self._to_strings(self.graph.nested_list),
            ['data', 'conv1', 'conv2'])

    def test_dependency(self):
        two_towers = """{
            "name" : "test",
            "layers" : {
                "data": {
                    "parents": []
                },
                "conv1": {
                    "parents": ["data"]
                },
                "conv2": {
                    "parents": ["data"]
                },
                "output": {
                    "parents" : ["conv1", "conv2"]
                }
            }
        }
        """
        self.graph.load_from_string(two_towers)
        nested_list = self._to_strings(self.graph.nested_list)
        self.assertEqual(nested_list[0], 'data')
        six.assertCountEqual(self, nested_list[1], (['conv1'], ['conv2']))
        self.assertEqual(nested_list[2], 'output')

    def test_dependency2(self):
        two_towers = """{
            "name" : "test",
            "layers" : {
                "data": {
                    "parents": []
                },
                "conv1": {
                    "parents": ["data"]
                },
                "output": {
                    "parents" : ["data", "conv1"]
                }
            }
        }
        """
        self.graph.load_from_string(two_towers)
        nested_list = self._to_strings(self.graph.nested_list)
        self.assertEqual(nested_list[0], 'data')
        six.assertCountEqual(self, nested_list[1], (['conv1'], ))
        self.assertEqual(nested_list[2], 'output')

    def test_block(self):
        two_towers = """{
            "name" : "test",
            "layers" : {
                "data": {
                    "parents": []
                },
                "conv1": {
                    "parents": ["data"]
                },
                "conv2": {
                    "type": "Block",
                    "parents": ["conv1"],
                    "endpoint": "concat",
                    "layers": {
                        "conv2a": {
                            "parents": []
                        },
                        "conv2b" : {
                            "parents": []
                        },
                        "concat": {
                            "parents": ["conv2a", "conv2b"]
                        }
                    }
                },
                "output": {
                    "parents" : ["conv2"]
                }
            }
        }
        """
        self.graph.load_from_string(two_towers)
        nested_list = self._to_strings(self.graph.nested_list)
        self.assertEqual(nested_list[0], 'data')
        self.assertEqual(nested_list[1], 'conv1')
        six.assertCountEqual(self, nested_list[2], (['conv2/conv2a'],
                                                    ['conv2/conv2b']))
        self.assertEqual(nested_list[3], 'conv2/concat')
        self.assertEqual(nested_list[4], 'output')

    def test_model_parallel(self):
        two_towers = """{
            "name" : "test",
            "layers" : {
                "data": {
                    "parents": []
                },
                "conv1": {
                    "parents": ["data"]
                },
                "conv2": {
                    "type": "ModelParallel",
                    "parents": ["conv1"],
                    "splits": 2,
                    "layers": {
                        "conv2a": {
                            "parents": []
                        },
                        "mix": {
                            "parents": ["conv2a@all"]
                        }
                    }
                },
                "output": {
                    "parents" : ["conv2/mix@all"]
                }
            }
        }
        """
        self.graph.load_from_string(two_towers)
        nested_list = self._to_strings(self.graph.nested_list)
        self.assertEqual(nested_list[0], 'data')
        self.assertEqual(nested_list[1], 'conv1')
        six.assertCountEqual(self, nested_list[2], (['conv2/conv2a@0'],
                                                    ['conv2/conv2a@1']))
        six.assertCountEqual(self, nested_list[3],
                             (['conv2/mix@0'], ['conv2/mix@1']))
        self.assertEqual(nested_list[4], 'output')


if __name__ == '__main__':
    unittest.main()
