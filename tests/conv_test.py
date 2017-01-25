"""Tests for Conv2D."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from paleo.layers import conv


class Conv2DTest(unittest.TestCase):
    def setUp(self):
        self.layer = conv.Conv2d(
            name='alex_v2/conv1',
            inputs=[128, 224, 224, 3],
            filters=[11, 11, 3, 96],
            strides=[1, 4, 4, 1],
            padding='VALID')

    def test_output_shape(self):
        self.assertEqual(self.layer.outputs, [128, 54, 54, 96])

    def test_grad_inputs(self):
        layer = self.layer
        back_layer = layer.gradients()
        self.assertListEqual(back_layer.outputs, layer.inputs)

    def test_grad_filters(self):
        layer = self.layer
        back_layer = layer.gradients(wrt='filters')
        self.assertListEqual(back_layer.outputs[1:3], layer.filters[0:2])


class Conv2DValidStride1Test(unittest.TestCase):
    def setUp(self):
        self.layer = conv.Conv2d(
            name='alex_v2/fc6',
            inputs=[128, 5, 5, 256],
            filters=[5, 5, 256, 4096],
            strides=[1, 1, 1, 1],
            padding='VALID')

    def test_grad_inputs(self):
        layer = self.layer
        back_layer = layer.gradients()
        self.assertListEqual(back_layer.outputs, layer.inputs)

    def test_grad_filters(self):
        layer = self.layer
        back_layer = layer.gradients(wrt='filters')
        self.assertListEqual(back_layer.outputs[1:3], layer.filters[0:2])


class Conv2DSameStride1Test(unittest.TestCase):
    def setUp(self):
        self.layer = conv.Conv2d(
            name='alex_v2/conv2',
            inputs=[128, 26, 26, 64],
            filters=[5, 5, 64, 192],
            strides=[1, 1, 1, 1],
            padding='SAME')

    def test_grad_inputs(self):
        layer = self.layer
        back_layer = layer.gradients()
        self.assertListEqual(back_layer.outputs, layer.inputs)

    def test_grad_filters(self):
        layer = self.layer
        back_layer = layer.gradients(wrt='filters')
        self.assertListEqual(back_layer.outputs[1:3], layer.filters[0:2])


class Conv2DValidStrideGt1Test(unittest.TestCase):
    def setUp(self):
        self.layer = conv.Conv2d(
            name='inception/Conv2d_1a_3x3',
            inputs=[64, 299, 299, 3],
            filters=[3, 3, 3, 32],
            strides=[1, 2, 2, 1],
            padding='VALID')

    def test_grad_inputs(self):
        layer = self.layer
        back_layer = layer.gradients()
        self.assertListEqual(back_layer.outputs, layer.inputs)

    def test_grad_filters(self):
        layer = self.layer
        back_layer = layer.gradients(wrt='filters')
        self.assertListEqual(back_layer.outputs[1:3], layer.filters[0:2])


class Conv2DAsymmetricTest(unittest.TestCase):
    def setUp(self):
        self.layer = conv.Conv2d(
            name='Inception/Mixed_6b/Branch_1/Conv2d_0b_1x7',
            inputs=[64, 17, 17, 128],
            filters=[1, 7, 128, 128],
            strides=[1, 1, 1, 1],
            padding='SAME')

    def test_grad_inputs(self):
        layer = self.layer
        back_layer = layer.gradients()
        self.assertListEqual(back_layer.outputs, layer.inputs)

    def test_grad_filters(self):
        layer = self.layer
        back_layer = layer.gradients(wrt='filters')
        self.assertListEqual(back_layer.outputs[1:3], layer.filters[0:2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
