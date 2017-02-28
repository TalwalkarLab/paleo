"""Flops-based computation time estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from paleo.profilers.base import BaseProfiler, TimeMeasure

_BYTES_FLOAT = 4
_BYTES_COMPLEX = 8


class FlopsProfiler(BaseProfiler):
    def __init__(self, options, device):
        super(FlopsProfiler, self).__init__('FlopsProfiler', options)
        self._device = device
        if not self._device.is_gpu:
            self.options.use_cudnn_heuristics = False

        if self.options.use_cudnn_heuristics:
            from paleo.profilers import cudnn_profiler as cudnn
            self.cudnn = cudnn

    def profile(self,
                layer,
                current_device=0,
                parent_devices=[0],
                cross_device_bandwidth=None):
        time = TimeMeasure()
        if layer.layertype == 'conv2d':
            time += self._profile_conv2d(layer)
        elif layer.layertype == 'deconv2d':
            time += self._profile_deconv2d(layer)
        elif layer.layertype == 'innerproduct':
            time += self._profile_innerproduct(layer)
        elif layer.layertype == 'pool2d':
            time += self._profile_pool2d(layer)
        elif layer.layertype == 'dropout':
            time += self._profile_dropout(layer)
        else:
            self._logger.debug('Unimplemented \'%s\'' % layer.layertype)

        time += self._estimate_remote_fetch(
            layer, current_device, parent_devices, cross_device_bandwidth)
        return time

    def _estimate_remote_fetch(self, layer, current_device, parent_devices,
                               bandwidth):
        fetch_time = 0
        if len(parent_devices) == 1:
            if current_device != parent_devices[0]:
                num_bytes = np.prod(layer.inputs) * _BYTES_FLOAT
                self._logger.debug('Remote fetch %s from device %s to %s %s' %
                                   (layer.name, parent_devices[0],
                                    current_device, str(layer.inputs)))
                fetch_time += self._estimate_comm_time(
                    num_bytes, bandwidth, ppp=self.options.ppp_comm)
        else:
            for i, parent in enumerate(parent_devices):
                if parent != current_device:
                    if (int(current_device) // 4) != (int(parent) // 4):
                        # penalize 50% bandwidth.
                        bandwidth /= 2
                    num_bytes = np.prod(layer.inputs[i]) * _BYTES_FLOAT
                    self._logger.debug(
                        'Remote fetch %s from device %s to %s %s' %
                        (layer.name, parent, current_device,
                         str(layer.inputs[i])))
                    fetch_time += self._estimate_comm_time(
                        num_bytes, bandwidth, ppp=self.options.ppp_comm)
        return TimeMeasure(comm_time=fetch_time)

    def _estimate_comp_time(self, flops):
        """Return estimated time in milliseconds."""
        if not flops or flops == 0:
            return 0
        gflops = flops / (10 ** 9)
        time_in_sec = gflops / self._device.peek_gflop
        time_in_ms = time_in_sec * (10 ** 3)
        clock_time_in_ms = 1 / self._device.clock / (10 ** 3)
        # PPP for computation, since we estimated form the full time.
        time_in_ms /= self.options.ppp_comp
        return max(clock_time_in_ms, time_in_ms)

    def _estimate_comm_time(self, comm_in_bytes, bandwidth=None, ppp=None):
        if bandwidth is None:
            bandwidth = self._device.mem_bandwidth
        clock_time_in_ms = 1 / self._device.clock / (10 ** 3)
        time_in_ms = comm_in_bytes / 2 ** 30 / bandwidth * 10 ** 3
        # PPP for computation, since we estimated form the full time.
        if ppp is None:
            time_in_ms /= self.options.ppp_comp
        else:
            time_in_ms /= ppp
        return max(clock_time_in_ms, time_in_ms)

    def _profile_innerproduct(self, layer):
        def _innerproduct(X, W, Y):
            assert X[-1] == W[0], ("Shape mismatch: {}x{}={}".format(X, W, Y))
            flops = 2 * np.prod(X) * W[-1]
            comm_time = self._estimate_comm_time(np.prod(X) * _BYTES_FLOAT)
            comm_time += self._estimate_comm_time(np.prod(W) * _BYTES_FLOAT)
            comm_time += self._estimate_comm_time(np.prod(Y) * _BYTES_FLOAT)
            comp_time = self._estimate_comp_time(flops)
            return TimeMeasure(comp_time=comp_time, comm_time=comm_time)

        def _transpose_shape(X):
            return [X[1], X[0]]

        if self.options.direction == 'backward':
            t_data = TimeMeasure()
            t_filter = TimeMeasure()
            assert self.options.gradient_wrt is None or (
                self.options.gradient_wrt in ('data', 'filter'))
            if (not self.options.gradient_wrt or
                    self.options.gradient_wrt == 'data'):
                t_data = _innerproduct(layer.outputs,
                                       _transpose_shape(layer.weights),
                                       layer.inputs)
            if (not self.options.gradient_wrt or
                    self.options.gradient_wrt == 'filter'):
                t_filter = _innerproduct(
                    _transpose_shape(layer.inputs), layer.outputs,
                    layer.weights)
            return t_data + t_filter

        return _innerproduct(layer.inputs, layer.weights, layer.outputs)

    def _profile_deconv2d(self, layer):
        if self.options.direction == 'backward':
            t_data, t_filter = TimeMeasure(), TimeMeasure()
            assert self.options.gradient_wrt is None or (
                self.options.gradient_wrt in ('data', 'filter'))
            if (not self.options.gradient_wrt or
                    self.options.gradient_wrt == 'data'):
                t_data = self._profile_conv2d(
                    layer._transposed, force_fwd=True)
            if (not self.options.gradient_wrt or
                    self.options.gradient_wrt == 'filter'):
                t_filter = self._profile_conv2d_backprop_filter(layer)
            return t_data + t_filter
        # The forward pass of decov is equivalent to the backword pass of
        # the transposed conv.
        return self._profile_conv2d_backprop_data(layer._transposed)

    def _profile_conv2d(self, layer, force_fwd=False):
        if not force_fwd and self.options.direction == 'backward':
            t_data, t_filter = TimeMeasure(), TimeMeasure()
            assert self.options.gradient_wrt is None or (
                self.options.gradient_wrt in ('data', 'filter'))
            if (not self.options.gradient_wrt or
                    self.options.gradient_wrt == 'data'):
                t_data = self._profile_conv2d_backprop_data(layer)
            if (not self.options.gradient_wrt or
                    self.options.gradient_wrt == 'filter'):
                t_filter = self._profile_conv2d_backprop_filter(layer)
            return t_data + t_filter
        # Forward pass.
        t_conv, t_bias, t_relu = TimeMeasure(), TimeMeasure(), TimeMeasure()
        if not self.options.use_cudnn_heuristics:
            self.message = 'Heuristic disabled.'
            t_conv = self._profile_conv2d_gemm(layer)
        else:
            # Use cudnn heuristics to get the algorithm used.
            algo, ws_size = self.cudnn.get_convolution_fwd_algorithm(
                layer.inputs, layer.filters, layer.strides, layer._pad_h,
                layer._pad_w)
            algorithm_name = self.cudnn.CONV_ALGO_FWD_NAME[algo]
            self.message = '%s %f MB' % (algorithm_name, ws_size / 10 ** 6)

            if layer.filters[0:2] == [1, 1]:
                self.message = 'GEMM 1x1'
                t_conv = self._profile_conv2d_gemm(layer)
            elif algorithm_name == 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM':
                t_conv = self._profile_conv2d_gemm(layer)
            elif (algorithm_name ==
                  'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM'):
                t_conv = self._profile_conv2d_gemm(layer)
            elif algorithm_name == 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM':
                t_conv = self._profile_conv2d_gemm(layer)
            elif algorithm_name == 'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT':
                t_conv = self._profile_conv2d_gemm(layer)
            elif algorithm_name == 'CUDNN_CONVOLUTION_FWD_ALGO_FFT':
                t_conv = self._profile_conv2d_fft(layer)
            elif algorithm_name == 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING':
                t_conv = self._profile_conv2d_fft(layer, tiling=True)
            elif algorithm_name == 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD':
                self._logger.warning('Unsupported algorithm: %s' %
                                     algorithm_name)

        if self.options.include_bias_and_activation:
            raise ValueError(
                'We choose not to include bias and activation for'
                'simplicity. And they are by no mean the bottleneck.')
            t_bias = self._profile_bias(layer)
            if layer.activation_fn:
                t_relu = self._profile_relu(layer)
        t_total = t_conv + t_bias + t_relu
        return t_total

    def _profile_conv2d_backprop_data(self, layer):
        dummy_layer = layer.gradients()
        self._logger.debug(
            'BWD DATA: %s (%.2f), %s (%.2f) => %s\n  Padding: %d %d %s\n'
            '  Stride: %s' %
            (dummy_layer.inputs, dummy_layer.percent_holes_in_inputs,
             dummy_layer.filters, dummy_layer.percent_holes_in_filters,
             dummy_layer.outputs, dummy_layer._pad_h, dummy_layer._pad_w,
             dummy_layer.padding, str(dummy_layer.strides)))
        assert dummy_layer.outputs == layer.inputs, (
            '%s: Grad shall match original shape [grad] %s != %s [inputs]' %
            (layer.name, dummy_layer.outputs, layer.inputs))

        if not layer.backprop:
            self._logger.debug('Skipped backprop on data for %s' % layer.name)
            return TimeMeasure()
        if not self.options.use_cudnn_heuristics:
            self.message = 'Heuristic disabled.'
            return self._profile_conv2d_gemm(dummy_layer)

        # Use cudnn heuristics to get the algorithm used.
        algo, ws_size = self.cudnn.get_convolution_bwd_data_algorithm(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)
        algorithm_name = self.cudnn.CONV_ALGO_BWD_DATA_NAME[algo]
        self.message = '%s %f MB' % (algorithm_name, ws_size / 10 ** 6)

        if layer.filters[0:2] == [1, 1]:
            self.message = 'GEMM 1x1'
            return self._profile_conv2d_gemm(dummy_layer)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0':
            # implicit gemm
            return self._profile_conv2d_gemm(dummy_layer)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1':
            # precomp gemm
            return self._profile_conv2d_gemm(dummy_layer, additional_mem=True)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT':
            return self._profile_conv2d_fft(dummy_layer)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING':
            return self._profile_conv2d_fft(dummy_layer, tiling=True)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD':
            pass
        self._logger.warning('Unsupported algorithm: %s' % algorithm_name)
        return TimeMeasure()

    def _profile_conv2d_backprop_filter(self, layer):
        # Dummy conv layer in which backprop is implemented.
        dummy_layer = layer.gradients(wrt='filters')
        self._logger.debug(
            'BWD FILTER: %s, %s => %s' %
            (dummy_layer.inputs, dummy_layer.filters, dummy_layer.outputs))
        assert dummy_layer.outputs[1:3] == layer.filters[0:2], (
            '%s: Grad shall match original shape [grad] %s != %s [filters]' %
            (layer.name, dummy_layer.outputs, layer.filters))

        if not self.options.use_cudnn_heuristics:
            self.message = 'Heuristic disabled.'
            return self._profile_conv2d_gemm(dummy_layer)

        # Use cudnn heuristics to get the algorithm used.
        algo, ws_size = self.cudnn.get_convolution_bwd_filter_algorithm(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)
        algorithm_name = self.cudnn.CONV_ALGO_BWD_FILTER_NAME[algo]
        self.message = '%s %f MB' % (algorithm_name, ws_size / 10 ** 6)

        if layer.filters[0:2] == [1, 1]:
            self.message = 'GEMM 1x1'
            return self._profile_conv2d_gemm(dummy_layer)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0':
            return self._profile_conv2d_gemm(dummy_layer)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1':
            return self._profile_conv2d_gemm(dummy_layer)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT':
            return self._profile_conv2d_fft(dummy_layer)
        elif algorithm_name == 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3':
            return self._profile_conv2d_gemm(dummy_layer)
        self._logger.warning('Unsupported algorithm: %s' % algorithm_name)
        return TimeMeasure()

    def _profile_conv2d_gemm(self, layer, additional_mem=False):
        """Returns the flops of convolution 2d.
        Assume
            inputs: [N, H, W, C]
            filters: [H, W, C_in, C_out]
        """
        # Mul and add per output pixel: kernel_w x kernel_h x in_channel
        flops = 2 * layer.filters[0] * layer.filters[1] * layer.filters[2]

        # Flops per output map.
        flops *= layer.outputs[1] * layer.outputs[2] * layer.filters[3]

        # Flops across multiple input patches.
        flops *= layer.inputs[0]

        flops *= (1.0 - layer.percent_holes_in_filters)

        if layer.percent_holes_in_inputs > 0:
            # Move every element in the input tensor.
            flops += 2 * np.prod(layer.inputs) * (
                1.0 - layer.percent_holes_in_inputs)

        self._logger.debug('GEMM flops: %d\n  holes filter: %.2f\n'
                           '  holes inputs: %.2f' %
                           (flops, layer.percent_holes_in_filters,
                            layer.percent_holes_in_inputs))

        input_size = (layer.inputs[0] * (layer.inputs[1] + 2 * layer._pad_h) *
                      (layer.inputs[2] + 2 * layer._pad_w) * layer.inputs[3])
        comm_time = self._estimate_comm_time(input_size * _BYTES_FLOAT)
        comm_time += self._estimate_comm_time(
            np.prod(layer.filters) * (1.0 - layer.percent_holes_in_filters) *
            _BYTES_FLOAT)
        comm_time += self._estimate_comm_time(
            np.prod(layer.outputs) * _BYTES_FLOAT)
        comp_time = self._estimate_comp_time(flops)

        if additional_mem:
            # [batch, out_height, out_width, filter_height * filter_width *
            #  in_channels]
            mem = ((layer.inputs[0] * layer.outputs[1] * layer.outputs[2] *
                    layer.filters[0] * layer.filters[1] * layer.filters[2]) *
                   _BYTES_FLOAT) * 2
            comm_time += self._estimate_comm_time(mem)

            # Read the shared weights for each patch.
            # mem = ((layer.filters[0] * layer.filters[1] * layer.filters[3]) *
            #        layer.outputs[1] * layer.outputs[2]) * _BYTES_FLOAT
            # comm_time += self._estimate_comm_time(mem)

        self._logger.debug('GEMM estimates: %f = %f + %f' %
                           (comp_time + comm_time, comp_time, comm_time))
        return TimeMeasure(comp_time=comp_time, comm_time=comm_time)

    def _profile_bias(self, layer):
        flops = np.prod(layer.outputs)

        comm_time = self._estimate_comm_time(
            np.prod(layer.outputs) * _BYTES_FLOAT)
        comm_time += self._estimate_comm_time(
            np.prod(layer.bias) * _BYTES_FLOAT)
        comm_time += self._estimate_comm_time(
            np.prod(layer.outputs) * _BYTES_FLOAT)
        comp_time = self._estimate_comp_time(flops)

        return TimeMeasure(comp_time=comp_time, comm_time=comm_time)

    def _profile_relu(self, layer):
        # ReLU simply requires 1 FLOP per element.
        flops = np.prod(layer.outputs)

        comm_time = 2 * self._estimate_comm_time(
            np.prod(layer.outputs) * _BYTES_FLOAT)
        comp_time = self._estimate_comp_time(flops)

        return TimeMeasure(comp_time=comp_time, comm_time=comm_time)

    def _profile_conv2d_fft(self, layer, tiling=False):
        """Returns the flops of convolution 2d."""

        def _fft_flops(fft_dim, mode='r2c', filter_1d=False):
            # Note this is not an accurate flops count.
            # Pad to the nearest power of 2.
            tile_size = math.sqrt(fft_dim)
            tile_size_2 = _to_pow2(tile_size)
            f = 2 * tile_size_2 * 5 * tile_size_2 * (math.log(tile_size_2) /
                                                     math.log(2))
            if filter_1d:
                f /= 2
            if mode == 'r2c':
                f /= 2
            return f

        def _to_pow2(n):
            return math.pow(2, math.ceil(math.log(n) / math.log(2)))

        filter_1d = False
        filter_size = layer.filters[0] * layer.filters[1]
        if filter_size in [layer.filters[0], layer.filters[1]]:
            # one of the filter dimension is 1.
            filter_1d = True

        if tiling:
            _TILE_SIZE = 32
            h_tiles = (layer.inputs[1] + _TILE_SIZE - 1) // _TILE_SIZE
            w_tiles = (layer.inputs[2] + _TILE_SIZE - 1) // _TILE_SIZE
            fft_size = _TILE_SIZE ** 2
            num_tiles = h_tiles * w_tiles
            self._logger.info('Tile FFT: %d (%dx%d) 1D: %s' %
                              (num_tiles, _TILE_SIZE, _TILE_SIZE, filter_1d))
            tile_size = _TILE_SIZE
        else:
            # Filters and inputs are padded to the same size.
            # padded_h = (layer.inputs[1] + layer._pad_h +
            #             layer.filters[0] // 2)
            # padded_w = (layer.inputs[2] + layer._pad_w +
            #             layer.filters[1] // 2)
            padded_h = max(layer.inputs[1] + layer._pad_h * 2,
                           layer.filters[0])
            padded_w = max(layer.inputs[2] + layer._pad_w * 2,
                           layer.filters[1])
            fft_size = padded_h * padded_w
            num_tiles = 1
            self._logger.debug('FFT size: %dx%d (%dx%d) 1D: %s' %
                               (_to_pow2(padded_h), _to_pow2(padded_w),
                                padded_h, padded_w, filter_1d))
            tile_size = max(padded_h, padded_w)

        # Calculate time for filters separateily.
        comp_time, comm_time = 0, 0
        comp_time_filters, comm_time_filters = 0, 0

        # (1) fft2d r2c.
        inputs_nc = layer.inputs[0] * layer.inputs[3]
        filters_ck = layer.filters[2] * layer.filters[3]
        comp_time += num_tiles * self._estimate_comp_time(
            inputs_nc * _fft_flops(fft_size, filter_1d=filter_1d))
        comp_time_filters += self._estimate_comp_time(filters_ck * _fft_flops(
            fft_size, filter_1d=filter_1d))

        # Read inputs and write transformed inputs.
        comm_time += num_tiles * self._estimate_comm_time(
            inputs_nc * fft_size * _BYTES_FLOAT)
        comm_time += num_tiles * self._estimate_comm_time(
            inputs_nc * fft_size * _BYTES_COMPLEX)
        # Read filters and write transformed filters.
        # Padding time are not considered here.
        comm_time_filters += self._estimate_comm_time(filters_ck * fft_size *
                                                      _BYTES_FLOAT)
        if filter_1d:
            comm_time_filters += self._estimate_comm_time(
                filters_ck * tile_size * _BYTES_COMPLEX)
        else:
            comm_time_filters += self._estimate_comm_time(
                filters_ck * fft_size * _BYTES_COMPLEX)

        # (2) Elementwise multiplication.
        # Complex number: add 2, muliply 4
        comp_time += num_tiles * self._estimate_comp_time(
            4 * layer.inputs[0] * layer.inputs[3] * layer.filters[3] *
            fft_size)

        # Pipe: Writing results while doing FFT for the next set of tiles?
        # Read transformed inputs.
        comm_time += num_tiles * self._estimate_comm_time(
            inputs_nc * fft_size * _BYTES_COMPLEX)
        # Read transformed filters.
        if filter_1d:
            comm_time_filters += num_tiles * self._estimate_comm_time(
                filters_ck * tile_size * _BYTES_COMPLEX)
        else:
            comm_time_filters += num_tiles * self._estimate_comm_time(
                filters_ck * fft_size * _BYTES_COMPLEX)
        # Write results.
        comm_time += num_tiles * self._estimate_comm_time(
            layer.inputs[0] * layer.filters[3] * fft_size * _BYTES_COMPLEX)

        # Assume additional memory needed is only for one tile.
        # FIXME: how to match this number with cuDNN suggestion?
        mem_inputs = inputs_nc * fft_size * _BYTES_COMPLEX
        mem_filters = filters_ck * fft_size * _BYTES_COMPLEX
        mem_outputs = (layer.inputs[0] * layer.filters[3] * fft_size *
                       _BYTES_COMPLEX)
        self._logger.debug('FFT mem: %d MB (%d, %d, %d)' % (
            (mem_inputs + mem_filters + mem_outputs) / 2 ** 20, mem_inputs / 2
            ** 20, mem_filters / 2 ** 20, mem_outputs / 2 ** 20))

        # (3) fft2d c2r on num_tiles.
        comp_time += num_tiles * self._estimate_comp_time(
            layer.inputs[0] * layer.filters[3] * _fft_flops(
                fft_size, 'c2r', filter_1d=filter_1d))

        # Read complex.
        comm_time += self._estimate_comm_time(
            layer.inputs[0] * layer.filters[3] * fft_size * _BYTES_COMPLEX)
        # Write outputs. Ignore clipping time here.
        comm_time += self._estimate_comm_time(
            layer.outputs[0] * layer.outputs[1] * layer.outputs[2] *
            layer.outputs[3] * _BYTES_FLOAT)

        # Do not multiple batch size for filters.
        comp_time += comp_time_filters
        comm_time += comm_time_filters

        self._logger.debug('FFT estimates: %f = %f + %f' %
                           (comp_time + comm_time, comp_time, comm_time))
        return TimeMeasure(comp_time=comp_time, comm_time=comm_time)

    def _profile_pool2d(self, layer):
        """Returns the flops."""
        if self.options.direction == 'backward':
            if self.options.gradient_wrt == 'filter':
                return TimeMeasure()

        # Per output pixel: kernel_w x kernel_h x in_channel
        flops = 2 * layer.kernel[1] * layer.kernel[2] * layer.inputs[3]

        # Flops per output map.
        flops *= layer.outputs[1] * layer.outputs[2]

        # Flops across multiple input patches.
        flops *= layer.inputs[0]

        self._logger.debug('Pool2d flops: %d' % flops)

        comm_time = self._estimate_comm_time(
            np.prod(layer.inputs) * _BYTES_FLOAT)
        comm_time += self._estimate_comm_time(
            np.prod(layer.outputs) * _BYTES_FLOAT)

        comp_time = self._estimate_comp_time(flops)

        return TimeMeasure(comp_time=comp_time, comm_time=comm_time)

    def profile_apply_updates(self, params_in_bytes):
        """Time for update all model parameters."""
        # w = w - alpha \Delta w
        num_parameters = params_in_bytes // 4
        flops = 2 * num_parameters
        comp_time = self._estimate_comp_time(flops)

        # Read weights, read updates, write weights.
        comm_time = 3 * self._estimate_comm_time(params_in_bytes)
        return TimeMeasure(comp_time=comp_time, comm_time=comm_time)

    def _profile_dropout(self, layer):
        if self.options.direction == 'backward':
            if self.options.gradient_wrt == 'filter':
                return TimeMeasure()

        flops = np.prod(layer.inputs)
        comp_time = self._estimate_comp_time(flops)

        comm_time = self._estimate_comm_time(
            np.prod(layer.inputs) * _BYTES_FLOAT)
        comm_time += self._estimate_comm_time(
            np.prod(layer.outputs) * _BYTES_FLOAT)

        return TimeMeasure(comp_time=comp_time, comm_time=comm_time)
