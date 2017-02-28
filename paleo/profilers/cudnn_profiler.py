"""cuDNN based profiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
from paleo.profilers.base import BaseProfiler, TimeMeasure
from paleo.third_party import libcudnn


class CudnnProfiler(BaseProfiler):
    def __init__(self, options):
        super(CudnnProfiler, self).__init__('CudnnProfiler', options)

    def profile(self, layer):
        self.clear_msg()
        t = TimeMeasure()
        if layer.layertype == 'conv2d':
            if self.options.direction == 'forward':
                t += self._profile_conv2d(layer, self.options.num_iter,
                                          self.options.num_warmup)
            elif self.options.direction == 'backward':
                # FIXME: filter or data.
                if (not self.options.gradient_wrt or
                        self.options.gradient_wrt == 'data'):
                    if layer.backprop:
                        t += self._profile_conv2d_bwd_data(
                            layer, self.options.num_iter,
                            self.options.num_warmup)

                if (not self.options.gradient_wrt or
                        self.options.gradient_wrt == 'filter'):
                    t += self._profile_conv2d_bwd_filter(
                        layer, self.options.num_iter, self.options.num_warmup)
        elif layer.layertype == 'deconv2d':
            if self.options.direction == 'forward':
                t += self._profile_conv2d_bwd_data(layer._transposed,
                                                   self.options.num_iter,
                                                   self.options.num_warmup)
            elif self.options.direction == 'backward':
                # FIXME: filter or data.
                if (not self.options.gradient_wrt or
                        self.options.gradient_wrt == 'data'):
                    if layer.backprop:
                        t += self._profile_conv2d(layer._transposed,
                                                  self.options.num_iter,
                                                  self.options.num_warmup)

                if (not self.options.gradient_wrt or
                        self.options.gradient_wrt == 'filter'):
                    t += self._profile_conv2d_bwd_filter(
                        layer._transposed, self.options.num_iter,
                        self.options.num_warmup)
        return t

    def _profile_conv2d(self, layer, num_iter, num_warmup):
        # Use heuristics to find algorithm.
        # This shall be equivalent to the scenario when tf_auto_tune is off.
        algo_heuristic, ws_size = get_convolution_fwd_algorithm(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)
        self.message = '%s' % CONV_ALGO_FWD_NAME[algo_heuristic]

        # Run cudnn profiler to get time.
        cudnn_context, X_desc, filters_desc, conv_desc, Y_desc = cudnn_prepare(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)

        trails = []
        for i in range(num_warmup + num_iter):
            num_results = 7
            algos = libcudnn.cudnnFindConvolutionForwardAlgorithm(
                cudnn_context, X_desc, filters_desc, conv_desc, Y_desc,
                num_results)

            # Print the exhustive search result once in verbose mode.
            if i == num_warmup:
                for al in algos:
                    self._logger.debug("%s, %s, %f" % (
                        CONV_ALGO_FWD_NAME[al.algo],
                        str(libcudnn.cudnnError(al.status)), al.time))

            # Always use the time returned with the heuristic-chosen algorithm.
            if i >= num_warmup:
                for al in algos:
                    if al.algo == algo_heuristic:
                        time = al.time
                trails.append(time)
        cudnn_cleanup(cudnn_context, X_desc, Y_desc, filters_desc, conv_desc)
        mean_time = np.mean(trails)
        return TimeMeasure(total_time=mean_time)

    def _profile_conv2d_bwd_filter(self, layer, num_iter, num_warmup):
        # Use heuristics to find algorithm.
        # This shall be equivalent to the scenario when tf_auto_tune is off.
        algo_heuristic, ws_size = get_convolution_bwd_filter_algorithm(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)
        self.message = '%s' % CONV_ALGO_BWD_FILTER_NAME[algo_heuristic]

        # Run cudnn profiler to get time.
        cudnn_context, X_desc, filters_desc, conv_desc, Y_desc = cudnn_prepare(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)

        trails = []
        for i in range(num_warmup + num_iter):
            num_results = len(CONV_ALGO_BWD_FILTER_NAME)
            algos = libcudnn.cudnnFindConvolutionBackwardFilterAlgorithm(
                cudnn_context, X_desc, Y_desc, conv_desc, filters_desc,
                num_results)

            # Print the exhustive search result once in verbose mode.
            if i == num_warmup:
                for al in algos:
                    self._logger.debug("%s, %s, %f" % (
                        CONV_ALGO_BWD_FILTER_NAME[al.algo],
                        str(libcudnn.cudnnError(al.status)), al.time))

            # Always use the time returned with the heuristic-chosen algorithm.
            if i >= num_warmup:
                for al in algos:
                    if al.algo == algo_heuristic:
                        time = al.time
                trails.append(time)
        cudnn_cleanup(cudnn_context, X_desc, Y_desc, filters_desc, conv_desc)
        mean_time = np.mean(trails)
        self._logger.debug('BWD FILTER: %s %f' % (
            CONV_ALGO_BWD_FILTER_NAME[algo_heuristic], mean_time))
        return TimeMeasure(total_time=mean_time)

    def _profile_conv2d_bwd_data(self, layer, num_iter, num_warmup):
        # Use heuristics to find algorithm.
        # This shall be equivalent to the scenario when tf_auto_tune is off.
        algo_heuristic, ws_size = get_convolution_bwd_data_algorithm(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)
        self.message = '%s' % CONV_ALGO_BWD_DATA_NAME[algo_heuristic]

        # Run cudnn profiler to get time.
        cudnn_context, X_desc, filters_desc, conv_desc, Y_desc = cudnn_prepare(
            layer.inputs, layer.filters, layer.strides, layer._pad_h,
            layer._pad_w)

        trails = []
        for i in range(num_warmup + num_iter):
            num_results = len(CONV_ALGO_BWD_DATA_NAME)
            algos = libcudnn.cudnnFindConvolutionBackwardDataAlgorithm(
                cudnn_context, filters_desc, Y_desc, conv_desc, X_desc,
                num_results)

            # Print the exhustive search result once in verbose mode.
            if i == num_warmup:
                for al in algos:
                    self._logger.debug("%s, %s, %f ms, %d Bytes" %
                                       (CONV_ALGO_BWD_DATA_NAME[al.algo],
                                        str(libcudnn.cudnnError(al.status)),
                                        al.time, al.memory))

            # Always use the time returned with the heuristic-chosen algorithm.
            if i >= num_warmup:
                for al in algos:
                    if al.algo == algo_heuristic:
                        time = al.time
                trails.append(time)
        cudnn_cleanup(cudnn_context, X_desc, Y_desc, filters_desc, conv_desc)
        mean_time = np.mean(trails)
        self._logger.debug('BWD DATA: %s %f ms' % (
            CONV_ALGO_BWD_DATA_NAME[algo_heuristic], mean_time))
        return TimeMeasure(total_time=mean_time)


CONV_ALGO_FWD_NAME = {
    0: 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
    1: 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM',
    2: 'CUDNN_CONVOLUTION_FWD_ALGO_GEMM',
    3: 'CUDNN_CONVOLUTION_FWD_ALGO_DIRECT',
    4: 'CUDNN_CONVOLUTION_FWD_ALGO_FFT',
    5: 'CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING',
    6: 'CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD'
}

CONV_ALGO_BWD_DATA_NAME = {
    0: 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_0',  # non-deterministic
    1: 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
    2: 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT',
    3: 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING',
    4: 'CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD'
}

CONV_ALGO_BWD_FILTER_NAME = {
    0: 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0',  # non-deterministic
    1: 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1',
    2: 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT',
    3: 'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3'  # non-determ, 0 with workspace
}


def cudnn_prepare(inputs, filters, strides, pad_h, pad_w):
    # Data layout: cuDNN uses NCHW, while tensorflow uses NHWC.
    tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
    data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']

    # Mode: CUDNN_CROSS_CORRELATION, CUDNN_CONVOLUTION
    # Always use cross correlation.
    convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']

    n_input, height_in, width_in, _ = inputs
    height_filter, width_filter, filters_in, filters_out = filters
    _, vertical_stride, horizontal_stride, _ = strides
    upscalex, upscaley = 1, 1

    # Create a cuDNN context
    cudnn_context = libcudnn.cudnnCreate()

    # Descriptor for inp
    X_desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(X_desc, tensor_format, data_type,
                                        n_input, filters_in, height_in,
                                        width_in)

    # Filter descriptor
    filters_desc = libcudnn.cudnnCreateFilterDescriptor()
    libcudnn.cudnnSetFilter4dDescriptor(filters_desc, data_type, tensor_format,
                                        filters_out, filters_in, height_filter,
                                        width_filter)

    # Convolution descriptor
    conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
    libcudnn.cudnnSetConvolution2dDescriptor(
        conv_desc, pad_h, pad_w, vertical_stride, horizontal_stride, upscalex,
        upscaley, convolution_mode)

    # Get output dimensions (first two values are n_input and filters_out)
    _, _, height_output, width_output = (
        libcudnn.cudnnGetConvolution2dForwardOutputDim(conv_desc, X_desc,
                                                       filters_desc))

    Y_desc = libcudnn.cudnnCreateTensorDescriptor()
    libcudnn.cudnnSetTensor4dDescriptor(Y_desc, tensor_format, data_type,
                                        n_input, filters_out, height_output,
                                        width_output)

    return cudnn_context, X_desc, filters_desc, conv_desc, Y_desc


def get_convolution_fwd_algorithm(inputs,
                                  filters,
                                  strides,
                                  pad_h,
                                  pad_w,
                                  has_space_limit=True):
    cudnn_context, X_desc, filters_desc, conv_desc, Y_desc = cudnn_prepare(
        inputs, filters, strides, pad_h, pad_w)

    # CHoices for preference:
    #   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
    #   CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
    #   CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT

    if has_space_limit:
        # 1 << 32 = 4 GB
        convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference[
            'CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT']
        space_limit = 1 << 32
    else:
        convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference[
            'CUDNN_CONVOLUTION_FWD_NO_WORKSPACE']
        space_limit = 0

    algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(
        cudnn_context, X_desc, filters_desc, conv_desc, Y_desc,
        convolution_fwd_pref, space_limit)

    ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_context, X_desc, filters_desc, conv_desc, Y_desc, algo)

    cudnn_cleanup(cudnn_context, X_desc, Y_desc, filters_desc, conv_desc)
    return algo.value, ws_size.value


def get_convolution_bwd_data_algorithm(inputs,
                                       filters,
                                       strides,
                                       pad_h,
                                       pad_w,
                                       has_space_limit=True):
    cudnn_context, X_desc, filters_desc, conv_desc, Y_desc = cudnn_prepare(
        inputs, filters, strides, pad_h, pad_w)

    if has_space_limit:
        # 1 << 32 = 4 GB
        convolution_bwd_pref = libcudnn.cudnnConvolutionBwdDataPreference[
            'CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT']
        space_limit = 1 << 32
    else:
        convolution_bwd_pref = libcudnn.cudnnConvolutionBwdDataPreference[
            'CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE']
        space_limit = 0

    algo = libcudnn.cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_context, filters_desc, Y_desc, conv_desc, X_desc,
        convolution_bwd_pref, space_limit)

    ws_size = libcudnn.cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_context, filters_desc, Y_desc, conv_desc, X_desc, algo)

    cudnn_cleanup(cudnn_context, X_desc, Y_desc, filters_desc, conv_desc)
    return algo.value, ws_size.value


def get_convolution_bwd_filter_algorithm(inputs,
                                         filters,
                                         strides,
                                         pad_h,
                                         pad_w,
                                         has_space_limit=True):
    cudnn_context, X_desc, filters_desc, conv_desc, Y_desc = cudnn_prepare(
        inputs, filters, strides, pad_h, pad_w)

    if has_space_limit:
        # 1 << 32 = 4 GB
        convolution_bwd_pref = libcudnn.cudnnConvolutionBwdFilterPreference[
            'CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT']
        space_limit = 1 << 32
    else:
        convolution_bwd_pref = libcudnn.cudnnConvolutionBwdFilterPreference[
            'CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE']
        space_limit = 0

    algo = libcudnn.cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn_context, X_desc, Y_desc, conv_desc, filters_desc,
        convolution_bwd_pref, space_limit)

    ws_size = libcudnn.cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_context, X_desc, Y_desc, conv_desc, filters_desc, algo)

    cudnn_cleanup(cudnn_context, X_desc, Y_desc, filters_desc, conv_desc)
    return algo.value, ws_size.value


def cudnn_cleanup(cudnn_context, X_desc, Y_desc, filters_desc, conv_desc):
    # Clean up
    libcudnn.cudnnDestroyTensorDescriptor(X_desc)
    libcudnn.cudnnDestroyTensorDescriptor(Y_desc)
    libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
    libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
    libcudnn.cudnnDestroy(cudnn_context)
