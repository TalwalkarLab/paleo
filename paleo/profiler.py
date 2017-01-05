# This script analyze neural network architectures.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import math

import click
import numpy as np

from paleo.graph import OperationGraph
from paleo import device, layers
from paleo import profilers
from paleo import comm
from paleo import simulation
from paleo.utils import save_layer

FORMAT = "%(levelname)s %(pathname)s:%(lineno)d] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger("paleo")
logger.setLevel(logging.INFO)


class Profiler():
    def __init__(self, filename, separator='\t'):
        """Initialize a profiler for the given network architecture."""
        self._filename = filename

        # Parse the net spec and flatten into a list in topology order.
        self.graph = OperationGraph(filename)
        logger.debug('Net spec loaded from %s.' % filename)
        logger.info('Dependencies: %s' % str(self.graph.nested_list))
        self._separator = separator

    def print_static_summary(self):
        """Print a static summary about the network."""
        print('A summary of static characteristics of network.')
        print('  LAYER\tOUTPUTS')
        num_params = 0
        weights_in_bytes = 0
        num_activations = 0
        for layer_spec in self.graph.topology_order:
            layer = layer_spec.layer_op
            print('  %s' % layer)
            num_params += layer.num_params
            weights_in_bytes += layer.weights_in_bytes
            num_activations += np.prod(layer.outputs)
        print('Number of params: {:,} ({:,} Bytes)'.format(num_params,
                                                           weights_in_bytes))
        print('Activation: {:,} Bytes'.format(num_activations * 4))

    def save_conv_layers(self, save_dir):
        """Save convolution layers into separate files."""
        for layer_spec in self.graph.topology_order:
            if layer_spec['type'] != 'Convolution':
                continue
            layer = layer_spec.layer_op
            outfilename = os.path.join(save_dir, "%s.json" % layer_spec.name)
            save_layer.save_conv_layer(outfilename, layer)

    def profile(self, device_name, options, executor=None):
        """Profile the network with the given device spec.

        Returns:
            A dictionary contains the following keys:
              (layers, flops, executor, executor_std, flops_message,
              executor_msg)
        """
        device_spec = device.DEVICES[device_name]
        logger.info('Profiling for device %s' % device_spec.name)

        results = []
        for layer_spec in self.graph.topology_order:
            layer = layer_spec.layer_op

            # Always run flop-based profiler.
            if executor == 'tensorflow':
                # Here we disable the cudnn heuristics.
                # Tensorflow requires creating a cuda stream and does not allow
                # multiple context under one process.
                # We cannot use cuda stream because of the python wrapper.
                options.use_cudnn_heuristics = False

            flops_profiler = profilers.FlopsProfiler(options, device_spec)
            flop_based_time = flops_profiler.profile(layer)

            logger.info('Layer: %s' % layer_spec.name)
            logger.info('- %s: %s  %s' % (flops_profiler.name, flop_based_time,
                                          flops_profiler.message))

            if device_spec.is_gpu:
                profiler = None
                if executor == 'cudnn':
                    profiler = profilers.CudnnProfiler(options)
                elif executor == 'tensorflow':
                    import TensorFlowProfiler from profilers.tensorflow_profiler
                    profiler = profilers.TensorFlowProfiler(options)

                if profiler:
                    executor_time = profiler.profile(layer)
                    logger.info('- %s: %s  %s' % (profiler.name, executor_time,
                                                  profiler.message))

                    results.append(
                        (layer_spec.name, flop_based_time.total_time,
                         executor_time.total_time, 0, flops_profiler.message,
                         profiler.message))
        return results

    def profile_full_pass(self, device, num_warmup, num_iter, batch_size):
        """Profile full pass execution with tensorflow."""
        options = profilers.ProfilerOptions()
        options.num_warmup = num_warmup
        options.num_iter = num_iter
        options.include_bias_and_activation = False
        import TensorFlowProfiler from profilers.tensorflow_profiler
        profiler = profilers.TensorFlowProfiler(options)

        if batch_size:
            for l in self.graph.topology_order:
                l.layer_op.batch_size = batch_size

        layers = [layer_spec.layer_op
                  for layer_spec in self.graph.topology_order]

        return profiler.profile_full_pass(layers)

    def simulate(self, device_name, network_name, batch_size, use_pipeline,
                 use_only_gemm, worker_sizes, scaling, ppp_comp, ppp_comm,
                 parallel, hybrid_workers):
        device_spec = device.DEVICES[device_name]
        network_spec = device.NETWORKS[network_name]

        if parallel == 'data':
            for scaling_option in scaling.split(','):
                # Estimate time for weights update.
                # Weak scaling.
                print('=' * 10)
                headers, scaling_times = simulation.simulate_scaling(
                    self.graph.nested_list, self.graph.topology_order,
                    worker_sizes, scaling_option, batch_size, device_spec,
                    network_spec, use_pipeline, use_only_gemm, ppp_comp,
                    ppp_comm)
                print('%s scaling' % scaling_option)
                print('Profiling for device %s and %s (%f GB/s)' %
                      (device_spec.name, network_spec.name,
                       network_spec.bandwidth / 8))
                print('Use pipelining: %s' % use_pipeline)
                print('Use gemm: %s  PPP comp: %f   PPP comm: %f' %
                      (use_only_gemm, ppp_comp, ppp_comm))
                print(self._separator.join(headers))
                for times in scaling_times:
                    print(self._separator.join([str(t) for t in times]))
        elif parallel == 'model':
            # Estimate time for weights update.
            # Weak scaling.
            print('=' * 10)
            print('Model parallel')
            headers, result_times = simulation.simulate_model_parallel(
                self.graph.nested_list, self.graph.topology_order, batch_size,
                device_spec, network_spec, use_pipeline, use_only_gemm,
                ppp_comp, ppp_comm)
            print('Profiling for device %s and %s (%f GB/s)' %
                  (device_spec.name, network_spec.name,
                   network_spec.bandwidth / 8))
            print('Use pipelining: %s' % use_pipeline)
            print('Use gemm: %s  PPP comp: %f   PPP comm: %f' %
                  (use_only_gemm, ppp_comp, ppp_comm))
            print(self._separator.join(headers))
            for times in result_times:
                print(self._separator.join([str(t) for t in times]))
        elif parallel == 'hybrid':
            # Estimate time for weights update.
            print('=' * 10)
            print('Hybrid parallel')
            headers, result_times = simulation.simulate_hybrid_parallel(
                self.graph.nested_list, self.graph.topology_order, batch_size,
                device_spec, network_spec, use_pipeline, use_only_gemm,
                ppp_comp, ppp_comm, hybrid_workers)
            print('Profiling for device %s and %s (%f GB/s)' %
                  (device_spec.name, network_spec.name,
                   network_spec.bandwidth / 8))
            print('Use pipelining: %s' % use_pipeline)
            print('Hybrid workers: %d' % hybrid_workers)
            print('Use gemm: %s  PPP comp: %f   PPP comm: %f' %
                  (use_only_gemm, ppp_comp, ppp_comm))
            print(self._separator.join(headers))
            for times in result_times:
                print(self._separator.join([str(t) for t in times]))


HELP_VERBOSE = 'Whether to display debug level log messages.'
HELP_DEVICE_NAME = 'Device to estimate.'


@click.group()
@click.option('--verbose', is_flag=True, help=HELP_VERBOSE)
def cli(verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)


@cli.command()
@click.argument('netspec_files', nargs=-1)
@click.option('--device_name', default='TITAN_X', help=HELP_DEVICE_NAME)
@click.option('--network_name', default='ethernet')
@click.option('--batch_size', default=128)
@click.option('--use_pipeline', is_flag=True)
@click.option('--use_only_gemm', is_flag=True)
@click.option('--num_workers', default='1,2,4,8,16,32,64,128')
@click.option('--scaling', default='weak,strong')
@click.option('--ppp_comp', default=1.0)
@click.option('--ppp_comm', default=1.0)
@click.option('--separator', default='\t')
@click.option(
    '--parallel',
    default='data',
    type=click.Choice(['data', 'model', 'hybrid']))
@click.option('--hybrid_workers', default=1)
def simulate(netspec_files, device_name, network_name, batch_size,
             use_pipeline, use_only_gemm, num_workers, scaling, ppp_comp,
             ppp_comm, parallel, hybrid_workers, separator):
    """Simulate distributed training of a neural network."""

    num_workers = [int(x) for x in num_workers.split(',')]

    for netspec_file in netspec_files:
        profiler = Profiler(netspec_file, separator=separator)
        profiler.simulate(device_name, network_name, batch_size, use_pipeline,
                          use_only_gemm, num_workers, scaling, ppp_comp,
                          ppp_comm, parallel, hybrid_workers)


HELP_EXECUTIOR = 'Which executor to use.'
HELP_WARMUP = 'Iterations to burn in.'
HELP_ITER = 'Iterations to run for profiling.'
HELP_EXTRACT_CONV_DIR = 'Path to extract conv layers.'


@cli.command()
@click.argument('netspec_files', nargs=-1)
@click.option('--device_name', default='TITAN_X', help=HELP_DEVICE_NAME)
@click.option('--num_warmup', default=10, help=HELP_WARMUP)
@click.option('--num_iter', default=50, help=HELP_ITER)
@click.option('--batch_size', type=int)
def fullpass(netspec_files, device_name, num_warmup, num_iter, batch_size):
    """Profile full pass with TensorFlow."""
    for netspec_file in netspec_files:
        profiler = Profiler(netspec_file)
        fwd_time, softmax_time, bwd_time = profiler.profile_full_pass(
            device_name, num_warmup, num_iter, batch_size)
        print('Fullpass profiling with Tensorflow. Customize batch size = %s' %
              str(batch_size))
        print('Forward time\t %s' % fwd_time.total_time)
        print('Backward time\t %s' % bwd_time.total_time)
        print('Softmax time\t %s' % softmax_time.total_time)


@cli.command()
@click.argument('netspec_files', nargs=-1)
@click.option('--device_name', default='TITAN_X', help=HELP_DEVICE_NAME)
@click.option('--num_warmup', default=10, help=HELP_WARMUP)
@click.option('--num_iter', default=50, help=HELP_ITER)
@click.option('--extract_conv_dir', help=HELP_EXTRACT_CONV_DIR)
@click.option('--direction', default='forward')
@click.option('--gradient_wrt', default='data')
@click.option('--use_only_gemm', is_flag=True)
@click.option('--ppp_comp', default=1.0)
@click.option('--executor')
@click.option('--separator', default='\t')
def profile(netspec_files, device_name, num_warmup, num_iter, extract_conv_dir,
            direction, gradient_wrt, use_only_gemm, executor, ppp_comp,
            separator):
    """Profiling a neural network."""

    def _print_tabular(cudnn_result, tensorflow_result):
        assert len(cudnn_result) == len(tensorflow_result)

        print(separator.join(['layer', 'ours', 'cudnn', 'tensorflow',
                              'ours_alg', 'cu_alg']))
        sum_ours, sum_cu, sum_tf = 0, 0, 0
        for cudnn_prof, tf_prof in zip(cudnn_result, tensorflow_result):
            (layer_name, ours_time, cudnn_time, tf_time, our_msg,
             cu_msg) = ['', 0, 0, 0, '', '']
            if cudnn_prof:
                layer_name, ours_time, cudnn_time, _, our_msg, cu_msg = cudnn_prof
            if tf_prof:
                layer_name, ours_time, tf_time, _, our_msg, _ = tf_prof

            our_msg = our_msg.replace('CUDNN_CONVOLUTION_', '')
            cu_msg = cu_msg.replace('CUDNN_CONVOLUTION_', '')

            if layer_name == 'data':
                continue

            sum_ours += ours_time
            sum_cu += cudnn_time
            sum_tf += tf_time

            print(separator.join([
                str(x)
                for x in (layer_name, ours_time, cudnn_time, tf_time, our_msg,
                          cu_msg)
            ]))
        print(separator.join(['Sum', str(sum_ours), str(sum_cu), str(sum_tf)]))

    all_results = dict()
    for netspec_file in netspec_files:
        profiler = Profiler(netspec_file, separator=separator)

        if extract_conv_dir:
            profiler.save_conv_layers(extract_conv_dir)

        if profile:
            options = profilers.ProfilerOptions()
            options.direction = direction
            options.gradient_wrt = gradient_wrt
            options.num_iter = num_iter
            options.num_warmup = num_warmup
            options.ppp_comp = ppp_comp

            tensorflow_result, cudnn_result = None, None
            if executor == 'tensorflow':
                options.use_cudnn_heuristics = False
                tensorflow_result = profiler.profile(
                    device_name, options, executor='tensorflow')

            if not use_only_gemm:
                options.use_cudnn_heuristics = True

            if executor == 'cudnn':
                cudnn_result = profiler.profile(
                    device_name, options, executor='cudnn')

            if cudnn_result:
                tensorflow_result = [None] * len(cudnn_result)
            elif tensorflow_result:
                cudnn_result = [None] * len(tensorflow_result)
            all_results[netspec_file] = (cudnn_result, tensorflow_result)

    for net in all_results:
        print('Network: %s' % net)
        print('Direction: %s' % direction)
        if direction == 'backward':
            print('Gradient wrt: %s' % gradient_wrt)
        (cu, tf) = all_results[net]
        _print_tabular(cu, tf)


@cli.command()
@click.argument('netspec_files', nargs=-1)
def summary(netspec_files):
    """Summarize a neural network."""
    for netspec_file in netspec_files:
        profiler = Profiler(netspec_file)
        profiler.print_static_summary()


if __name__ == '__main__':
    cli()
