"""Simulate distributed setups."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
import numpy as np

from paleo import comm, profilers
from paleo.profilers.base import TimeMeasure

logger = logging.getLogger("paleo")


def _profile_for_batch_size(layer_list,
                            direction,
                            device,
                            batch_size,
                            use_only_gemm,
                            ppp_comp,
                            ppp_comm,
                            cross_device_bandwidth=None):
    """Use flops profiler to estiamte execution with under the given spec."""
    logger.debug('Profile for\n  pass: %s\n  device: %s\n  batch size: %s' %
                 (direction, device.name, batch_size))
    times = []
    params_in_bytes = 0

    # Estimate forward time for each layer.
    for layer_spec in layer_list:
        layer = layer_spec.layer_op
        if batch_size:
            layer.batch_size = batch_size

        options = profilers.ProfilerOptions()
        options.direction = direction
        options.gradient_wrt = None
        if use_only_gemm:
            options.use_cudnn_heuristics = False
        # FIXME: we don't include bias and activation for simplicity.
        options.include_bias_and_activation = False
        options.ppp_comp = ppp_comp
        options.ppp_comm = ppp_comm
        flops_profiler = profilers.FlopsProfiler(options, device)

        layer_time = flops_profiler.profile(
            layer, layer_spec.device_id,
            [p.device_id for p in layer_spec.parents], cross_device_bandwidth)
        params_in_bytes += layer.weights_in_bytes
        times.append(layer_time)

    return times, params_in_bytes


def _profile_for_apply_updates(params_in_bytes, device):
    flops_profiler = profilers.FlopsProfiler(profilers.ProfilerOptions(),
                                             device)
    return flops_profiler.profile_apply_updates(params_in_bytes)


def _sum_with_parallel(layer_dependencies, layer_list, times):
    """This function sums the time but allows parallel execution.
    Layer dependencies and layer_list are LayerSpec objects.
    Here we only need name. need refactor later.
    """
    assert len(layer_list) == len(times)
    layers_to_times = dict()
    for i, l in enumerate(layer_list):
        t = times[i]
        if isinstance(t, TimeMeasure):
            t = t.total_time
        layers_to_times[l.name] = t

    lower_bound, upper_bound = 0, 0
    for block in layer_dependencies:
        if isinstance(block, tuple):
            # Parallel nodes, each element in the tuple shall be a list.
            parallel_times = []
            for sequence in block:
                sum_of_time = sum(
                    [layers_to_times.get(l.name, 0) for l in sequence])
                parallel_times.append(sum_of_time)
            lower_bound += max(parallel_times)
            upper_bound += sum(parallel_times)
        else:
            lower_bound += layers_to_times.get(block.name, 0)
            upper_bound += layers_to_times.get(block.name, 0)

    return lower_bound, upper_bound


def simulate_model_parallel(nested_list, layer_list, batch_size, device,
                            network, use_pipeline, use_only_gemm, ppp_comp,
                            ppp_comm):
    """Run simulation for model parallel."""
    # Get times for different batch_sizes:
    forward_times, params_in_bytes = _profile_for_batch_size(
        layer_list, 'forward', device, batch_size, use_only_gemm, ppp_comp,
        ppp_comm)
    backward_times, _ = _profile_for_batch_size(layer_list, 'backward', device,
                                                batch_size, use_only_gemm,
                                                ppp_comp, ppp_comm)

    result_times = []
    for direction, times in [('Fwd', forward_times), ('Bwd', backward_times)]:
        lower, upper = _sum_with_parallel(nested_list, layer_list, times)
        result_times.append([direction, lower, upper])
    headers = ['Pass', 'Lowerbound, Upperbound']
    return headers, result_times


def simulate_hybrid_parallel(nested_list, layer_list, batch_size, device,
                             network, use_pipeline, use_only_gemm, ppp_comp,
                             ppp_comm, hybrid_workers):
    """Run simulation for hybrid parallel described as in OneWeird Trick."""

    # Pretty hacky implementation. Need organize and refactor later.

    def _simulate_data_parallel_layers(layers):
        # Forward and backward time for data parallel layers.
        forward_times, params_in_bytes = _profile_for_batch_size(
            layers, 'forward', device, batch_size, use_only_gemm, ppp_comp,
            ppp_comm, network.bandwidth / 8)
        backward_times, _ = _profile_for_batch_size(
            layers, 'backward', device, batch_size, use_only_gemm, ppp_comp,
            ppp_comm, network.bandwidth / 8)

        # Weight sync between data parallel layers.
        comm_scheme = comm.TreeAllReduce(hybrid_workers, network, ppp_comm)
        time_sync_weights = comm_scheme.all_reduce(params_in_bytes)
        time_apply_updates = _profile_for_apply_updates(params_in_bytes,
                                                        device)

        time_fwd, time_bwd = sum(forward_times), sum(backward_times)
        time_fwd, time_bwd = time_fwd.total_time, time_bwd.total_time
        time_apply = time_apply_updates.total_time
        if use_pipeline:
            time_fwd = sum([t.lowerbound for t in forward_times])
            time_bwd = sum([t.lowerbound for t in backward_times])
            time_apply = time_apply_updates.lowerbound
        return (time_fwd, time_bwd, time_apply, time_sync_weights)

    # Get times for different batch_sizes:
    def _simulate_model_parallel_layers(layers):
        forward_times, params_in_bytes = _profile_for_batch_size(
            layers, 'forward', device, batch_size, use_only_gemm, ppp_comp,
            ppp_comm, network.bandwidth / 8)
        backward_times, _ = _profile_for_batch_size(
            layers, 'backward', device, batch_size, use_only_gemm, ppp_comp,
            ppp_comm, network.bandwidth / 8)  # Use GB/s in profiler.
        time_apply_updates = _profile_for_apply_updates(params_in_bytes,
                                                        device)
        fwd_lower, fwd_upper = _sum_with_parallel(nested_list, layers,
                                                  forward_times)
        bwd_lower, bwd_upper = _sum_with_parallel(nested_list, layers,
                                                  backward_times)
        return fwd_lower, bwd_lower, time_apply_updates.total_time

    effective_batch_size = batch_size * hybrid_workers

    # Separate data model parallel layers.
    data_parallel_layers = filter(lambda l: '@' not in l.name, layer_list)
    model_parallel_layers = filter(lambda l: '@' in l.name, layer_list)
    logger.debug('Data parallel layers: %s' % data_parallel_layers)
    logger.debug('Model parallel layers: %s' % model_parallel_layers)

    (d_time_fwd, d_time_bwd, d_time_apply,
     d_time_sync) = _simulate_data_parallel_layers(data_parallel_layers)
    (m_time_fwd, m_time_bwd,
     m_time_apply) = _simulate_model_parallel_layers(model_parallel_layers)

    # Estimate data fetch time for the last data parallel layer.
    time_fetch = 0
    if len(model_parallel_layers) > 0:
        parent_layer = model_parallel_layers[0].parents[0]
        bytes_to_transfer = np.prod(parent_layer.layer_op.outputs)
        time_fetch = bytes_to_transfer / 2 ** 30 / (network.bandwidth /
                                                    8) * 10 ** 3
        time_fetch *= (hybrid_workers - 1)

        # Multiple model parallel stages.
        # Only one fetch_time because of pipeline.
        m_time_fwd = (m_time_fwd - time_fetch) * hybrid_workers + time_fetch
        m_time_bwd = (m_time_bwd - time_fetch) * hybrid_workers + time_fetch
        logger.info('Saved fetch time from %s by pipelining: %f' %
                    (parent_layer, time_fetch * (hybrid_workers - 1)))

    headers = [
        'workers', 'batch_size', 'fwd_time', 'bwd_time', 'apply_time',
        'sync_time(tree)'
    ]
    results = []
    results.append([
        hybrid_workers, effective_batch_size, d_time_fwd, d_time_bwd,
        d_time_apply, d_time_sync
    ])
    results.append([
        hybrid_workers, effective_batch_size, m_time_fwd, m_time_bwd,
        m_time_apply, 0
    ])
    return headers, results


def simulate_scaling(layer_dependencies, layer_list, worker_sizes,
                     scaling_type, batch_size, device, network, use_pipeline,
                     use_only_gemm, ppp_comp, ppp_comm):
    """Run simulation for data parallel."""
    logger.debug(
        'Simulate scaling:\n  type: %s\n  device: %s\n  batch size: %s' %
        (scaling_type, device.name, batch_size))

    if scaling_type == 'weak':
        # Get times for different batch_sizes:
        forward_times, params_in_bytes = _profile_for_batch_size(
            layer_list, 'forward', device, batch_size, use_only_gemm, ppp_comp,
            ppp_comm)
        backward_times, _ = _profile_for_batch_size(
            layer_list, 'backward', device, batch_size, use_only_gemm,
            ppp_comp, ppp_comm)

    all_times = []
    for num_workers in worker_sizes:
        if scaling_type == 'weak':
            num_iterations = 1
            batch_size_per_node = batch_size
            batch_size_per_iteration = batch_size
            effective_batch_size = batch_size * num_workers
        elif scaling_type == 'strong':
            batch_size_per_node = int(math.ceil(batch_size / num_workers))
            # Maximum batch size per worker is 128.
            # batch_size_per_iteration = min(batch_size_per_node, 128)
            batch_size_per_iteration = batch_size_per_node
            num_iterations = int(
                math.ceil(batch_size_per_node / batch_size_per_iteration))
            effective_batch_size = batch_size

            # Get times for different batch_sizes:
            forward_times, params_in_bytes = _profile_for_batch_size(
                layer_list, 'forward', device, batch_size_per_iteration,
                use_only_gemm, ppp_comp, ppp_comm)
            backward_times, _ = _profile_for_batch_size(
                layer_list, 'backward', device, batch_size_per_iteration,
                use_only_gemm, ppp_comp, ppp_comm)
        else:
            raise ValueError('Unknown scaling type: %s' % scaling_type)

        time_apply_updates = _profile_for_apply_updates(params_in_bytes,
                                                        device)

        time_fwd, time_bwd = sum(forward_times), sum(backward_times)
        time_fwd, time_bwd = time_fwd.total_time, time_bwd.total_time
        time_apply = time_apply_updates.total_time
        if use_pipeline:
            time_fwd = sum([t.lowerbound for t in forward_times])
            time_bwd = sum([t.lowerbound for t in backward_times])
            time_apply = time_apply_updates.lowerbound

        # For each node size, we collect
        #     workers,effective_batch_size, num_iter, fwd_time, bwd_time,
        #     apply_time, comp_time, [sync_times...]
        times = [
            num_workers, effective_batch_size, num_iterations, time_fwd,
            time_bwd, time_apply, time_fwd + time_bwd + time_apply
        ]
        times.extend([
            c.all_reduce(params_in_bytes)
            for c in comm.get_all_comm_schemes(num_workers, network, ppp_comm)
        ])
        all_times.append(times)

    headers = [
        'workers', 'batch_size', 'iter', 'time_fwd', 'time_bwd', 'time_apply',
        'time_comp'
    ]
    headers.extend(
        [c.name for c in comm.get_all_comm_schemes(1, network, ppp_comm)])
    assert len(headers) == len(all_times[0])

    return headers, all_times
