import os
from typing import Dict, List, Optional, Tuple, Callable
import copy

from partitioning.pipeline_predictor import PredictorBundle, load_predictor
from partitioning.pipeline_simulator import PipelineSimulator

SIM_OUTPUT_LEN = 16


def sim_ee(
    predictors: Dict[str, PredictorBundle],
    simulator: PipelineSimulator,
    devices: List[str],
    deployment: List[int],
    batch_size: int
):
    pairs_predicted = []
    for idx, device in enumerate(devices):
        pairs_predicted.append(predictors[device].predict(deployment[idx], batch_size))
    ee = simulator.sim(
        latency_power_pairs=pairs_predicted,
        output_len=SIM_OUTPUT_LEN,
    )
    return ee


def search_optimal_partition(
    model: str,
    devices: List[str],
    nlayer_range: List[Tuple[int, int]],
    total_nlayer: int,
    max_batch_size_callback: Callable,
) -> List[int]:
    print(f"[search_optimal_partition] model: {model}")
    print(f"[search_optimal_partition] devices: {devices}")
    print(f"[search_optimal_partition] nlayer_range: {nlayer_range}")
    print(f"[search_optimal_partition] total_nlayer: {total_nlayer}")
    # Input Example:
    ## model = "Llama-2-7b-chat-hf"
    ## devices = ['jetson-64g-4', 'jetson-16g-2', 'jetson-16g-8', 'jetson-8g-1']
    ## nlayer_range = [[1,12], [1,12], [1,12], [1,10]]

    assert (
        len(devices) == len(nlayer_range)
    ), "len(devices) does not fit len(nlayer_range)"
    assert (
        len(devices) < total_nlayer
    ), "too many devices for partitioning"

    predictors: Dict[str, PredictorBundle] = {}
    for device in devices:
        predictors[device] = load_predictor(
            checkpoint_path=f"/home/austin/repos/AdaptSplit/AdaptSplit/partitioning/checkpoints/{model}-{device}.pt",
            device="cuda"
        )
    simulator = PipelineSimulator(devices)

    split_int = lambda m, n: [m//n + (i < m%n) for i in range(n)]
    partition_scheme = split_int(total_nlayer, len(devices))
    EE_current = sim_ee(
        predictors=predictors,
        simulator=simulator,
        devices=devices,
        deployment=partition_scheme,
        batch_size=max_batch_size_callback(partition_scheme)
    )

    print(f"[search_optimal_partition] partition_scheme init: {partition_scheme} (batch_size: {max_batch_size_callback(partition_scheme)})")
    bs = max_batch_size_callback(partition_scheme)

    while True:
        best_gain = 0
        best_move = {"from": -1, "to": -1}

        for i, device_i in enumerate(devices):
            for j, device_j in enumerate(devices):
                if partition_scheme[i] > nlayer_range[i][0] and partition_scheme[j] < nlayer_range[j][1]:
                    scheme = copy.deepcopy(partition_scheme)
                    scheme[i] = scheme[i] - 1
                    scheme[j] = scheme[j] + 1
                    EE_new = sim_ee(
                        predictors=predictors,
                        simulator=simulator,
                        devices=devices,
                        deployment=scheme,
                        batch_size=bs
                    )
                    gain = EE_new - EE_current
                    if gain > best_gain:
                        best_gain = gain
                        best_move["from"] = i
                        best_move["to"] = j
        
        if best_gain > 0:
            partition_scheme[best_move["from"]] -= 1
            partition_scheme[best_move["to"]] += 1
            EE_current = EE_current + best_gain
            print(f"[search_optimal_partition] partition_scheme updated: {partition_scheme} (batch_size: {max_batch_size_callback(partition_scheme)})")
        else:
            break
    
    return partition_scheme