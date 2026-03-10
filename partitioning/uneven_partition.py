import os
from typing import Dict, List, Optional, Tuple, Callable
import copy

from partitioning.pipeline_predictor import PredictorBundle, load_predictor
from partitioning.pipeline_simulator import PipelineSimulator

SIM_OUTPUT_LEN = 16

TOPS = {
    "jetson-64g": 275,
    "jetson-16g": 157,
    "jetson-8g": 67
}

def get_tops(device_name: str):
    for prefix, value in TOPS.items():
        if prefix in device_name:
            return value
    raise ValueError("Unknown Device TOPS")

def allocate_by_ratio(value: int, ratios: List[float], ranges: List[int]) -> List[int]:
    if not ratios:
        return []
    if len(ratios) != len(ranges):
        raise ValueError("ratios 和 ranges 长度必须一致")
    n = len(ratios)
    if value < n:
        raise ValueError("value 太小，无法保证每个结果都至少为 1")
    if any(x < 1 for x in ranges):
        raise ValueError("ranges 中每个最大值都必须 >= 1")
    if sum(ranges) < value:
        raise ValueError("ranges 的总上限不足，无法分配到 value")
    total = sum(ratios)
    ratios = [r / total for r in ratios] if abs(total - 1.0) > 1e-10 else ratios
    # 先保证每个位置至少为 1
    res = [1] * n
    caps = [mx - 1 for mx in ranges]
    remain = value - n
    while remain > 0:
        idxs = [i for i in range(n) if caps[i] > 0]
        if not idxs:
            raise ValueError("无可分配空间，但 remain > 0")
        s = sum(ratios[i] for i in idxs)
        quotas = {i: remain * ratios[i] / s for i in idxs}
        adds = {i: min(caps[i], int(quotas[i])) for i in idxs}
        used = sum(adds.values())
        if used == 0:
            # 所有整数部分都为 0 时，按最大余额法补 1
            i = max(idxs, key=lambda x: quotas[x] - int(quotas[x]))
            res[i] += 1
            caps[i] -= 1
            remain -= 1
        else:
            for i, a in adds.items():
                res[i] += a
                caps[i] -= a
            remain -= used
    return res


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

    # partition_scheme = allocate_by_ratio(total_nlayer, [get_tops(device) for device in devices], [r[1] for r in nlayer_range])
    partition_scheme = allocate_by_ratio(
        total_nlayer,
        [1 for _ in range(len(devices))],
        [r[1] for r in nlayer_range]
    )
    batch_size_current = max_batch_size_callback(partition_scheme)
    EE_current = sim_ee(
        predictors=predictors,
        simulator=simulator,
        devices=devices,
        deployment=partition_scheme,
        batch_size=batch_size_current
    )

    print(f"[search_optimal_partition] partition_scheme init: {partition_scheme} (batch_size: {batch_size_current})")
    bs = batch_size_current

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
                    if gain > best_gain and max_batch_size_callback(scheme) >= batch_size_current:
                        best_gain = gain
                        best_move["from"] = i
                        best_move["to"] = j
        
        if best_gain > 0:
            partition_scheme[best_move["from"]] -= 1
            partition_scheme[best_move["to"]] += 1
            EE_current = EE_current + best_gain
            batch_size_current = max_batch_size_callback(partition_scheme)
            print(f"[search_optimal_partition] partition_scheme updated: {partition_scheme} (batch_size: {batch_size_current})")
        else:
            break
    
    return partition_scheme