import os
from typing import Dict, List, Optional, Tuple, Callable
import copy
from functools import lru_cache

from partitioning.pipeline_predictor import PredictorBundle, load_predictor
from partitioning.pipeline_simulator import PipelineSimulator

SIM_OUTPUT_LEN = 32

TOPS = {
    "jetson-64g-4": 275,
    "jetson-16g-2": 157,
    "jetson-16g-7": 98,
    "jetson-16g-8": 157,
    "jetson-8g-1": 67
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
        batch_size=batch_size,
        output_len=SIM_OUTPUT_LEN,
    )
    return ee



'''
贪心找到局部最优解，初始时按设备算力分配层数开始搜索，找到最大能效策略
'''
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

    partition_scheme = allocate_by_ratio(
        total_nlayer,
        [get_tops(device) for device in devices],
        [r[1] for r in nlayer_range]
    )
    # partition_scheme = allocate_by_ratio(
    #     total_nlayer,
    #     [1 for _ in range(len(devices))],
    #     [r[1] for r in nlayer_range]
    # )
    init_size_current = max_batch_size_callback(partition_scheme)
    EE_current = sim_ee(
        predictors=predictors,
        simulator=simulator,
        devices=devices,
        deployment=partition_scheme,
        batch_size=init_size_current
    )

    print(f"[search_optimal_partition] partition_scheme init: {partition_scheme} (EE_current: {EE_current}, batch_size: {init_size_current})")

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
                        batch_size=max_batch_size_callback(scheme)
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
            print(f"[search_optimal_partition] partition_scheme updated: {partition_scheme} (EE_current: {EE_current}, batch_size: {max_batch_size_callback(partition_scheme)})")
        else:
            break
    
    return partition_scheme


'''
枚举所有可行解，并使用缓存方法优化效率，找到最大能效策略
'''
# def enumerate_partitions(
#     nlayer_range: List[Tuple[int, int]],
#     total_nlayer: int,
# ) -> List[List[int]]:
#     results = []
#     n = len(nlayer_range)

#     mins = [r[0] for r in nlayer_range]
#     maxs = [r[1] for r in nlayer_range]

#     if sum(mins) > total_nlayer or sum(maxs) < total_nlayer:
#         return results

#     def dfs(idx: int, remain: int, current: List[int]):
#         if idx == n:
#             if remain == 0:
#                 results.append(current.copy())
#             return

#         future_min = sum(mins[idx + 1:]) if idx + 1 < n else 0
#         future_max = sum(maxs[idx + 1:]) if idx + 1 < n else 0

#         low = max(mins[idx], remain - future_max)
#         high = min(maxs[idx], remain - future_min)

#         for x in range(low, high + 1):
#             current.append(x)
#             dfs(idx + 1, remain - x, current)
#             current.pop()

#     dfs(0, total_nlayer, [])
#     return results


# def normalize(value: float, vmin: float, vmax: float) -> float:
#     if vmax <= vmin:
#         return 1.0
#     return (value - vmin) / (vmax - vmin)


# def search_optimal_partition(
#     model: str,
#     devices: List[str],
#     nlayer_range: List[Tuple[int, int]],
#     total_nlayer: int,
#     max_batch_size_callback: Callable,
#     alpha: float = 1.0,                    # 越大越偏向 EE，越小越偏向 batch
#     min_batch_required: Optional[int] = None,
#     min_batch_ratio: Optional[float] = None,   # 如 0.3，表示 batch 至少达到全局最大 batch 的 30%
# ) -> List[int]:
#     print(f"[search_optimal_partition] model: {model}")
#     print(f"[search_optimal_partition] devices: {devices}")
#     print(f"[search_optimal_partition] nlayer_range: {nlayer_range}")
#     print(f"[search_optimal_partition] total_nlayer: {total_nlayer}")

#     assert len(devices) == len(nlayer_range), "len(devices) does not fit len(nlayer_range)"
#     assert len(devices) < total_nlayer, "too many devices for partitioning"
#     assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"

#     predictors: Dict[str, PredictorBundle] = {}
#     for device in devices:
#         predictors[device] = load_predictor(
#             checkpoint_path=f"/home/austin/repos/AdaptSplit/AdaptSplit/partitioning/checkpoints/{model}-{device}.pt",
#             device="cuda"
#         )

#     simulator = PipelineSimulator(devices)

#     all_partitions = enumerate_partitions(nlayer_range, total_nlayer)
#     print(f"[search_optimal_partition] total candidate partitions: {len(all_partitions)}")

#     @lru_cache(maxsize=None)
#     def evaluate_partition(partition_tuple: Tuple[int, ...]) -> Tuple[int, float]:
#         partition = list(partition_tuple)
#         batch_size = max_batch_size_callback(partition)
#         ee = sim_ee(
#             predictors=predictors,
#             simulator=simulator,
#             devices=devices,
#             deployment=partition,
#             batch_size=batch_size
#         )
#         return batch_size, ee

#     # 先把所有候选方案都算出来
#     candidates = []
#     for partition in all_partitions:
#         batch_size, ee = evaluate_partition(tuple(partition))
#         candidates.append({
#             "partition": partition,
#             "batch_size": batch_size,
#             "ee": ee,
#         })

#     if not candidates:
#         raise ValueError("No feasible partition found.")

#     # 如果使用比例阈值，则先求全局最大 batch
#     global_max_batch = max(x["batch_size"] for x in candidates)

#     effective_min_batch = min_batch_required
#     if min_batch_ratio is not None:
#         ratio_batch = max(1, int(global_max_batch * min_batch_ratio))
#         if effective_min_batch is None:
#             effective_min_batch = ratio_batch
#         else:
#             effective_min_batch = max(effective_min_batch, ratio_batch)

#     # 过滤掉 batch 太小的方案
#     if effective_min_batch is not None:
#         filtered = [x for x in candidates if x["batch_size"] >= effective_min_batch]
#     else:
#         filtered = candidates

#     if not filtered:
#         raise ValueError("No feasible partition found after batch-size filtering.")

#     # 在过滤后的集合上做归一化
#     ee_min = min(x["ee"] for x in filtered)
#     ee_max = max(x["ee"] for x in filtered)
#     batch_min = min(x["batch_size"] for x in filtered)
#     batch_max = max(x["batch_size"] for x in filtered)

#     best = None
#     best_score = None

#     for x in filtered:
#         ee_norm = normalize(x["ee"], ee_min, ee_max)
#         batch_norm = normalize(x["batch_size"], batch_min, batch_max)

#         score = alpha * ee_norm + (1.0 - alpha) * batch_norm

#         x["score"] = score
#         x["ee_norm"] = ee_norm
#         x["batch_norm"] = batch_norm

#         if best is None or score > best_score:
#             best = x
#             best_score = score

#     print(
#         f"[search_optimal_partition] best partition: {best['partition']}, "
#         f"batch_size: {best['batch_size']}, ee: {best['ee']}, score: {best['score']}"
#     )

#     return best["partition"]