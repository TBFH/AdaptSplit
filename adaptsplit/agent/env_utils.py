from typing import List, Tuple
import numpy as np
from typing import List
import json

from adaptsplit.lifetime import LifetimeEvent, LifetimeEventType, json_decode_lifetime_events

class RequestResult:
    """
    A class for storing the results of a single request
    """
    
    def __init__(
        self,
        prompt_len: int,
        output_len: int,
        start_time: float,
        end_time: float,
        token_timestamps: List[float],
        lifetime_events: List[LifetimeEvent] = None
    ):
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.start_time = start_time
        self.end_time = end_time
        self.token_timestamps = token_timestamps
        self.lifecycle_events = lifetime_events
        
        self.latency = end_time - start_time
        self.ttft = (token_timestamps[0] - start_time) * 1000   # s -> ms
        self.tpot = 0 if output_len == 1 else (token_timestamps[-1] - token_timestamps[0]) / (output_len-1)
        self.tpot = self.tpot * 1000    # s -> ms

def read_request_results(path: str) -> List[RequestResult]:
    with open(path, "r") as f:
        request_results: List[RequestResult] = [
            RequestResult(
                item["prompt_len"],
                item["output_len"],
                item["start_time"],
                item["end_time"],
                item["token_timestamps"],
                json_decode_lifetime_events(item["lifecycle_events"]) if item.get("lifecycle_events", None) is not None else None
            )
            for item in json.load(f)
        ]
    return request_results

def count_valid_results(request_results: list[RequestResult], ttft: float, tpot: float) -> int:
    """
    count_valid_results: Count the number of requests that satisfy the given TTFT and TPOT.
    """
    count = 0
    for req in request_results:
        if req.ttft <= ttft and req.tpot <= tpot:
            count += 1
    return count

def get_slo_attainment(request_results: list[RequestResult], ttft: float, tpot: float) -> float:
    """
    get_slo_attainment: Get the SLO attainment of the given request results under the given TTFT and TPOT.
    """
    return count_valid_results(request_results, ttft, tpot) / len(request_results)

def slo_percentile(request_results: list[RequestResult], percentile: int) -> Tuple[float, float]:
    '''
    get percentile of TTFT and TPOT like P90, P95, P99
    '''
    ttft, tpot = [], []
    for req in request_results:
        ttft.append(req.ttft)
        tpot.append(req.tpot)
    return np.percentile(ttft, percentile), np.percentile(tpot, percentile)