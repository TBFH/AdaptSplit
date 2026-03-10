from typing import Dict, List, Optional, Tuple
import time
import csv
import os
from datetime import datetime

class PipelineSimulator:
    def __init__(
        self,
        devices: List[str],
    ):
        self.devices = devices

        self.interval_multipler = 0.1
        self.interval_between_tokens = 2

        self.num_batches = len(devices)
        self.pp_records = [[] for _ in range(len(devices))]

    def _record(
        self,
        stage_id: int,
        req_id: int,
        desc: int,
        start_time: float,
        end_time: float,
    ):
        self.pp_records[stage_id].append({
            "stage_id": stage_id,
            "req_id": req_id,
            "desc": desc,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time
        })

    def _collect(self, record_dir: str):
        all_records = [record for stage_record in self.pp_records for record in stage_record]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(record_dir, f"pp-records_{timestamp}.csv")
        if record_dir and not os.path.exists(record_dir):
            os.makedirs(record_dir, exist_ok=True)
        fieldnames = list(all_records[0].keys())
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        print(f"Simulator pp_gantte saved to {filename}")

    def _plot(self):
        pass

    def sim(
        self,
        latency_power_pairs: List[Tuple[float, float]],
        output_len: int = 8,
        enable_record: bool = False,
        record_dir: Optional[str] = None
    ):
        assert (
            len(self.devices) == len(latency_power_pairs)
        ), "len(devices) does not fit len(latency_power_pairs)"

        if enable_record and not record_dir:
            record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pp_sim_records")

        latencys = {}
        powers = {}
        for device, (latency, power) in zip(self.devices, latency_power_pairs):
            latencys[device] = latency
            powers[device] = power

        finish_time = -1
        batch_latest = [0] * self.num_batches
        device_latest = [0] * self.num_batches

        for output_id in range(output_len):
            for batch_id in range(self.num_batches):
                for device_id, device in enumerate(self.devices):
                    latest = batch_latest[batch_id] if batch_latest[batch_id] > device_latest[device_id] else device_latest[device_id]
                    start_time = latest
                    end_time = latest + latencys[device]

                    if enable_record:
                        stage_id = device_id
                        req_id = batch_id
                        desc = output_id
                        self._record(stage_id, req_id, desc, start_time / 1000, end_time / 1000)  # ms -> s
                    
                    interval = self.interval_between_tokens if device_id == len(self.devices) - 1 else latencys[device] * self.interval_multipler
                    batch_latest[batch_id] = end_time + interval
                    device_latest[device_id] = end_time + latencys[device] * self.interval_multipler

                    finish_time = end_time / 1000  # ms -> s
        
        if enable_record:
            self._collect(record_dir=record_dir)
        
        num_tokens = self.num_batches * output_len
        throughput = num_tokens / finish_time
        avg_power = sum([power for power in powers.values()])
        energy_efficiency = throughput / avg_power

        return energy_efficiency


if __name__ == "__main__":
    from pipeline_predictor import PredictorBundle, load_predictor

    model = "Llama-2-7b-chat-hf"
    devices = ['jetson-64g-4', 'jetson-16g-2', 'jetson-16g-8', 'jetson-8g-1']

    predictors: Dict[str, PredictorBundle] = {}
    for device in devices:
        predictors[device] = load_predictor(
            checkpoint_path=f"/home/austin/repos/AdaptSplit/AdaptSplit/uneven_partitioning/checkpoints/{model}-{device}.pt",
            device="cuda"
        )

    simulator = PipelineSimulator(devices)

    def sim_ee(deployment: List[int], batch_size: int):
        pairs_predicted = []
        for idx, device in enumerate(devices):
            pairs_predicted.append(predictors[device].predict(deployment[idx], batch_size))
        ee = simulator.sim(
            latency_power_pairs=pairs_predicted,
            output_len=16,
            enable_record=True
        )
        return ee

    # Testing
    deployment = [12,9,9,2]
    batch_size = 54
    ee = sim_ee(
        deployment=deployment,
        batch_size=batch_size
    )
    print(
        f"model: {model} \t devices: {devices} \n"
        f"deployment: {deployment} \t batch_size: {batch_size} \n"
        f"simulated energy_efficiency: {ee}"
    )
