from pipeline_predictor import load_predictor

class Simulator:
    def __init__(self):
        pass

if __name__ == "__main__":
    predictor = load_predictor("/home/austin/repos/AdaptSplit/AdaptSplit/uneven_partitioning/checkpoints/opt-1.3b-jetson-64g-4.pt", device="cuda")
    latency_ms, power_w = predictor.predict(num_layer=7, batch_size=100)
    print("latency_ms =", latency_ms)
    print("power_w    =", power_w)