import torch
import time
import numpy as np


class Benchmark:
    statistics = {}
    current_times = {}

    @classmethod
    def start_measure(cls, name):
        if name in cls.current_times:
            raise ValueError(f"Measurement for {name} start second time!")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cls.current_times[name] = time.time()
        if not name in cls.statistics:
            cls.statistics[name] = []

    @classmethod
    def end_measure(cls, name):
        if name not in cls.current_times:
            raise ValueError(f"Measurement for {name} ends, but not started!")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cls.statistics[name].append(time.time() - cls.current_times[name])
        cls.current_times.pop(name)

    @classmethod
    def print_results(cls):
        for name, stat in cls.statistics.items():
            print(f"{name}:: all: {np.sum(stat):.4f} runs: {len(stat)}")
