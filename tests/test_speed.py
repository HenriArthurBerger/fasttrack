"""Speed budget: learned models may at most double tracking time (benchmarks/inference.py tracks it on real data).

The seed finder replaces the classical centroid step and adds one encoder pass, so the pass may cost at most as
much as the whole classical pipeline. Timed here on synthetic frames, so it runs on any GPU. Calibrated on an idle
RTX 3090 (synthetic matches real 1152 px frames of the 2024-12-15 plate closely):

                         real (inference.py)   synthetic (here)
  stride 1 encoder        1.7x classical        1.6-1.9x      fails
  stride 2 encoder        0.43x                 0.40-0.49x    passes

Run it on an idle GPU: another job on the same card slows the classical pipeline's many small steps more than the
encoder. The real-data numbers are tracked in benchmarks/results/inference.csv."""

import time

import pytest
import torch

from fasttrack.config import make_config
from fasttrack.synthetic.generate import generate
from fasttrack.track import centroid_net
from fasttrack.track.batch import process_batch

LIMIT = 1.0   # encoder pass / classical pipeline


def _best(fn, repeat=3):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="timing needs a CUDA GPU")
def test_encoder_pass_keeps_learned_tracking_within_twice_classical():
    frames, _, _ = generate({"frames": 40, "size": 1024, "seed": 3})
    cfg = make_config({"device": "cuda:0"})
    frames = frames.to("cuda:0")
    classical = _best(lambda: process_batch(frames, cfg))
    net = centroid_net.build().to("cuda:0").eval().to(memory_format=torch.channels_last)
    nuclei = frames[:, cfg["nuclear_channel"]:cfg["nuclear_channel"] + 1].float()
    encoder = _best(lambda: centroid_net.score_map(net, nuclei))
    assert encoder <= LIMIT * classical, (f"encoder pass {encoder:.2f}s vs classical pipeline {classical:.2f}s on 40 synthetic frames "
                                         f"({encoder / classical:.2f}x, limit {LIMIT}x): learned tracking would go over 2x")
