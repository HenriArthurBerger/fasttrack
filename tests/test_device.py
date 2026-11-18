"""One experiment file runs on any machine: a GPU it lacks falls back to the CPU."""
import pytest
import torch

from fasttrack import device


def test_auto_and_missing_gpus_resolve_to_something_this_machine_has():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert device.resolve("cpu") == "cpu"
    assert device.resolve("auto").startswith("cuda:" if n else "cpu")
    with pytest.warns(RuntimeWarning):
        assert device.resolve(f"cuda:{n + 3}") == "cpu"
    if not n:
        assert device.resolve("cuda:0") == "cpu"
    assert device.resolve("mps") in ("mps", "cpu")
    assert "torch" in device.describe()


def test_pipeline_config_holds_a_usable_device():
    from fasttrack.experiment import new_experiment, pipeline_config
    cfg = pipeline_config(new_experiment(pipeline={"device": "cuda:99"}))
    torch.zeros(1, device=cfg["device"])
