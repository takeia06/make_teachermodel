import os
import subprocess
import sys
import pytest


def _run(cmd):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True)


def test_imports_cpu():
    # Keep this extremely light; no heavy frameworks here
    import json  # noqa: F401
    import math  # noqa: F401


@pytest.mark.gpu
def test_cuda_and_optional_demo():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch not installed: {e}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this runner.")

    # Optional mini demo: run your own script if present.
    demo = os.path.join("mmdetection", "demo", "image_demo_tokens_compat.py")
    img = os.path.join("assets", "samples", "sample.png")
    cfg = os.path.join("mmdetection", "configs", "anomaly_finetune.py")
    ckpt = os.environ.get("SMOKE_CKPT", "")

    if not (
        os.path.exists(demo) and os.path.exists(img) and os.path.exists(cfg) and ckpt
    ):
        pytest.skip("Demo assets or ckpt not found; skipping actual detector smoke.")

    _run(
        [
            sys.executable,
            demo,
            img,
            cfg,
            "--weights",
            ckpt,
            "--texts",
            "red polyp",
            "--custom-entities",
            "--pred-score-thr",
            "0.25",
            "--print-result",
        ]
    )
