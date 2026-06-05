"""GPU + sentiment smoke test.

Confirms the installed PyTorch build actually runs on this machine's GPU
(RTX 5070 Ti, Blackwell sm_120) and that FinBERT can classify finance text on
it. Two ways this is more than a `cuda.is_available()` check:

* it runs a real CUDA matmul, which is what surfaces an "unsupported arch /
  no kernel image" error if the torch build lacks sm_120 kernels;
* it then loads FinBERT and runs inference *on the GPU*.

On any CUDA problem it prints a diagnosis and exits non-zero.

Run:  uv run python scripts/smoke_test_gpu.py
"""

from __future__ import annotations

import torch

SAMPLE_SENTENCES = [
    "The company reported record quarterly profits, beating analyst expectations.",
    "Shares plunged after the firm slashed its full-year revenue guidance.",
    "The central bank left interest rates unchanged, in line with forecasts.",
]


def check_cuda() -> int:
    print(f"torch.__version__      : {torch.__version__}")
    print(f"torch.version.cuda     : {torch.version.cuda}")
    print(f"cuda.is_available()    : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\nDIAGNOSIS: CUDA is not available to PyTorch.")
        print("  - Confirm `nvidia-smi` works and shows the RTX 5070 Ti.")
        print("  - Confirm a CUDA-enabled torch build is installed (expected: +cu128).")
        print(f"  - Installed build reports CUDA = {torch.version.cuda!r} (None ⇒ CPU-only wheel).")
        return 1

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device name            : {name}")
    print(f"compute capability     : sm_{major}{minor}")
    print(f"total VRAM             : {total_gb:.1f} GiB")
    print(f"torch arch list        : {torch.cuda.get_arch_list()}")

    # The real test: actually launch a kernel on the GPU.
    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        print(f"GPU matmul check       : OK (result={c:.2f})")
    except RuntimeError as exc:
        print(f"\nDIAGNOSIS: a CUDA kernel failed to launch on {name} (sm_{major}{minor}).")
        print(f"  error: {exc}")
        print("  This usually means the torch build lacks kernels for this GPU arch.")
        print(f"  Installed arch list: {torch.cuda.get_arch_list()} (need sm_{major}{minor}).")
        return 1

    return 0


def check_finbert() -> int:
    print("\nLoading FinBERT (ProsusAI/finbert) on the GPU (first run downloads ~440 MB)...")
    try:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=0,  # GPU
        )
        results = classifier(SAMPLE_SENTENCES)
    except Exception as exc:  # noqa: BLE001 - report any model/download failure clearly
        print(f"\nDIAGNOSIS: FinBERT failed to load or run: {type(exc).__name__}: {exc}")
        print("  - First run needs internet to download the model from Hugging Face.")
        return 1

    print("\nFinBERT sentiment results:")
    for sentence, res in zip(SAMPLE_SENTENCES, results, strict=True):
        print(f"  [{res['label']:>8}  {res['score']:.3f}]  {sentence}")
    return 0


def main() -> int:
    rc = check_cuda()
    if rc != 0:
        return rc
    rc = check_finbert()
    if rc != 0:
        return rc
    print("\nGPU + FinBERT smoke test OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
