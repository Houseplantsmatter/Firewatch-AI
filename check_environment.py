"""Print the installed YOLO/PyTorch environment and available accelerators."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    try:
        import torch
        import ultralytics
    except ImportError as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        print(
            "Install the project requirements, then run this command again.",
            file=sys.stderr,
        )
        return 2

    print(f"Python:      {platform.python_version()}")
    print(f"Platform:    {platform.platform()}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA built:  {torch.version.cuda or 'no'}")
    print(f"CUDA ready:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA GPUs:   {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"  [{index}] {torch.cuda.get_device_name(index)}")

    mps = getattr(torch.backends, "mps", None)
    print(f"MPS ready:   {bool(mps is not None and mps.is_available())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())