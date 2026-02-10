from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


def _src_dir() -> Path:
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_uvcc_cuda_ext(*, verbose: bool = False) -> Any:
    """
    Build+load the UVCC CUDA extension (DPF/DCF kernels).

    This is intentionally lazy-loaded so CPU-only environments can still import `uvcc_party`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (torch.cuda.is_available() == False)")
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"torch.utils.cpp_extension not available: {e}") from e
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is not set; cannot build CUDA extension")

    here = _src_dir()
    cpp = here / "uvcc_cuda_ext.cpp"
    cu = here / "uvcc_dpf_dcf_kernels.cu"
    cu_oplut = here / "uvcc_oplut_kernels.cu"
    cu_matmul = here / "uvcc_matmul_u64_kernels.cu"
    if not cpp.exists() or not cu.exists():
        raise RuntimeError("missing cuda extension sources")
    if not cu_oplut.exists():
        raise RuntimeError("missing cuda extension sources (uvcc_oplut_kernels.cu)")
    if not cu_matmul.exists():
        raise RuntimeError("missing cuda extension sources (uvcc_matmul_u64_kernels.cu)")

    # Stable build id based on sources path; torch caches under ~/.cache/torch_extensions.
    name = "uvcc_cuda_ext_v1"

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "-lineinfo"]

    # Allow users to inject extra flags.
    if os.environ.get("UVCC_CUDA_EXT_DEBUG", "0") == "1":
        extra_cflags += ["-g"]
        extra_cuda_cflags += ["-G"]

    return load(
        name=name,
        sources=[str(cpp), str(cu), str(cu_oplut), str(cu_matmul)],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=bool(verbose),
    )


