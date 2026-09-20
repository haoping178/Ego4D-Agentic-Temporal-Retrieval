from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_REPO = Path(
    os.environ.get(
        "QWEN3_VL_EMBEDDING_REPO",
        str(PROJECT_DIR.parent / "Qwen3-VL-Embedding")
    )
)


def import_embedder_class():
    """Load Qwen's official Qwen3VLEmbedder class from the cloned repository."""
    repo = DEFAULT_REPO.expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(
            "找不到 Qwen3-VL-Embedding 官方 repository："
            f"{repo}\n"
            "請先 git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git，"
            "或設定環境變數 QWEN3_VL_EMBEDDING_REPO。"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    except Exception as error:
        raise ImportError(
            "無法匯入 Qwen3VLEmbedder。請依官方 repository 安裝環境與相依套件。"
        ) from error
    return Qwen3VLEmbedder


def load_qwen_embedder(
    model_name_or_path: str,
    *,
    max_length: int = 8192,
    min_pixels: int = 4096,
    max_pixels: int = 448 * 448,
    total_pixels: int = 8 * 448 * 448,
    fps: float = 8.0,
    max_frames: int = 8,
):
    Qwen3VLEmbedder = import_embedder_class()

    kwargs: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "max_length": max_length,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "total_pixels": total_pixels,
        "fps": fps,
        "max_frames": max_frames,
    }

    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        # Windows 上沒有 flash-attn 時，SDPA 最穩定。
        kwargs["attn_implementation"] = os.environ.get(
            "QWEN_ATTN_IMPLEMENTATION", "sdpa"
        )

    print(f"Model：{model_name_or_path}")
    print(f"Device：{'cuda' if torch.cuda.is_available() else 'cpu'}")
    return Qwen3VLEmbedder(**kwargs)


def to_normalized_numpy(output: Any) -> np.ndarray:
    if isinstance(output, dict):
        for key in ("embeddings", "embedding", "features", "last_hidden_state"):
            if key in output:
                output = output[key]
                break
    if isinstance(output, (tuple, list)) and output and not isinstance(output[0], (float, int)):
        # 官方 process 通常直接回傳 Tensor；此處只處理包裝型輸出。
        if len(output) == 1:
            output = output[0]
    if isinstance(output, np.ndarray):
        tensor = torch.from_numpy(output)
    elif isinstance(output, torch.Tensor):
        tensor = output.detach().float().cpu()
    else:
        tensor = torch.as_tensor(output, dtype=torch.float32)

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Embedding shape 錯誤：{tuple(tensor.shape)}")

    tensor = F.normalize(tensor, p=2, dim=-1)
    return tensor.numpy().astype(np.float32)
