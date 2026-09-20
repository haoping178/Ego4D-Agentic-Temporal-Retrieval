import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from qwen_embedder_utils import load_qwen_embedder, to_normalized_numpy


NLQ_JSON = Path("nlq_val.json")
VIDEO_DIR = Path("F:/ego4d_data/v2/clips")

OUTPUT_DIR = Path("second_embeddings")
OUTPUT_EMBEDDING_NPY = OUTPUT_DIR / "qwen_second_embeddings.npy"
OUTPUT_METADATA_JSON = OUTPUT_DIR / "qwen_second_metadata.json"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
VIDEO_INSTRUCTION = "Represent this one-second egocentric video moment for text-to-video retrieval."

SECOND_SIZE = 1.0
SECOND_STRIDE = 1.0
SAMPLE_FRAMES = 8

# 一次送幾個「1 秒影片片段」給 Qwen。2B 建議 2；OOM 時改 1。
SECONDS_PER_CHUNK = 16
QWEN_BATCH_SIZE = 128

VIDEO_EXTENSIONS = {".mp4"}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def configure_runtime():
    cv2.setNumThreads(0)

    if DEVICE.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    print("=" * 60)
    print("CUDA 資訊")
    print("=" * 60)
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"CUDA：{torch.version.cuda}")
    print(f"PyTorch：{torch.__version__}")


def count_questions(clip: dict[str, Any]) -> int:
    total = 0

    for annotation in clip.get("annotations", []):
        if not isinstance(annotation, dict):
            continue

        queries = annotation.get("language_queries", [])

        if isinstance(queries, list):
            total += len(queries)

    return total


def load_available_clips() -> list[dict[str, Any]]:
    if not NLQ_JSON.exists():
        raise FileNotFoundError(
            f"找不到：{NLQ_JSON.resolve()}"
        )

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"找不到影片資料夾：{VIDEO_DIR.resolve()}"
        )

    with NLQ_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)

    local_videos = {
        path.stem: path
        for path in VIDEO_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    }

    clips = []
    seen = set()

    for video in data.get("videos", []):
        if not isinstance(video, dict):
            continue

        video_uid = str(
            video.get("video_uid", "")
        ).strip()

        for clip in video.get("clips", []):
            if not isinstance(clip, dict):
                continue

            clip_uid = str(
                clip.get("clip_uid", "")
            ).strip()

            if not clip_uid or clip_uid in seen:
                continue

            seen.add(clip_uid)

            question_count = count_questions(clip)

            if question_count <= 0:
                continue

            video_path = local_videos.get(clip_uid)

            if video_path is None:
                continue

            clips.append({
                "video_uid": video_uid,
                "clip_uid": clip_uid,
                "question_count": question_count,
                "video_path": video_path,
            })

    if not clips:
        raise RuntimeError(
            "沒有找到可處理影片。"
        )

    return clips


def ask_question_count(total_questions: int) -> int:
    while True:
        raw = input(
            f"本機影片共涵蓋 {total_questions} 題，"
            "請輸入本次要處理幾題"
            "（Enter = 全部）："
        ).strip()

        if raw == "":
            return total_questions

        try:
            value = int(raw)
        except ValueError:
            print("請輸入整數。")
            continue

        if value <= 0:
            print("題數必須大於 0。")
            continue

        return min(value, total_questions)


def select_clips_by_questions() -> list[dict[str, Any]]:
    clips = load_available_clips()

    total_questions = sum(
        int(item["question_count"])
        for item in clips
    )

    target = ask_question_count(total_questions)

    selected = []
    accumulated = 0

    for item in clips:
        if accumulated >= target:
            break

        selected.append(item)
        accumulated += int(item["question_count"])

    print("=" * 60)
    print(f"目標題數：{target}")
    print(f"實際涵蓋題數：{accumulated}")
    print(f"處理影片數：{len(selected)}")
    print("=" * 60)

    return selected


def load_model():
    # 每個輸入是一秒、8 張 PIL frames。Qwen 直接輸出該秒的單一向量。
    return load_qwen_embedder(
        MODEL_NAME,
        max_pixels=448 * 448,
        total_pixels=SAMPLE_FRAMES * 448 * 448,
        fps=float(SAMPLE_FRAMES),
        max_frames=SAMPLE_FRAMES,
    )


def encode_chunk(
    model,
    flat_frames: list[Image.Image],
    second_count: int,
) -> np.ndarray:
    expected_count = second_count * SAMPLE_FRAMES
    if len(flat_frames) != expected_count:
        raise RuntimeError(
            f"Frame 數量錯誤：{len(flat_frames)} != {expected_count}"
        )

    second_frame_groups = [
        flat_frames[i * SAMPLE_FRAMES:(i + 1) * SAMPLE_FRAMES]
        for i in range(second_count)
    ]
    outputs = []
    for start in range(0, second_count, QWEN_BATCH_SIZE):
        groups = second_frame_groups[start:start + QWEN_BATCH_SIZE]
        inputs = [
            {
                "video": frames,
                "instruction": VIDEO_INSTRUCTION,
                "fps": float(SAMPLE_FRAMES),
                "max_frames": SAMPLE_FRAMES,
            }
            for frames in groups
        ]
        outputs.append(to_normalized_numpy(model.process(inputs)))

    matrix = np.concatenate(outputs, axis=0).astype(np.float32)
    if len(matrix) != second_count:
        raise RuntimeError(
            f"Second embedding 數量錯誤：{len(matrix)} != {second_count}"
        )
    return matrix


def get_video_info(
    capture: cv2.VideoCapture,
    video_path: Path,
) -> dict[str, float]:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0 or frame_count <= 0:
        raise ValueError(
            f"影片資訊無效：{video_path.name}"
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps,
    }


def create_second_windows(
    duration: float,
) -> list[tuple[float, float]]:
    windows = []
    start_sec = 0.0

    while start_sec < duration:
        end_sec = min(
            start_sec + SECOND_SIZE,
            duration,
        )

        windows.append((
            round(start_sec, 4),
            round(end_sec, 4),
        ))

        start_sec += SECOND_STRIDE

    return windows


def get_sample_times(
    start_sec: float,
    end_sec: float,
) -> list[float]:
    edges = np.linspace(
        start_sec,
        end_sec,
        SAMPLE_FRAMES + 1,
        dtype=np.float64,
    )

    return [
        float(value)
        for value in (
            edges[:-1] + edges[1:]
        ) / 2.0
    ]


def read_chunk_frames(
    capture: cv2.VideoCapture,
    video_path: Path,
    chunk_windows: list[tuple[float, float]],
) -> tuple[list[Image.Image], list[list[float]]]:
    flat_frames = []
    sampled_times_per_second = []
    last_valid_frame = None

    for start_sec, end_sec in chunk_windows:
        second_frames = []
        used_times = []

        for timestamp in get_sample_times(
            start_sec,
            end_sec,
        ):
            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000.0,
            )

            ok, frame_bgr = capture.read()

            if ok and frame_bgr is not None:
                frame_rgb = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                current_frame = Image.fromarray(
                    frame_rgb
                )

                last_valid_frame = current_frame
                second_frames.append(current_frame)
                used_times.append(timestamp)

            elif last_valid_frame is not None:
                second_frames.append(
                    last_valid_frame.copy()
                )
                used_times.append(timestamp)

        if not second_frames:
            raise RuntimeError(
                f"讀不到影格：{video_path.name} "
                f"{start_sec}-{end_sec}s"
            )

        while len(second_frames) < SAMPLE_FRAMES:
            second_frames.append(
                second_frames[-1].copy()
            )
            used_times.append(
                used_times[-1]
            )

        flat_frames.extend(
            second_frames[:SAMPLE_FRAMES]
        )

        sampled_times_per_second.append(
            used_times[:SAMPLE_FRAMES]
        )

    return flat_frames, sampled_times_per_second


def process_clip(
    clip_info: dict[str, Any],
    model,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    video_path = Path(
        clip_info["video_path"]
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"無法開啟：{video_path}"
        )

    try:
        info = get_video_info(
            capture,
            video_path,
        )

        windows = create_second_windows(
            info["duration"]
        )

        embeddings = []
        metadata = []

        for chunk_start in tqdm(
            range(
                0,
                len(windows),
                SECONDS_PER_CHUNK,
            ),
            desc=clip_info["clip_uid"],
            leave=False,
        ):
            chunk_end = min(
                chunk_start + SECONDS_PER_CHUNK,
                len(windows),
            )

            chunk_windows = windows[
                chunk_start:chunk_end
            ]

            try:
                flat_frames, sampled_times = (
                    read_chunk_frames(
                        capture,
                        video_path,
                        chunk_windows,
                    )
                )

                chunk_embeddings = encode_chunk(
                    model,
                    flat_frames,
                    len(chunk_windows),
                )

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA 記憶體不足。請降低 QWEN_BATCH_SIZE，"
                    "必要時也降低 SAMPLE_FRAMES。"
                )

            for local_index, embedding in enumerate(
                chunk_embeddings
            ):
                second_index = (
                    chunk_start + local_index
                )

                start_sec, end_sec = chunk_windows[
                    local_index
                ]

                embeddings.append(embedding)

                metadata.append({
                    "video_uid": clip_info[
                        "video_uid"
                    ],
                    "clip_uid": clip_info[
                        "clip_uid"
                    ],
                    "question_count": clip_info[
                        "question_count"
                    ],
                    "video_filename": video_path.name,
                    "second_index": second_index,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "sample_frames": SAMPLE_FRAMES,
                    "sampled_times": [
                        round(value, 6)
                        for value in sampled_times[
                            local_index
                        ]
                    ],
                    "fps": float(info["fps"]),
                    "video_duration": float(
                        info["duration"]
                    ),
                })

        return embeddings, metadata

    finally:
        capture.release()


def save_results(
    embeddings: list[np.ndarray],
    metadata: list[dict[str, Any]],
):
    if not embeddings:
        raise RuntimeError(
            "沒有產生任何每秒 Embedding"
        )

    matrix = np.stack(
        embeddings,
        axis=0,
    ).astype(np.float32)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_EMBEDDING_NPY,
        matrix,
    )

    output = {
        "model": MODEL_NAME,
        "video_instruction": VIDEO_INSTRUCTION,
        "second_size": SECOND_SIZE,
        "second_stride": SECOND_STRIDE,
        "sample_frames": SAMPLE_FRAMES,
        "seconds_per_chunk": SECONDS_PER_CHUNK,
        "qwen_batch_size": QWEN_BATCH_SIZE,
        "normalized": True,
        "embedding_shape": list(
            matrix.shape
        ),
        "items": metadata,
    }

    with OUTPUT_METADATA_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n完成")
    print(
        f"Embedding：{OUTPUT_EMBEDDING_NPY.resolve()}"
    )
    print(
        f"Metadata：{OUTPUT_METADATA_JSON.resolve()}"
    )
    print(f"Shape：{matrix.shape}")


def main():
    configure_runtime()

    selected_clips = (
        select_clips_by_questions()
    )

    model = load_model()

    all_embeddings = []
    all_metadata = []

    for clip_info in selected_clips:
        embeddings, metadata = process_clip(
            clip_info,
            model,
        )

        all_embeddings.extend(embeddings)
        all_metadata.extend(metadata)

    save_results(
        all_embeddings,
        all_metadata,
    )


if __name__ == "__main__":
    main()