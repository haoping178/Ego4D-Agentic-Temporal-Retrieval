import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from qwen_embedder_utils import load_qwen_embedder, to_normalized_numpy


# ============================================================
# 路徑與設定
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
STAGE1_DIR = BASE_DIR / "stage1"
STAGE1_RESULTS_JSON = STAGE1_DIR / "retrieval_results/qwen_top30_candidates.json"
SECOND_EMBEDDING_NPY = STAGE1_DIR / "second_embeddings/qwen_second_embeddings.npy"
SECOND_METADATA_JSON = STAGE1_DIR / "second_embeddings/qwen_second_metadata.json"
OUTPUT_JSON = Path(__file__).resolve().parent / "retrieval_results/qwen_query_segments.json"

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
QUERY_INSTRUCTION = "Retrieve the video moment that answers this egocentric query."

QUERY_BATCH_SIZE = 64

# 每秒分數平滑；1 代表不平滑。
SMOOTH_KERNEL = 1

# 以每秒高分位置建立不同長度的候選區間。
CANDIDATE_DURATIONS = (
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    12.0,
    16.0,
)

# 取分數最高的前 N 個每秒位置建立候選。
PEAK_COUNT = 30

TOP_K_SEGMENTS = 30

# 保留更多重疊但邊界不同的候選。
NMS_IOU_THRESHOLD = 0.3


# ============================================================
# Top-1 Re-ranking 設定
# ============================================================
RERANK_PROPOSAL_WEIGHT = 0.4
RERANK_CONTRAST_WEIGHT = 0.6
RERANK_BACKGROUND_MARGIN_SEC = 3.0

SAVE_DEBUG_TIMELINE = False


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"找不到：{path.resolve()}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def extract_items(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError("Metadata 格式錯誤")


def clean_query(text: str) -> str:
    text = text.strip()
    for prefix in ("Query Text:", "query text:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def load_queries() -> list[dict]:
    """讀取 Stage 1 的 query 與 Top-30 秒級候選。"""
    data = load_json(STAGE1_RESULTS_JSON)
    if not isinstance(data, list):
        raise ValueError("Stage 1 retrieval JSON 必須是 list")

    queries: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        query = clean_query(str(item.get("query", "") or ""))
        template = clean_query(str(item.get("template", "") or ""))
        queries.append({
            "query_id": str(item.get("query_id", "")),
            "clip_uid": str(item.get("clip_uid", "")),
            "template": template,
            "query": query,
            "encoder_query": query or template or "unknown action",
            "stage1_candidates": item.get("candidates", []),
        })
    return queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ego4D Query → temporal segments（Qwen3-VL，左側完整流程，無 Memory）"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="評估題數；省略時會互動詢問，Enter 表示全部",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("sequential", "random", "stratified"),
        default="sequential",
        help="題目抽樣方式。正式比較建議 random 或 stratified",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-timeline",
        action="store_true",
        help="保存每秒分數（檔案會顯著變大）",
    )
    return parser.parse_args()


def ask_query_count(total: int) -> int:
    while True:
        raw = input(
            f"可評估 Query 共 {total} 題，請輸入題數（Enter = 全部）："
        ).strip()
        if raw == "":
            return total
        try:
            value = int(raw)
        except ValueError:
            print("請輸入整數")
            continue
        if value <= 0:
            print("題數必須大於 0")
            continue
        return min(value, total)


def select_queries(
    queries: list[dict], count: int, mode: str, seed: int
) -> list[dict]:
    count = min(count, len(queries))
    if count == len(queries) or mode == "sequential":
        return queries[:count]

    rng = np.random.default_rng(seed)
    if mode == "random":
        indices = np.sort(rng.choice(len(queries), size=count, replace=False))
        return [queries[int(i)] for i in indices]

    # 依 template 比例分層抽樣，降低只取前 N 題造成的偏差。
    groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(queries):
        groups[str(item.get("template", ""))].append(index)

    selected: list[int] = []
    remaining: list[int] = []
    for indices in groups.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        quota = int(np.floor(count * len(indices) / len(queries)))
        selected.extend(shuffled[:quota].tolist())
        remaining.extend(shuffled[quota:].tolist())

    if len(selected) < count:
        remaining_array = np.asarray(remaining, dtype=np.int64)
        rng.shuffle(remaining_array)
        selected.extend(remaining_array[:count - len(selected)].tolist())

    return [queries[i] for i in sorted(selected[:count])]


# ============================================================
# Qwen3-VL Query Encoder
# ============================================================
def encode_queries(model, queries: list[dict]) -> np.ndarray:
    """
    使用 Qwen3-VL-Embedding-2B 編碼 Query。

    題目來源與左側流程一致：優先使用 query；query 為空時使用
    template；兩者皆空時使用固定占位文字。
    """
    texts = [
        item.get("encoder_query") or item.get("query") or "unknown action"
        for item in queries
    ]
    all_embeddings: list[np.ndarray] = []

    for start in tqdm(
        range(0, len(texts), QUERY_BATCH_SIZE),
        desc="Qwen Query Encoding",
    ):
        batch_texts = texts[start:start + QUERY_BATCH_SIZE]
        inputs = [
            {"text": text, "instruction": QUERY_INSTRUCTION}
            for text in batch_texts
        ]

        embeddings = to_normalized_numpy(model.process(inputs))
        if len(embeddings) != len(batch_texts):
            raise RuntimeError(
                f"Query embedding 數量錯誤：{len(embeddings)} != {len(batch_texts)}"
            )
        all_embeddings.append(embeddings.astype(np.float32, copy=False))

    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


# ============================================================
# Direct Query-Video 每秒計分
# ============================================================
def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-8, None)


def smooth_scores(scores: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or len(scores) < 2:
        return scores.copy()
    kernel_size = min(kernel_size, len(scores))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    left = kernel_size // 2
    right = kernel_size - 1 - left
    padded = np.pad(scores, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def score_clip_timeline(
    query_embedding: np.ndarray,
    clip_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    使用 Query embedding 與每秒 Video embedding 的 cosine similarity 計分。

    Returns:
        final_scores: 經平滑後的每秒分數。
        direct_scores: 原始 Query-Video cosine similarity。
    """
    query = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-8)
    clip = normalize_rows(clip_embeddings)

    direct_scores = (clip @ query).astype(np.float32)
    final_scores = smooth_scores(direct_scores, SMOOTH_KERNEL)

    return final_scores, direct_scores


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-8:
        return np.ones_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def calculate_local_contrast(
    timeline: list[dict],
    start_sec: float,
    end_sec: float,
    margin_sec: float,
) -> float:
    inside_scores: list[float] = []
    background_scores: list[float] = []

    background_start = start_sec - margin_sec
    background_end = end_sec + margin_sec

    for item in timeline:
        item_start = float(item["start_sec"])
        item_end = float(item["end_sec"])
        item_score = float(item["score"])

        overlaps_candidate = item_end > start_sec and item_start < end_sec
        overlaps_background_window = (
            item_end > background_start and item_start < background_end
        )

        if overlaps_candidate:
            inside_scores.append(item_score)
        elif overlaps_background_window:
            background_scores.append(item_score)

    if not inside_scores or not background_scores:
        return 0.0

    return float(np.mean(inside_scores) - np.mean(background_scores))


# ============================================================
# Timeline → 多長度候選 Segment
# ============================================================
def temporal_iou(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return 0.0 if union <= 0 else intersection / union


def build_candidate(
    timeline: list[dict],
    peak_index: int,
    target_duration: float,
) -> dict:
    peak_time = (
        float(timeline[peak_index]["start_sec"])
        + float(timeline[peak_index]["end_sec"])
    ) / 2.0
    clip_start = float(timeline[0]["start_sec"])
    clip_end = float(timeline[-1]["end_sec"])

    wanted_start = max(clip_start, peak_time - target_duration / 2.0)
    wanted_end = min(clip_end, peak_time + target_duration / 2.0)

    included = [
        item
        for item in timeline
        if float(item["end_sec"]) > wanted_start
        and float(item["start_sec"]) < wanted_end
    ]
    if not included:
        included = [timeline[peak_index]]

    scores = np.asarray(
        [item["score"] for item in included],
        dtype=np.float32,
    )
    direct_scores = np.asarray(
        [item["direct_query_video_score"] for item in included],
        dtype=np.float32,
    )

    best_second = max(included, key=lambda item: item["score"])
    start_sec = float(included[0]["start_sec"])
    end_sec = float(included[-1]["end_sec"])
    duration = max(end_sec - start_sec, 1e-6)

    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))

    # 候選區間原始分數：最高秒分數與區間平均分數的加權。
    proposal_score = 0.7 * max_score + 0.3 * mean_score

    top_k = min(3, len(scores))
    topk_scores = np.partition(scores, len(scores) - top_k)[-top_k:]
    topk_mean_score = float(np.mean(topk_scores))

    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration": duration,
        "proposal_score": proposal_score,
        "segment_score": proposal_score,
        "max_score": max_score,
        "mean_score": mean_score,
        "topk_mean_score": topk_mean_score,
        "score_std": float(np.std(scores)),
        "direct_mean_score": float(np.mean(direct_scores)),
        "peak_second_index": best_second["second_index"],
    }


def rerank_candidates(
    timeline: list[dict],
    candidates: list[dict],
) -> list[dict]:
    if not candidates:
        return []

    proposal_scores = np.asarray(
        [float(candidate["proposal_score"]) for candidate in candidates],
        dtype=np.float32,
    )

    contrast_scores = np.asarray(
        [
            calculate_local_contrast(
                timeline=timeline,
                start_sec=float(candidate["start_sec"]),
                end_sec=float(candidate["end_sec"]),
                margin_sec=RERANK_BACKGROUND_MARGIN_SEC,
            )
            for candidate in candidates
        ],
        dtype=np.float32,
    )

    proposal_normalized = minmax_normalize(proposal_scores)
    contrast_normalized = minmax_normalize(contrast_scores)

    for index, candidate in enumerate(candidates):
        rerank_score = (
            RERANK_PROPOSAL_WEIGHT * float(proposal_normalized[index])
            + RERANK_CONTRAST_WEIGHT * float(contrast_normalized[index])
        )
        candidate["local_contrast_score"] = float(contrast_scores[index])
        candidate["rerank_score"] = rerank_score
        candidate["segment_score"] = rerank_score

    candidates.sort(key=lambda item: item["segment_score"], reverse=True)
    return candidates


def select_segments(
    timeline: list[dict],
    candidate_seconds: set[int] | None = None,
) -> list[dict]:
    if not timeline:
        return []

    timeline = sorted(timeline, key=lambda item: float(item["start_sec"]))
    if candidate_seconds:
        peak_indices = [
            index
            for index, item in enumerate(timeline)
            if int(item["second_index"]) in candidate_seconds
        ]
        peak_indices.sort(
            key=lambda index: float(timeline[index]["score"]),
            reverse=True,
        )
        peak_indices = peak_indices[:min(PEAK_COUNT, len(peak_indices))]
    else:
        peak_indices = np.argsort(
            [float(item["score"]) for item in timeline]
        )[::-1][:min(PEAK_COUNT, len(timeline))]

    candidates = [
        build_candidate(timeline, int(peak), duration)
        for peak in peak_indices
        for duration in CANDIDATE_DURATIONS
    ]

    # 先以原始 proposal 分數排序，再做二階段重排。
    candidates.sort(key=lambda item: item["proposal_score"], reverse=True)
    candidates = rerank_candidates(timeline, candidates)

    selected: list[dict] = []
    rejected: list[dict] = []
    for candidate in candidates:
        if any(
            temporal_iou(
                candidate["start_sec"],
                candidate["end_sec"],
                kept["start_sec"],
                kept["end_sec"],
            ) >= NMS_IOU_THRESHOLD
            for kept in selected
        ):
            rejected.append(candidate)
            continue

        selected.append(candidate)
        if len(selected) >= TOP_K_SEGMENTS:
            break

    # 候選重疊過多時仍補滿 Top-30，讓下一階段取得固定數量的區間。
    if len(selected) < TOP_K_SEGMENTS:
        for candidate in rejected:
            selected.append(candidate)
            if len(selected) >= TOP_K_SEGMENTS:
                break

    return selected


def main() -> None:
    args = parse_args()

    queries = load_queries()
    second_embeddings = np.load(SECOND_EMBEDDING_NPY).astype(np.float32)
    second_items = extract_items(load_json(SECOND_METADATA_JSON))

    if len(second_embeddings) != len(second_items):
        raise ValueError("Second embedding 與 metadata 數量不同")
    if second_embeddings.ndim != 2:
        raise ValueError(
            f"Second embedding 必須是二維陣列，目前 shape={second_embeddings.shape}"
        )

    seconds_by_clip: dict[str, list[dict]] = defaultdict(list)
    for index, item in enumerate(second_items):
        clip_uid = str(item.get("clip_uid", ""))
        if not clip_uid:
            continue
        seconds_by_clip[clip_uid].append({
            **item,
            "embedding_index": index,
        })

    # 以 nlq_val.json 的全部 Query 為準，不因 clip 缺少 embedding 而刪題。
    count = (
        ask_query_count(len(queries))
        if args.count is None
        else min(max(args.count, 1), len(queries))
    )
    selected_queries = select_queries(
        queries,
        count,
        args.sample_mode,
        args.seed,
    )

    print(
        f"選取 {len(selected_queries)} 題；"
        f"模式={args.sample_mode}；seed={args.seed}"
    )
    if args.sample_mode == "sequential" and count < len(queries):
        print(
            "提醒：sequential 是前 N 題，"
            "跨題數比較建議使用 --sample-mode stratified"
        )

    model = load_qwen_embedder(MODEL_NAME)
    query_embeddings = encode_queries(model, selected_queries)

    if query_embeddings.ndim != 2 or query_embeddings.shape[0] != len(selected_queries):
        raise ValueError(
            "Query embedding shape 錯誤："
            f"{query_embeddings.shape}，預期第一維為 {len(selected_queries)}"
        )
    if query_embeddings.shape[1] != second_embeddings.shape[1]:
        raise ValueError(
            "Query 與 Video embedding 維度不同："
            f"query={query_embeddings.shape[1]}, "
            f"video={second_embeddings.shape[1]}。"
            "請確認兩者皆使用相同的 Qwen3-VL-Embedding-2B 重新產生。"
        )
    results: list[dict] = []

    for query_info, query_embedding in tqdm(
        zip(selected_queries, query_embeddings),
        total=len(selected_queries),
        desc="Query → Segment",
    ):
        clip_uid = query_info["clip_uid"]

        # 題目仍保留；若此 clip 沒有 Video Embedding，輸出空 predictions。
        if clip_uid not in seconds_by_clip:
            results.append({
                **query_info,
                "sampling": {
                    "mode": args.sample_mode,
                    "seed": args.seed,
                },
                "embedding_available": False,
                "predictions": [],
            })
            continue

        clip_seconds = sorted(
            seconds_by_clip[clip_uid],
            key=lambda item: float(item["start_sec"]),
        )
        clip_indices = np.asarray(
            [item["embedding_index"] for item in clip_seconds],
            dtype=np.int64,
        )
        clip_embeddings = second_embeddings[clip_indices]

        final_scores, direct_scores = score_clip_timeline(
            query_embedding=query_embedding,
            clip_embeddings=clip_embeddings,
        )

        timeline: list[dict] = []
        for i, second_item in enumerate(clip_seconds):
            timeline.append({
                "second_index": second_item["second_index"],
                "start_sec": float(second_item["start_sec"]),
                "end_sec": float(second_item["end_sec"]),
                "score": float(final_scores[i]),
                "direct_query_video_score": float(direct_scores[i]),
            })

        stage1_candidates = query_info.get("stage1_candidates", [])
        candidate_seconds = {
            int(candidate["second"])
            for candidate in stage1_candidates
            if isinstance(candidate, dict) and candidate.get("second") is not None
        }
        segments = select_segments(timeline, candidate_seconds)
        for rank, segment in enumerate(segments, start=1):
            segment["rank"] = rank

        result = {
            **query_info,
            "sampling": {
                "mode": args.sample_mode,
                "seed": args.seed,
            },
            "embedding_available": True,
            "scoring": {
                "architecture": "direct_query_video_with_left_reranking",
                "model": MODEL_NAME,
                "query_instruction": QUERY_INSTRUCTION,
                "smooth_kernel": SMOOTH_KERNEL,
                "proposal_formula": "0.7 * max + 0.3 * mean",
                "rerank_weights": {
                    "proposal": RERANK_PROPOSAL_WEIGHT,
                    "local_contrast": RERANK_CONTRAST_WEIGHT,
                },
            },
            "predictions": segments,
        }
        if SAVE_DEBUG_TIMELINE or args.debug_timeline:
            result["timeline"] = timeline
        results.append(result)

    save_json(OUTPUT_JSON, results)
    print(f"完成：{OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()