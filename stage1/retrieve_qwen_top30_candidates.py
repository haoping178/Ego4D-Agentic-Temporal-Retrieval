import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from qwen_embedder_utils import (
    load_qwen_embedder,
    to_normalized_numpy,
)


# ============================================================
# 路徑與設定
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

NLQ_JSON = Path(
    "F:/Qwen_agent 20260815/nlq_val.json"
)

SECOND_EMBEDDING_NPY = (
    BASE_DIR
    / "second_embeddings"
    / "qwen_second_embeddings.npy"
)

SECOND_METADATA_JSON = (
    BASE_DIR
    / "second_embeddings"
    / "qwen_second_metadata.json"
)

OUTPUT_JSON = (
    BASE_DIR
    / "retrieval_results"
    / "qwen_top30_candidates.json"
)


# ============================================================
# 模型
# ============================================================
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

QUERY_INSTRUCTION = (
    "Retrieve the video moment that answers "
    "this egocentric query."
)

QUERY_BATCH_SIZE = 64

TOP_K = 30


# ============================================================
# 基本工具
# ============================================================
def load_json(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"找不到：{path.resolve()}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def extract_items(
    data: Any,
) -> list[dict]:

    if isinstance(data, list):
        return data

    if (
        isinstance(data, dict)
        and isinstance(
            data.get("items"),
            list,
        )
    ):
        return data["items"]

    raise ValueError(
        "Metadata 格式錯誤"
    )


def clean_query(
    text: str,
) -> str:

    text = text.strip()

    for prefix in (
        "Query Text:",
        "query text:",
    ):

        if text.startswith(prefix):

            text = text[
                len(prefix):
            ].strip()

    return text


# ============================================================
# 讀取 Ego4D NLQ Query
# ============================================================
def load_queries() -> list[dict]:

    data = load_json(
        NLQ_JSON
    )

    queries: list[dict] = []

    query_index = 0

    for video in data.get(
        "videos",
        [],
    ):

        for clip in video.get(
            "clips",
            [],
        ):

            clip_uid = str(
                clip.get(
                    "clip_uid",
                    "",
                )
            )

            for annotation in clip.get(
                "annotations",
                [],
            ):

                for item in annotation.get(
                    "language_queries",
                    [],
                ):

                    template = clean_query(
                        str(
                            item.get(
                                "template",
                                "",
                            )
                            or ""
                        )
                    )

                    query = clean_query(
                        str(
                            item.get(
                                "query",
                                "",
                            )
                            or ""
                        )
                    )

                    encoder_query = (
                        query
                        or template
                        or "unknown action"
                    )

                    queries.append(
                        {
                            "query_id": (
                                f"query_"
                                f"{query_index + 1:04d}"
                            ),
                            "clip_uid": clip_uid,
                            "query": query,
                            "template": template,
                            "encoder_query": (
                                encoder_query
                            ),
                        }
                    )

                    query_index += 1

    return queries


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Ego4D Query → "
            "Top-30 candidate seconds "
            "+ full second scores"
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "只處理前 N 題；"
            "省略則處理全部 Query"
        ),
    )

    return parser.parse_args()


# ============================================================
# Qwen3-VL Query Encoder
# ============================================================
def encode_queries(
    model,
    queries: list[dict],
) -> np.ndarray:

    texts = [
        item["encoder_query"]
        for item in queries
    ]

    all_embeddings: list[
        np.ndarray
    ] = []

    for start in tqdm(
        range(
            0,
            len(texts),
            QUERY_BATCH_SIZE,
        ),
        desc="Qwen Query Encoding",
    ):

        batch_texts = texts[
            start:
            start + QUERY_BATCH_SIZE
        ]

        inputs = [
            {
                "text": text,
                "instruction": (
                    QUERY_INSTRUCTION
                ),
            }
            for text in batch_texts
        ]

        embeddings = (
            to_normalized_numpy(
                model.process(inputs)
            )
        )

        all_embeddings.append(
            embeddings.astype(
                np.float32,
                copy=False,
            )
        )

    if not all_embeddings:

        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    return np.concatenate(
        all_embeddings,
        axis=0,
    )


# ============================================================
# Normalize
# ============================================================
def normalize_rows(
    array: np.ndarray,
) -> np.ndarray:

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    return array / np.clip(
        norms,
        1e-8,
        None,
    )


# ============================================================
# 計算該 Query 對 Clip 每一秒 similarity
#
# 回傳：
#
# [
#   {
#       "second": 0,
#       "similarity": 0.123
#   },
#   ...
# ]
# ============================================================
def get_second_scores(
    query_embedding: np.ndarray,
    clip_seconds: list[dict],
    clip_embeddings: np.ndarray,
) -> tuple[
    np.ndarray,
    list[dict],
]:

    # --------------------------------------------------------
    # Query normalize
    # --------------------------------------------------------
    query_embedding = (
        query_embedding
        / max(
            float(
                np.linalg.norm(
                    query_embedding
                )
            ),
            1e-8,
        )
    )

    # --------------------------------------------------------
    # Video per-second normalize
    # --------------------------------------------------------
    clip_embeddings = (
        normalize_rows(
            clip_embeddings
        )
    )

    # --------------------------------------------------------
    # Cosine similarity
    #
    # 因為兩邊都 L2 normalize：
    #
    # cosine = dot product
    # --------------------------------------------------------
    similarities = (
        clip_embeddings
        @ query_embedding
    )

    # --------------------------------------------------------
    # 建立完整每秒 similarity
    # --------------------------------------------------------
    second_scores: list[
        dict
    ] = []

    for index, second_item in enumerate(
        clip_seconds
    ):

        second = second_item.get(
            "second_index"
        )

        if second is None:

            second = int(
                round(
                    float(
                        second_item.get(
                            "start_sec",
                            0,
                        )
                    )
                )
            )

        second_scores.append(
            {
                "second": int(second),
                "similarity": float(
                    similarities[index]
                ),
            }
        )

    return (
        similarities,
        second_scores,
    )


# ============================================================
# 從完整 similarity 取得 Top-K
# ============================================================
def get_top_k_candidates(
    similarities: np.ndarray,
    clip_seconds: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:

    k = min(
        top_k,
        len(similarities),
    )

    # similarity 高 → 低
    top_indices = np.argsort(
        similarities
    )[::-1][:k]

    candidates: list[
        dict
    ] = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        second_item = (
            clip_seconds[
                int(index)
            ]
        )

        second = second_item.get(
            "second_index"
        )

        if second is None:

            second = int(
                round(
                    float(
                        second_item.get(
                            "start_sec",
                            0,
                        )
                    )
                )
            )

        candidates.append(
            {
                "rank": rank,
                "second": int(second),
                "similarity": float(
                    similarities[
                        int(index)
                    ]
                ),
            }
        )

    return candidates


# ============================================================
# Main
# ============================================================
def main() -> None:

    args = parse_args()

    # ========================================================
    # Query
    # ========================================================
    queries = load_queries()

    if args.count is not None:

        queries = queries[
            :max(
                0,
                args.count,
            )
        ]

    print()
    print("=" * 70)
    print("Stage 1 Retrieval")
    print("=" * 70)

    print(
        f"Query 數量      : "
        f"{len(queries)}"
    )

    print(
        f"Top-K           : "
        f"{TOP_K}"
    )

    print(
        "輸出內容        : "
        "Top30 + full second_scores"
    )

    print("=" * 70)


    # ========================================================
    # Load video second embeddings
    # ========================================================
    second_embeddings = np.load(
        SECOND_EMBEDDING_NPY
    ).astype(
        np.float32
    )

    second_items = extract_items(
        load_json(
            SECOND_METADATA_JSON
        )
    )

    if (
        len(second_embeddings)
        != len(second_items)
    ):

        raise ValueError(
            "Second embedding "
            "與 metadata 數量不同"
        )


    # ========================================================
    # Clip UID → 每秒 metadata
    # ========================================================
    seconds_by_clip: dict[
        str,
        list[dict],
    ] = defaultdict(
        list
    )

    for (
        embedding_index,
        item,
    ) in enumerate(
        second_items
    ):

        clip_uid = str(
            item.get(
                "clip_uid",
                "",
            )
        )

        if not clip_uid:
            continue

        seconds_by_clip[
            clip_uid
        ].append(
            {
                **item,
                "embedding_index": (
                    embedding_index
                ),
            }
        )


    # ========================================================
    # Load Qwen
    # ========================================================
    model = load_qwen_embedder(
        MODEL_NAME
    )


    # ========================================================
    # Encode all queries
    # ========================================================
    query_embeddings = (
        encode_queries(
            model,
            queries,
        )
    )


    # ========================================================
    # Check dimension
    # ========================================================
    if (
        query_embeddings.shape[1]
        != second_embeddings.shape[1]
    ):

        raise ValueError(
            "Query 與 Video embedding "
            "維度不同："
            f"query="
            f"{query_embeddings.shape[1]}, "
            f"video="
            f"{second_embeddings.shape[1]}"
        )


    # ========================================================
    # Results
    # ========================================================
    results: list[
        dict
    ] = []


    # ========================================================
    # 每個 Query
    # ========================================================
    for (
        query_info,
        query_embedding,
    ) in tqdm(

        zip(
            queries,
            query_embeddings,
        ),

        total=len(
            queries
        ),

        desc=(
            "Query → "
            "Top30 + Second Scores"
        ),
    ):

        clip_uid = (
            query_info[
                "clip_uid"
            ]
        )


        # ----------------------------------------------------
        # Clip embedding 不存在
        # ----------------------------------------------------
        if (
            clip_uid
            not in seconds_by_clip
        ):

            results.append(
                {
                    "query_id": (
                        query_info[
                            "query_id"
                        ]
                    ),
                    "clip_uid": (
                        clip_uid
                    ),
                    "query": (
                        query_info[
                            "query"
                        ]
                    ),
                    "template": (
                        query_info[
                            "template"
                        ]
                    ),

                    "top_k": TOP_K,

                    "candidates": [],

                    "second_scores": [],

                    "second_score_count": 0,
                }
            )

            continue


        # ----------------------------------------------------
        # 該 Clip 所有秒數
        # ----------------------------------------------------
        clip_seconds = sorted(
            seconds_by_clip[
                clip_uid
            ],
            key=lambda item: float(
                item.get(
                    "start_sec",
                    0,
                )
            ),
        )


        # ----------------------------------------------------
        # 對應 embedding index
        # ----------------------------------------------------
        clip_indices = np.asarray(
            [
                item[
                    "embedding_index"
                ]
                for item
                in clip_seconds
            ],
            dtype=np.int64,
        )


        # ----------------------------------------------------
        # 該 Clip 每秒 embedding
        # ----------------------------------------------------
        clip_embeddings = (
            second_embeddings[
                clip_indices
            ]
        )


        # ====================================================
        # 1. 計算完整 Second Scores
        # ====================================================
        (
            similarities,
            second_scores,
        ) = get_second_scores(

            query_embedding=(
                query_embedding
            ),

            clip_seconds=(
                clip_seconds
            ),

            clip_embeddings=(
                clip_embeddings
            ),
        )


        # ====================================================
        # 2. 從完整 similarity 取 Top30
        # ====================================================
        candidates = (
            get_top_k_candidates(

                similarities=(
                    similarities
                ),

                clip_seconds=(
                    clip_seconds
                ),

                top_k=TOP_K,
            )
        )


        # ====================================================
        # 3. 儲存
        # ====================================================
        results.append(
            {
                "query_id": (
                    query_info[
                        "query_id"
                    ]
                ),

                "clip_uid": (
                    clip_uid
                ),

                "query": (
                    query_info[
                        "query"
                    ]
                ),

                "template": (
                    query_info[
                        "template"
                    ]
                ),

                # --------------------------------------------
                # Top30
                # --------------------------------------------
                "top_k": TOP_K,

                "candidates": (
                    candidates
                ),

                # --------------------------------------------
                # 完整 similarity curve
                # --------------------------------------------
                "second_scores": (
                    second_scores
                ),

                "second_score_count": (
                    len(
                        second_scores
                    )
                ),
            }
        )


    # ========================================================
    # Save
    # ========================================================
    save_json(
        OUTPUT_JSON,
        results,
    )


    # ========================================================
    # Print
    # ========================================================
    print()
    print("=" * 70)

    print(
        "Stage 1 完成"
    )

    print("=" * 70)

    print(
        f"Output : "
        f"{OUTPUT_JSON.resolve()}"
    )

    print(
        f"Queries: "
        f"{len(results)}"
    )

    print()

    print(
        "每個 Query 現在包含："
    )

    print(
        "1. candidates      "
        "→ Top-30"
    )

    print(
        "2. second_scores   "
        "→ 該 Clip 全部秒數 similarity"
    )

    print("=" * 70)


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    main()