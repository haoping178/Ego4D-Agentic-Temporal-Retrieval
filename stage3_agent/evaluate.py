from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from conf import EVAL_DIR, NLQ_VAL_JSON, OUTPUT_JSON


def norm_text(s: Any) -> str:
    text = str(s or "").strip().lower()
    text = re.sub(r"^query\s*text\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    return text.strip()


def temporal_iou(
    a0: float,
    a1: float,
    b0: float,
    b1: float,
) -> float:
    inter = max(
        0.0,
        min(a1, b1) - max(a0, b0),
    )

    union = max(a1, b1) - min(a0, b0)

    return inter / union if union > 0 else 0.0


def load(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save(
    path: Path,
    obj: Any,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def flatten_gt(
    nlq: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    依照 nlq_val.json 原始順序，
    將所有 Query 展平成一個 list。
    """

    rows = []
    global_index = 0

    for video in nlq.get("videos", []):
        video_uid = video.get("video_uid")

        for clip in video.get("clips", []):
            clip_uid = clip.get("clip_uid")

            for ann in clip.get("annotations", []):
                annotation_uid = ann.get(
                    "annotation_uid"
                )

                for query_idx, q in enumerate(
                    ann.get(
                        "language_queries",
                        [],
                    )
                ):
                    raw_query = q.get(
                        "query",
                        "",
                    )

                    # global_index 必須維持 nlq_val.json 的原始位置，
                    # 因此即使這題沒有 query，也仍先佔用原始 index。
                    current_global_index = global_index
                    global_index += 1

                    # Ego4D NLQ v2 validation 有 2 筆缺少有效自然語言 Query。
                    # 正式評估時直接排除，不建立 GT matcher entry。
                    if not norm_text(raw_query):
                        continue

                    rows.append(
                        {
                            "global_index": current_global_index,
                            "video_uid": video_uid,
                            "clip_uid": clip_uid,
                            "annotation_uid": annotation_uid,
                            "query_idx": query_idx,
                            "query": raw_query,
                            "query_norm": norm_text(
                                raw_query
                            ),
                            "gt_start_sec": float(
                                q["clip_start_sec"]
                            ),
                            "gt_end_sec": float(
                                q["clip_end_sec"]
                            ),
                            "template": q.get(
                                "template"
                            ),
                        }
                    )

    return rows


def build_gt_matcher(
    gt_rows: list[dict[str, Any]],
):
    """
    建立 GT 配對索引。
    """

    by_ann = {}
    by_clip_query = defaultdict(list)
    by_global_index = {}

    for r in gt_rows:

        if r.get("annotation_uid") is not None:
            by_ann[
                (
                    r["annotation_uid"],
                    int(r["query_idx"]),
                )
            ] = r

        by_clip_query[
            (
                r["clip_uid"],
                r["query_norm"],
            )
        ].append(r)

        by_global_index[
            int(r["global_index"])
        ] = r

    return (
        by_ann,
        by_clip_query,
        by_global_index,
    )


def match_gt(
    item: dict[str, Any],
    by_ann: dict,
    by_clip_query: dict,
    by_global_index: dict,
    occurrence_counter: dict,
) -> tuple[dict[str, Any] | None, str]:
    """
    將 Stage3 結果與 nlq_val GT 配對。

    優先：
    query_id 對應 nlq_val 原始 global_index

    其次：
    annotation_uid + query_idx

    最後才使用：
    clip_uid + query 文字
    """

    # --------------------------------------------------
    # 方法 1：
    # query_0001 -> global_index 0
    # query_0002 -> global_index 1
    #
    # Stage3 的 query_id 是 1-based，
    # nlq_val 展平後 global_index 是 0-based。
    # --------------------------------------------------

    query_id = str(
        item.get(
            "query_id",
            "",
        )
    ).strip()

    match = re.fullmatch(
        r"query_(\d+)",
        query_id,
    )

    if match is not None:

        global_index = (
            int(match.group(1))
            - 1
        )

        r = by_global_index.get(
            global_index
        )

        if r is not None:

            # 額外確認 clip 與 query 文字，
            # 避免 query_id 和 GT 順序不一致時誤配。
            same_clip = (
                str(r.get("clip_uid"))
                ==
                str(item.get("clip_uid"))
            )

            same_query = (
                r.get("query_norm")
                ==
                norm_text(
                    item.get("query")
                )
            )

            if (
                same_clip
                and same_query
            ):
                return (
                    r,
                    "query_id+global_index",
                )

    ann = item.get(
        "annotation_uid"
    )

    qidx = item.get(
        "query_idx"
    )

    # --------------------------------------------------
    # 方法 2：
    # annotation_uid + query_idx
    # --------------------------------------------------

    if (
        ann is not None
        and qidx is not None
    ):
        r = by_ann.get(
            (
                ann,
                int(qidx),
            )
        )

        if r is not None:
            return (
                r,
                "annotation_uid+query_idx",
            )

    # --------------------------------------------------
    # 方法 3：
    # clip_uid + query
    # 只當 fallback 使用。
    # --------------------------------------------------

    key = (
        item.get("clip_uid"),
        norm_text(
            item.get("query")
        ),
    )

    matches = by_clip_query.get(
        key,
        [],
    )

    if not matches:
        return (
            None,
            "not_found",
        )

    occ = occurrence_counter[key]

    occurrence_counter[key] += 1

    if occ < len(matches):
        return (
            matches[occ],
            "clip_uid+query_occurrence",
        )

    return (
        matches[-1],
        "clip_uid+query_last_fallback",
    )


def evaluate(args):

    # ==================================================
    # 1. 讀取檔案
    # ==================================================

    result_path = (
        Path(args.results)
        if args.results
        else OUTPUT_JSON
    )

    gt_path = (
        Path(args.gt)
        if args.gt
        else NLQ_VAL_JSON
    )

    results = load(
        result_path
    )

    nlq = load(
        gt_path
    )

    # ==================================================
    # 2. GT 展平
    # ==================================================

    gt_rows = flatten_gt(
        nlq
    )

    (
        by_ann,
        by_clip_query,
        by_global_index,
    ) = (
        build_gt_matcher(
            gt_rows
        )
    )

    occurrence_counter = defaultdict(
        int
    )

    # ==================================================
    # 3. 評估紀錄
    # ==================================================

    details = []

    counts = {

        # --------------------------------------------------
        # 基本統計
        # --------------------------------------------------

        "evaluated": 0,

        "skipped_empty_query": 0,

        "errors": 0,

        "gt_not_found": 0,

        # --------------------------------------------------
        # Stage2
        # --------------------------------------------------

        "stage2_r1_03": 0,

        "stage2_r5_03": 0,

        "stage2_r1_05": 0,

        "stage2_r5_05": 0,

        # --------------------------------------------------
        # Stage3 Top1
        # boundary 修改前
        # --------------------------------------------------

        "agent_r1_before_trim_03": 0,

        "agent_r1_before_trim_05": 0,

        # --------------------------------------------------
        # Stage3 Top1
        # boundary 修改後
        # --------------------------------------------------

        "agent_r1_after_trim_03": 0,

        "agent_r1_after_trim_05": 0,

        # --------------------------------------------------
        # Stage3 Top5
        # --------------------------------------------------

        "agent_r5_03": 0,

        "agent_r5_05": 0,

        # --------------------------------------------------
        # Boundary 統計
        # --------------------------------------------------

        "trim_applied": 0,

        "agent_top5_boundary_changed": 0,
    }

    # ==================================================
    # 4. 開始逐題評估
    # ==================================================

    for item in results:

        # ==================================================
        # ★ 新增：
        # 已經成功評估指定題數就停止
        #
        # 例如：
        # --count 500
        #
        # 就會成功評估滿 500 題後停止。
        # ==================================================

        if (
            args.count is not None
            and counts["evaluated"] >= args.count
        ):
            break

        # --------------------------------------------------
        # 必須是 dictionary
        # --------------------------------------------------

        if not isinstance(
            item,
            dict,
        ):
            continue

        # --------------------------------------------------
        # 缺少有效 Natural Language Query：直接跳過
        # 不計入 errors、GT miss，也不進 Recall 分母。
        # --------------------------------------------------
        if not norm_text(item.get("query")):
            counts["skipped_empty_query"] += 1
            continue

        # --------------------------------------------------
        # Stage3 執行失敗
        # --------------------------------------------------

        if (
            item.get("status")
            != "success"
        ):
            counts["errors"] += 1
            continue

        # --------------------------------------------------
        # 找 GT
        # --------------------------------------------------

        gt, source = match_gt(
            item,
            by_ann,
            by_clip_query,
            by_global_index,
            occurrence_counter,
        )

        if gt is None:
            counts[
                "gt_not_found"
            ] += 1
            continue

        # --------------------------------------------------
        # GT 時間區間
        # --------------------------------------------------

        g0 = gt[
            "gt_start_sec"
        ]

        g1 = gt[
            "gt_end_sec"
        ]

        # --------------------------------------------------
        # Stage2 原始 Top5
        # --------------------------------------------------

        original = item.get(
            "original_stage2_top5",
            [],
        )[:5]

        # --------------------------------------------------
        # Stage3 Agent Top5
        # --------------------------------------------------

        agent_top5 = item.get(
            "agent_top5",
            [],
        )[:5]

        # --------------------------------------------------
        # Stage3 最終 Top1
        # --------------------------------------------------

        final = item.get(
            "final_top1",
            {},
        )

        if (
            not original
            or not agent_top5
            or not final
        ):
            counts[
                "errors"
            ] += 1
            continue

        # ==================================================
        # 5. Stage2 Top5 IoU
        # ==================================================

        original_ious = [
            temporal_iou(
                float(
                    p["start_sec"]
                ),
                float(
                    p["end_sec"]
                ),
                g0,
                g1,
            )
            for p in original
        ]

        # ==================================================
        # 6. Stage3 Agent Top5 IoU
        # ==================================================

        agent_top5_ious = [
            temporal_iou(
                float(
                    p["start_sec"]
                ),
                float(
                    p["end_sec"]
                ),
                g0,
                g1,
            )
            for p in agent_top5
        ]

        # ==================================================
        # 7. Agent Top1
        # Boundary 修改前
        # ==================================================

        before_iou = temporal_iou(
            float(
                final[
                    "original_start_sec"
                ]
            ),
            float(
                final[
                    "original_end_sec"
                ]
            ),
            g0,
            g1,
        )

        # ==================================================
        # 8. Agent Top1
        # Boundary 修改後
        # ==================================================

        after_iou = temporal_iou(
            float(
                final[
                    "start_sec"
                ]
            ),
            float(
                final[
                    "end_sec"
                ]
            ),
            g0,
            g1,
        )

        # ==================================================
        # 9. 成功評估題數 +1
        # ==================================================

        counts[
            "evaluated"
        ] += 1

        # ==================================================
        # 10. Stage2 R@1 / R@5
        # IoU = 0.3
        # ==================================================

        counts[
            "stage2_r1_03"
        ] += (
            original_ious[0]
            >= 0.3
        )

        counts[
            "stage2_r5_03"
        ] += (
            max(
                original_ious
            )
            >= 0.3
        )

        # ==================================================
        # Stage2 IoU = 0.5
        # ==================================================

        counts[
            "stage2_r1_05"
        ] += (
            original_ious[0]
            >= 0.5
        )

        counts[
            "stage2_r5_05"
        ] += (
            max(
                original_ious
            )
            >= 0.5
        )

        # ==================================================
        # 11. Stage3 Top1
        # boundary 修改前
        # ==================================================

        counts[
            "agent_r1_before_trim_03"
        ] += (
            before_iou
            >= 0.3
        )

        counts[
            "agent_r1_before_trim_05"
        ] += (
            before_iou
            >= 0.5
        )

        # ==================================================
        # 12. Stage3 Top1
        # boundary 修改後
        # ==================================================

        counts[
            "agent_r1_after_trim_03"
        ] += (
            after_iou
            >= 0.3
        )

        counts[
            "agent_r1_after_trim_05"
        ] += (
            after_iou
            >= 0.5
        )

        # ==================================================
        # 13. Stage3 Top5
        # ==================================================

        counts[
            "agent_r5_03"
        ] += (
            max(
                agent_top5_ious
            )
            >= 0.3
        )

        counts[
            "agent_r5_05"
        ] += (
            max(
                agent_top5_ious
            )
            >= 0.5
        )

        # ==================================================
        # 14. 最終 Top1 是否修改 Boundary
        # ==================================================

        counts[
            "trim_applied"
        ] += bool(
            final.get(
                "boundary_refined",
                False,
            )
        )

        # ==================================================
        # 15. 計算 Stage3 Top5 中
        # 有幾個候選真的修改了 Boundary
        # ==================================================

        original_by_id = {
            int(
                p.get(
                    "candidate_id",
                    i + 1,
                )
            ): p
            for i, p
            in enumerate(
                original
            )
        }

        boundary_changed_this_query = 0

        for i, p in enumerate(
            agent_top5
        ):

            cid = int(
                p.get(
                    "candidate_id",
                    i + 1,
                )
            )

            ref = original_by_id.get(
                cid,
                {},
            )

            # --------------------------------------------------
            # 原本 start/end
            # --------------------------------------------------

            o0 = float(
                p.get(
                    "original_start_sec",
                    ref.get(
                        "start_sec",
                        p["start_sec"],
                    ),
                )
            )

            o1 = float(
                p.get(
                    "original_end_sec",
                    ref.get(
                        "end_sec",
                        p["end_sec"],
                    ),
                )
            )

            # --------------------------------------------------
            # Agent 修改後
            # --------------------------------------------------

            a0 = float(
                p["start_sec"]
            )

            a1 = float(
                p["end_sec"]
            )

            if (
                abs(
                    a0 - o0
                ) > 1e-9
                or
                abs(
                    a1 - o1
                ) > 1e-9
            ):
                boundary_changed_this_query += 1

        counts[
            "agent_top5_boundary_changed"
        ] += (
            boundary_changed_this_query
        )

        # ==================================================
        # 16. 儲存每題詳細資料
        # ==================================================

        details.append(
            {
                "evaluation_index": counts[
                    "evaluated"
                ],

                "query_id": item.get(
                    "query_id"
                ),

                "global_index": gt.get(
                    "global_index"
                ),

                "clip_uid": item.get(
                    "clip_uid"
                ),

                "query": item.get(
                    "query"
                ),

                "gt_source": source,

                "gt_start_sec": g0,

                "gt_end_sec": g1,

                "gt_duration": (
                    g1 - g0
                ),

                # ------------------------------------------
                # Stage2
                # ------------------------------------------

                "stage2_original_top5_ious":
                    original_ious,

                "stage2_top1_iou":
                    original_ious[0],

                "stage2_best_top5_iou":
                    max(
                        original_ious
                    ),

                # ------------------------------------------
                # Agent Ranking
                # ------------------------------------------

                "agent_ranked_candidate_ids":
                    item[
                        "agent_ranking"
                    ][
                        "ranked_candidate_ids"
                    ],

                # ------------------------------------------
                # Stage3 Top1
                # ------------------------------------------

                "agent_top1_before_trim_iou":
                    before_iou,

                "agent_top1_after_trim_iou":
                    after_iou,

                # ------------------------------------------
                # Stage3 Top5
                # ------------------------------------------

                "agent_top5_ious":
                    agent_top5_ious,

                "agent_best_top5_iou":
                    max(
                        agent_top5_ious
                    ),

                # ------------------------------------------
                # Boundary
                # ------------------------------------------

                "agent_top5_boundary_changed_count":
                    boundary_changed_this_query,

                "boundary_refined":
                    bool(
                        final.get(
                            "boundary_refined",
                            False,
                        )
                    ),
            }
        )

    # ==================================================
    # 17. 評估結果
    # ==================================================

    n = counts[
        "evaluated"
    ]

    if n <= 0:
        raise RuntimeError(
            "No evaluable queries"
        )

    def rate(key):
        return round(
            counts[key] / n,
            6,
        )

    # ==================================================
    # 18. Summary
    # ==================================================

    summary = {

        "result_json":
            str(
                result_path.resolve()
            ),

        "gt_json":
            str(
                gt_path.resolve()
            ),

        # --------------------------------------------------
        # 使用者要求評估幾題
        # --------------------------------------------------

        "requested_query_count":
            args.count,

        # --------------------------------------------------
        # 實際成功評估
        # --------------------------------------------------

        "evaluated_query_count":
            n,

        "skipped_empty_query_count":
            counts["skipped_empty_query"],

        "error_query_count":
            counts[
                "errors"
            ],

        "gt_not_found_count":
            counts[
                "gt_not_found"
            ],

        # ==================================================
        # Stage2
        # ==================================================

        "Stage2_original": {

            "R@1_IoU=0.3":
                rate(
                    "stage2_r1_03"
                ),

            "R@5_IoU=0.3":
                rate(
                    "stage2_r5_03"
                ),

            "R@1_IoU=0.5":
                rate(
                    "stage2_r1_05"
                ),

            "R@5_IoU=0.5":
                rate(
                    "stage2_r5_05"
                ),
        },

        # ==================================================
        # Stage3
        # Agent Ranking 後
        # Boundary 修改前
        # ==================================================

        "Stage3_agent_before_boundary": {

            "R@1_IoU=0.3":
                rate(
                    "agent_r1_before_trim_03"
                ),

            "R@1_IoU=0.5":
                rate(
                    "agent_r1_before_trim_05"
                ),
        },

        # ==================================================
        # Stage3
        # Boundary 修改後
        # ==================================================

        "Stage3_agent_after_boundary": {

            "R@1_IoU=0.3":
                rate(
                    "agent_r1_after_trim_03"
                ),

            "R@5_IoU=0.3":
                rate(
                    "agent_r5_03"
                ),

            "R@1_IoU=0.5":
                rate(
                    "agent_r1_after_trim_05"
                ),

            "R@5_IoU=0.5":
                rate(
                    "agent_r5_05"
                ),
        },

        # ==================================================
        # Boundary 統計
        # ==================================================

        "boundary_refined_count":
            counts[
                "trim_applied"
            ],

        "boundary_refined_rate":
            round(
                counts[
                    "trim_applied"
                ] / n,
                6,
            ),

        "Stage3_agent_top5_boundary_changed_count":
            counts[
                "agent_top5_boundary_changed"
            ],

        "Stage3_agent_top5_total_candidate_slots":
            n * 5,

        "Stage3_agent_top5_boundary_changed_rate":
            round(
                counts[
                    "agent_top5_boundary_changed"
                ]
                / (n * 5),
                6,
            ),
    }

    # ==================================================
    # 19. 儲存
    # ==================================================

    save(
        EVAL_DIR
        / "stage3_top5_evaluation_details.json",
        details,
    )

    save(
        EVAL_DIR
        / "stage3_top5_evaluation_summary.json",
        summary,
    )

    # ==================================================
    # 20. Terminal 顯示
    # ==================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "Stage3 Top-5 Evaluation"
    )

    print(
        "=" * 72
    )

    if args.count is None:

        print(
            "Requested: ALL"
        )

    else:

        print(
            f"Requested: {args.count}"
        )

    print(
        f"Evaluated: {n}"
    )

    print(
        f"Skipped empty query: "
        f"{counts['skipped_empty_query']}"
    )

    print(
        f"Errors   : "
        f"{counts['errors']}"
    )

    print(
        f"GT miss  : "
        f"{counts['gt_not_found']}"
    )

    # --------------------------------------------------
    # 如果資料不足
    # --------------------------------------------------

    if (
        args.count is not None
        and n < args.count
    ):
        print()

        print(
            "[WARNING]"
        )

        print(
            f"Requested {args.count} "
            f"evaluable queries, "
            f"but only {n} were available."
        )

    print()

    # ==================================================
    # Stage2
    # ==================================================

    print(
        "[Stage2 original]"
    )

    for (
        k,
        v,
    ) in summary[
        "Stage2_original"
    ].items():

        print(
            f"  {k}: "
            f"{v:.4f} "
            f"({v * 100:.2f}%)"
        )

    print()

    # ==================================================
    # Stage3 before boundary
    # ==================================================

    print(
        "[Stage3 Agent - before boundary trim]"
    )

    for (
        k,
        v,
    ) in summary[
        "Stage3_agent_before_boundary"
    ].items():

        print(
            f"  {k}: "
            f"{v:.4f} "
            f"({v * 100:.2f}%)"
        )

    print()

    # ==================================================
    # Stage3 after boundary
    # ==================================================

    print(
        "[Stage3 Agent - after boundary trim]"
    )

    for (
        k,
        v,
    ) in summary[
        "Stage3_agent_after_boundary"
    ].items():

        print(
            f"  {k}: "
            f"{v:.4f} "
            f"({v * 100:.2f}%)"
        )

    print()

    # ==================================================
    # Boundary 修改比例
    # ==================================================

    print(
        f"Top-1 boundary refined: "
        f"{counts['trim_applied']}/{n} "
        f"("
        f"{summary['boundary_refined_rate'] * 100:.2f}%"
        f")"
    )

    print(
        "Stage3 Agent Top-5 boundary changed: "
        f"{counts['agent_top5_boundary_changed']} "
        f"/ {n * 5} candidate slots "
        f"("
        f"{summary['Stage3_agent_top5_boundary_changed_rate'] * 100:.2f}%"
        f")"
    )

    print(
        "=" * 72
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage2 and Stage3 "
            "temporal localization results."
        )
    )

    # ==================================================
    # Stage3 預測結果
    # ==================================================

    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help=(
            "Stage3 result JSON path. "
            "Default uses OUTPUT_JSON from conf.py."
        ),
    )

    # ==================================================
    # Ego4D NLQ GT
    # ==================================================

    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help=(
            "nlq_val.json path. "
            "Default uses NLQ_VAL_JSON from conf.py."
        ),
    )

    # ==================================================
    # ★ 想評估幾題
    # ==================================================

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "Number of successfully evaluable "
            "queries to evaluate. "
            "Example: --count 500. "
            "Default=None means evaluate all."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 防止輸入 0 或負數
    # --------------------------------------------------

    if (
        args.count is not None
        and args.count <= 0
    ):
        parser.error(
            "--count must be greater than 0"
        )

    evaluate(
        args
    )


if __name__ == "__main__":
    main()