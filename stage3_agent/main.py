from __future__ import annotations

import argparse
import asyncio
import traceback
from collections import Counter
from typing import Any

from agent import inspect_candidate, rank_top5
from conf import CONTINUE_ON_ERROR, OUTPUT_JSON, RESUME, SAVE_EVERY, STAGE2_JSON
from tool import (
    Candidate,
    apply_one_shot_trim,
    find_video_path,
    get_stage2_top5,
    load_json,
    make_refined_candidate,
    sample_keyframes_for_candidates,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage3 Top-5 visual inspection -> per-candidate one-shot trim -> "
            "multimodal Top-5 reranking"
        )
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def candidate_dict(c: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": c.candidate_id,
        "stage2_rank": c.stage2_rank,
        "start_sec": c.start_sec,
        "end_sec": c.end_sec,
        "duration": c.end_sec - c.start_sec,
        "rerank_score": c.rerank_score,
    }


def upsert(records: list[dict[str, Any]], new: dict[str, Any]) -> None:
    qid = new.get("query_id")
    for i, old in enumerate(records):
        if old.get("query_id") == qid:
            if old.get("status") == "success" and new.get("status") != "success":
                return
            records[i] = new
            return
    records.append(new)


async def process_query(item: dict[str, Any]) -> dict[str, Any]:
    query_id = str(item.get("query_id") or "")
    clip_uid = str(item["clip_uid"])
    query = str(item["query"])
    predictions = item.get("predictions", [])

    print("\n" + "=" * 72)
    print(f"Query ID : {query_id}")
    print(f"Clip UID : {clip_uid}")
    print(f"Query    : {query}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Original Stage2 Top-5. These objects are NEVER mutated.
    # ------------------------------------------------------------------
    candidates = get_stage2_top5(predictions)
    if len(candidates) < 5:
        raise RuntimeError(f"Need 5 valid Stage2 candidates, got {len(candidates)}")

    print("[1] Original Stage2 Top-5:")
    for c in candidates:
        print(
            f"    #{c.stage2_rank}: {c.start_sec:.3f} ~ {c.end_sec:.3f} "
            f"score={c.rerank_score:.6f}"
        )

    # Save this before any refinement. It stays completely separated from Stage3.
    original_top5 = [candidate_dict(c) for c in candidates]
    original_candidate_map = {c.candidate_id: c for c in candidates}

    # ------------------------------------------------------------------
    # 2. Find video + uniformly sample 8 keyframes per ORIGINAL candidate.
    # ------------------------------------------------------------------
    video_path = find_video_path(clip_uid)
    print(f"[2] Video: {video_path}")

    keyframes_by_candidate = sample_keyframes_for_candidates(video_path, candidates)
    print("[3] 8 keyframes sampled for each Top-5 candidate.")

    # ------------------------------------------------------------------
    # 3. Independent visual inspection + boundary proposal for every candidate.
    # ------------------------------------------------------------------
    judgements = {}
    for c in candidates:
        print(f"[4] Inspect Candidate {c.candidate_id}/5 ...")
        j = await inspect_candidate(
            query=query,
            candidate=c,
            keyframes=keyframes_by_candidate[c.candidate_id],
        )
        judgements[c.candidate_id] = j
        print(
            f"    object={j.object_match}/4, action/state={j.action_state_match}/4, "
            f"temporal={j.temporal_intent_match}/4, complete={j.completeness}/4, "
            f"should_trim={j.should_trim}, trim_conf={j.trim_confidence}/4"
        )

    # ------------------------------------------------------------------
    # 4. Apply the SAME conservative one-shot trim to ALL five candidates.
    #    No candidate is mutated; five new refined Candidate objects are created.
    # ------------------------------------------------------------------
    refinements: dict[int, dict[str, Any]] = {}
    refined_candidates: list[Candidate] = []

    for c in candidates:
        boundary = apply_one_shot_trim(
            c,
            keyframes_by_candidate[c.candidate_id],
            judgements[c.candidate_id],
        )
        refinements[c.candidate_id] = boundary
        refined = make_refined_candidate(c, boundary)
        refined_candidates.append(refined)

        print(
            f"[5] Refine Candidate {c.candidate_id}: "
            f"{c.start_sec:.3f}~{c.end_sec:.3f} -> "
            f"{refined.start_sec:.3f}~{refined.end_sec:.3f} "
            f"({'APPLIED' if boundary['refinement_applied'] else 'UNCHANGED'})"
        )

    requested_trim_count = sum(
        bool(refinements[c.candidate_id].get("should_trim_requested", False))
        for c in candidates
    )
    applied_trim_count = sum(
        bool(refinements[c.candidate_id]["refinement_applied"]) for c in candidates
    )
    blocked_trim_count = sum(
        bool(refinements[c.candidate_id].get("should_trim_requested", False))
        and not bool(refinements[c.candidate_id]["refinement_applied"])
        for c in candidates
    )
    blocked_breakdown = Counter(
        str(refinements[c.candidate_id].get("blocked_by"))
        for c in candidates
        if bool(refinements[c.candidate_id].get("should_trim_requested", False))
        and not bool(refinements[c.candidate_id]["refinement_applied"])
    )
    refinement_source_breakdown = Counter(
        str(refinements[c.candidate_id].get("refinement_source", "none"))
        for c in candidates
        if bool(refinements[c.candidate_id]["refinement_applied"])
    )

    # Requested diagnostic: should_trim=True but safety gates blocked refinement.
    print(
        "[Trim diagnostic] "
        f"requested={requested_trim_count}/5, applied={applied_trim_count}/5, "
        f"should_trim=True but blocked={blocked_trim_count}/5"
    )
    if blocked_breakdown:
        print(f"    blocked_by={dict(blocked_breakdown)}")
    if refinement_source_breakdown:
        print(f"    refinement_source={dict(refinement_source_breakdown)}")

    # ------------------------------------------------------------------
    # 5. Final listwise ranking uses REFINED durations + text + up to 2 images/candidate.
    # ------------------------------------------------------------------
    ranking = await rank_top5(
        query=query,
        candidates=refined_candidates,
        judgements=judgements,
        keyframes_by_candidate=keyframes_by_candidate,
    )

    print(f"[6] Agent reranked refined Top-5: {ranking.ranked_candidate_ids}")
    print(
        f"    Selected Top-1: Candidate {ranking.selected_candidate_id}, "
        f"confidence={ranking.confidence}/4"
    )

    refined_candidate_map = {c.candidate_id: c for c in refined_candidates}
    selected_refined = refined_candidate_map[ranking.selected_candidate_id]
    selected_original = original_candidate_map[ranking.selected_candidate_id]
    selected_boundary = refinements[ranking.selected_candidate_id]

    print(
        f"[7] Final Top-1: {selected_refined.start_sec:.3f} ~ "
        f"{selected_refined.end_sec:.3f} sec"
    )
    print(
        f"    Original: {selected_original.start_sec:.3f} ~ "
        f"{selected_original.end_sec:.3f}, "
        f"trim={'YES' if selected_boundary['refinement_applied'] else 'NO'}"
    )

    # ------------------------------------------------------------------
    # 6. agent_top5 = NEW ranking + NEW refined boundaries for all five.
    # ------------------------------------------------------------------
    agent_top5 = []
    for final_rank, cid in enumerate(ranking.ranked_candidate_ids, start=1):
        c = refined_candidate_map[cid]
        original_c = original_candidate_map[cid]
        j = judgements[cid]
        boundary = refinements[cid]
        agent_top5.append(
            {
                "final_rank": final_rank,
                **candidate_dict(c),
                "original_start_sec": original_c.start_sec,
                "original_end_sec": original_c.end_sec,
                "boundary_refined": boundary["refinement_applied"],
                "refinement_source": boundary.get("refinement_source", "none"),
                "keep_start_keyframe": boundary["keep_start_keyframe"],
                "keep_end_keyframe": boundary["keep_end_keyframe"],
                "trim_left_keyframes": boundary["trim_left_keyframes"],
                "trim_right_keyframes": boundary["trim_right_keyframes"],
                "total_trimmed_keyframes": boundary["total_trimmed_keyframes"],
                "trim_confidence": boundary["confidence"],
                "trim_reason": boundary["reason"],
                "object_match": j.object_match,
                "action_state_match": j.action_state_match,
                "temporal_intent_match": j.temporal_intent_match,
                "completeness": j.completeness,
                "visual_summary": j.visual_summary,
            }
        )

    # Detailed independent inspections. Original boundaries stay visible; the actual
    # applied refinement is recorded separately for analysis/debugging.
    visual_judgements = []
    for c in candidates:
        j = judgements[c.candidate_id]
        boundary = refinements[c.candidate_id]
        visual_judgements.append(
            {
                "candidate_id": c.candidate_id,
                "stage2_rank": c.stage2_rank,
                "start_sec": c.start_sec,
                "end_sec": c.end_sec,
                "object_match": j.object_match,
                "action_state_match": j.action_state_match,
                "temporal_intent_match": j.temporal_intent_match,
                "completeness": j.completeness,
                "visual_summary": j.visual_summary,
                "keyframes": [k.model_dump() for k in j.keyframes],
                "boundary_proposal": {
                    "should_trim": j.should_trim,
                    "keep_start_keyframe": j.keep_start_keyframe,
                    "keep_end_keyframe": j.keep_end_keyframe,
                    "trim_confidence": j.trim_confidence,
                    "trim_reason": j.trim_reason,
                },
                "boundary_refinement": {
                    "refinement_applied": boundary["refinement_applied"],
                    "refinement_source": boundary.get("refinement_source", "none"),
                    "refined_start_sec": boundary["refined_start_sec"],
                    "refined_end_sec": boundary["refined_end_sec"],
                    "blocked_by": boundary.get("blocked_by"),
                },
            }
        )

    return {
        "query_id": query_id,
        "clip_uid": clip_uid,
        "video_uid": item.get("video_uid"),
        "annotation_uid": item.get("annotation_uid"),
        "query_idx": item.get("query_idx"),
        "template": item.get("template"),
        "query": query,

        # Intentionally unchanged Stage2 snapshot.
        "original_stage2_top5": original_top5,
        "visual_judgements": visual_judgements,

        "trim_diagnostics": {
            "should_trim_true_count": requested_trim_count,
            "refinement_applied_count": applied_trim_count,
            "should_trim_true_but_blocked_count": blocked_trim_count,
            "blocked_by": dict(blocked_breakdown),
            "refinement_source": dict(refinement_source_breakdown),
        },

        "agent_ranking": {
            "selected_candidate_id": ranking.selected_candidate_id,
            "ranked_candidate_ids": ranking.ranked_candidate_ids,
            "confidence": ranking.confidence,
            "reasoning": ranking.reasoning,
        },
        "agent_top5": agent_top5,

        # Field names are kept compatible with the original output format.
        # Values now come directly from the already-refined ranked Top-1.
        "final_top1": {
            "candidate_id": selected_refined.candidate_id,
            "stage2_rank": selected_refined.stage2_rank,
            "original_start_sec": selected_original.start_sec,
            "original_end_sec": selected_original.end_sec,
            "start_sec": selected_refined.start_sec,
            "end_sec": selected_refined.end_sec,
            "duration": selected_refined.end_sec - selected_refined.start_sec,
            "boundary_refined": selected_boundary["refinement_applied"],
            "refinement_source": selected_boundary.get("refinement_source", "none"),
            "keep_start_keyframe": selected_boundary["keep_start_keyframe"],
            "keep_end_keyframe": selected_boundary["keep_end_keyframe"],
            "trim_left_keyframes": selected_boundary["trim_left_keyframes"],
            "trim_right_keyframes": selected_boundary["trim_right_keyframes"],
            "total_trimmed_keyframes": selected_boundary["total_trimmed_keyframes"],
            "trim_confidence": selected_boundary["confidence"],
            "trim_reason": selected_boundary["reason"],
        },
    }


async def main():
    args = parse_args()
    print("=" * 72)
    print("Stage 3 - Visual Inspect -> Refine All Top-5 -> Multimodal Rerank")
    print("=" * 72)
    print(f"Input : {STAGE2_JSON}")
    print(f"Output: {OUTPUT_JSON}")

    data = load_json(STAGE2_JSON)
    if not isinstance(data, list):
        raise TypeError("Stage2 JSON must be a list")

    start = max(0, args.start)
    selected = data[start:] if args.count is None else data[start : start + args.count]

    if OUTPUT_JSON.exists():
        output = load_json(OUTPUT_JSON)
        if not isinstance(output, list):
            output = []
    else:
        output = []

    done = {
        x.get("query_id")
        for x in output
        if isinstance(x, dict) and x.get("status") == "success"
    }
    pending = (
        [x for x in selected if x.get("query_id") not in done]
        if RESUME and not args.no_resume
        else list(selected)
    )

    print(f"Selected: {len(selected)}")
    print(f"Pending : {len(pending)}")

    since_save = 0
    for idx, item in enumerate(pending, start=1):
        qid = item.get("query_id", f"unknown_{idx}")
        try:
            result = await process_query(item)
            result["status"] = "success"
            upsert(output, result)
        except Exception as exc:
            print(f"\n[ERROR] {qid}\n{exc}")
            traceback.print_exc()
            upsert(
                output,
                {
                    "query_id": qid,
                    "clip_uid": item.get("clip_uid"),
                    "query": item.get("query"),
                    "template": item.get("template"),
                    "status": "error",
                    "error": str(exc),
                },
            )
            if not CONTINUE_ON_ERROR:
                save_json(OUTPUT_JSON, output)
                raise

        since_save += 1
        if since_save >= SAVE_EVERY:
            save_json(OUTPUT_JSON, output)
            since_save = 0
            print("\n[Checkpoint saved]")

    save_json(OUTPUT_JSON, output)
    print("\nFinished")
    print(f"Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
