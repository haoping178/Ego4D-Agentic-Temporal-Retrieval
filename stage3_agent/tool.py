from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

from conf import (
    JPEG_QUALITY,
    KEYFRAME_MAX_SIDE,
    KEYFRAMES_PER_SEGMENT,
    MAX_TOTAL_TRIM_KEYFRAMES,
    MIN_EVENT_EVIDENCE_FRAMES_FOR_TRIM,
    STAGE3_TOP_K,
    TRIM_MIN_CONFIDENCE,
    VIDEO_DIR,
    VIDEO_EXTENSIONS,
)


@dataclass
class Candidate:
    candidate_id: int
    start_sec: float
    end_sec: float
    stage2_rank: int
    rerank_score: float
    original_prediction: dict[str, Any]


@dataclass
class KeyframeSample:
    index: int
    timestamp_sec: float
    bin_start_sec: float
    bin_end_sec: float
    image_bytes: bytes
    media_type: str = "image/jpeg"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def get_stage2_top5(predictions: list[dict[str, Any]]) -> list[Candidate]:
    """
    完全按照 Stage2 rank 取原始 Top-5。
    Stage3 不做 NMS、不去重、不重新產生候選。
    """
    ranked = sorted(predictions, key=lambda x: int(x.get("rank", 999999)))
    output: list[Candidate] = []

    for pred in ranked:
        try:
            start = float(pred["start_sec"])
            end = float(pred["end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue

        output.append(
            Candidate(
                candidate_id=len(output) + 1,
                start_sec=start,
                end_sec=end,
                stage2_rank=int(pred.get("rank", len(output) + 1)),
                rerank_score=float(pred.get("rerank_score", 0.0)),
                original_prediction=pred,
            )
        )
        if len(output) >= STAGE3_TOP_K:
            break

    return output


@lru_cache(maxsize=2048)
def find_video_path(clip_uid: str) -> Path:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"VIDEO_DIR does not exist: {VIDEO_DIR}")

    # 完全同名
    for ext in VIDEO_EXTENSIONS:
        p = VIDEO_DIR / f"{clip_uid}{ext}"
        if p.is_file():
            return p

    # 子資料夾完全同名
    for ext in VIDEO_EXTENSIONS:
        for p in VIDEO_DIR.rglob(f"{clip_uid}{ext}"):
            if p.is_file():
                return p

    # 模糊檔名
    for ext in VIDEO_EXTENSIONS:
        for p in VIDEO_DIR.rglob(f"{clip_uid}*{ext}"):
            if p.is_file():
                print(f"[VIDEO MATCH] {clip_uid} -> {p}")
                return p

    raise FileNotFoundError(
        "\nCannot find video.\n"
        f"clip_uid = {clip_uid}\n"
        f"VIDEO_DIR = {VIDEO_DIR}\n"
    )


def _get_duration(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return n / fps if fps > 0 and n > 0 else 0.0


def _read_frame_at(cap: cv2.VideoCapture, sec: float):
    for offset in (0.0, -0.05, 0.05, -0.10, 0.10, -0.25, 0.25):
        t = max(0.0, sec + offset)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    return None


def _resize_frame(frame):
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= KEYFRAME_MAX_SIDE:
        return frame
    scale = KEYFRAME_MAX_SIDE / m
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _encode_jpeg(frame) -> bytes:
    frame = _resize_frame(frame)
    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
    )
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return buf.tobytes()


def sample_keyframes_for_candidates(
    video_path: Path,
    candidates: list[Candidate],
) -> dict[int, list[KeyframeSample]]:
    """
    每個候選切成 8 個等長 bin，各取 bin 中心一張。
    因此之後若 Agent 選 K2~K7，就等於只往內裁一次。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    result: dict[int, list[KeyframeSample]] = {}
    try:
        video_duration = _get_duration(cap)

        for c in candidates:
            start = max(0.0, c.start_sec)
            end = c.end_sec
            if video_duration > 0:
                end = min(end, video_duration)
            if end <= start:
                raise ValueError(f"Invalid candidate {c.candidate_id}: {start}~{end}")

            bin_size = (end - start) / KEYFRAMES_PER_SEGMENT
            frames: list[KeyframeSample] = []
            previous_bytes: bytes | None = None

            for i in range(KEYFRAMES_PER_SEGMENT):
                bs = start + i * bin_size
                be = start + (i + 1) * bin_size
                ts = (bs + be) / 2.0
                frame = _read_frame_at(cap, ts)

                if frame is None:
                    if previous_bytes is None:
                        raise RuntimeError(
                            f"Failed first keyframe candidate={c.candidate_id}"
                        )
                    image_bytes = previous_bytes
                else:
                    image_bytes = _encode_jpeg(frame)
                    previous_bytes = image_bytes

                frames.append(
                    KeyframeSample(
                        index=i + 1,
                        timestamp_sec=ts,
                        bin_start_sec=bs,
                        bin_end_sec=be,
                        image_bytes=image_bytes,
                    )
                )

            result[c.candidate_id] = frames
    finally:
        cap.release()

    return result


def apply_one_shot_trim(
    candidate: Candidate,
    keyframes: list[KeyframeSample],
    judgement: Any,
) -> dict[str, Any]:
    """Apply Agent-driven inward boundary refinement to any candidate.

    Boundary sources, in priority order:
    1. explicit Agent proposal (should_trim + keep_start/end_keyframe);
    2. Agent frame-level evidence fallback: if strong event evidence is concentrated
       inside the candidate, derive a continuous kept range from the first/last
       strong evidence keyframe.

    No GT timestamp or IoU is used here.  The function only consumes the candidate,
    sampled keyframes, and the Agent's visual judgement.
    """
    from conf import (
        AGENT_EVIDENCE_AUTO_TRIM,
        AGENT_EVIDENCE_MIN_FRAMES,
        AGENT_EVIDENCE_MIN_RELEVANCE,
        ALLOW_STRONG_EVIDENCE_CONFIDENCE_OVERRIDE,
        STRONG_EVIDENCE_OVERRIDE_MIN_FRAMES,
    )

    original_start = float(candidate.start_sec)
    original_end = float(candidate.end_sec)
    n = len(keyframes)
    confidence = int(getattr(judgement, "trim_confidence", 0))
    should_trim = bool(getattr(judgement, "should_trim", False))

    judgement_frames = list(getattr(judgement, "keyframes", []) or [])
    event_frames = [
        k for k in judgement_frames
        if bool(getattr(k, "is_event_evidence", False))
    ]
    strong_event_frames = [
        k for k in event_frames
        if int(getattr(k, "relevance", 0)) >= AGENT_EVIDENCE_MIN_RELEVANCE
    ]

    fallback = {
        "refinement_applied": False,
        "refinement_source": "none",
        "original_start_sec": original_start,
        "original_end_sec": original_end,
        "refined_start_sec": original_start,
        "refined_end_sec": original_end,
        "keep_start_keyframe": 1,
        "keep_end_keyframe": n,
        "trim_left_keyframes": 0,
        "trim_right_keyframes": 0,
        "total_trimmed_keyframes": 0,
        "confidence": confidence,
        "reason": str(getattr(judgement, "trim_reason", "No refinement.")),
        "should_trim_requested": should_trim,
        "event_evidence_count": len(event_frames),
        "strong_event_evidence_count": len(strong_event_frames),
        "blocked_by": None,
    }

    if n <= 0:
        fallback["blocked_by"] = "no_keyframes"
        return fallback

    # ------------------------------------------------------------------
    # 1) Prefer the Agent's explicit kept range.
    # ------------------------------------------------------------------
    source = "agent_explicit"
    ks = int(getattr(judgement, "keep_start_keyframe", 1))
    ke = int(getattr(judgement, "keep_end_keyframe", n))

    use_explicit = should_trim

    # Confidence >=2 is accepted directly.  A lower confidence can still pass
    # when the Agent itself marked multiple strong event frames; confidence is
    # therefore a risk signal, not an unconditional veto.
    confidence_ok = confidence >= TRIM_MIN_CONFIDENCE
    if (
        use_explicit
        and not confidence_ok
        and ALLOW_STRONG_EVIDENCE_CONFIDENCE_OVERRIDE
        and len(strong_event_frames) >= STRONG_EVIDENCE_OVERRIDE_MIN_FRAMES
    ):
        confidence_ok = True
        source = "agent_explicit_strong_evidence_override"

    if use_explicit and not confidence_ok:
        use_explicit = False
        fallback["blocked_by"] = "trim_confidence"

    if use_explicit and len(event_frames) < MIN_EVENT_EVIDENCE_FRAMES_FOR_TRIM:
        use_explicit = False
        fallback["blocked_by"] = "event_evidence_count"

    if use_explicit and not (1 <= ks <= ke <= n):
        use_explicit = False
        fallback["blocked_by"] = "invalid_keep_range"

    # ------------------------------------------------------------------
    # 2) Evidence fallback.  This still comes from the Agent: it uses the
    #    Agent's is_event_evidence + relevance labels to infer the event span.
    # ------------------------------------------------------------------
    if not use_explicit and AGENT_EVIDENCE_AUTO_TRIM:
        strong_indices = sorted(
            int(getattr(k, "keyframe_index", 0))
            for k in strong_event_frames
            if 1 <= int(getattr(k, "keyframe_index", 0)) <= n
        )
        if len(strong_indices) >= AGENT_EVIDENCE_MIN_FRAMES:
            auto_ks = min(strong_indices)
            auto_ke = max(strong_indices)
            # Only useful when it actually removes an outer bin.  For safety,
            # every bin that would be removed must have been labelled by the
            # Agent as NOT event evidence.  This prevents a weak but genuine
            # outer event frame from being discarded merely because its
            # relevance score is below the strong-evidence threshold.
            outside_agent_frames = [
                k for k in judgement_frames
                if int(getattr(k, "keyframe_index", 0)) < auto_ks
                or int(getattr(k, "keyframe_index", 0)) > auto_ke
            ]
            outside_is_clean = all(
                not bool(getattr(k, "is_event_evidence", False))
                for k in outside_agent_frames
            )
            if (auto_ks > 1 or auto_ke < n) and outside_is_clean:
                ks, ke = auto_ks, auto_ke
                use_explicit = True
                source = "agent_frame_evidence"
                fallback["blocked_by"] = None

    if not use_explicit:
        if fallback["blocked_by"] is None:
            fallback["blocked_by"] = (
                "should_trim_false_or_no_strong_internal_evidence"
                if not should_trim
                else "no_valid_agent_boundary"
            )
        return fallback

    trim_left = ks - 1
    trim_right = n - ke
    total = trim_left + trim_right

    if total <= 0:
        fallback["blocked_by"] = "no_actual_trim"
        return fallback
    if total > MAX_TOTAL_TRIM_KEYFRAMES:
        fallback["blocked_by"] = "max_total_trim_keyframes"
        return fallback
    if ke - ks + 1 < 1:
        fallback["blocked_by"] = "empty_keep_range"
        return fallback

    # Keep full represented bins, not keyframe center timestamps.  This makes
    # the new boundary conservative and consistent with the eight-bin sampling.
    refined_start = float(keyframes[ks - 1].bin_start_sec)
    refined_end = float(keyframes[ke - 1].bin_end_sec)

    if refined_end <= refined_start:
        fallback["blocked_by"] = "invalid_refined_boundary"
        return fallback
    if refined_start < original_start - 1e-9 or refined_end > original_end + 1e-9:
        fallback["blocked_by"] = "would_expand_candidate"
        return fallback

    reason = str(getattr(judgement, "trim_reason", "Boundary refined."))
    if source == "agent_frame_evidence":
        reason = (
            f"Agent frame-evidence fallback kept K{ks}~K{ke}. "
            f"Original Agent reason: {reason}"
        )

    return {
        "refinement_applied": True,
        "refinement_source": source,
        "original_start_sec": original_start,
        "original_end_sec": original_end,
        "refined_start_sec": refined_start,
        "refined_end_sec": refined_end,
        "keep_start_keyframe": ks,
        "keep_end_keyframe": ke,
        "trim_left_keyframes": trim_left,
        "trim_right_keyframes": trim_right,
        "total_trimmed_keyframes": total,
        "confidence": confidence,
        "reason": reason,
        "should_trim_requested": should_trim,
        "event_evidence_count": len(event_frames),
        "strong_event_evidence_count": len(strong_event_frames),
        "blocked_by": None,
    }

def make_refined_candidate(
    candidate: Candidate,
    refinement: dict[str, Any],
) -> Candidate:
    """
    Wrap ``apply_one_shot_trim`` output as a new Candidate.

    The returned Candidate keeps the same candidate_id / Stage2 metadata, while
    ``start_sec`` and ``end_sec`` come from the refined boundary.  The input
    Candidate is never modified, so ``original_stage2_top5`` can remain intact.
    """
    start = float(refinement.get("refined_start_sec", candidate.start_sec))
    end = float(refinement.get("refined_end_sec", candidate.end_sec))

    if end <= start:
        # Defensive fallback; apply_one_shot_trim should already prevent this.
        start = float(candidate.start_sec)
        end = float(candidate.end_sec)

    return Candidate(
        candidate_id=candidate.candidate_id,
        start_sec=start,
        end_sec=end,
        stage2_rank=candidate.stage2_rank,
        rerank_score=candidate.rerank_score,
        original_prediction=candidate.original_prediction,
    )
