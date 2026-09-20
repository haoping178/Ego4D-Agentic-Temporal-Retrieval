from __future__ import annotations
from pathlib import Path

STAGE3_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STAGE3_DIR.parent

# Stage2 output: each query contains ranked predictions.
STAGE2_JSON = PROJECT_DIR / "stage2" / "retrieval_results" / "qwen_query_segments.json"

# Ego4D NLQ validation GT.  This is used ONLY by evaluate.py, never by Stage3 agents.
NLQ_VAL_JSON = PROJECT_DIR / "nlq_val.json"

OUTPUT_DIR = STAGE3_DIR / "retrieval_results"
OUTPUT_JSON = OUTPUT_DIR / "qwen_stage3_top5_results.json"
EVAL_DIR = STAGE3_DIR / "evaluation_results"

# Video
VIDEO_DIR = Path("F:/ego4d_data/v2/clips")
VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mkv", ".MKV", ".webm", ".avi", ".mov")

# Ollama / PydanticAI
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen3-vl:8b-instruct"
MODEL_TEMPERATURE = 0.1

# Stage3 only consumes Stage2 Top-5.
STAGE3_TOP_K = 5

# Each candidate is represented by eight chronological, uniform bins.
KEYFRAMES_PER_SEGMENT = 8
KEYFRAME_MAX_SIDE = 384
JPEG_QUALITY = 88

# Retry
CANDIDATE_RETRY = 2
FINAL_RANK_RETRY = 2

# Final listwise ranking: up to two strongest images per candidate.
FINAL_RANK_IMAGES_PER_CANDIDATE = 2

# -----------------------------------------------------------------------------
# Agent-driven boundary refinement
# -----------------------------------------------------------------------------
# Keep at least one bin.  With eight bins this means at most seven may be removed.
MAX_TOTAL_TRIM_KEYFRAMES = 7

# Previous code required confidence >=3 and all 31-query trim proposals were blocked.
# 2 means "reasonable visual evidence" and lets the Agent actually refine boundaries.
TRIM_MIN_CONFIDENCE = 2

# At least this many Agent-labelled event frames must exist for an explicit proposal.
MIN_EVENT_EVIDENCE_FRAMES_FOR_TRIM = 1

# If should_trim=False but the Agent's own strong frame-level evidence clearly lies
# inside the candidate, derive a conservative kept range from that evidence.
AGENT_EVIDENCE_AUTO_TRIM = True
AGENT_EVIDENCE_MIN_RELEVANCE = 3
AGENT_EVIDENCE_MIN_FRAMES = 2

# Low-confidence explicit proposals may still be accepted only when supported by
# multiple strong event-evidence frames.  This avoids making confidence a hard veto.
ALLOW_STRONG_EVIDENCE_CONFIDENCE_OVERRIDE = True
STRONG_EVIDENCE_OVERRIDE_MIN_FRAMES = 2

# Runtime
SAVE_EVERY = 5
RESUME = True
CONTINUE_ON_ERROR = True
