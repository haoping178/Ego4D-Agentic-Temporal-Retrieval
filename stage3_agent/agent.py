from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider

from conf import (
    CANDIDATE_RETRY,
    FINAL_RANK_RETRY,
    FINAL_RANK_IMAGES_PER_CANDIDATE,
    KEYFRAMES_PER_SEGMENT,
    MODEL_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from tool import Candidate, KeyframeSample


class KeyframeEvidence(BaseModel):
    keyframe_index: int = Field(ge=1, le=KEYFRAMES_PER_SEGMENT)
    relevance: int = Field(ge=0, le=4)
    is_event_evidence: bool
    description: str = Field(min_length=1, max_length=180)


class CandidateVisualJudgement(BaseModel):
    candidate_id: int = Field(ge=1, le=5)

    # 明確拆開，避免舊版 semantic_score 只輸出 0/1
    object_match: int = Field(ge=0, le=4)
    action_state_match: int = Field(ge=0, le=4)
    temporal_intent_match: int = Field(ge=0, le=4)
    completeness: int = Field(ge=0, le=4)

    keyframes: list[KeyframeEvidence]

    # 給 final ranking agent 的短摘要 / caption
    visual_summary: str = Field(min_length=1, max_length=450)

    # Boundary proposal：這裡先提出，main.py 會對每個候選各自套用相同安全門檻
    should_trim: bool
    keep_start_keyframe: int = Field(ge=1, le=KEYFRAMES_PER_SEGMENT)
    keep_end_keyframe: int = Field(ge=1, le=KEYFRAMES_PER_SEGMENT)
    trim_confidence: int = Field(ge=0, le=4)
    trim_reason: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def validate_output(self):
        ids = sorted(k.keyframe_index for k in self.keyframes)
        expected = list(range(1, KEYFRAMES_PER_SEGMENT + 1))
        if ids != expected:
            raise ValueError(f"Need exactly K1~K{KEYFRAMES_PER_SEGMENT}")
        if self.keep_start_keyframe > self.keep_end_keyframe:
            raise ValueError("keep_start_keyframe > keep_end_keyframe")
        return self


class FinalTop5Ranking(BaseModel):
    selected_candidate_id: int = Field(ge=1, le=5)
    ranked_candidate_ids: list[int]
    confidence: int = Field(ge=0, le=4)
    reasoning: str = Field(min_length=1, max_length=700)

    @model_validator(mode="after")
    def validate_rank(self):
        if len(self.ranked_candidate_ids) != 5:
            raise ValueError("ranked_candidate_ids must contain exactly 5 IDs")
        if sorted(self.ranked_candidate_ids) != [1, 2, 3, 4, 5]:
            raise ValueError("ranked_candidate_ids must contain 1,2,3,4,5 exactly once")
        if self.selected_candidate_id != self.ranked_candidate_ids[0]:
            self.selected_candidate_id = self.ranked_candidate_ids[0]
        return self


provider = OllamaProvider(base_url=OLLAMA_BASE_URL)
model = OllamaModel(OLLAMA_MODEL, provider=provider)


CANDIDATE_SYSTEM_PROMPT = f"""
You are the Stage-3 visual verifier for Ego4D Natural Language Query temporal localization.

You receive ONE candidate segment at a time, the original natural-language query,
and exactly {KEYFRAMES_PER_SEGMENT} uniformly sampled chronological keyframes.

Your job has two parts.

A. VISUAL INTENT VERIFICATION
Judge from visible evidence only:
- object_match: is the object/person/location named by the query present and correct?
- action_state_match: is the queried action or state actually visible?
- temporal_intent_match: does the sequence match intent such as put, pick, before,
  after, last seen, carry, drop, etc.?
- completeness: does this candidate contain enough of the event to answer the query?

Use integer 0~4:
0 = absent/wrong
1 = weak
2 = plausible/uncertain
3 = clear
4 = very clear

For every keyframe K1~K{KEYFRAMES_PER_SEGMENT}, return relevance 0~4,
is_event_evidence, and a short factual description.

B. AGENT-DRIVEN INWARD BOUNDARY PROPOSAL
Your boundary judgement is important and WILL be used to refine every candidate before final ranking.
First identify the earliest keyframe bin that contains direct evidence for the queried event,
and the latest keyframe bin that still contains direct evidence for that same event.
Then return that continuous range as keep_start_keyframe~keep_end_keyframe.

Rules:
- You may trim the beginning, the end, or both ends.
- The kept keyframes MUST form one continuous range.
- Never remove frames in the middle.
- Never expand beyond the candidate.
- Do not trim merely because an outer frame looks less relevant; trim it when it is visibly before
  the queried event, after the queried event, or unrelated background/context.
- If K1 or K8 still contains meaningful direct event evidence, keep it.
- If direct event evidence is concentrated inside the candidate (for example K3~K6), set
  should_trim=true and keep K3~K6 instead of conservatively keeping K1~K8.
- should_trim=false is reserved for cases where the current boundaries are already appropriate or
  the visible evidence truly cannot support a safer inward boundary.
- At most 7 keyframe bins may be removed in total.

trim_confidence meaning:
0 = no usable boundary evidence
1 = weak/ambiguous
2 = reasonable visual evidence
3 = clear boundary evidence
4 = very clear boundary evidence

IMPORTANT:
- Do not use Stage2 rerank score; you are not given it.
- Do not invent timestamps.
- Do not guess unseen content.
- Keep visual_summary compact and factual; it will be used by a second Agent to compare all Top-5.
"""

candidate_agent = Agent(
    model,
    output_type=NativeOutput(CandidateVisualJudgement),
    system_prompt=CANDIDATE_SYSTEM_PROMPT,
    model_settings={"temperature": MODEL_TEMPERATURE},
)


FINAL_SYSTEM_PROMPT = """
You are the final Stage-3 selector for Ego4D NLQ.

Stage2 has already produced exactly five best temporal candidates. Each candidate
has been independently inspected by a vision-language model using 8 chronological
keyframes, and each candidate may already have received a conservative one-shot
inward boundary refinement.

For EACH candidate you now receive:
- its REFINED duration;
- structured visual scores and text summary from the first visual inspection;
- keyframe evidence descriptions;
- a small set of the strongest visual-evidence keyframe IMAGES (normally up to 2).

Your task:
1. compare all five candidates against the original query intent;
2. explicitly compare the attached images, not only the text summaries;
3. rank all five from best to worst;
4. select the candidate that most completely and precisely matches the query.

Do NOT simply preserve Stage2 order.
Do NOT use the Stage2 rerank score; it is intentionally hidden.
Use visible evidence for the queried object/entity, action/state, temporal intent,
and event completeness. If multiple candidates contain the same object, prefer the
one showing the exact queried action/state/temporal relation rather than mere object
presence. Prefer precise boundaries when visual evidence is otherwise comparable.
If evidence is ambiguous, use the candidate with the most complete direct evidence.

Return IDs 1,2,3,4,5 exactly once.
"""

final_agent = Agent(
    model,
    output_type=NativeOutput(FinalTop5Ranking),
    system_prompt=FINAL_SYSTEM_PROMPT,
    model_settings={"temperature": MODEL_TEMPERATURE},
)


def _candidate_prompt_content(
    query: str,
    candidate: Candidate,
    keyframes: list[KeyframeSample],
) -> list:
    content: list = [
        f"Natural-language query:\n{query}\n\n"
        f"Candidate ID: {candidate.candidate_id}\n"
        f"Candidate duration: {candidate.end_sec - candidate.start_sec:.3f} sec\n"
        "Frames are chronological."
    ]

    for k in keyframes:
        content.append(
            f"K{k.index}: represented bin {k.bin_start_sec:.3f}~{k.bin_end_sec:.3f} sec"
        )
        content.append(BinaryContent(data=k.image_bytes, media_type=k.media_type))

    return content


async def inspect_candidate(
    *,
    query: str,
    candidate: Candidate,
    keyframes: list[KeyframeSample],
) -> CandidateVisualJudgement:
    last_exc: Exception | None = None
    for attempt in range(1, CANDIDATE_RETRY + 1):
        try:
            result = await candidate_agent.run(
                _candidate_prompt_content(query, candidate, keyframes)
            )
            out = result.output
            if out.candidate_id != candidate.candidate_id:
                raise RuntimeError(
                    f"Expected candidate {candidate.candidate_id}, got {out.candidate_id}"
                )
            return out
        except Exception as exc:
            last_exc = exc
            print(
                f"    [WARN] Candidate {candidate.candidate_id} "
                f"inspection attempt {attempt} failed: {type(exc).__name__}"
            )

    raise RuntimeError(
        f"Candidate {candidate.candidate_id} inspection failed after retries: {last_exc}"
    )

#-----------------------------------------------------------------------------------------------------

def _select_final_rank_keyframes(
    *,
    judgement: CandidateVisualJudgement,
    keyframes: list[KeyframeSample],
) -> list[tuple[KeyframeEvidence, KeyframeSample]]:
    """
    Pick a small visual subset for final ranking.

    Priority:
    1) is_event_evidence=True
    2) higher relevance
    3) earlier keyframe index for deterministic tie-breaking

    The selected set is then re-sorted chronologically before being sent to the
    final Agent, so image order remains easy to interpret.
    """
    sample_by_index = {k.index: k for k in keyframes}
    evidence_sorted = sorted(
        judgement.keyframes,
        key=lambda x: (
            -int(bool(x.is_event_evidence)),
            -int(x.relevance),
            int(x.keyframe_index),
        ),
    )

    selected: list[tuple[KeyframeEvidence, KeyframeSample]] = []
    for ev in evidence_sorted:
        sample = sample_by_index.get(ev.keyframe_index)
        if sample is None:
            continue
        selected.append((ev, sample))
        if len(selected) >= FINAL_RANK_IMAGES_PER_CANDIDATE:
            break

    return sorted(selected, key=lambda pair: pair[0].keyframe_index)


def _final_rank_prompt_content(
    *,
    query: str,
    candidates: list[Candidate],
    judgements: dict[int, CandidateVisualJudgement],
    keyframes_by_candidate: dict[int, list[KeyframeSample]],
) -> list:
    content: list = [
        f"Natural-language query:\n{query}\n\n"
        "Compare all five refined candidates. Each candidate includes text evidence "
        "plus a small number of strongest visual-evidence images."
    ]

    for c in candidates:
        j = judgements[c.candidate_id]
        evidence_frames = [
            f"K{k.keyframe_index}(rel={k.relevance}, evidence={k.is_event_evidence}): "
            f"{k.description}"
            for k in j.keyframes
        ]
        content.append(
            "\n"
            f"=== Candidate {c.candidate_id} ===\n"
            f"refined_duration={c.end_sec - c.start_sec:.3f} sec\n"
            f"object_match={j.object_match}/4\n"
            f"action_state_match={j.action_state_match}/4\n"
            f"temporal_intent_match={j.temporal_intent_match}/4\n"
            f"completeness={j.completeness}/4\n"
            f"visual_summary={j.visual_summary}\n"
            "keyframe_evidence:\n" + "\n".join(evidence_frames) +
            "\nAttached strongest evidence images follow:"
        )

        selected_frames = _select_final_rank_keyframes(
            judgement=j,
            keyframes=keyframes_by_candidate[c.candidate_id],
        )
        for ev, sample in selected_frames:
            content.append(
                f"Candidate {c.candidate_id} K{ev.keyframe_index}: "
                f"relevance={ev.relevance}/4, event_evidence={ev.is_event_evidence}; "
                f"{ev.description}"
            )
            content.append(
                BinaryContent(data=sample.image_bytes, media_type=sample.media_type)
            )

    return content


async def rank_top5(
    *,
    query: str,
    candidates: list[Candidate],
    judgements: dict[int, CandidateVisualJudgement],
    keyframes_by_candidate: dict[int, list[KeyframeSample]],
) -> FinalTop5Ranking:
    """
    Listwise final ranking using refined candidate durations + text + images.

    ``candidates`` must already contain the one-shot-refined boundaries.
    The images come from the original 8 uniformly sampled keyframes, selected by
    first-stage event-evidence/relevance scores. With the default configuration,
    at most 5 * 2 = 10 images are sent.
    """
    prompt_content = _final_rank_prompt_content(
        query=query,
        candidates=candidates,
        judgements=judgements,
        keyframes_by_candidate=keyframes_by_candidate,
    )

    last_exc: Exception | None = None
    for attempt in range(1, FINAL_RANK_RETRY + 1):
        try:
            result = await final_agent.run(prompt_content)
            return result.output
        except Exception as exc:
            last_exc = exc
            print(
                f"    [WARN] Final Top-5 ranking attempt {attempt} failed: "
                f"{type(exc).__name__}"
            )

    # Safe fallback: preserve Stage2 candidate IDs/order rather than failing query.
    fallback_ids = [c.candidate_id for c in candidates]
    return FinalTop5Ranking(
        selected_candidate_id=fallback_ids[0],
        ranked_candidate_ids=fallback_ids,
        confidence=0,
        reasoning=f"Final ranking failed; kept candidate order. Error: {str(last_exc)[:300]}",
    )
