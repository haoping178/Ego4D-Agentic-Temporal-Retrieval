# Ego4D Three-Stage NLQ Framework

A three-stage framework for Natural Language Query Localization (NLQ) on Ego4D, combining multimodal embedding-based temporal retrieval, multi-scale proposal generation, and VLM-agent-based temporal refinement.

## Overview

Natural Language Query Localization (NLQ) aims to localize the temporal segment in a video that best answers a natural-language query.

This project implements a three-stage pipeline:

### Stage 1 — Temporal Retrieval
- Divide each video into one-second temporal units.
- Sample 8 frames per second.
- Extract multimodal representations using `Qwen3-VL-Embedding-2B`.
- Encode the natural-language query into the same embedding space.
- Compute cosine similarity between the query and per-second video embeddings.
- Retrieve the Top-30 candidate seconds.

### Stage 2 — Temporal Proposal Generation and Reranking
- Generate multi-scale temporal proposals around retrieved candidate seconds.
- Use candidate durations of `2, 3, 4, 5, 6, 8, 12, 16` seconds.
- Aggregate per-second similarity scores into proposal scores.
- Measure local temporal contrast using surrounding background regions.
- Apply Non-Maximum Suppression (NMS).
- Rerank candidate temporal segments and retain the Top-5 candidates.

### Stage 3 — VLM Agent Refinement
- Sample chronological keyframes from each Top-5 candidate.
- Use a Qwen3-VL-based agent with PydanticAI and Ollama.
- Inspect visual evidence relevant to the query.
- Refine temporal boundaries.
- Perform multimodal reranking.
- Produce the final temporal predictions.

## Pipeline

```text
Natural Language Query
        |
        v
+--------------------------------+
| Stage 1: Temporal Retrieval    |
| Qwen3-VL-Embedding-2B          |
| Query <-> Per-second Video     |
| Cosine Similarity              |
+---------------+----------------+
                |
        Top-30 Candidate Seconds
                |
                v
+--------------------------------+
| Stage 2: Temporal Proposals    |
| Multi-scale Durations          |
| Proposal Score                 |
| Local Contrast + NMS           |
| Temporal Reranking             |
+---------------+----------------+
                |
          Top-5 Segments
                |
                v
+--------------------------------+
| Stage 3: VLM Agent             |
| Keyframe Inspection            |
| Boundary Refinement            |
| Multimodal Reranking           |
+---------------+----------------+
                |
                v
       Final Temporal Prediction
```

## Repository Structure

```text
.
├── stage1/
│   ├── build_qwen_video_embeddings.py
│   ├── qwen_embedder_utils.py
│   └── retrieve_qwen_top30_candidates.py
├── stage2/
│   ├── qwen_embedder_utils.py
│   └── retrieve_qwen_query_segments.py
├── stage3_agent/
│   ├── agent.py
│   ├── conf.py
│   ├── evaluate.py
│   ├── main.py
│   └── tool.py
├── nlq_val.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Stage 1: Per-Second Temporal Retrieval

Each video is represented at one-second temporal resolution. Eight frames are sampled from every second and encoded using `Qwen3-VL-Embedding-2B`.

The natural-language query is encoded into the same embedding space. Cosine similarity is calculated between the query representation and every per-second video representation.

The Top-30 highest-scoring seconds are retained as candidate moments for the next stage.

## Stage 2: Multi-Scale Temporal Proposal Generation

Stage 2 converts candidate seconds into temporal segments using multiple candidate durations:

| Candidate Duration |
| ---: |
| 2 s |
| 3 s |
| 4 s |
| 5 s |
| 6 s |
| 8 s |
| 12 s |
| 16 s |

The temporal proposals are evaluated using proposal scores and local temporal contrast.

Current configuration:

```text
Proposal Weight       = 0.4
Local Contrast Weight = 0.6
Background Margin     = ±3 seconds
NMS IoU Threshold     = 0.3
```

After reranking and NMS, the Top-5 temporal candidates are passed to Stage 3.

## Stage 3: Agent-Based Temporal Refinement

Stage 3 uses a Vision-Language Model (VLM) agent to inspect the Top-5 candidate segments.

For each candidate:

1. Sample chronological keyframes.
2. Provide the query and visual evidence to the VLM agent.
3. Examine relevant objects, actions, events, and temporal completeness.
4. Refine temporal boundaries when appropriate.
5. Compare refined candidates using multimodal evidence.
6. Produce the final ranking.

The agent implementation uses:

```text
PydanticAI
Ollama
Qwen3-VL
```

## Dataset

The framework is designed for the Natural Language Queries (NLQ) benchmark of Ego4D.

Large video files, model weights, extracted embeddings, and intermediate experimental outputs are not included in this repository.

Users should obtain the Ego4D dataset separately and configure the corresponding local paths.

## Evaluation

The framework supports standard temporal localization metrics:

- R@1, IoU = 0.3
- R@1, IoU = 0.5
- R@5, IoU = 0.3
- R@5, IoU = 0.5

Evaluation utilities are available in:

```text
stage3_agent/evaluate.py
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Ollama and the required Qwen models should be installed/configured separately according to the local inference environment.

## Notes

The repository focuses on source code and the experimental pipeline. The following data are intentionally excluded:

- Ego4D video files
- Model weights
- Per-second embedding files
- Intermediate retrieval results
- Evaluation outputs
- Python cache files

## Citation

If you use this repository in academic work, please cite the corresponding paper when available.

```bibtex
@article{three_stage_nlq_2026,
  title={A Training-Free Agentic Temporal Retrieval Framework for Natural Language Query Localization},
  year={2026}
}
```

## Acknowledgements

This project uses the Ego4D NLQ benchmark and Qwen multimodal models.
