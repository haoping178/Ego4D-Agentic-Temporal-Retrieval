Ego4D 三階段自然語言影片時間定位框架

Ego4D Three-Stage Natural Language Query Localization Framework

本專案實作一套用於 Ego4D Natural Language Queries（NLQ） 的三階段影片時間定位架構。
系統結合 Qwen3-VL-Embedding-2B、Cosine Similarity、多尺度時間區間生成、Local Contrast、NMS，以及 Qwen3-VL VLM Agent，由自然語言 Query 找出影片中最符合語意的時間區間。

1. 研究目標

Natural Language Query Localization（NLQ，自然語言查詢時間定位）的目標是：

給定一段長影片以及一個自然語言問題，找出影片中最能回答該問題的開始時間與結束時間。

例如：

Query:
"What did I put into the drawer?"

Video:
0s ------------------------------------------------------ 300s

Prediction:
                     [ 125s -------- 132s ]

本專案將整個定位流程拆成三個階段：

Stage 1：候選秒檢索（Temporal Retrieval）

Stage 2：時間區間生成與重排（Temporal Proposal & Reranking）

Stage 3：VLM Agent 視覺驗證、邊界修正與最終重排

2. 整體架構

                    Natural Language Query
                              │
                              ▼
              ┌─────────────────────────────┐
              │ Stage 1                     │
              │ Temporal Retrieval          │
              │                             │
              │ Qwen3-VL-Embedding-2B       │
              │ Query / Video Embedding     │
              │ Cosine Similarity           │
              └──────────────┬──────────────┘
                             │
                    Top-30 Candidate Seconds
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Stage 2                     │
              │ Temporal Proposal           │
              │                             │
              │ Multi-scale Durations       │
              │ Proposal Score              │
              │ Local Contrast              │
              │ NMS + Reranking             │
              └──────────────┬──────────────┘
                             │
                       Top-5 Segments
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Stage 3                     │
              │ VLM Agent Refinement        │
              │                             │
              │ Keyframe Inspection         │
              │ Boundary Refinement         │
              │ Multimodal Reranking        │
              └──────────────┬──────────────┘
                             │
                             ▼
                   Final Temporal Prediction

3. Stage 1：候選秒檢索

第一階段的目標是從完整影片中快速找出與 Query 最相關的時間點。

影片處理

影片以 1 秒為單位建立表示，每秒取樣 8 Frames，再利用：

Qwen/Qwen3-VL-Embedding-2B

產生每秒影片的 multimodal embedding。

Query Encoding

Natural Language Query 同樣使用 Qwen3-VL-Embedding-2B 編碼，使 Query 與 Video 位於相同的 embedding space。

Similarity

使用 Cosine Similarity：

Similarity(q, v) = cosine(Query Embedding, Video Embedding)

計算 Query 與影片每一秒的語意相似度。

最後保留：

Top-K = 30

個相似度最高的候選秒，交給 Stage 2。

4. Stage 2：多尺度時間區間生成與重排

Stage 1 得到的是離散的候選秒，但 NLQ 最終需要輸出：

[start_time, end_time]

因此 Stage 2 會將候選秒轉換成不同長度的 temporal proposals。

Candidate Durations

目前使用：

Duration

2 秒

3 秒

4 秒

5 秒

6 秒

8 秒

12 秒

16 秒

Proposal Score

每個候選區間會根據區間內的 per-second similarity scores 計算 Proposal Score。

Local Contrast

為了避免只選到「整段影片都很像 Query」但缺乏局部辨識力的區間，系統額外比較候選區間與前後背景。

目前背景範圍：

±3 秒

Reranking

目前 Stage 2 使用：

Final Score
= 0.4 × Proposal Score
+ 0.6 × Local Contrast

並使用：

NMS IoU Threshold = 0.3

降低高度重疊的候選區間。

最後保留 Top-5 temporal segments 送入 Stage 3。

5. Stage 3：VLM Agent 邊界修正與最終重排

Stage 3 不只依賴 embedding similarity，而是讓 Vision-Language Model 直接檢查候選區間的視覺內容。

目前 Agent 架構使用：

PydanticAI
Ollama
Qwen3-VL

目前 conf.py 設定的 Ollama 模型為：

qwen3-vl:8b-instruct

Candidate Inspection

對 Stage 2 的 Top-5 candidates：

取得候選時間區間。

從區間中依時間順序取樣 Keyframes。

將 Query 與 Keyframes 一起交給 VLM Agent。

判斷物件、動作、事件與 Query 是否一致。

檢查事件是否完整。

必要時進行保守的 temporal boundary refinement。

Final Reranking

完成各候選區間的視覺檢查與邊界修正後，再比較 Top-5 candidates 的 multimodal evidence，產生最終排序與時間預測。

6. 專案目錄

.
├── stage1/
│   ├── build_qwen_video_embeddings.py
│   ├── qwen_embedder_utils.py
│   └── retrieve_qwen_top30_candidates.py
│
├── stage2/
│   ├── qwen_embedder_utils.py
│   └── retrieve_qwen_query_segments.py
│
├── stage3_agent/
│   ├── agent.py
│   ├── conf.py
│   ├── evaluate.py
│   ├── main.py
│   └── tool.py
│
├── nlq_val.json
├── requirements.txt
├── .gitignore
└── README.md

7. 各程式功能

Stage 1

build_qwen_video_embeddings.py

讀取 Ego4D 影片

以每秒 8 Frames 取樣

使用 Qwen3-VL-Embedding-2B 建立 per-second video embeddings

儲存 embedding 與 metadata

qwen_embedder_utils.py

載入官方 Qwen3-VL-Embedding repository

建立 Qwen3VLEmbedder

處理 embedding normalization

retrieve_qwen_top30_candidates.py

編碼 NLQ Query

計算 Query / Video cosine similarity

搜尋 Top-30 candidate seconds

輸出 Stage 1 retrieval results

Stage 2

retrieve_qwen_query_segments.py

讀取 Stage 1 Top-30 candidates

建立多尺度 temporal proposals

計算 Proposal Score

計算 Local Contrast

執行 NMS

重新排序候選區間

輸出 Top-5 temporal candidates

Stage 3

agent.py

定義 VLM Agent

分析候選區間 Keyframes

執行候選比較與最終排序

tool.py

候選區間資料結構

Keyframe 擷取

時間區間處理

Agent 所需工具函式

conf.py

Stage 3 路徑

Ollama 模型

Agent 與推論參數設定

main.py

Stage 3 主程式

讀取 Stage 2 candidates

呼叫 VLM Agent

執行 boundary refinement

輸出最終結果

evaluate.py

計算 NLQ temporal localization evaluation metrics

8. 評估指標

本專案使用 Ego4D NLQ 常見的 Recall / Temporal IoU 指標：

R@1, IoU = 0.3
R@1, IoU = 0.5
R@5, IoU = 0.3
R@5, IoU = 0.5

其中 Temporal IoU 用於衡量：

Prediction Segment
與
Ground Truth Segment

之間的時間重疊程度。

9. 環境需求

建議環境：

Python 3.11
CUDA-compatible NVIDIA GPU
PyTorch 2.8.0

安裝 Python dependencies：

pip install -r requirements.txt

Qwen3-VL-Embedding

Stage 1 / Stage 2 使用官方 Qwen3-VL-Embedding repository。

git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git

程式預設會從本機 Qwen3-VL-Embedding repository 匯入：

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

模型：

Qwen/Qwen3-VL-Embedding-2B

Ollama

Stage 3 使用本機 Ollama 服務執行 VLM。

Ollama 本體不包含在 requirements.txt 中，需要另外安裝並準備：

qwen3-vl:8b-instruct

10. requirements.txt 版本說明

為了提高實驗環境的可重現性，本 Repository 的 requirements.txt 使用固定版本。

其中 Qwen3-VL-Embedding 相關核心版本依官方專案需求設定，包括：

torch==2.8.0
torchvision==0.23.0
transformers==4.57.3
accelerate==1.12.0
qwen-vl-utils==0.0.14
decord==0.6.0

其餘套件則用於數值運算、影像處理、進度顯示與 Stage 3 Agent。

11. Dataset

本專案使用：

Ego4D Natural Language Queries (NLQ)

Repository 不提供 Ego4D 原始影片。

使用者需要自行取得 Ego4D Dataset，並修改程式中的 dataset/video path。


12. Acknowledgements

本專案使用或參考以下專案與資源：

Ego4D

Qwen3-VL

Qwen3-VL-Embedding

PydanticAI

Ollama
