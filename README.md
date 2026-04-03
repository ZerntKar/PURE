# PURE 🔍
### Path-aware User preference Reasoning for Explainable Recommendation

> Knowledge-graph-grounded explainable recommendation via RGAT path retrieval and LoRA-fine-tuned LLM generation.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![torch-geometric](https://img.shields.io/badge/PyG-2.4+-orange.svg)](https://pyg.org/)

---

## 📖 Overview

Given a user's interaction history and a target item, PURE explains *why* the item is recommended by reasoning over a knowledge graph and generating a natural-language explanation grounded in the user's actual preferences.

```
User History ──► Multi-hop KG Path Retrieval (RGAT)
                        │
                        ▼
              Node Specificity Scoring (λ_s · λ_m · λ_p)
                        │
                        ▼
         Graph-Transformer Cross-Attention Projector
                        │
                        ▼
          LoRA-fine-tuned LLM ──► Natural Language Explanation
```

---

## ✨ Key Features

- 🧠 **Relation-aware RGAT encoder** — heterogeneous KG message passing with per-edge relation matrices and multi-head attention
- 🛤️ **MMR path retrieval** — diversity-aware top-k path selection balancing relevance and coverage (γ = 0.6)
- 📐 **Node specificity scorer** — jointly models semantic rarity (λ_s), membership signal (λ_m), and structural position (λ_p)
- 🔗 **Graph-Transformer projector** — soft-prompt injection bridging KG subgraph embeddings into LLM token space
- ⚡ **Efficient inference** — AMP mixed-precision + LoRA, < 200 ms per explanation on a single A100
- 📊 **Faithfulness-first evaluation** — P-EHR and F-EHR metrics explicitly penalize preference-misaligned hallucination

---

## 🗂️ Repository Structure

```
PURE/
├── config.py                  # PUREConfig dataclass
├── train.py                   # Training entry point
├── inference.py               # Evaluate / interactive / profile modes
│
├── data/
│   └── dataset.py             # KnowledgeGraph, RecommendationDataset, collate_fn
│
├── models/
│   ├── pure_model.py          # Top-level PUREModel
│   ├── rgat.py                # RGATConv (single-layer relation-aware GAT)
│   ├── path_retrieval.py      # PathEncoder + MMR retrieval
│   └── semantic_index.py      # StructureEnhancedIndex, NodeSpecificityScorer
│
├── evaluation/
│   └── metrics.py             # PUREEvaluator, P-EHR, F-EHR, BLEU-4, ROUGE-L, FMR, FCR, DIV
│
└── data/
    ├── movies/
    │   ├── kg_triples.json
    │   ├── entity_texts.json
    │   ├── id2entity.json
    │   ├── id2relation.json
    │   ├── train.json
    │   ├── valid.json
    │   └── test.json
    ├── books/
    └── yelp/
```

---

## 📦 Requirements

```bash
pip install torch>=2.1 torchvision
pip install torch-geometric>=2.4
pip install transformers>=4.40
pip install peft>=0.10
pip install sentence-transformers>=2.7
pip install rouge-score nltk numpy tqdm
```

Or install all at once:

```bash
pip install -r requirements.txt
```

---


## 📊 Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **F-EHR** | Feature-level explanation hallucination rate | ↓ lower is better |
| **P-EHR** | Preference-aligned hallucination rate (primary) | ↓ lower is better |
| **BLEU-4** | 4-gram text quality vs. reference | ↑ |
| **ROUGE-L** | Longest common subsequence overlap | ↑ |
| **FMR** | Feature mention rate in generated text | ↑ |
| **FCR** | Feature coverage rate across all item features | ↑ |
| **DIV** | Intra-list diversity of explanations | ↓ lower is better |

**P-EHR** is the primary model selection criterion — it measures how often generated explanations mention features *misaligned* with the user's positive preference profile, penalizing hallucination at the semantic level.

---
