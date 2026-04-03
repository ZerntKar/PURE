# PURE 🔍
### Path-aware User preference Reasoning for Explainable Recommendation

> Knowledge-graph-grounded explainable recommendation via RGAT path retrieval and LoRA-fine-tuned LLM generation.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![torch-geometric](https://img.shields.io/badge/PyG-2.4+-orange.svg)](https://pyg.org/)

---

## Repository Structure

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

## Requirements

```bash
pip install torch>=2.1 torchvision
pip install torch-geometric>=2.4
pip install transformers>=4.40
pip install peft>=0.10
pip install sentence-transformers>=2.7
pip install rouge-score nltk numpy tqdm
```

---


## Evaluation Metrics

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
