# PURE

> Knowledge-graph-grounded explainable recommendation with RGAT indexing,
> preference-aware path retrieval, and LoRA-tuned explanation generation.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![torch-geometric](https://img.shields.io/badge/PyG-2.4+-orange.svg)](https://pyg.org/)

Source code for **Preference-Aware Evidence Selection for Faithful LLM-Based
Recommendation Explanations**. The repository provides data preparation,
offline graph and path indexing, model training, inference, and evaluation.
Processed datasets, model weights, and human evaluation annotations are
external artifacts; the input formats required by this code are listed below.

## Code structure

| Paper component | Implementation |
| --- | --- |
| Frozen PLM + four-layer RGAT index | `models/semantic_index.py`, `models/rgat.py` |
| Offline three-hop RGAT search vectors and BERT path vectors | `models/path_retrieval.py:PathIndex` |
| Target-aware history attention | `TargetAwareUserIntent` |
| Inverse-degree, neighborhood-entropy, preference specificity | `NodeSpecificityScorer` |
| Multiplicative score, threshold, bounded path retrieval, MMR | `PreferenceAwarePathRetrieval` |
| Four-layer Graph Transformer and soft prompts | `models/graph_transformer.py`, `models/pure_model.py` |
| LoRA generation plus graph/text alignment | `PUREModel.forward` |
| F-EHR and P-EHR with Sentires-Guide features | `evaluation/sentires_report.py` |

Paths are enumerated and encoded around each target item when the index is
built. RGAT node vectors form structure-aware path vectors for candidate
search; BERT vectors provide path semantics for the paper's score and MMR.
Training and inference load this index; they do not traverse the KG per
request. The index covers the target and sampled negative items present in the
supplied split files. Build a new index when the item catalog changes. For a
new target that was not indexed, retrieval returns no paths until the index is
rebuilt.

## Required data

Create `data/{books,movies,yelp}/` with:

- `kg_entities.json`: entity name to contiguous integer ID;
- `kg_relations.json`: relation name to contiguous integer ID;
- `kg_triples.json`: `[[head_id, relation_id, tail_id], ...]`;
- `entity_texts.json`: entity texts in ID order;
- `id2entity.json`: ID to display name;
- `train.json`, `valid.json`, `test.json`: lists of records containing
  `user_id`, `history` (item entity IDs), `target_item`,
  `explanation`, `item_features`, `user_positive_features`, and optional
  `negative_items`.

The split files can be built from chronologically ordered, KG-aligned
interactions with:

```bash
python -m data.prepare_data --interactions interactions.jsonl \
  --item-features item_features.json --sentires-output reviews.pickle \
  --output-dir data/books
```

`item_features.json` must map every aligned candidate item entity ID to its
normalized KG attributes, including catalog items without interactions. The
interaction JSONL requires `user_id`, `item_id`,
`timestamp`, and `text`; the optional `explanation` field supplies the
generation target. The preparation script uses the last interaction for test,
the previous one for validation, earlier ones for training, and only past
reviews to build each user's positive preference features. It samples up to 39
negative items from the aligned item catalog for a pool of at most 40 items,
excluding items previously interacted with by that user. The Sentires output
must use the same user and aligned item IDs as the interaction file.

## Train and evaluate

```bash
python train.py --dataset books --data_dir ./data --output_dir ./checkpoints/books
python inference.py --dataset books --data_dir ./data \
  --checkpoint ./checkpoints/books/best_model.pt \
  --output_path ./results/books.json
```

`inference.py` writes generated explanations, selected paths, text metrics,
ranking metrics when candidate items are present, and latency measurements.
Feature-level metrics use Sentires-Guide quadruples. Process the generated
explanations from `results/books.json` with
[Sentires-Guide](https://github.com/lileipisces/Sentires-Guide), then run:

```bash
python -m evaluation.sentires_report --results results/books.json \
  --sentires-output generated_reviews.pickle \
  --output results/books_sentires_metrics.json
```

The Sentires output must contain a record for every generated explanation;
each record has `user`, `item`, `text`, and a `sentence` list of
`(feature, opinion, sentence, score)` quadruples. The P-EHR threshold is 0.35.
Only load trusted pickle files.

## Implementation details

The offline RGAT index is trained with KG link reconstruction before storing
node vectors. Sampled-candidate ranking uses the strongest preference-aligned
path score for each item. Validation selects the generation checkpoint by
ROUGE-L. The path pruning threshold defaults to 0.0 and can be set with
`--path_score_threshold` after validation on a given dataset.
Graph/text alignment uses frozen `all-MiniLM-L6-v2` explanation embeddings.

FMR is the fraction of explanations mentioning a supported item feature; FCR
is mean per-item attribute coverage; DIV is pairwise self-BLEU-2 multiplied
by 100. F-EHR and P-EHR use extracted Sentires-Guide features, with the P-EHR
threshold set to 0.35. BLEU-4 and ROUGE-L are reported on a 0–100 scale.
The specificity weights are `(0.27, 0.31, 0.42)` with 64 semantic clusters.
P-EHR excludes instances that mention features but have no historically
positive features, because their semantic preference centroid is undefined;
the output reports the number of evaluable instances.
