# Business Entity Resolution — ML_Code_3

## Methodology
Normalize the provided business names/addresses without external identity lookup or geocoding.
Preserve every source ID. Ground-truth connected components assign train/validation clusters;
shared references cannot bridge the two supervised splits. Countries are open-set strings,
including France at test time. No external business data is added.

## Encoder
Apache-2.0 sentence-transformers/paraphrase-multilingual-mpnet-base-v2, revision
79f2382ceacceacdf38563d7c5d16b9ff8d725d6. The model card describes 768-dimensional outputs.
LoRA rank 8 on query/value projections, one epoch by default; cached contrastive loss with
unique clusters per batch. BF16 training on supported GPUs. Validation monitor selects a
checkpoint from the initial state and fine-tuned states. A small monitor is a selection signal,
not the final full-pool retrieval metric. All records are embedded with the same selected encoder.
The offline smoke-test encoder is randomly initialized and is never the production model.

## Candidate generation
Country/source FAISS indexes use IVF-PQ for large partitions and exact inner-product search
for small partitions. Original normalized float16 vectors rerank the shortlist. Unknown countries
receive fallback searches. Lexical blocks include names, address components, character n-grams,
and words; BM25 shortlists are reranked with hashed TF-IDF fit on training references only.
Neural and lexical candidates are merged with diversity quotas, capped at 64 per S1.
The emitted candidate_pairs.tsv is exactly the set scored by the matcher.

## Features and matcher
Name/address edit, token, character and TF-IDF similarities, address components, missingness,
retrieval provenance/rank and neural cosine feed binary LightGBM. Missing evidence stays NaN.
IDs, country categories and split labels are not model features. All non-holdout training candidate
pairs are used through disk-backed staging; memory estimates can stop a run rather than silently
sampling. LightGBM uses CPU in the Lightning profile, avoiding custom CUDA compilation.

## Validation and inference
Full-validation macro F0.5 includes singletons, zero-candidate queries and unretrieved positives.
Threshold selection uses validation; this selected score is not an unbiased test estimate.
Test inference scores all final candidates. Every test S1 appears once, including France and
empty-match queries. The supplied official validator must PASS before outputs are copied to output/.

## Reproduction
See README.md. Use the packaged configs/reproduction_model.json with pipeline.py, the original
challenge dataset and a compatible environment. The package script appends measured run metadata.
No leaderboard score, full-data runtime or optimal accuracy is claimed without a completed run.
