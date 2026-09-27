# Amazon ML Challenge 2026 — Business Entity Resolution

## Overview

This project implements a business entity resolution pipeline for matching reference entities from Source 1 (S1) against entities from Sources 2 and 3 (S2/S3).

The system supports one-to-zero, one-to-one, and one-to-many relationships. For each S1 entity, the pipeline generates candidate S2/S3 entities, computes similarity features, applies a trained machine-learning matcher, and produces the final matched entity IDs.

## Pipeline

1. Load and normalize S1, S2, and S3 business records.
2. Generate candidate pairs using blocking strategies.
3. Apply inexpensive candidate filtering.
4. Extract name, address, country, and address-number similarity features.
5. Score candidate pairs using a trained LightGBM classifier.
6. Retain all candidate matches whose probability meets the production threshold.
7. Aggregate predictions by S1 entity.
8. Produce `matching_results.tsv` and `candidate_pairs.tsv`.

## Feature Engineering

The matcher uses features including:

- Business-name similarity
- Address token-sort similarity
- Address token-set similarity
- Address partial similarity
- Address exact-match indicator
- Address-number overlap
- Country compatibility
- Combined name/address signals

Country values are canonicalized to support equivalent country representations.

Address-number extraction uses conservative handling of isolated single-digit numbers to reduce spurious matches caused by common numeric tokens.

## Matching Model

A LightGBM-based binary classifier is used to estimate whether a candidate pair represents the same business entity.

The model is evaluated using entity-grouped validation so that records belonging to the same S1 entity do not leak across training and validation groups.

The production inference threshold is:

`0.93`

All candidates meeting the threshold are retained. The pipeline does not impose a one-match-per-S1 constraint or an arbitrary top-K restriction.

## Multi-Match Support

The dataset can contain multiple valid S2/S3 matches for a single S1 entity. Therefore, predictions are represented as a comma-separated list of matched entity IDs for each S1 entity.

S1 entities with no accepted matches remain in the output with an empty matched-ID field.

## Production Inference

The production pipeline supports streaming candidate processing and multiprocessing to handle the large test dataset without loading the complete candidate set into memory.

The final pipeline uses multiple worker processes and processes candidate data in chunks.

## Output Format

### matching_results.tsv

```text
source1_entity_id    matched_entity_ids

