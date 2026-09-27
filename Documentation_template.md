# Business Entity Resolution --- Methodology Documentation

## 1. Problem Statement

The task is to perform business entity resolution across multiple source
systems.

For every reference entity in Source 1 (S1), the system identifies all
corresponding records in Source 2 (S2) and Source 3 (S3). An S1 entity
may have no match, one match, or multiple matches.

The solution uses a two-stage architecture:

1.  Candidate generation / blocking to reduce the search space.
2.  Pairwise machine-learning classification to distinguish likely
    matches from hard negatives.

The system is precision-oriented because the evaluation metric is F0.5.

## 2. Overall Architecture

``` mermaid
flowchart TD
    A[Source 1 Reference Entities] --> B[Normalization]
    B --> C[Candidate Generation / Blocking]
    D[Source 2 + Source 3] --> E[Normalization]
    E --> C
    C --> F[Candidate Pairs]
    F --> G[Feature Engineering]
    G --> H[LightGBM Classifier]
    H --> I[Threshold Filtering]
    I --> J[Final Matches]
    J --> K[Post-run Validation]
    K --> L[matching_results.tsv]
```

## 3. Dataset Analysis

  Dataset             S1          S2          S3
  ---------- ----------- ----------- -----------
  Training     2,206,821   5,034,616   5,285,603
  Test         1,732,544   4,887,273   5,082,316

Training ground truth is multi-match:

-   Training S1 entities: 2,206,821
-   Total true target pairs: 7,638,365
-   S1 entities with zero true matches: 123,247 (5.58%)
-   Exactly one true match: 119,157 (5.40%)
-   More than one true match: 1,964,417 (89.02%)
-   Mean true matches per S1: 3.4613
-   Median: 3
-   Maximum: 11

This means the problem cannot be treated as one-to-one matching.

## 4. Normalization

Normalization is applied before similarity computation and candidate
generation. Processing includes lowercasing, whitespace and punctuation
normalization, Unicode/transliteration handling where applicable,
business-name/address normalization, and extraction of useful address
and numeric tokens.

Country representations are handled for feature computation using
country aliases/canonical forms.

The blocking implementation has its own normalized country
representation, while feature computation uses the country-alias
mapping. Therefore, country normalization is not implemented through one
identical function at every stage.

## 5. Candidate Generation and Blocking

Direct comparison of every S1 entity against every S2/S3 record is
computationally infeasible. Blocking therefore generates a manageable
candidate set.

The blocking strategy combines signals including:

-   normalized business-name information,
-   country information,
-   legal suffix handling,
-   transliteration-aware matching,
-   address-derived keys,
-   address token overlap,
-   house/address number information,
-   controlled fuzzy signatures,
-   domain/root information where available.

The objective is high candidate recall while keeping the candidate set
manageable for downstream feature computation.

## 6. Blocking Experiments

Representative candidate-recall results were:

  Blocking strategy                                   Candidate recall
  ------------------------------------------------- ------------------
  Exact name + country                                          22.03%
  Legal suffix handling                                         39.36%
  Transliteration                                               43.89%
  Tight address blocking                                        74.78%
  Controlled fuzzy signature                                    77.73%
  Safe baseline                                               95.9571%
  Compact union                                               96.8951%
  Best compact union                                          97.0419%
  Top-8 address                                               97.6145%
  Number + strong address token                               98.3598%
  Larger candidate depth                                      98.7051%
  Larger candidate depth                                      98.8404%
  Larger candidate depth                                      99.0475%
  Final compact candidate set used in experiments         **98.0893%**

## 7. Candidate Recall Evaluation

On a sampled set of 10,000 training S1 entities:

-   Candidate pairs: 19,116,887
-   True pairs: 34,752
-   Retrieved true pairs: 34,088
-   Candidate recall: **98.0893%**
-   S1 entities with at least one missed true target: 567
-   Missed true pairs: 664

Candidate generation is evaluated separately from classification because
a true pair that is never generated cannot be recovered by the
classifier.

## 8. Feature Engineering

Pairwise features are calculated for every generated candidate. The
feature set includes:

-   business-name similarity,
-   address similarity,
-   token-based address similarity,
-   country agreement,
-   phone-related similarity,
-   email-related similarity,
-   website/domain similarity,
-   numeric/address-number signals,
-   exact and partial agreement indicators,
-   fuzzy similarity measurements,
-   combined similarity signals.

The implementation also includes composite signals such as `comb_score`
and `number_mismatch`.

RapidFuzz is used for efficient fuzzy string comparison.

## 9. Hard Negatives

Random negatives are insufficient because many non-matching businesses
can look highly similar.

The training process therefore includes difficult candidate pairs
generated by blocking, including records with similar names, shared
locality/address tokens, similar numeric information, and
transliteration similarities.

## 10. Machine Learning Model

The primary classifier is LightGBM with:

-   300 trees
-   learning rate: 0.05
-   31 leaves

If LightGBM is unavailable, the implementation falls back to
scikit-learn's `HistGradientBoostingClassifier`.

The classifier outputs a match probability for each candidate pair.

## 11. Validation Strategy

Validation uses 5-fold GroupKFold, grouping by S1 entity. This prevents
candidate pairs belonging to the same reference entity from being split
arbitrarily between training and validation folds.

A probability-threshold sweep is used to evaluate the precision/recall
trade-off.

## 12. Ablation Study

The corrected multi-match ground truth was used for the ablation
experiments.

The cached evaluation sample contained:

-   1,738,488 candidate pairs
-   34,088 positive pairs
-   1,704,400 hard negatives

Representative results:

  Configuration           Macro F0.5    Precision       Recall
  --------------------- ------------ ------------ ------------
  Baseline                    0.7379       0.8365       0.5868
  Feature engineering     **0.8659**   **0.9059**   **0.8095**

## 13. Threshold Selection

The validation threshold sweep was:

    Threshold   Macro F0.5
  ----------- ------------
         0.50       0.8658
         0.60       0.8676
         0.65   **0.8681**
         0.70       0.8657
         0.75       0.8624
         0.80       0.8588
         0.85       0.8526
         0.90       0.8422
         0.92       0.8338
         0.95       0.8102

The validation experiment peaked around 0.65.

The large-scale test distribution showed higher candidate density,
especially for France, so additional threshold and candidate-hygiene
diagnostics were performed before production inference.

## 14. Test Distribution Shift

The training experiments were dominated by US and India, while the test
set also contained France.

Approximate test distribution observed during diagnostics:

-   India: 46.7%
-   United States: 38.3%
-   France: 15.0%

Test candidate density was also substantially higher than the training
subsample. Low-probability candidates could therefore accumulate into
very large predicted match sets, particularly for French records.

## 15. Address and Number Hygiene

Diagnostics showed that generic French address tokens and single-digit
numeric collisions could create excessive candidate or prediction
density.

Address-number hygiene was introduced to reduce these pathological
combinations.

In a 10K-S1 test experiment using the original address representation
for fuzzy metrics:

    Threshold   Predictions   Empty S1   Mean matches   P99
  ----------- ------------- ---------- -------------- -----
         0.90        33,239      8.10%           3.32    20
         0.92        30,750      9.20%           3.08    18
         0.93        29,613      9.86%           2.96    16
         0.94        28,378     10.56%           2.84    15
         0.95        26,924     11.53%           2.69    14
         0.96        25,099     13.06%           2.51    12
         0.97        22,675     15.35%           2.27     9

The production configuration uses a 0.93 match threshold and retains
every candidate at or above that threshold.

The 0.93 threshold was selected as a practical production configuration
after considering distribution-shift and prediction-density diagnostics.
It was not claimed to be the maximum point in the earlier validation
sweep.

## 16. Why Strict Address Rules Were Not Used

Hard address-number rules reduced some false positives but also removed
legitimate matches involving incomplete addresses, formatting
differences, transliteration, DBA/business-name variations, and partial
address information.

The final approach therefore uses address and number information
primarily as model features and candidate-generation signals rather than
requiring exact agreement.

## 17. Production Inference Pipeline

The inference pipeline is designed for large-scale processing and uses:

-   8 workers,
-   20,000 S1 entities per chunk,
-   checkpoint files for completed chunks,
-   model preloading,
-   vectorized candidate retrieval using DuckDB,
-   pairwise feature computation,
-   probability thresholding,
-   final result consolidation.

The pipeline performs post-run checks for row count, duplicate IDs, ID
validity, and submission format.

## 18. Candidate and Matching Outputs

### `matching_results.tsv`

The required format is:

``` text
source1_entity_id    matched_entity_ids
```

There must be exactly one row per test S1 entity. Matched IDs are
represented as a comma-separated list. An S1 entity with no match must
still be present with an empty matched-ID field.

### `candidate_pairs.tsv`

The required grouped format is:

``` text
source1_entity_id    candidate_entity_ids
```

Candidate IDs are represented as a comma-separated list. This file
represents the final candidate set supplied to the matching model before
final scoring.

## 19. Submission Validation

The repository includes the challenge validator at:

``` text
utils/validate_submission.py
```

A final submission can be checked with:

``` bash
python3 utils/validate_submission.py   --matching output/matching_results.tsv   --candidate output/candidate_pairs.tsv   --test-dir dataset/test
```

## 20. Running the Pipeline

The source package uses module imports such as
`from src.features import ...`. Therefore, execute it from the package
directory using module execution:

``` bash
cd code/business_entity_resolution

DATA_DIR=/path/to/dataset VALIDATOR_SCRIPT=../../utils/validate_submission.py python -m src.pipeline --workers 8
```

For a smoke test:

``` bash
cd code/business_entity_resolution

DATA_DIR=/path/to/dataset VALIDATOR_SCRIPT=../../utils/validate_submission.py python -m src.pipeline --workers 8 --smoke
```

Candidate generation is a separate preprocessing stage; the pipeline
consumes generated candidate data rather than creating the complete test
candidate set itself.

## 21. AWS Processing

Large-scale processing experiments were performed using AWS
infrastructure. EC2 was used for compute-intensive candidate generation
and inference, while S3 was used for intermediate candidate and
diagnostic data.

The repository does not contain AWS credentials, private keys, account
identifiers, instance IP addresses, or private machine paths.

## 22. Error Analysis

False-positive patterns included:

-   businesses in the same building,
-   similar businesses with different house numbers,
-   generic address-token collisions,
-   highly similar records representing different entities.

False-negative patterns included:

-   transliteration differences,
-   DBA/business-name variations,
-   partial addresses,
-   incomplete source records,
-   useful identifying information distributed across different fields.

These observations guided subsequent blocking and feature-engineering
iterations.

## 23. Design Decisions

### Multi-match formulation

The ground truth contains multiple valid matches for most S1 entities,
so the system does not force one-to-one matching.

### Blocking before ML

Blocking reduces the pairwise search space and makes large-scale
matching practical.

### Precision-oriented thresholding

Because the evaluation metric is F0.5, false positives are particularly
costly. Candidate hygiene and thresholding therefore emphasize precision
while retaining useful recall.

### Grouped validation

Grouping by S1 prevents leakage between candidate pairs belonging to the
same reference entity.

### Checkpointed inference

Large-scale inference is chunked so completed work can be reused after
interruptions.

## 24. Limitations

1.  Candidate recall is below 100%, so some true pairs can be lost
    before classification.
2.  Test data contains distribution differences relative to the training
    experiments.
3.  Address formats and transliteration patterns create difficult edge
    cases.
4.  Generic tokens can create high candidate density.
5.  The production threshold was selected using validation and
    test-distribution diagnostics rather than assuming that the
    validation optimum transfers unchanged to the full test set.
6.  The large-scale inference pipeline depends on pre-generated
    candidate data and sufficient compute/storage resources.

No external entity lookup, geocoding service, or external business
database is used.

## 25. Reproducibility Notes

The core source code is under:

``` text
code/business_entity_resolution/src/
```

The project includes blocking utilities, feature engineering, model
training/evaluation, inference, submission validation, and
EDA/normalization notebooks.

Large raw datasets and generated intermediate artifacts are separate
from the core source code. Candidate generation and its generated
artifacts should be treated as preprocessing inputs to the final
matching pipeline when reproducing inference.

## 26. Key Lessons

1.  Candidate recall is a first-class metric: a classifier cannot
    recover a true pair that blocking never generates.
2.  Multi-match ground truth changes evaluation: one-to-one assumptions
    are inappropriate.
3.  Hard negatives matter because random negatives do not sufficiently
    represent difficult cases.
4.  Feature engineering combining name, address, contact, country,
    domain, and numeric signals significantly improves classification.
5.  Distribution shift matters: thresholds can behave differently when
    country mix and candidate density change.
6.  Generic address tokens require care because they can generate large
    numbers of superficially plausible candidates.
7.  Checkpointing is important for large inference jobs.
8.  Post-run validation is necessary because correct predictions are not
    enough if output format or IDs are invalid.

## 27. Conclusion

The final system uses a scalable **blocking → feature engineering →
machine-learning classification → thresholding → validation**
architecture for business entity resolution.

The development process combined multi-match ground-truth analysis,
high-recall candidate generation, fuzzy and structural feature
engineering, hard-negative training, grouped cross-validation, threshold
analysis, distribution-shift diagnostics, address/number hygiene, and
checkpointed inference.

The strongest reported sampled candidate-generation result achieved
**98.0893% candidate recall**, while the feature-engineered validation
experiment achieved a **Macro F0.5 of 0.8659**.

These are controlled development and validation results and should not
be interpreted as a claim of a particular final leaderboard score.
