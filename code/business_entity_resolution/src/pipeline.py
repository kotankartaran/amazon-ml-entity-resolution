"""
High-Performance Multiprocessing Production Pipeline for Amazon ML Challenge 2026
Orchestrates 8 parallel worker processes streaming 1.7M test S1 candidate pairs
with instant chunk checkpointing and post-inference validation audits.
"""

import os
import sys
import time
import zipfile
import subprocess
import joblib
import multiprocessing as mp
import numpy as np
import pandas as pd
import duckdb
from src.blocking_utils import canonicalize_country
from src.features import batch_extract_features
from src.model import EntityResolutionMatcher

DATA_DIR = os.environ.get("DATA_DIR", "dataset")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = "output"
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "matching_chunks")
MODEL_PATH = os.path.join(OUTPUT_DIR, "matcher_model.joblib")

MATCHING_OUTPUT = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_OUTPUT = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
ZIP_OUTPUT = os.path.join(OUTPUT_DIR, "submission.zip")
VALIDATOR_SCRIPT = os.environ.get("VALIDATOR_SCRIPT", "utils/validate_submission.py")

PROD_THRESHOLD = 0.93
NUM_WORKERS = 8
CHUNK_S1_SIZE = 20_000

# Worker Process Function
def process_chunk_worker(task_args):
    """
    Worker function executed in parallel across 8 processes.
    Task args: (chunk_idx, s1_id_batch, s1_names_dict, s1_addrs_dict, s1_countries_dict, t_names_dict, t_addrs_dict, t_countries_dict, cand_pairs_path, offset, feature_cols, threshold)
    """
    chunk_idx, s1_id_batch, s1_names_dict, s1_addrs_dict, s1_countries_dict, t_names_dict, t_addrs_dict, t_countries_dict, cand_pairs_path, offset, feature_cols, threshold = task_args
    chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{chunk_idx:04d}.tsv")

    # Checkpoint check: skip if already computed
    if os.path.exists(chunk_file) and os.path.getsize(chunk_file) > 0:
        with open(chunk_file, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        return chunk_idx, len(s1_id_batch), 0, sum(len(l.split("\t")[1].split(",")) for l in lines if "\t" in l and l.split("\t")[1].strip()), 0.0

    t0 = time.time()
    
    # Load pre-trained model once in worker process (0.007s)
    matcher = joblib.load(MODEL_PATH)

    matched_dict = {s1: [] for s1 in s1_id_batch}

    start_line = offset + 2  # Line 1 is header
    end_line = start_line + len(s1_id_batch) - 1

    batch_pairs = []
    with open(cand_pairs_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx < start_line:
                continue
            if idx > end_line:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1].strip():
                s1 = parts[0]
                for c in parts[1].split(","):
                    c_clean = c.strip()
                    if c_clean:
                        batch_pairs.append((s1, c_clean))

    cands_count = len(batch_pairs)
    total_matches = 0

    # Extract features & score in sub-batches of 500,000 pairs
    BATCH_SUB_SIZE = 500_000
    for sub_idx in range(0, cands_count, BATCH_SUB_SIZE):
        sub_pairs = batch_pairs[sub_idx:sub_idx+BATCH_SUB_SIZE]

        s1_names = [s1_names_dict.get(p[0], "") for p in sub_pairs]
        s1_addrs = [s1_addrs_dict.get(p[0], "") for p in sub_pairs]
        s1_countries = [s1_countries_dict.get(p[0], "") for p in sub_pairs]

        t_names = [t_names_dict.get(p[1], "") for p in sub_pairs]
        t_addrs = [t_addrs_dict.get(p[1], "") for p in sub_pairs]
        t_countries = [t_countries_dict.get(p[1], "") for p in sub_pairs]

        pair_df = pd.DataFrame({
            "s1_name": s1_names, "s1_address": s1_addrs, "s1_country": s1_countries,
            "target_name": t_names, "target_address": t_addrs, "target_country": t_countries
        })

        feat_df = batch_extract_features(pair_df)
        probs = matcher.predict_proba(feat_df[feature_cols].values)

        above_indices = np.where(probs >= threshold)[0]
        for idx in above_indices:
            s1_id, target_id = sub_pairs[idx]
            if target_id not in matched_dict[s1_id]:
                matched_dict[s1_id].append(target_id)
                total_matches += 1

    # Write chunk checkpoint file
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    with open(chunk_file, "w", encoding="utf-8", newline="") as f:
        for s1 in s1_id_batch:
            mids = matched_dict.get(s1, [])
            unique_mids = list(dict.fromkeys(mids))
            mids_str = ",".join(unique_mids) if unique_mids else ""
            f.write(f"{s1}\t{mids_str}\n")

    return chunk_idx, len(s1_id_batch), cands_count, total_matches, time.time() - t0

def run_parallel_inference(threshold=PROD_THRESHOLD, num_workers=NUM_WORKERS, max_s1_limit=None, cand_pairs_path="output/test_candidate_pairs.tsv"):
    print("\n" + "="*80)
    print(f"PARALLEL PRODUCTION INFERENCE ({num_workers} Workers, Threshold = {threshold})")
    print("="*80)

    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    con = duckdb.connect()

    print("Loading test S1 entity dataset...")
    s1_df = con.execute(f"""
        SELECT entity_id, business_name, business_address, country 
        FROM read_csv('{os.path.join(TEST_DIR, "test_source1.tsv")}', delim='\\t', header=true)
    """).df()

    test_s1_ids = list(s1_df["entity_id"].values)
    if max_s1_limit:
        test_s1_ids = test_s1_ids[:max_s1_limit]
        print(f"[SMOKE TEST MODE] Limiting to first {max_s1_limit:,} S1 entities.")

    s1_df["canon_country"] = s1_df["country"].apply(canonicalize_country)
    
    s1_names_dict = dict(zip(s1_df["entity_id"], s1_df["business_name"].fillna("")))
    s1_addrs_dict = dict(zip(s1_df["entity_id"], s1_df["business_address"].fillna("")))
    s1_countries_dict = dict(zip(s1_df["entity_id"], s1_df["country"].fillna("")))
    s1_country_map = dict(zip(s1_df["entity_id"], s1_df["canon_country"]))

    total_s1 = len(test_s1_ids)
    print(f"Loaded {total_s1:,} test S1 entities.")

    print("Loading test S2 & S3 target entity details...")
    s2_df = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{os.path.join(TEST_DIR, 'test_source2.tsv')}', delim='\\t', header=true)").df()
    s3_df = con.execute(f"SELECT entity_id, business_name, business_address, country FROM read_csv('{os.path.join(TEST_DIR, 'test_source3.tsv')}', delim='\\t', header=true)").df()
    
    target_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    t_names_dict = dict(zip(target_df["entity_id"], target_df["business_name"].fillna("")))
    t_addrs_dict = dict(zip(target_df["entity_id"], target_df["business_address"].fillna("")))
    t_countries_dict = dict(zip(target_df["entity_id"], target_df["country"].fillna("")))

    print(f"Loaded {len(t_names_dict):,} target entities (S2 + S3).")
    con.close()

    train_parquet = "output/candidate_features_sample.parquet"
    train_df = pd.read_parquet(train_parquet)
    ignore_cols = {"source1_entity_id", "candidate_entity_id", "is_match"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    # Pre-train and cache model once
    if not os.path.exists(MODEL_PATH):
        print("Pre-training and saving LightGBM matcher model...")
        matcher = EntityResolutionMatcher(n_estimators=300, learning_rate=0.05)
        matcher.train(train_df[feature_cols].values, train_df["is_match"].values)
        joblib.dump(matcher, MODEL_PATH)

    # Prepare Chunk Tasks
    tasks = []
    chunk_idx = 0
    for offset in range(0, total_s1, CHUNK_S1_SIZE):
        s1_batch = test_s1_ids[offset:offset+CHUNK_S1_SIZE]
        task = (
            chunk_idx, s1_batch, s1_names_dict, s1_addrs_dict, s1_countries_dict,
            t_names_dict, t_addrs_dict, t_countries_dict,
            cand_pairs_path, offset, feature_cols, threshold
        )
        tasks.append(task)
        chunk_idx += 1

    print(f"Prepared {len(tasks)} chunk tasks across {num_workers} parallel workers.")

    # Execute Tasks via multiprocessing Pool
    total_processed_s1 = 0
    total_cands_scored = 0
    total_matches_found = 0

    pool = mp.Pool(processes=num_workers)
    
    results = []
    for res in pool.imap_unordered(process_chunk_worker, tasks):
        c_idx, c_s1, c_cands, c_matches, c_time = res
        results.append(res)
        total_processed_s1 += c_s1
        total_cands_scored += c_cands
        total_matches_found += c_matches

        elapsed = time.time() - start_time
        s1_per_min = (total_processed_s1 / elapsed) * 60 if elapsed > 0 else 0
        rem_s1 = total_s1 - total_processed_s1
        rem_min = (rem_s1 / s1_per_min) if s1_per_min > 0 else 0

        print(f"  Completed Chunk {c_idx+1}/{len(tasks)} | S1: {total_processed_s1:,}/{total_s1:,} ({total_processed_s1/total_s1:.1%}) | Candidates: {total_cands_scored:,} | Matches: {total_matches_found:,} | Rate: {s1_per_min:.0f} S1/min | Est Rem: {rem_min:.1f} mins", flush=True)

    pool.close()
    pool.join()

    # Combine Chunk Files deterministically in S1 order
    print(f"\nCombining {len(tasks)} chunk files into {MATCHING_OUTPUT}...")
    matched_dict = {}
    
    with open(MATCHING_OUTPUT, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        for idx in range(len(tasks)):
            chunk_file = os.path.join(CHUNKS_DIR, f"chunk_{idx:04d}.tsv")
            with open(chunk_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    parts = line.rstrip("\n").split("\t")
                    s1 = parts[0]
                    mids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
                    matched_dict[s1] = mids

    # Format output/candidate_pairs.tsv
    if not os.path.exists(CANDIDATE_OUTPUT) or os.path.abspath(cand_pairs_path) != os.path.abspath(CANDIDATE_OUTPUT):
        if not os.path.exists(CANDIDATE_OUTPUT):
            print(f"Linking candidate pairs file to {CANDIDATE_OUTPUT}...")
            os.symlink(os.path.abspath(cand_pairs_path), os.path.abspath(CANDIDATE_OUTPUT))

    # Post-Inference Audits
    if not max_s1_limit:
        run_post_inference_audits(test_s1_ids, s1_country_map, matched_dict, t_names_dict)

    return MATCHING_OUTPUT, CANDIDATE_OUTPUT

def run_post_inference_audits(test_s1_ids, s1_country_map, matched_dict, target_dict):
    print("\n" + "="*80)
    print("RUNNING POST-INFERENCE AUDITS & VERIFICATION")
    print("="*80)

    # Audit 1: Line Count Check
    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    print(f"Audit 1: matching_results.tsv Line Count: {num_lines:,} (Expected: 1,732,545)")
    assert num_lines == 1732545, f"Line count mismatch! Got {num_lines}, expected 1732545."

    # Audit 2 & 3: Check S1 row uniqueness & completeness
    print("Audit 2 & 3: S1 Row Uniqueness & Completeness Check...")
    s1_rows_seen = set()
    dup_s1 = set()
    invalid_targets = set()
    intra_dups = set()
    total_matches = 0

    with open(MATCHING_OUTPUT, "r", encoding="utf-8") as f:
        header = f.readline()
        assert header.strip() == "source1_entity_id\tmatched_entity_ids", f"Invalid header: {header}"

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""

            if s1_id in s1_rows_seen:
                dup_s1.add(s1_id)
            s1_rows_seen.add(s1_id)

            if rest:
                mids = [m.strip() for m in rest.split(",") if m.strip()]
                total_matches += len(mids)
                if len(mids) != len(set(mids)):
                    intra_dups.add(s1_id)
                for mid in mids:
                    if mid not in target_dict:
                        invalid_targets.add(mid)

    assert len(dup_s1) == 0, f"Duplicate S1 rows found: {len(dup_s1)}"
    assert len(s1_rows_seen) == len(test_s1_ids), f"Missing S1 rows! Found {len(s1_rows_seen)}, expected {len(test_s1_ids)}"
    assert len(invalid_targets) == 0, f"Matched IDs not in test S2/S3: {len(invalid_targets)}"
    assert len(intra_dups) == 0, f"Duplicate matched IDs inside an S1 row: {len(intra_dups)}"

    print("  [PASSED] All row uniqueness, completeness, and ID-existence checks cleared!")

    # Audit 4: Statistical Metrics Breakdown
    match_counts = np.array([len(matched_dict.get(s1, [])) for s1 in test_s1_ids])
    empty_cnt = (match_counts == 0).sum()
    empty_pct = empty_cnt / len(test_s1_ids)

    print("\n--- OVERALL PREDICTION STATISTICS ---")
    print(f"  Total Test S1: {len(test_s1_ids):,}")
    print(f"  Total Matched Pairs: {total_matches:,}")
    print(f"  Empty Match Predictions: {empty_cnt:,} ({empty_pct:.2%})")
    print(f"  Mean Matches / S1: {match_counts.mean():.4f}")
    print(f"  Median Matches / S1: {np.median(match_counts):.1f}")
    print(f"  P90 Matches / S1: {np.percentile(match_counts, 90):.1f}")
    print(f"  P99 Matches / S1: {np.percentile(match_counts, 99):.1f}")
    print(f"  Max Matches / Single S1: {match_counts.max():,}")

    # Country Level Distribution
    print("\n--- COUNTRY-LEVEL MATCH DISTRIBUTIONS ---")
    country_groups = {}
    for s1 in test_s1_ids:
        c = s1_country_map.get(s1, "unknown")
        if c not in country_groups:
            country_groups[c] = []
        country_groups[c].append(len(matched_dict.get(s1, [])))

    for c in ["in", "us", "fr"]:
        counts = np.array(country_groups.get(c, [0]))
        print(f"  Country '{c:4s}' (N={len(counts):,} S1s): Mean={counts.mean():.2f}, Med={np.median(counts):.1f}, P90={np.percentile(counts, 90):.1f}, P99={np.percentile(counts, 99):.1f}, Max={counts.max():,}")

    # Official Submission Validator Check
    print("\n--- RUNNING OFFICIAL SUBMISSION VALIDATOR ---")
    val_cmd = [
        sys.executable, VALIDATOR_SCRIPT,
        "--matching", MATCHING_OUTPUT,
        "--candidate", CANDIDATE_OUTPUT,
        "--test-dir", TEST_DIR
    ]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True)
    print("Validator Output:\n", val_res.stdout)
    if val_res.returncode != 0:
        print("Validator Errors:\n", val_res.stderr)
        raise RuntimeError("Official Submission Validator FAILED!")

    print("\n[ALL AUDITS PASSED SUCCESSFULLY] Production matching_results.tsv is 100% compliant.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 20,000 S1 smoke test")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of worker processes")
    args = parser.parse_args()

    limit = 20000 if args.smoke else None
    run_parallel_inference(threshold=PROD_THRESHOLD, num_workers=args.workers, max_s1_limit=limit)
