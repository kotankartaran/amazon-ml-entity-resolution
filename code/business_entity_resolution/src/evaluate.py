"""
Exact Official Challenge Metric Evaluator
Macro F0.5 per S1, Macro Precision, Macro Recall, Singleton FP Rate, Empty Accuracy
Supports Multi-Match S1 Ground Truth (comma-separated target IDs unnested into sets)
"""

import os
import sys
import numpy as np
import duckdb

def evaluate_predictions(predictions_dict, ground_truth_dict, all_s1_ids):
    """
    Computes exact Macro F0.5 per S1.
    
    predictions_dict: dict {s1_id: set(predicted_target_ids)}
    ground_truth_dict: dict {s1_id: set(true_target_ids)}
    all_s1_ids: list/set of all S1 entity IDs being evaluated
    """
    total_s1 = len(all_s1_ids)
    if total_s1 == 0:
        return {}

    f05_list = []
    precision_list = []
    recall_list = []
    match_count_list = []
    gt_count_list = []
    singleton_fp_count = 0
    singleton_count = 0
    empty_pred_count = 0

    for s1_id in all_s1_ids:
        gt_set = ground_truth_dict.get(s1_id, set())
        pred_set = predictions_dict.get(s1_id, set())

        len_gt = len(gt_set)
        len_pred = len(pred_set)

        match_count_list.append(len_pred)
        gt_count_list.append(len_gt)

        if len_pred == 0:
            empty_pred_count += 1

        if len_gt == 0:
            singleton_count += 1
            if len_pred == 0:
                p, r, f05 = 1.0, 1.0, 1.0
            else:
                singleton_fp_count += 1
                p, r, f05 = 0.0, 1.0, 0.0
        else:
            if len_pred == 0:
                p, r, f05 = 0.0, 0.0, 0.0
            else:
                tp = len(pred_set.intersection(gt_set))
                p = tp / len_pred
                r = tp / len_gt
                if tp == 0:
                    f05 = 0.0
                else:
                    f05 = (1.25 * p * r) / (0.25 * p + r)

        f05_list.append(f05)
        precision_list.append(p)
        recall_list.append(r)

    f05_arr = np.array(f05_list)
    p_arr = np.array(precision_list)
    r_arr = np.array(recall_list)
    pred_counts = np.array(match_count_list)
    gt_counts = np.array(gt_count_list)

    stats = {
        "macro_f05": float(f05_arr.mean()),
        "macro_precision": float(p_arr.mean()),
        "macro_recall": float(r_arr.mean()),
        "singleton_fp_rate": float(singleton_fp_count / singleton_count) if singleton_count > 0 else 0.0,
        "empty_prediction_rate": float(empty_pred_count / total_s1),
        "total_s1": total_s1,
        "total_predictions": int(pred_counts.sum()),
        "mean_pred_matches": float(pred_counts.mean()),
        "median_pred_matches": float(np.median(pred_counts)),
        "p90_pred_matches": float(np.percentile(pred_counts, 90)),
        "p99_pred_matches": float(np.percentile(pred_counts, 99)),
        "max_pred_matches": int(pred_counts.max()) if len(pred_counts) > 0 else 0,
        "total_true_matches": int(gt_counts.sum()),
        "mean_true_matches": float(gt_counts.mean()),
    }
    return stats

def evaluate_files(matching_file, ground_truth_file, source1_file):
    """Reads TSV files directly using DuckDB and evaluates Macro F0.5."""
    con = duckdb.connect()
    print(f"Loading Ground Truth from {ground_truth_file}...")
    gt_rows = con.execute(f"""
        SELECT source1_entity_id, TRIM(UNNEST(STRING_SPLIT(matched_entity_ids, ','))) AS target_id
        FROM read_csv('{ground_truth_file}', delim='\\t', header=true)
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
    """).fetchall()
    gt_dict = {}
    for s1, target in gt_rows:
        if s1 not in gt_dict:
            gt_dict[s1] = set()
        gt_dict[s1].add(target)

    print(f"Loading S1 list from {source1_file}...")
    s1_rows = con.execute(f"""
        SELECT entity_id FROM read_csv('{source1_file}', delim='\\t', header=true)
    """).fetchall()
    all_s1_ids = [r[0] for r in s1_rows]

    print(f"Loading Predictions from {matching_file}...")
    pred_dict = {}
    with open(matching_file, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                pred_dict[parts[0]] = set(parts[1].split(","))
            else:
                pred_dict[parts[0]] = set()

    con.close()
    return evaluate_predictions(pred_dict, gt_dict, all_s1_ids)

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        results = evaluate_files(sys.argv[1], sys.argv[2], sys.argv[3])
        print("\nEvaluation Results:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v:,}")
