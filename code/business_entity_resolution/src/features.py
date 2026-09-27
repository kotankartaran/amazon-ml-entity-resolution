"""
Feature Engineering for Entity Resolution (Vectorized / Fast List Comprehensions)
Supports RapidFuzz string metrics, address number overlap, and canonical country compatibility.
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from src.blocking_utils import canonicalize_country, extract_address_numbers

def batch_extract_features(df_pairs):
    """
    Given a DataFrame with columns: 
    [s1_name, s1_address, s1_country, target_name, target_address, target_country]
    Returns a DataFrame of feature columns fast.
    """
    s1_names = df_pairs["s1_name"].fillna("").astype(str).tolist()
    t_names = df_pairs["target_name"].fillna("").astype(str).tolist()
    s1_addrs = df_pairs["s1_address"].fillna("").astype(str).tolist()
    t_addrs = df_pairs["target_address"].fillna("").astype(str).tolist()
    s1_countries = df_pairs["s1_country"].fillna("").astype(str).tolist()
    t_countries = df_pairs["target_country"].fillna("").astype(str).tolist()

    n_pairs = len(df_pairs)
    
    # Pre-canonicalize countries
    c1_list = [canonicalize_country(c) for c in s1_countries]
    c2_list = [canonicalize_country(c) for c in t_countries]
    
    c_match = [(1.0 if (c1 and c2 and c1 == c2) else 0.0) for c1, c2 in zip(c1_list, c2_list)]
    c_mismatch = [(1.0 if (c1 and c2 and c1 != c2) else 0.0) for c1, c2 in zip(c1_list, c2_list)]
    c_missing = [(1.0 if (not c1 or not c2) else 0.0) for c1, c2 in zip(c1_list, c2_list)]

    # Name metrics
    n_sort = [float(fuzz.token_sort_ratio(n1, n2)) for n1, n2 in zip(s1_names, t_names)]
    n_set = [float(fuzz.token_set_ratio(n1, n2)) for n1, n2 in zip(s1_names, t_names)]
    n_part = [float(fuzz.partial_ratio(n1, n2)) for n1, n2 in zip(s1_names, t_names)]
    n_w = [float(fuzz.WRatio(n1, n2)) for n1, n2 in zip(s1_names, t_names)]
    n_exact = [(1.0 if n1.casefold() == n2.casefold() and len(n1) > 0 else 0.0) for n1, n2 in zip(s1_names, t_names)]
    n_len_diff = [float(abs(len(n1) - len(n2))) for n1, n2 in zip(s1_names, t_names)]

    # Address metrics
    a_sort = [float(fuzz.token_sort_ratio(a1, a2)) for a1, a2 in zip(s1_addrs, t_addrs)]
    a_set = [float(fuzz.token_set_ratio(a1, a2)) for a1, a2 in zip(s1_addrs, t_addrs)]
    a_part = [float(fuzz.partial_ratio(a1, a2)) for a1, a2 in zip(s1_addrs, t_addrs)]
    a_exact = [(1.0 if a1.casefold() == a2.casefold() and len(a1) > 0 else 0.0) for a1, a2 in zip(s1_addrs, t_addrs)]

    # Address numbers (with conservative single-digit house-number suppression)
    num_overlap = []
    num_mismatch = []
    for a1, a2 in zip(s1_addrs, t_addrs):
        raw_nums1 = set(extract_address_numbers(a1))
        raw_nums2 = set(extract_address_numbers(a2))

        # Filter out single-digit numbers (0-9) to avoid low-entropy single-digit false positives
        nums1 = {n for n in raw_nums1 if len(n) > 1}
        nums2 = {n for n in raw_nums2 if len(n) > 1}

        # Fallback: if no multi-digit numbers exist, but single-digit numbers exist
        if not nums1 and not nums2:
            nums1 = raw_nums1
            nums2 = raw_nums2
            # A single matching single-digit number alone (e.g. '1' vs '1') is suppressed
            if len(nums1.union(nums2)) == 1 and nums1 == nums2:
                nums1 = set()
                nums2 = set()

        if nums1 and nums2:
            overlap = len(nums1.intersection(nums2))
            union = len(nums1.union(nums2))
            num_overlap.append(overlap / union if union > 0 else 0.0)
            num_mismatch.append(1.0 if overlap == 0 else 0.0)
        else:
            num_overlap.append(0.0)
            num_mismatch.append(0.0)

    comb_score = [(ns * as_) / 10000.0 for ns, as_ in zip(n_set, a_set)]

    return pd.DataFrame({
        "name_token_sort_ratio": n_sort,
        "name_token_set_ratio": n_set,
        "name_partial_ratio": n_part,
        "name_wratio": n_w,
        "name_exact_match": n_exact,
        "name_len_diff": n_len_diff,
        "address_token_sort_ratio": a_sort,
        "address_token_set_ratio": a_set,
        "address_partial_ratio": a_part,
        "address_exact_match": a_exact,
        "number_overlap_ratio": num_overlap,
        "number_mismatch": num_mismatch,
        "country_match": c_match,
        "country_mismatch": c_mismatch,
        "country_missing": c_missing,
        "comb_score": comb_score
    })
