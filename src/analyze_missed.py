import os
import duckdb


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/Users/kotankartaran/Downloads/student_resource/dataset"
)

TRAIN_DIR = os.path.join(DATA_DIR, "train")

S1_PATH = os.path.join(
    TRAIN_DIR,
    "train_source1.tsv"
)

S2_PATH = os.path.join(
    TRAIN_DIR,
    "train_source2.tsv"
)

S3_PATH = os.path.join(
    TRAIN_DIR,
    "train_source3.tsv"
)

MISSED_PATH = os.path.join(
    BASE_DIR,
    "output",
    "missed_true_pairs.tsv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "output",
    "missed_pairs_detailed.tsv"
)


con = duckdb.connect()


# ============================================================
# LOAD MISSED PAIRS
# ============================================================

print("Loading missed pairs...")

con.execute(
    f"""
    CREATE TABLE missed AS

    SELECT
        source1_entity_id,
        candidate_entity_id

    FROM read_csv(
        '{MISSED_PATH}',
        delim='\\t',
        header=true
    )
    """
)

count = con.execute(
    """
    SELECT COUNT(*)
    FROM missed
    """
).fetchone()[0]

print(
    f"Missed pairs: {count:,}"
)


# ============================================================
# LOAD S1
# ============================================================

print("Loading S1...")

con.execute(
    f"""
    CREATE TABLE s1 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM read_csv(
        '{S1_PATH}',
        delim='\\t',
        header=true
    )
    """
)


# ============================================================
# LOAD S2
# ============================================================

print("Loading S2...")

con.execute(
    f"""
    CREATE TABLE s2 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM read_csv(
        '{S2_PATH}',
        delim='\\t',
        header=true
    )
    """
)


# ============================================================
# LOAD S3
# ============================================================

print("Loading S3...")

con.execute(
    f"""
    CREATE TABLE s3 AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country

    FROM read_csv(
        '{S3_PATH}',
        delim='\\t',
        header=true
    )
    """
)


# ============================================================
# COMBINE S2 + S3
# ============================================================

con.execute(
    """
    CREATE TABLE targets AS

    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        'S2' AS source

    FROM s2

    UNION ALL

    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        'S3' AS source

    FROM s3
    """
)


# ============================================================
# JOIN MISSED PAIRS
# ============================================================

print("Joining missed pairs with records...")

con.execute(
    """
    CREATE TABLE detailed AS

    SELECT

        m.source1_entity_id,

        s.business_name
            AS s1_business_name,

        s.business_address
            AS s1_business_address,

        s.country
            AS s1_country,

        m.candidate_entity_id,

        t.source
            AS target_source,

        t.business_name
            AS target_business_name,

        t.business_address
            AS target_business_address,

        t.country
            AS target_country

    FROM missed m

    LEFT JOIN s1 s

        ON m.source1_entity_id =
           s.entity_id

    LEFT JOIN targets t

        ON m.candidate_entity_id =
           t.entity_id
    """
)


# ============================================================
# SAVE
# ============================================================

con.execute(
    f"""
    COPY detailed

    TO '{OUTPUT_PATH}'

    (
        FORMAT CSV,
        DELIMITER '\\t',
        HEADER
    )
    """
)


# ============================================================
# SAMPLE
# ============================================================

print()
print("=" * 70)
print("SAMPLE MISSED PAIRS")
print("=" * 70)

rows = con.execute(
    """
    SELECT *
    FROM detailed
    LIMIT 30
    """
).fetchall()

for row in rows:

    (
        s1_id,
        s1_name,
        s1_address,
        s1_country,
        target_id,
        target_source,
        target_name,
        target_address,
        target_country,
    ) = row

    print()
    print("-" * 70)

    print(
        f"S1 [{s1_country}]: {s1_name}"
    )

    print(
        f"   Address: {s1_address}"
    )

    print(
        f"{target_source} [{target_country}]: "
        f"{target_name}"
    )

    print(
        f"   Address: {target_address}"
    )

    print(
        f"Target ID: {target_id}"
    )


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print(
    f"Detailed file:\n{OUTPUT_PATH}"
)

con.close()