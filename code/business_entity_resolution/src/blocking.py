# ============================================================
# AMAZON ML ENTITY RESOLUTION - BLOCKING
# ============================================================

import os
import re
import time

import duckdb
from unidecode import unidecode


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.environ.get(
    "DATA_DIR",
    "dataset"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

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

GT_PATH = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "blocking.duckdb"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

# Keep this at 10,000 while testing blocking strategies.
S1_SAMPLE_SIZE = 10_000

# Maximum target frequency for fuzzy signatures.
FUZZY_SIGNATURE_MAX_FREQ = 500

# Maximum target frequency for address-pair blocking.
ADDRESS_PAIR_MAX_FREQ = 100

ADDRESS_TRIPLE_MAX_FREQ = 50

# Maximum target frequency for Top-8 address-pair blocking.
ADDRESS_TOP8_MAX_FREQ = 100

# Maximum target frequency for address-number + strong-token blocking.
NUMBER_TOKEN_MAX_FREQ = 100

# Maximum target frequency for domain root blocking.
DOMAIN_ROOT_MAX_FREQ = 100

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LEGAL SUFFIXES
# ============================================================

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "plc",
    "pvt",
    "private",
    "llp",
}


# ============================================================
# GENERIC ADDRESS TOKENS
# ============================================================

GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "drive",
    "dr",
    "lane",
    "ln",
    "court",
    "ct",
    "highway",
    "hwy",
    "parkway",
    "pkwy",
    "place",
    "pl",
    "terrace",
    "ter",
    "circle",
    "cir",
    "way",
    "unit",
    "suite",
    "ste",
    "apt",
    "apartment",
    "floor",
    "fl",
    "building",
    "bldg",
    "county",
    "city",
    "town",
    "township",
    "state",
    "province",
    "block",
    "sector",
    "near",
    "opposite",
    "opp",
    "po",
    "box",
}


ADDRESS_REPLACEMENTS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "highway": "hwy",
    "parkway": "pkwy",
    "place": "pl",
    "terrace": "ter",
    "circle": "cir",
}


TLDS = [
    ".com", ".org", ".net", ".co.in", ".in", ".io", ".us", ".info",
    ".biz", ".edu", ".gov", ".co", ".uk", ".ca", ".de"
]


# ============================================================
# DOMAIN ROOT KEY
# ============================================================

def create_domain_root_keys(name, country):
    """
    Create domain-root / concatenated business name blocking key.

    Strips TLDs (.com, .org, etc.) and legal suffixes, removes non-alphanumeric
    characters, and concatenates core brand tokens.
    """
    country = normalize_country(country)
    if not name or not country:
        return []

    raw = str(name).strip().casefold()
    raw = unidecode(raw)

    cleaned = raw
    for tld in TLDS:
        if tld in cleaned:
            cleaned = cleaned.replace(tld, "")

    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    tokens = [t for t in cleaned.split() if t not in LEGAL_SUFFIXES]
    if not tokens:
        return []

    concat_domain = "".join(tokens)
    if len(concat_domain) < 5:
        return []

    return [f"{country}|domainroot|{concat_domain}"]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value).strip().casefold()

    value = "".join(
        ch if ch.isalnum() else " "
        for ch in value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_name(value):
    return normalize_text(value)


def normalize_country(value):
    if value is None:
        return ""

    value = str(value).strip().casefold()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


# ============================================================
# LEGAL SUFFIX NORMALIZATION
# ============================================================

def remove_legal_suffix(value):

    if not value:
        return ""

    tokens = value.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def normalize_name_without_legal_suffix(value):

    return remove_legal_suffix(
        normalize_name(value)
    )


# ============================================================
# TRANSLITERATION
# ============================================================

def normalize_transliterated_name(value):

    if value is None:
        return ""

    value = str(value).strip().casefold()

    value = unidecode(value)

    value = "".join(
        ch if ch.isalnum() else " "
        for ch in value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_transliterated_core_name(value):

    return remove_legal_suffix(
        normalize_transliterated_name(value)
    )


# ============================================================
# ADDRESS NORMALIZATION
# ============================================================

def normalize_address(value):

    if value is None:
        return ""

    value = str(value).strip().casefold()

    value = "".join(
        ch if ch.isalnum() else " "
        for ch in value
    )

    tokens = value.split()

    result = []

    for token in tokens:

        result.append(
            ADDRESS_REPLACEMENTS.get(
                token,
                token
            )
        )

    return " ".join(result)


def extract_address_numbers(address):

    if not address:
        return []

    numbers = re.findall(
        r"(?<!\d)\d+(?!\d)",
        str(address)
    )

    result = []

    for number in numbers:

        try:
            result.append(
                str(int(number))
            )

        except ValueError:
            result.append(
                number
            )

    return result


def get_address_tokens(address):

    if not address:
        return []

    return normalize_address(
        address
    ).split()


def extract_strong_address_tokens(address):

    tokens = get_address_tokens(
        address
    )

    result = []

    for token in tokens:

        if token in GENERIC_ADDRESS_TOKENS:
            continue

        if len(token) < 4:
            continue

        if token.isdigit():
            continue

        result.append(
            token
        )

    return result


# ============================================================
# POSTAL / PIN
# ============================================================

def extract_postal_code(address, country):

    if not address:
        return ""

    address = str(address)

    country = normalize_country(
        country
    )

    # United States
    if country in {
        "us",
        "usa",
        "united states",
        "united states of america",
    }:

        match = re.search(
            r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)",
            address
        )

        if match:
            return match.group(1)

    # India
    if country in {
        "in",
        "india",
    }:

        matches = re.findall(
            r"(?<!\d)([1-9]\d{5})(?!\d)",
            address
        )

        if matches:
            return matches[-1]

    return ""


# ============================================================
# FUZZY NAME SIGNATURE
# ============================================================

def get_name_tokens(name):

    normalized = normalize_transliterated_core_name(
        name
    )

    if not normalized:
        return []

    return normalized.split()


def get_strong_name_tokens(name):

    tokens = get_name_tokens(
        name
    )

    result = []

    for token in tokens:

        if len(token) < 5:
            continue

        if token.isdigit():
            continue

        result.append(
            token
        )

    return result


def make_name_signature(token):
    """
    Controlled fuzzy signature.

    Example:

        strategic
        -> stratic

    Keeps first 4 + last 2 characters.
    """

    if not token:
        return ""

    if len(token) < 5:
        return ""

    if len(token) <= 6:
        return token

    return (
        token[:4]
        + token[-2:]
    )


def create_name_signature(name, country):
    """
    Use strongest/longest business-name token.
    """

    country = normalize_country(
        country
    )

    if not country:
        return ""

    tokens = get_strong_name_tokens(
        name
    )

    if not tokens:
        return ""

    token = sorted(
        set(tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[0]

    signature = make_name_signature(
        token
    )

    if not signature:
        return ""

    return (
        f"{country}|sig|{signature}"
    )


# ============================================================
# BLOCK 4 ADDRESS KEYS
# ============================================================

def create_house_token_key(address, country):
    """
    Country + normalized house number
    + strongest address token.
    """

    country_key = normalize_country(
        country
    )

    numbers = extract_address_numbers(
        address
    )

    if not country_key or not numbers:
        return ""

    house_number = numbers[0]

    tokens = extract_strong_address_tokens(
        address
    )

    if not tokens:
        return ""

    tokens = sorted(
        set(tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )

    return (
        f"{country_key}|"
        f"{house_number}|"
        f"{tokens[0]}"
    )


def create_house_two_token_key(address, country):
    """
    Country + house number + two strong
    address tokens.
    """

    country_key = normalize_country(
        country
    )

    numbers = extract_address_numbers(
        address
    )

    if not country_key or not numbers:
        return ""

    house_number = numbers[0]

    tokens = list(
        set(
            extract_strong_address_tokens(
                address
            )
        )
    )

    if len(tokens) < 2:
        return ""

    tokens = sorted(
        tokens,
        key=lambda x: (
            -len(x),
            x
        )
    )

    return (
        f"{country_key}|"
        f"{house_number}|"
        f"{tokens[0]}|"
        f"{tokens[1]}"
    )


def create_postal_key(address, country):

    country_key = normalize_country(
        country
    )

    postal = extract_postal_code(
        address,
        country
    )

    if not country_key or not postal:
        return ""

    return (
        f"{country_key}|postal|{postal}"
    )


# ============================================================
# IMPROVED NAME + ADDRESS SIGNATURES
# ============================================================

def normalize_signature_token(token):
    """
    Normalize a name token for typo-tolerant blocking.
    """

    if not token:
        return ""

    token = normalize_transliterated_name(
        token
    )

    if len(token) < 4:
        return ""

    return token


def token_signature(token):
    """
    Compact typo-tolerant signature.

    Example:

        strategic -> stratic
        hanson    -> hanson
        lending   -> lending
    """

    token = normalize_signature_token(
        token
    )

    if not token:
        return ""

    if len(token) <= 6:
        return token

    return (
        token[:4]
        + token[-2:]
    )


def get_name_signature_tokens(name):
    """
    Return strong business-name signatures.
    """

    tokens = get_name_tokens(
        name
    )

    result = []

    for token in tokens:

        token = normalize_signature_token(
            token
        )

        if not token:
            continue

        if token in LEGAL_SUFFIXES:
            continue

        signature = token_signature(
            token
        )

        if signature:
            result.append(
                signature
            )

    return sorted(
        set(result)
    )


# ============================================================
# BLOCK 5A KEY
# ============================================================

def create_name_address_keys(
    name,
    address,
    country
):
    """
    Name signature + strong address token.
    """

    country = normalize_country(
        country
    )

    if not country:
        return []

    name_tokens = get_name_signature_tokens(
        name
    )

    address_tokens = extract_strong_address_tokens(
        address
    )

    if not name_tokens or not address_tokens:
        return []

    address_tokens = sorted(
        set(address_tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[:3]

    keys = []

    for name_token in name_tokens:

        for address_token in address_tokens:

            keys.append(
                f"{country}|na|"
                f"{name_token}|"
                f"{address_token}"
            )

    return keys


# ============================================================
# BLOCK 5B KEY
# ============================================================

def create_house_name_key(
    name,
    address,
    country
):
    """
    House number + name signature.
    """

    country = normalize_country(
        country
    )

    numbers = extract_address_numbers(
        address
    )

    if not country or not numbers:
        return []

    house_number = numbers[0]

    name_tokens = get_name_signature_tokens(
        name
    )

    if not name_tokens:
        return []

    name_tokens = sorted(
        set(name_tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[:3]

    return [
        f"{country}|hn|{house_number}|{token}"
        for token in name_tokens
    ]


# ============================================================
# BLOCK 5C KEY
# ============================================================

def create_name_pair_keys(
    name,
    country
):
    """
    Order-independent two-token name blocking.

    Handles:
        Crystal Lending PC
        Crystal PC Lending
    """

    country = normalize_country(
        country
    )

    if not country:
        return []

    tokens = get_name_signature_tokens(
        name
    )

    if len(tokens) < 2:
        return []

    tokens = sorted(
        set(tokens)
    )

    # Maximum 5 tokens.
    tokens = tokens[:5]

    keys = []

    for i in range(
        len(tokens)
    ):

        for j in range(
            i + 1,
            len(tokens)
        ):

            token1 = tokens[i]
            token2 = tokens[j]

            keys.append(
                f"{country}|pair|"
                f"{token1}|{token2}"
            )

    return keys


# ============================================================
# BLOCK 5D KEY
# ============================================================

def create_address_pair_keys(
    address,
    country
):
    """
    Selective order-independent address pairs.

    Instead of generating all combinations from the
    entire address, keep only the four strongest
    informative tokens and create at most three
    pairs.

    This keeps candidate explosion under control.
    """

    country = normalize_country(
        country
    )

    if not country:
        return []

    tokens = extract_strong_address_tokens(
        address
    )

    if len(tokens) < 2:
        return []

    # Remove duplicates and keep strongest tokens.
    tokens = sorted(
        set(tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[:4]

    keys = []

    # Strongest + second strongest
    if len(tokens) >= 2:

        keys.append(
            f"{country}|addrpair|"
            f"{tokens[0]}|{tokens[1]}"
        )

    # Strongest + third
    if len(tokens) >= 3:

        keys.append(
            f"{country}|addrpair|"
            f"{tokens[0]}|{tokens[2]}"
        )

    # Strongest + fourth
    if len(tokens) >= 4:

        keys.append(
            f"{country}|addrpair|"
            f"{tokens[0]}|{tokens[3]}"
        )

    return keys

def create_address_triple_keys(address, country):
    """
    Create a small number of selective, order-independent
    3-token address blocking keys.
    """

    country_key = normalize_country(country)

    if not country_key:
        return []

    tokens = extract_strong_address_tokens(address)

    if len(tokens) < 3:
        return []

    # Keep the strongest / most informative tokens.
    # This prevents combinatorial explosion.
    tokens = sorted(
        set(tokens),
        key=lambda x: (-len(x), x)
    )[:6]

    keys = []

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            for k in range(j + 1, len(tokens)):

                triple = sorted(
                    [
                        tokens[i],
                        tokens[j],
                        tokens[k]
                    ]
                )

                keys.append(
                    f"{country_key}|addrtriple|"
                    f"{triple[0]}|"
                    f"{triple[1]}|"
                    f"{triple[2]}"
                )

    return list(set(keys))


# ============================================================
# BLOCK 7 KEY
# TOP-8 ADDRESS TOKEN PAIRS
# ============================================================

def create_top8_address_pair_keys(address, country):
    """
    Create all order-independent pairs among the top 8
    strongest address tokens.

    Target-side frequency filtering is applied in the
    blocking pipeline, not here.
    """

    country_key = normalize_country(country)

    if not country_key:
        return []

    tokens = extract_strong_address_tokens(address)

    if len(tokens) < 2:
        return []

    tokens = sorted(
        set(tokens),
        key=lambda x: (-len(x), x)
    )[:8]

    keys = []

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):

            keys.append(
                f"{country_key}|addrtop8|"
                f"{tokens[i]}|{tokens[j]}"
            )

    return keys


# ============================================================
# BLOCK 8 KEY
# ADDRESS NUMBER + STRONG ADDRESS TOKEN
# ============================================================

def create_number_token_keys(address, country):
    """
    Create country + address-number + strong-address-token keys.

    Examples:
        india|numtoken|204|bangalore
        us|numtoken|8066|haven
    """

    country_key = normalize_country(country)

    if not country_key:
        return []

    numbers = set(
        extract_address_numbers(address)
    )

    tokens = set(
        extract_strong_address_tokens(address)
    )

    if not numbers or not tokens:
        return []

    keys = []

    for number in numbers:
        for token in tokens:

            keys.append(
                f"{country_key}|numtoken|"
                f"{number}|{token}"
            )

    return keys


# ============================================================
# DUCKDB UDF REGISTRATION
# ============================================================

def register_functions(con):

    con.create_function(
        "normalize_name",
        normalize_name,
        ["VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "normalize_name_without_legal_suffix",
        normalize_name_without_legal_suffix,
        ["VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "normalize_transliterated_name",
        normalize_transliterated_name,
        ["VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "normalize_transliterated_core_name",
        normalize_transliterated_core_name,
        ["VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "normalize_country",
        normalize_country,
        ["VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "create_house_token_key",
        create_house_token_key,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "create_house_two_token_key",
        create_house_two_token_key,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "create_postal_key",
        create_postal_key,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "create_name_signature",
        create_name_signature,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR"
    )

    con.create_function(
        "create_name_address_keys",
        create_name_address_keys,
        [
            "VARCHAR",
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )

    con.create_function(
        "create_house_name_key",
        create_house_name_key,
        [
            "VARCHAR",
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )

    con.create_function(
        "create_name_pair_keys",
        create_name_pair_keys,
        [
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )

    con.create_function(
        "create_address_pair_keys",
        create_address_pair_keys,
        [
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )

    con.create_function(
        "create_address_triple_keys",
        create_address_triple_keys,
        [
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )


    con.create_function(
        "create_top8_address_pair_keys",
        create_top8_address_pair_keys,
        [
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )

    con.create_function(
        "create_number_token_keys",
        create_number_token_keys,
        [
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )

    con.create_function(
        "create_domain_root_keys",
        create_domain_root_keys,
        [
            "VARCHAR",
            "VARCHAR"
        ],
        "VARCHAR[]"
    )


# ============================================================
# PRINT CANDIDATE COUNT
# ============================================================

def print_candidate_count(
    con,
    label
):

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        """
    ).fetchone()[0]

    print(
        f"Candidates after {label}: "
        f"{count:,}"
    )

    return count


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print(
        "AMAZON ML ENTITY RESOLUTION - BLOCKING"
    )
    print("=" * 70)

    print()
    print(
        f"Data directory: {DATA_DIR}"
    )

    print(
        f"S1 sample size: "
        f"{S1_SAMPLE_SIZE:,}"
    )

    print(
        f"Fuzzy signature frequency limit: "
        f"{FUZZY_SIGNATURE_MAX_FREQ}"
    )

    print(
        f"Address pair frequency limit: "
        f"{ADDRESS_PAIR_MAX_FREQ}"
    )

    print(
        f"Address triple frequency limit: "
        f"{ADDRESS_TRIPLE_MAX_FREQ}"
    )

    print(
        f"Top-8 address-pair frequency limit: "
        f"{ADDRESS_TOP8_MAX_FREQ}"
    )

    print(
        f"Number-token frequency limit: "
        f"{NUMBER_TOKEN_MAX_FREQ}"
    )

    print(
        f"Domain-root frequency limit: "
        f"{DOMAIN_ROOT_MAX_FREQ}"
    )


    # ========================================================
    # REMOVE OLD DATABASE
    # ========================================================

    print()
    print(
        "Creating fresh DuckDB database..."
    )

    for path in [
        DB_PATH,
        DB_PATH + ".wal",
    ]:

        if os.path.exists(path):

            try:
                os.remove(path)

            except OSError:
                pass

    con = duckdb.connect(
        DB_PATH
    )

    register_functions(
        con
    )


    # ========================================================
    # LOAD S1
    # ========================================================

    print()
    print(
        "Loading Source 1 sample..."
    )

    con.execute(
        f"""
        CREATE TABLE s1_sample AS

        SELECT
            entity_id AS source1_entity_id,
            business_name,
            business_address,
            country

        FROM read_csv(
            '{S1_PATH}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"'
        )

        LIMIT {S1_SAMPLE_SIZE}
        """
    )

    s1_count = con.execute(
        """
        SELECT COUNT(*)
        FROM s1_sample
        """
    ).fetchone()[0]

    print(
        f"S1 sample loaded: "
        f"{s1_count:,}"
    )


    # ========================================================
    # LOAD S2
    # ========================================================

    print()
    print(
        "Loading Source 2..."
    )

    con.execute(
        f"""
        CREATE TABLE source2 AS

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            'S2' AS source

        FROM read_csv(
            '{S2_PATH}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"'
        )
        """
    )

    s2_count = con.execute(
        """
        SELECT COUNT(*)
        FROM source2
        """
    ).fetchone()[0]

    print(
        f"S2 loaded: "
        f"{s2_count:,}"
    )


    # ========================================================
    # LOAD S3
    # ========================================================

    print()
    print(
        "Loading Source 3..."
    )

    con.execute(
        f"""
        CREATE TABLE source3 AS

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            'S3' AS source

        FROM read_csv(
            '{S3_PATH}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"'
        )
        """
    )

    s3_count = con.execute(
        """
        SELECT COUNT(*)
        FROM source3
        """
    ).fetchone()[0]

    print(
        f"S3 loaded: "
        f"{s3_count:,}"
    )


    # ========================================================
    # COMBINE TARGETS
    # ========================================================

    print()
    print(
        "Combining Source 2 + Source 3..."
    )

    con.execute(
        """
        CREATE TABLE all_targets AS

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            source

        FROM source2

        UNION ALL

        SELECT
            entity_id,
            business_name,
            business_address,
            country,
            source

        FROM source3
        """
    )

    target_count = con.execute(
        """
        SELECT COUNT(*)
        FROM all_targets
        """
    ).fetchone()[0]

    print(
        f"Total targets: "
        f"{target_count:,}"
    )


    # ========================================================
    # CANDIDATE TABLE
    # ========================================================

    con.execute(
        """
        CREATE TABLE candidates (
            source1_entity_id VARCHAR,
            candidate_entity_id VARCHAR
        )
        """
    )


    # ========================================================
    # BLOCK 1
    # EXACT NORMALIZED NAME
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 1: EXACT NORMALIZED NAME + COUNTRY"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b1 AS

        SELECT
            source1_entity_id,

            normalize_name(business_name)
            || '|'
            || normalize_country(country)
            AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b1 AS

        SELECT
            entity_id,

            normalize_name(business_name)
            || '|'
            || normalize_country(country)
            AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b1 s

        JOIN target_b1 t
          ON s.block_key = t.block_key

        WHERE s.block_key <> '|'
        """
    )

    print_candidate_count(
        con,
        "Block 1"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 2
    # CORE NAME
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 2: NAME WITHOUT LEGAL SUFFIX + COUNTRY"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b2 AS

        SELECT
            source1_entity_id,

            normalize_name_without_legal_suffix(
                business_name
            )
            || '|'
            || normalize_country(country)
            AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b2 AS

        SELECT
            entity_id,

            normalize_name_without_legal_suffix(
                business_name
            )
            || '|'
            || normalize_country(country)
            AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b2 s

        JOIN target_b2 t
          ON s.block_key = t.block_key

        WHERE s.block_key <> '|'
        """
    )

    print_candidate_count(
        con,
        "Block 2"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 3
    # TRANSLITERATED NAME
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 3: TRANSLITERATED NAME + COUNTRY"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b3 AS

        SELECT
            source1_entity_id,

            normalize_transliterated_name(
                business_name
            )
            || '|'
            || normalize_country(country)
            AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b3 AS

        SELECT
            entity_id,

            normalize_transliterated_name(
                business_name
            )
            || '|'
            || normalize_country(country)
            AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b3 s

        JOIN target_b3 t
          ON s.block_key = t.block_key

        WHERE s.block_key <> '|'
        """
    )

    print_candidate_count(
        con,
        "Block 3"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 3B
    # TRANSLITERATED CORE NAME
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 3B: TRANSLITERATED CORE NAME + COUNTRY"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b3b AS

        SELECT
            source1_entity_id,

            normalize_transliterated_core_name(
                business_name
            )
            || '|'
            || normalize_country(country)
            AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b3b AS

        SELECT
            entity_id,

            normalize_transliterated_core_name(
                business_name
            )
            || '|'
            || normalize_country(country)
            AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b3b s

        JOIN target_b3b t
          ON s.block_key = t.block_key

        WHERE s.block_key <> '|'
        """
    )

    print_candidate_count(
        con,
        "Block 3B"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 4A
    # HOUSE NUMBER + ADDRESS TOKEN
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 4A: HOUSE NUMBER + STRONG ADDRESS TOKEN"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b4a AS

        SELECT
            source1_entity_id,

            create_house_token_key(
                business_address,
                country
            ) AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b4a AS

        SELECT
            entity_id,

            create_house_token_key(
                business_address,
                country
            ) AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b4a s

        JOIN target_b4a t
          ON s.block_key = t.block_key

        WHERE s.block_key <> ''
        """
    )

    print_candidate_count(
        con,
        "Block 4A"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 4B
    # HOUSE NUMBER + TWO ADDRESS TOKENS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 4B: HOUSE NUMBER + TWO ADDRESS TOKENS"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b4b AS

        SELECT
            source1_entity_id,

            create_house_two_token_key(
                business_address,
                country
            ) AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b4b AS

        SELECT
            entity_id,

            create_house_two_token_key(
                business_address,
                country
            ) AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b4b s

        JOIN target_b4b t
          ON s.block_key = t.block_key

        WHERE s.block_key <> ''
        """
    )

    print_candidate_count(
        con,
        "Block 4B"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 4C
    # POSTAL / PIN
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 4C: POSTAL / PIN + COUNTRY"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b4c AS

        SELECT
            source1_entity_id,

            create_postal_key(
                business_address,
                country
            ) AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b4c AS

        SELECT
            entity_id,

            create_postal_key(
                business_address,
                country
            ) AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b4c s

        JOIN target_b4c t
          ON s.block_key = t.block_key

        WHERE s.block_key <> ''
        """
    )

    print_candidate_count(
        con,
        "Block 4C"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 5
    # CONTROLLED FUZZY NAME SIGNATURE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 5: CONTROLLED FUZZY NAME SIGNATURE"
    )
    print("=" * 70)

    t = time.time()

    # Create signatures
    con.execute(
        """
        CREATE TABLE s1_b5_raw AS

        SELECT
            source1_entity_id,

            create_name_signature(
                business_name,
                country
            ) AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b5_raw AS

        SELECT
            entity_id,

            create_name_signature(
                business_name,
                country
            ) AS block_key

        FROM all_targets
        """
    )

    # Target signature frequencies
    con.execute(
        """
        CREATE TABLE target_b5_frequency AS

        SELECT
            block_key,
            COUNT(*) AS frequency

        FROM target_b5_raw

        WHERE block_key <> ''

        GROUP BY block_key
        """
    )

    # Keep selective signatures
    con.execute(
        """
        CREATE TABLE target_b5 AS

        SELECT
            t.entity_id,
            t.block_key

        FROM target_b5_raw t

        JOIN target_b5_frequency f
          ON t.block_key = f.block_key

        WHERE
            t.block_key <> ''

            AND

            f.frequency <= ?
        """,
        [
            FUZZY_SIGNATURE_MAX_FREQ
        ]
    )

    con.execute(
        """
        CREATE TABLE s1_b5 AS

        SELECT DISTINCT
            s.source1_entity_id,
            s.block_key

        FROM s1_b5_raw s

        JOIN target_b5 t
          ON s.block_key = t.block_key

        WHERE s.block_key <> ''
        """
    )

    # Join
    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b5 s

        JOIN target_b5 t
          ON s.block_key = t.block_key
        """
    )

    print_candidate_count(
        con,
        "Block 5"
    )

    selective_signatures = con.execute(
        """
        SELECT COUNT(*)
        FROM target_b5_frequency
        WHERE frequency <= ?
        """,
        [
            FUZZY_SIGNATURE_MAX_FREQ
        ]
    ).fetchone()[0]

    print(
        f"Selective signatures: "
        f"{selective_signatures:,}"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 5A
    # NAME SIGNATURE + ADDRESS TOKEN
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 5A: NAME SIGNATURE + ADDRESS TOKEN"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b5a AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_name_address_keys(
                    business_name,
                    business_address,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b5a AS

        SELECT
            entity_id,

            UNNEST(
                create_name_address_keys(
                    business_name,
                    business_address,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b5a s

        JOIN target_b5a t
          ON s.block_key = t.block_key

        WHERE s.block_key <> ''
        """
    )

    print_candidate_count(
        con,
        "Block 5A"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 5B
    # HOUSE NUMBER + NAME SIGNATURE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 5B: HOUSE NUMBER + NAME SIGNATURE"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b5b AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_house_name_key(
                    business_name,
                    business_address,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b5b AS

        SELECT
            entity_id,

            UNNEST(
                create_house_name_key(
                    business_name,
                    business_address,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b5b s

        JOIN target_b5b t
          ON s.block_key = t.block_key

        WHERE s.block_key <> ''
        """
    )

    print_candidate_count(
        con,
        "Block 5B"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 5C
    # TWO-TOKEN ORDER-INDEPENDENT NAME
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 5C: TWO-TOKEN ORDER-INDEPENDENT NAME"
    )
    print("=" * 70)

    t = time.time()

    con.execute(
        """
        CREATE TABLE s1_b5c AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_name_pair_keys(
                    business_name,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    con.execute(
        """
        CREATE TABLE target_b5c AS

        SELECT
            entity_id,

            UNNEST(
                create_name_pair_keys(
                    business_name,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    # Frequency control
    con.execute(
        """
        CREATE TABLE target_b5c_frequency AS

        SELECT
            block_key,
            COUNT(*) AS frequency

        FROM target_b5c

        WHERE block_key <> ''

        GROUP BY block_key
        """
    )

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b5c s

        JOIN target_b5c t
          ON s.block_key = t.block_key

        JOIN target_b5c_frequency f
          ON t.block_key = f.block_key

        WHERE
            s.block_key <> ''

            AND

            f.frequency <= ?
        """,
        [
            FUZZY_SIGNATURE_MAX_FREQ
        ]
    )

    print_candidate_count(
        con,
        "Block 5C"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 5D
    # SELECTIVE ADDRESS TOKEN PAIRS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 5D: SELECTIVE ADDRESS TOKEN PAIRS"
    )
    print("=" * 70)

    t = time.time()

    # --------------------------------------------------------
    # Generate S1 address-pair keys
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE s1_b5d AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_address_pair_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    # --------------------------------------------------------
    # Generate target address-pair keys
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE target_b5d_raw AS

        SELECT
            entity_id,

            UNNEST(
                create_address_pair_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    # --------------------------------------------------------
    # Count target frequencies
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE target_b5d_frequency AS

        SELECT
            block_key,
            COUNT(*) AS frequency

        FROM target_b5d_raw

        WHERE block_key <> ''

        GROUP BY block_key
        """
    )

    # --------------------------------------------------------
    # Keep only rare/selective address pairs
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE target_b5d AS

        SELECT
            t.entity_id,
            t.block_key

        FROM target_b5d_raw t

        JOIN target_b5d_frequency f
          ON t.block_key = f.block_key

        WHERE
            t.block_key <> ''

            AND

            f.frequency <= ?
        """,
        [
            ADDRESS_PAIR_MAX_FREQ
        ]
    )

    # --------------------------------------------------------
    # Join
    # --------------------------------------------------------

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b5d s

        JOIN target_b5d t
          ON s.block_key = t.block_key
        """
    )

    print_candidate_count(
        con,
        "Block 5D"
    )

    selective_address_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM target_b5d_frequency
        WHERE frequency <= ?
        """,
        [
            ADDRESS_PAIR_MAX_FREQ
        ]
    ).fetchone()[0]

    print(
        f"Selective address-pair keys: "
        f"{selective_address_pairs:,}"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )





    # ========================================================
    # BLOCK 6A
    # SELECTIVE ADDRESS TOKEN TRIPLES
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 6A: SELECTIVE ADDRESS TOKEN TRIPLES"
    )
    print("=" * 70)

    t = time.time()

    # --------------------------------------------------------
    # Generate S1 address-triple keys
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE s1_b6a AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_address_triple_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    # --------------------------------------------------------
    # Generate target address-triple keys
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE target_b6a_raw AS

        SELECT
            entity_id,

            UNNEST(
                create_address_triple_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    # --------------------------------------------------------
    # Count target frequencies
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE target_b6a_frequency AS

        SELECT
            block_key,
            COUNT(*) AS frequency

        FROM target_b6a_raw

        WHERE block_key <> ''

        GROUP BY block_key
        """
    )

    # --------------------------------------------------------
    # Keep only selective address triples
    # --------------------------------------------------------

    con.execute(
        """
        CREATE TABLE target_b6a AS

        SELECT
            t.entity_id,
            t.block_key

        FROM target_b6a_raw t

        JOIN target_b6a_frequency f
          ON t.block_key = f.block_key

        WHERE
            t.block_key <> ''

            AND

            f.frequency <= ?
        """,
        [
            ADDRESS_TRIPLE_MAX_FREQ
        ]
    )

    # --------------------------------------------------------
    # Join
    # --------------------------------------------------------

    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b6a s

        JOIN target_b6a t
          ON s.block_key = t.block_key
        """
    )

    print_candidate_count(
        con,
        "Block 6A"
    )

    selective_address_triples = con.execute(
        """
        SELECT COUNT(*)
        FROM target_b6a_frequency
        WHERE frequency <= ?
        """,
        [
            ADDRESS_TRIPLE_MAX_FREQ
        ]
    ).fetchone()[0]

    print(
        f"Selective address-triple keys: "
        f"{selective_address_triples:,}"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )



    # ========================================================
    # BLOCK 7
    # TOP-8 ADDRESS TOKEN PAIRS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 7: TOP-8 ADDRESS TOKEN PAIRS"
    )
    print("=" * 70)

    t = time.time()

    # Generate S1 keys.
    con.execute(
        """
        CREATE TABLE s1_b7 AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_top8_address_pair_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    # Generate target keys.
    con.execute(
        """
        CREATE TABLE target_b7_raw AS

        SELECT
            entity_id,

            UNNEST(
                create_top8_address_pair_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    # Count target frequencies.
    con.execute(
        """
        CREATE TABLE target_b7_frequency AS

        SELECT
            block_key,
            COUNT(*) AS frequency

        FROM target_b7_raw

        WHERE block_key <> ''

        GROUP BY block_key
        """
    )

    # Keep only selective Top-8 pairs.
    con.execute(
        """
        CREATE TABLE target_b7 AS

        SELECT
            t.entity_id,
            t.block_key

        FROM target_b7_raw t

        JOIN target_b7_frequency f
          ON t.block_key = f.block_key

        WHERE
            t.block_key <> ''

            AND

            f.frequency <= ?
        """,
        [
            ADDRESS_TOP8_MAX_FREQ
        ]
    )

    # Join S1 to targets.
    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b7 s

        JOIN target_b7 t
          ON s.block_key = t.block_key
        """
    )

    print_candidate_count(
        con,
        "Block 7"
    )

    selective_top8_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM target_b7_frequency
        WHERE frequency <= ?
        """,
        [
            ADDRESS_TOP8_MAX_FREQ
        ]
    ).fetchone()[0]

    print(
        f"Selective Top-8 pair keys: "
        f"{selective_top8_pairs:,}"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 8
    # ADDRESS NUMBER + STRONG ADDRESS TOKEN
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 8: ADDRESS NUMBER + STRONG TOKEN"
    )
    print("=" * 70)

    t = time.time()

    # Generate S1 keys.
    con.execute(
        """
        CREATE TABLE s1_b8 AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_number_token_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    # Generate target keys.
    con.execute(
        """
        CREATE TABLE target_b8_raw AS

        SELECT
            entity_id,

            UNNEST(
                create_number_token_keys(
                    business_address,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    # Count target frequencies.
    con.execute(
        """
        CREATE TABLE target_b8_frequency AS

        SELECT
            block_key,
            COUNT(*) AS frequency

        FROM target_b8_raw

        WHERE block_key <> ''

        GROUP BY block_key
        """
    )

    # Keep only selective number-token keys.
    con.execute(
        """
        CREATE TABLE target_b8 AS

        SELECT
            t.entity_id,
            t.block_key

        FROM target_b8_raw t

        JOIN target_b8_frequency f
          ON t.block_key = f.block_key

        WHERE
            t.block_key <> ''

            AND

            f.frequency <= ?
        """,
        [
            NUMBER_TOKEN_MAX_FREQ
        ]
    )

    # Join S1 to targets.
    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b8 s

        JOIN target_b8 t
          ON s.block_key = t.block_key
        """
    )

    print_candidate_count(
        con,
        "Block 8"
    )

    selective_number_token_keys = con.execute(
        """
        SELECT COUNT(*)
        FROM target_b8_frequency
        WHERE frequency <= ?
        """,
        [
            NUMBER_TOKEN_MAX_FREQ
        ]
    ).fetchone()[0]

    print(
        f"Selective number-token keys: "
        f"{selective_number_token_keys:,}"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # BLOCK 9
    # DOMAIN ROOT / CONCATENATED BRAND NAME
    # ========================================================

    print()
    print("=" * 70)
    print(
        "BLOCK 9: DOMAIN ROOT / CONCATENATED BRAND NAME"
    )
    print("=" * 70)

    t = time.time()

    # Generate S1 keys.
    con.execute(
        """
        CREATE TABLE s1_b9 AS

        SELECT
            source1_entity_id,

            UNNEST(
                create_domain_root_keys(
                    business_name,
                    country
                )
            ) AS block_key

        FROM s1_sample
        """
    )

    # Generate target keys.
    con.execute(
        """
        CREATE TABLE target_b9_raw AS

        SELECT
            entity_id,

            UNNEST(
                create_domain_root_keys(
                    business_name,
                    country
                )
            ) AS block_key

        FROM all_targets
        """
    )

    # Count target frequencies.
    con.execute(
        """
        CREATE TABLE target_b9_frequency AS

        SELECT
            block_key,
            COUNT(*) AS frequency

        FROM target_b9_raw

        WHERE block_key <> ''

        GROUP BY block_key
        """
    )

    # Keep selective domain root keys.
    con.execute(
        """
        CREATE TABLE target_b9 AS

        SELECT
            t.entity_id,
            t.block_key

        FROM target_b9_raw t

        JOIN target_b9_frequency f
          ON t.block_key = f.block_key

        WHERE
            t.block_key <> ''

            AND

            f.frequency <= ?
        """,
        [
            DOMAIN_ROOT_MAX_FREQ
        ]
    )

    # Join S1 to targets.
    con.execute(
        """
        INSERT INTO candidates

        SELECT DISTINCT
            s.source1_entity_id,
            t.entity_id

        FROM s1_b9 s

        JOIN target_b9 t
          ON s.block_key = t.block_key
        """
    )

    print_candidate_count(
        con,
        "Block 9"
    )

    selective_domain_root_keys = con.execute(
        """
        SELECT COUNT(*)
        FROM target_b9_frequency
        WHERE frequency <= ?
        """,
        [
            DOMAIN_ROOT_MAX_FREQ
        ]
    ).fetchone()[0]

    print(
        f"Selective domain-root keys: "
        f"{selective_domain_root_keys:,}"
    )

    print(
        f"Time: {time.time() - t:.2f}s"
    )


    # ========================================================
    # FINAL DEDUPLICATION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "DEDUPLICATING CANDIDATES"
    )
    print("=" * 70)

    con.execute(
        """
        CREATE TABLE final_candidates AS

        SELECT DISTINCT
            source1_entity_id,
            candidate_entity_id

        FROM candidates
        """
    )

    total_candidates = con.execute(
        """
        SELECT COUNT(*)
        FROM final_candidates
        """
    ).fetchone()[0]

    print(
        f"Final candidate pairs: "
        f"{total_candidates:,}"
    )


    # ========================================================
    # CANDIDATE DISTRIBUTION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CANDIDATE DISTRIBUTION"
    )
    print("=" * 70)

    stats = con.execute(
        """
        SELECT

            MIN(candidate_count),

            AVG(candidate_count),

            quantile_cont(
                candidate_count,
                0.50
            ),

            quantile_cont(
                candidate_count,
                0.90
            ),

            quantile_cont(
                candidate_count,
                0.95
            ),

            quantile_cont(
                candidate_count,
                0.99
            ),

            MAX(candidate_count)

        FROM (

            SELECT
                source1_entity_id,
                COUNT(*) AS candidate_count

            FROM final_candidates

            GROUP BY source1_entity_id

        )
        """
    ).fetchone()

    print(
        f"Minimum : {stats[0]:,.2f}"
    )

    print(
        f"Average : {stats[1]:,.2f}"
    )

    print(
        f"Median  : {stats[2]:,.2f}"
    )

    print(
        f"P90     : {stats[3]:,.2f}"
    )

    print(
        f"P95     : {stats[4]:,.2f}"
    )

    print(
        f"P99     : {stats[5]:,.2f}"
    )

    print(
        f"Maximum : {stats[6]:,.2f}"
    )


    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print()
    print("=" * 70)
    print(
        "GROUND TRUTH EVALUATION"
    )
    print("=" * 70)

    con.execute(
        f"""
        CREATE TABLE ground_truth AS

        SELECT
            source1_entity_id,
            matched_entity_ids

        FROM read_csv(
            '{GT_PATH}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"'
        )
        """
    )

    con.execute(
        """
        CREATE TABLE sampled_ground_truth AS

        SELECT
            gt.source1_entity_id,
            gt.matched_entity_ids

        FROM ground_truth gt

        JOIN s1_sample s

          ON gt.source1_entity_id =
             s.source1_entity_id
        """
    )

    con.execute(
        """
        CREATE TABLE true_pairs AS

        SELECT DISTINCT

            gt.source1_entity_id,

            TRIM(
                UNNEST(
                    STRING_SPLIT(
                        gt.matched_entity_ids,
                        ','
                    )
                )
            ) AS candidate_entity_id

        FROM sampled_ground_truth gt

        WHERE
            gt.matched_entity_ids IS NOT NULL

            AND

            TRIM(
                gt.matched_entity_ids
            ) <> ''
        """
    )

    total_true_pairs = con.execute(
        """
        SELECT COUNT(*)
        FROM true_pairs
        """
    ).fetchone()[0]

    found_true_pairs = con.execute(
        """
        SELECT COUNT(*)

        FROM true_pairs gt

        JOIN final_candidates c

          ON gt.source1_entity_id =
             c.source1_entity_id

         AND gt.candidate_entity_id =
             c.candidate_entity_id
        """
    ).fetchone()[0]

    missed_true_pairs = (
        total_true_pairs
        -
        found_true_pairs
    )

    recall = (
        found_true_pairs
        /
        total_true_pairs
        if total_true_pairs > 0
        else 0.0
    )


    # ========================================================
    # MISSED TRUE PAIRS
    # ========================================================

    con.execute(
        """
        CREATE TABLE missed_true_pairs AS

        SELECT

            gt.source1_entity_id,

            gt.candidate_entity_id

        FROM true_pairs gt

        LEFT JOIN final_candidates c

          ON gt.source1_entity_id =
             c.source1_entity_id

         AND gt.candidate_entity_id =
             c.candidate_entity_id

        WHERE
            c.candidate_entity_id IS NULL
        """
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    candidate_output = os.path.join(
        OUTPUT_DIR,
        "candidate_pairs.tsv"
    )

    missed_output = os.path.join(
        OUTPUT_DIR,
        "missed_true_pairs.tsv"
    )

    con.execute(
        f"""
        COPY final_candidates

        TO '{candidate_output}'

        (
            FORMAT CSV,
            DELIMITER '\\t',
            HEADER
        )
        """
    )

    con.execute(
        f"""
        COPY missed_true_pairs

        TO '{missed_output}'

        (
            FORMAT CSV,
            DELIMITER '\\t',
            HEADER
        )
        """
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print(
        "FINAL RESULTS"
    )
    print("=" * 70)

    print(
        f"S1 sample             : "
        f"{s1_count:,}"
    )

    print(
        f"S2 records            : "
        f"{s2_count:,}"
    )

    print(
        f"S3 records            : "
        f"{s3_count:,}"
    )

    print(
        f"Target records        : "
        f"{target_count:,}"
    )

    print()

    print(
        f"Final candidate pairs : "
        f"{total_candidates:,}"
    )

    print(
        f"True pairs            : "
        f"{total_true_pairs:,}"
    )

    print(
        f"True pairs found      : "
        f"{found_true_pairs:,}"
    )

    print(
        f"Missed true pairs     : "
        f"{missed_true_pairs:,}"
    )

    print(
        f"Candidate recall      : "
        f"{recall:.4%}"
    )

    print()

    print(
        f"Runtime               : "
        f"{elapsed / 60:.2f} minutes"
    )

    print()

    print(
        "Candidate file:"
    )

    print(
        candidate_output
    )

    print()

    print(
        "Missed-pair file:"
    )

    print(
        missed_output
    )

    print()
    print("=" * 70)
    print(
        "BLOCKING COMPLETE"
    )
    print("=" * 70)

    con.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()