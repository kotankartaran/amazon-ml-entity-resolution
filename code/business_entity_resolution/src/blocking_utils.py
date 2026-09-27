# ============================================================
# AMAZON ML ENTITY RESOLUTION
# BLOCKING UTILITIES
# ============================================================

import re

from unidecode import unidecode


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


# ============================================================
# ADDRESS REPLACEMENTS
# ============================================================

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


# ============================================================
# DOMAIN TLDs
# ============================================================

TLDS = [
    ".com",
    ".org",
    ".net",
    ".co.in",
    ".in",
    ".io",
    ".us",
    ".info",
    ".biz",
    ".edu",
    ".gov",
    ".co",
    ".uk",
    ".ca",
    ".de",
]


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


COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "united states": "us", "united states of america": "us", "u s": "us", "u s a": "us",
    "fr": "fr", "fra": "fr", "france": "fr", "republique francaise": "fr", "frankreich": "fr",
    "in": "in", "ind": "in", "india": "in", "bharat": "in",
    "de": "de", "deu": "de", "germany": "de", "deutschland": "de",
    "gb": "gb", "gbr": "gb", "uk": "gb", "united kingdom": "gb", "great britain": "gb", "england": "gb",
    "ca": "ca", "can": "ca", "canada": "ca",
    "au": "au", "aus": "au", "australia": "au",
    "jp": "jp", "jpn": "jp", "japan": "jp", "nippon": "jp",
    "cn": "cn", "chn": "cn", "china": "cn",
    "br": "br", "bra": "br", "brazil": "br", "brasil": "br",
    "it": "it", "ita": "it", "italy": "it", "italia": "it",
    "es": "es", "esp": "es", "spain": "es", "espana": "es",
    "nl": "nl", "nld": "nl", "netherlands": "nl", "nederland": "nl",
    "mx": "mx", "mex": "mx", "mexico": "mx",
    "sg": "sg", "sgp": "sg", "singapore": "sg",
    "ae": "ae", "are": "ae", "uae": "ae", "united arab emirates": "ae",
    "ch": "ch", "che": "ch", "switzerland": "ch", "schweiz": "ch",
    "se": "se", "swe": "se", "sweden": "se",
    "no": "no", "nor": "no", "norway": "no",
    "fi": "fi", "fin": "fi", "finland": "fi",
    "dk": "dk", "dnk": "dk", "denmark": "dk",
    "pl": "pl", "pol": "pl", "poland": "pl", "polska": "pl",
    "be": "be", "bel": "be", "belgium": "be",
    "at": "at", "aut": "at", "austria": "at", "osterreich": "at",
    "nz": "nz", "nzl": "nz", "new zealand": "nz",
    "ie": "ie", "irl": "ie", "ireland": "ie",
}

def canonicalize_country(value):
    if value is None:
        return ""
    val = str(value).strip().casefold()
    val = unidecode(val)
    val = re.sub(r"\s+", " ", val).strip()
    return COUNTRY_ALIASES.get(val, val)

def normalize_country(value):
    return canonicalize_country(value)


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

    # Generic international postal code.
    # Keep this conservative to avoid extracting ordinary
    # house numbers as postal codes.
    matches = re.findall(
        r"\b[A-Za-z0-9][A-Za-z0-9 -]{2,9}\b",
        address
    )

    for value in reversed(matches):

        value = value.strip()

        if any(ch.isdigit() for ch in value):

            digits = sum(
                ch.isdigit()
                for ch in value
            )

            if digits >= 3:
                return value

    return ""


# ============================================================
# NAME TOKENS
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


# ============================================================
# BASIC NAME SIGNATURE
# ============================================================

def make_name_signature(token):

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
# HOUSE + ADDRESS TOKEN
# ============================================================

def create_house_token_key(address, country):

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


# ============================================================
# HOUSE + TWO ADDRESS TOKENS
# ============================================================

def create_house_two_token_key(address, country):

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


# ============================================================
# POSTAL KEY
# ============================================================

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
# SIGNATURE TOKEN
# ============================================================

def normalize_signature_token(token):

    if not token:
        return ""

    token = normalize_transliterated_name(
        token
    )

    if len(token) < 4:
        return ""

    return token


def token_signature(token):

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
# NAME + ADDRESS KEYS
# ============================================================

def create_name_address_keys(
    name,
    address,
    country
):

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

    return sorted(
        set(keys)
    )


# ============================================================
# HOUSE + NAME KEYS
# ============================================================

def create_house_name_key(
    name,
    address,
    country
):

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
# NAME PAIR KEYS
# ============================================================

def create_name_pair_keys(
    name,
    country
):

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
    )[:5]

    keys = []

    for i in range(
        len(tokens)
    ):

        for j in range(
            i + 1,
            len(tokens)
        ):

            keys.append(
                f"{country}|pair|"
                f"{tokens[i]}|"
                f"{tokens[j]}"
            )

    return keys


# ============================================================
# ADDRESS PAIR KEYS
# ============================================================

def create_address_pair_keys(
    address,
    country
):

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

    tokens = sorted(
        set(tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[:4]

    keys = []

    if len(tokens) >= 2:

        keys.append(
            f"{country}|addrpair|"
            f"{tokens[0]}|{tokens[1]}"
        )

    if len(tokens) >= 3:

        keys.append(
            f"{country}|addrpair|"
            f"{tokens[0]}|{tokens[2]}"
        )

    if len(tokens) >= 4:

        keys.append(
            f"{country}|addrpair|"
            f"{tokens[0]}|{tokens[3]}"
        )

    return sorted(
        set(keys)
    )


# ============================================================
# ADDRESS TRIPLE KEYS
# ============================================================

def create_address_triple_keys(
    address,
    country
):

    country = normalize_country(
        country
    )

    if not country:
        return []

    tokens = extract_strong_address_tokens(
        address
    )

    if len(tokens) < 3:
        return []

    tokens = sorted(
        set(tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[:5]

    keys = []

    # Strongest three
    if len(tokens) >= 3:

        keys.append(
            f"{country}|addrtriple|"
            f"{tokens[0]}|"
            f"{tokens[1]}|"
            f"{tokens[2]}"
        )

    # Strongest + second + fourth
    if len(tokens) >= 4:

        keys.append(
            f"{country}|addrtriple|"
            f"{tokens[0]}|"
            f"{tokens[1]}|"
            f"{tokens[3]}"
        )

    # Strongest + second + fifth
    if len(tokens) >= 5:

        keys.append(
            f"{country}|addrtriple|"
            f"{tokens[0]}|"
            f"{tokens[1]}|"
            f"{tokens[4]}"
        )

    return sorted(
        set(keys)
    )


# ============================================================
# TOP-8 ADDRESS PAIR KEYS
# ============================================================

def create_top8_address_pair_keys(
    address,
    country
):

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

    # Keep at most 8 informative tokens.
    tokens = sorted(
        set(tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[:8]

    keys = []

    # Generate selected pairs rather than
    # every possible pair.
    for i in range(
        len(tokens)
    ):

        for j in range(
            i + 1,
            len(tokens)
        ):

            keys.append(
                f"{country}|top8|"
                f"{tokens[i]}|"
                f"{tokens[j]}"
            )

    return sorted(
        set(keys)
    )


# ============================================================
# NUMBER + STRONG ADDRESS TOKEN
# ============================================================

def create_number_token_keys(
    address,
    country
):

    country = normalize_country(
        country
    )

    if not country:
        return []

    numbers = extract_address_numbers(
        address
    )

    if not numbers:
        return []

    tokens = extract_strong_address_tokens(
        address
    )

    if not tokens:
        return []

    tokens = sorted(
        set(tokens),
        key=lambda x: (
            -len(x),
            x
        )
    )[:5]

    keys = []

    for number in numbers[:2]:

        for token in tokens:

            keys.append(
                f"{country}|numtoken|"
                f"{number}|{token}"
            )

    return sorted(
        set(keys)
    )


# ============================================================
# DOMAIN ROOT
# ============================================================

def create_domain_root_keys(
    name,
    country
):

    country = normalize_country(
        country
    )

    if not name or not country:
        return []

    raw = str(
        name
    ).strip().casefold()

    raw = unidecode(
        raw
    )

    cleaned = raw

    # Remove TLDs.
    for tld in TLDS:

        if tld in cleaned:
            cleaned = cleaned.replace(
                tld,
                ""
            )

    cleaned = re.sub(
        r"[^a-z0-9\s]",
        " ",
        cleaned
    )

    tokens = [
        token
        for token in cleaned.split()
        if token not in LEGAL_SUFFIXES
    ]

    if not tokens:
        return []

    concat_domain = "".join(
        tokens
    )

    if len(concat_domain) < 5:
        return []

    return [
        f"{country}|domainroot|{concat_domain}"
    ]


# ============================================================
# DUCKDB FUNCTION REGISTRATION
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
        "create_name_signature",
        create_name_signature,
        ["VARCHAR", "VARCHAR"],
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
        "create_name_address_keys",
        create_name_address_keys,
        ["VARCHAR", "VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )

    con.create_function(
        "create_house_name_key",
        create_house_name_key,
        ["VARCHAR", "VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )

    con.create_function(
        "create_name_pair_keys",
        create_name_pair_keys,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )

    con.create_function(
        "create_address_pair_keys",
        create_address_pair_keys,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )

    con.create_function(
        "create_address_triple_keys",
        create_address_triple_keys,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )

    con.create_function(
        "create_top8_address_pair_keys",
        create_top8_address_pair_keys,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )

    con.create_function(
        "create_number_token_keys",
        create_number_token_keys,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )

    con.create_function(
        "create_domain_root_keys",
        create_domain_root_keys,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR[]"
    )


# ============================================================
# END
# ============================================================