"""Build structured retrieval documents from the cleaned PilgrimAI dataset.

This is the ingestion/document-normalization stage of the pipeline.
"""

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

# Always resolve paths relative to this Python file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ZIP_PATH = RAW_DIR / "pilgrimai_cleaned.zip"
OUTPUT_JSONL = PROCESSED_DIR / "phase2_documents.jsonl"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(x: Any) -> str:
    """
    Convert different JSON value types into clean text.

    Handles:
    - None
    - strings
    - numbers
    - dictionaries
    - lists
    """

    if x is None:
        return ""

    # Handle dictionaries safely
    if isinstance(x, dict):
        # Try common text/name keys first
        for key in ["name", "value", "text", "label"]:
            if key in x and x[key] not in (None, ""):
                result = clean_text(x[key])
                if result:
                    return result

        return ""

    # Handle lists
    if isinstance(x, list):
        values = []

        for item in x:
            cleaned = clean_text(item)

            if cleaned:
                values.append(cleaned)

        return ", ".join(values)

    # Convert everything else to string
    s = str(x)

    # Remove unwanted contentReference artifacts
    s = re.sub(
        r":contentReference\[[^\]]*\]\{[^}]*\}",
        "",
        s
    )

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ============================================================
# FLATTEN JSON FOR PAGE CONTENT
# ============================================================

def flatten_for_text(x: Any, prefix=""):
    """
    Convert nested dictionaries/lists into readable text lines.
    """

    parts = []

    if x is None:
        return parts

    if isinstance(x, dict):

        for k, v in x.items():

            # Skip fields that are not useful for semantic retrieval
            if k in {
                "images",
                "wikidata",
                "knowledge_graph",
                "audio",
                "sources",
                "references"
            }:
                continue

            parts.extend(
                flatten_for_text(
                    v,
                    f"{prefix}{k}."
                )
            )

    elif isinstance(x, list):

        for v in x:
            parts.extend(
                flatten_for_text(
                    v,
                    prefix
                )
            )

    else:

        s = clean_text(x)

        if s:
            field_name = prefix[:-1] if prefix.endswith(".") else prefix

            parts.append(
                f"{field_name}: {s}"
            )

    return parts


# ============================================================
# NUMBER PARSING
# ============================================================

def parse_number(s):
    """
    Extract the first numeric value from a value.
    """

    if s is None:
        return None

    if isinstance(s, (int, float)):
        return float(s)

    m = re.search(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?",
        str(s)
    )

    if m:
        return float(
            m.group(0).replace(",", "")
        )

    return None


# ============================================================
# PRICE RANGE
# ============================================================

def parse_price_range(s):
    """
    Convert values such as:

        ₹500 - ₹1000
        500 to 1000
        Rs. 500-1000

    into:

        (500, 1000)
    """

    if not s:
        return None, None

    nums = [
        float(
            x.replace(",", "")
        )
        for x in re.findall(
            r"\d+(?:,\d{3})*(?:\.\d+)?",
            str(s)
        )
    ]

    if not nums:
        return None, None

    return min(nums), max(nums)


# ============================================================
# RATING PARSING
# ============================================================

def parse_rating(s):
    """
    Extract numeric rating.

    Example:

        "4.5/5" -> 4.5
    """

    if s is None:
        return None

    if isinstance(s, (int, float)):
        return float(s)

    m = re.search(
        r"\d+(?:\.\d+)?",
        str(s)
    )

    if m:
        return float(m.group(0))

    return None


# ============================================================
# BOOLEAN NORMALIZATION
# ============================================================

def truthy(x):
    """
    Convert common textual boolean values into True/False.
    """

    if isinstance(x, bool):
        return x

    if x is None:
        return None

    s = str(x).strip().lower()

    if s in {
        "yes",
        "true",
        "available",
        "open 24 hours",
        "24x7",
        "24/7"
    }:
        return True

    if s in {
        "no",
        "false",
        "not available"
    }:
        return False

    return None


# ============================================================
# NESTED JSON PATH
# ============================================================

def get_path(d, path):
    """
    Get a nested value using dot notation.

    Example:

        get_path(data, "location.city")
    """

    cur = d

    for p in path.split("."):

        if isinstance(cur, dict):
            cur = cur.get(p)

        else:
            return None

    return cur


# ============================================================
# FIRST AVAILABLE VALUE
# ============================================================

def first_value(d, paths):
    """
    Return the first non-empty value from multiple possible paths.
    """

    if isinstance(paths, str):
        paths = [paths]

    for p in paths:

        if p.startswith("root."):
            p = p[5:]

        v = get_path(d, p)

        if v not in (
            None,
            "",
            [],
            {}
        ):
            return v

    return None


# ============================================================
# PAGE CONTENT BUILDER
# ============================================================

def make_text(title, data, skip_keys=None):
    """
    Build the semantic-search text for a Document.

    IMPORTANT:
    Every item added to parts is explicitly converted
    into a string.
    """

    skip_keys = skip_keys or set()

    # Prevent dictionary/list title errors
    title = clean_text(title)

    if not title:
        title = "Document"

    parts = [title]

    for k, v in data.items():

        if k in skip_keys:
            continue

        if isinstance(v, (dict, list)):

            for line in flatten_for_text(
                v,
                f"{k}."
            ):

                if isinstance(line, str) and line.strip():
                    parts.append(line)

        else:

            s = clean_text(v)

            if s:
                parts.append(
                    f"{k}: {s}"
                )

    # Final safety check
    parts = [
        str(part)
        for part in parts
        if part not in (None, "")
    ]

    return "\n".join(parts)


# ============================================================
# BUILD DOCUMENTS
# ============================================================

def build_documents():

    docs = []

    if not ZIP_PATH.exists():

        raise FileNotFoundError(
            f"\nZIP file not found:\n{ZIP_PATH}\n\n"
            f"Expected structure:\n"
            f"{PROJECT_ROOT}\\src\\ingestion\\loader.py\n"
            f"{ZIP_PATH}"
        )

    print("=" * 60)
    print("PHASE 2 DOCUMENT BUILDER")
    print("=" * 60)

    print(f"ZIP file : {ZIP_PATH}")
    print(f"Output   : {OUTPUT_JSONL}")
    print()

    with zipfile.ZipFile(ZIP_PATH) as z:

        filenames = z.namelist()

        print(
            f"Files inside ZIP: {len(filenames)}"
        )
        print()

        for fname in filenames:

            # Only process JSON files
            if not fname.endswith(".json"):
                continue

            if fname.endswith("/"):
                continue

            # Read JSON
            raw = z.read(fname)

            data = json.loads(
                raw.decode("utf-8")
            )

            # ------------------------------------------------
            # TEMPLE REGISTRY
            # ------------------------------------------------

            # Registry is a lookup resource,
            # not a vector document.
            if fname.endswith(
                "temple_registry.json"
            ):
                continue

            # ------------------------------------------------
            # TEMPLES
            # ------------------------------------------------

            if fname.startswith(
                "data/temples/"
            ):

                d = data

                meta = {

                    "source_type": "temple",

                    "temple_id":
                        clean_text(
                            d.get("temple_id")
                        ),

                    "name":
                        clean_text(
                            d.get("name")
                        ),

                    "location":
                        clean_text(
                            get_path(
                                d,
                                "location.address"
                            )
                            or
                            get_path(
                                d,
                                "location.city"
                            )
                            or
                            get_path(
                                d,
                                "location.district"
                            )
                        ),

                    "religion":
                        clean_text(
                            d.get("religion")
                        ),

                    "temple_type":
                        clean_text(
                            d.get("temple_type")
                        ),

                    "category":
                        clean_text(
                            d.get("category")
                        ),

                    "deity":
                        clean_text(
                            get_path(
                                d,
                                "deity.presiding_deity"
                            )
                            or
                            d.get("deity")
                        ),

                    "latitude":
                        get_path(
                            d,
                            "location.latitude"
                        ),

                    "longitude":
                        get_path(
                            d,
                            "location.longitude"
                        )
                }

                text = make_text(
                    d.get(
                        "name",
                        "Temple"
                    ),
                    d,
                    {
                        "images",
                        "wikidata",
                        "knowledge_graph",
                        "audio",
                        "sources"
                    }
                )

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            k: v
                            for k, v in meta.items()
                            if v not in (
                                None,
                                ""
                            )
                        }
                    )
                )

            # ------------------------------------------------
            # ACCOMMODATION
            # ------------------------------------------------

            elif fname.startswith(
                "data/accommodations/"
            ):

                hotels = data.get(
                    "hotels"
                ) or [{}]

                hotel = hotels[0]

                hotel_id = (
                    hotel.get("hotel_id")
                    or
                    data.get("hotel_id")
                )

                temple_id = (
                    hotel.get("temple_id")
                    or
                    data.get("temple_id")
                )

                pmin, pmax = parse_price_range(
                    hotel.get("price_range")
                )

                rating = parse_rating(
                    get_path(
                        hotel,
                        "ratings.rating"
                    )
                )

                meta = {

                    "source_type":
                        "accommodation",

                    "hotel_id":
                        clean_text(
                            hotel_id
                        ),

                    "temple_id":
                        clean_text(
                            temple_id
                        ),

                    "location":
                        clean_text(
                            hotel.get("address")
                        ),

                    "hotel_type":
                        clean_text(
                            hotel.get("hotel_type")
                        ),

                    "price_min":
                        pmin,

                    "price_max":
                        pmax,

                    "rating":
                        rating,

                    "amenities":
                        json.dumps(
                            hotel.get(
                                "amenities",
                                []
                            ),
                            ensure_ascii=False
                        ),

                    "room_types":
                        json.dumps(
                            hotel.get(
                                "room_types",
                                []
                            ),
                            ensure_ascii=False
                        ),

                    "suitable_for":
                        json.dumps(
                            hotel.get(
                                "suitable_for",
                                []
                            ),
                            ensure_ascii=False
                        ),

                    "food_facility":
                        json.dumps(
                            hotel.get(
                                "food_facility",
                                {}
                            ),
                            ensure_ascii=False
                        ),

                    "payment_modes":
                        json.dumps(
                            hotel.get(
                                "payment_modes",
                                []
                            ),
                            ensure_ascii=False
                        )
                }

                text = make_text(
                    hotel.get(
                        "hotel_name",
                        "Accommodation"
                    ),
                    hotel,
                    {
                        "google_map_link",
                        "official_website",
                        "booking_links"
                    }
                )

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            k: v
                            for k, v in meta.items()
                            if v not in (
                                None,
                                ""
                            )
                        }
                    )
                )

            # ------------------------------------------------
            # RESTAURANTS
            # ------------------------------------------------

            elif fname.startswith(
                "data/restaurants/"
            ):

                d = data

                dist = first_value(
                    d,
                    [
                        "travelInformation.distanceFromTempleKm.value",
                        "distanceFromTemple"
                    ]
                )

                price1 = first_value(
                    d,
                    [
                        "priceInformation.averageCostForOne"
                    ]
                )

                price2 = first_value(
                    d,
                    [
                        "priceInformation.averageCostForTwo"
                    ]
                )

                pmin = parse_number(
                    price1
                )

                pmax = parse_number(
                    price2
                )

                if pmin is None and pmax is not None:
                    pmin = pmax

                if pmax is None and pmin is not None:
                    pmax = pmin

                rating = first_value(
                    d,
                    [
                        "customerInformation.rating",
                        "rating"
                    ]
                )

                # Safely extract restaurant name
                restaurant_name = clean_text(
                    d.get("restaurantName")
                )

                if not restaurant_name:
                    restaurant_name = "Restaurant"

                meta = {

                    "source_type":
                        "restaurant",

                    "restaurant_id":
                        clean_text(
                            d.get("restaurantId")
                        ),

                    "temple_id":
                        clean_text(
                            get_path(
                                d,
                                "associatedTemple.templeId"
                            )
                        ),

                    "location":
                        clean_text(
                            get_path(
                                d,
                                "location.fullAddress"
                            )
                            or
                            get_path(
                                d,
                                "location.city"
                            )
                        ),

                    "cuisine":
                        json.dumps(
                            first_value(
                                d,
                                [
                                    "basicInformation.cuisineType",
                                    "cuisine"
                                ]
                            )
                            or [],
                            ensure_ascii=False
                        ),

                    "dietary_options":
                        json.dumps(
                            d.get(
                                "dietaryInformation"
                            )
                            or
                            d.get(
                                "foodType"
                            )
                            or {},
                            ensure_ascii=False
                        ),

                    "price_min":
                        pmin,

                    "price_max":
                        pmax,

                    "distance_km":
                        parse_number(
                            dist
                        ),

                    "facilities":
                        json.dumps(
                            d.get(
                                "facilities"
                            )
                            or {},
                            ensure_ascii=False
                        ),

                    "opening_time":
                        clean_text(
                            get_path(
                                d,
                                "workingHours.openingTime"
                            )
                        ),

                    "closing_time":
                        clean_text(
                            get_path(
                                d,
                                "workingHours.closingTime"
                            )
                        ),

                    "rating":
                        parse_rating(
                            rating
                        )
                }

                text = make_text(
                    restaurant_name,
                    d,
                    {
                        "googleMapsLink",
                        "references"
                    }
                )

                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            k: v
                            for k, v in meta.items()
                            if v not in (
                                None,
                                ""
                            )
                        }
                    )
                )

            # ------------------------------------------------
            # EMERGENCY
            # ------------------------------------------------

            elif fname.startswith(
                "data/emergency/"
            ):

                temple_id = data.get(
                    "temple_id"
                )

                emergency = data.get(
                    "emergency",
                    {}
                )

                if not isinstance(
                    emergency,
                    dict
                ):
                    continue

                for facility_type, items in emergency.items():

                    if not isinstance(
                        items,
                        list
                    ):
                        continue

                    for idx, item in enumerate(items):

                        if not isinstance(
                            item,
                            dict
                        ):
                            continue

                        meaningful = any(
                            v not in (
                                None,
                                "",
                                [],
                                {}
                            )
                            for k, v in item.items()
                            if k not in {
                                "source_url",
                                "last_verified_at"
                            }
                        )

                        if not meaningful:
                            continue

                        fid = (
                            f"{temple_id}_"
                            f"{facility_type}_"
                            f"{idx + 1}"
                        )

                        distance = first_value(
                            item,
                            [
                                "distance_from_temple_km",
                                "distance_from_temple"
                            ]
                        )

                        open24 = first_value(
                            item,
                            [
                                "24x7_emergency",
                                "availability",
                                "opening_hours"
                            ]
                        )

                        facility_name = clean_text(
                            item.get("name")
                            or
                            item.get("service_name")
                            or
                            item.get("police_station_name")
                            or
                            item.get("pharmacy_name")
                            or
                            item.get("fire_station_name")
                        )

                        meta = {

                            "source_type":
                                "emergency",

                            "facility_id":
                                fid,

                            "temple_id":
                                clean_text(
                                    temple_id
                                ),

                            "facility_type":
                                clean_text(
                                    facility_type
                                ),

                            "facility_name":
                                facility_name,

                            "location":
                                clean_text(
                                    item.get(
                                        "address"
                                    )
                                ),

                            "distance_km":
                                parse_number(
                                    distance
                                ),

                            "emergency_available":
                                truthy(
                                    item.get(
                                        "emergency_department"
                                    )
                                ),

                            "ambulance_available":
                                truthy(
                                    item.get(
                                        "ambulance_available"
                                    )
                                ),

                            "trauma_care":
                                truthy(
                                    item.get(
                                        "trauma_care"
                                    )
                                ),

                            "icu_available":
                                truthy(
                                    item.get(
                                        "icu_available"
                                    )
                                ),

                            "pharmacy_available":
                                truthy(
                                    item.get(
                                        "pharmacy_available"
                                    )
                                ),

                            "open_24x7":
                                truthy(
                                    open24
                                ),

                            "verification_status":
                                clean_text(
                                    item.get(
                                        "verification_status"
                                    )
                                )
                        }

                        title = (
                            facility_name
                            or
                            clean_text(
                                facility_type
                            )
                            or
                            "Emergency Facility"
                        )

                        text = make_text(
                            title,
                            item,
                            {
                                "google_maps_url",
                                "source_url"
                            }
                        )

                        docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    k: v
                                    for k, v in meta.items()
                                    if v not in (
                                        None,
                                        ""
                                    )
                                }
                            )
                        )

    return docs


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    docs = build_documents()

    # --------------------------------------------------------
    # SERIALIZE DOCUMENTS
    # --------------------------------------------------------

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8"
    ) as f:

        for d in docs:

            f.write(
                json.dumps(
                    {
                        "page_content":
                            d.page_content,

                        "metadata":
                            d.metadata
                    },
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 2 COMPLETED")
    print("=" * 60)

    print(
        f"Created {len(docs)} LangChain Documents."
    )

    print(
        f"Serialized to:\n{OUTPUT_JSONL}"
    )

    print("=" * 60)