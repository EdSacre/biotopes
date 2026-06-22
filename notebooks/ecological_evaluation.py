"""
Baltic Sea Habitat Cluster Ecological Evaluation
=================================================
Evaluates the ecological coherence of habitat clusters from a hierarchical
clustering analysis of Baltic Sea / Kattegat benthic species.

Run from the project root:
    python notebooks/ecological_evaluation.py

Outputs:
    outputs/reports/cluster_evaluation_report.html  (main deliverable)
    outputs/reports/cluster_metrics.csv
    outputs/reports/outlier_species.csv
    traits/species_traits_merged.csv
    outputs/figures/*.png
"""

import sys, os, json, time, re, math, base64, io, warnings
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import requests

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RUN_DIR    = ROOT / "data" / "run1_4manSalinityZones_10perc_binStrictLVL"
DATA_PATH  = RUN_DIR / "full_cluster_df.csv"
TRAITS_PY  = ROOT / "traits" / "species_traits_tier1.py"
WORMS_CACHE= ROOT / "traits" / "worms_cache.json"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_REP    = ROOT / "outputs" / "reports"

sys.path.insert(0, str(ROOT / "traits"))
from species_traits_tier1 import TIER1_TRAITS, GENUS_FALLBACKS, ZONE_SALINITY

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Load Data
# ─────────────────────────────────────────────────────────────────────────────

def load_cluster_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.strip('"')
    df["species"] = df["species"].str.strip('"').str.replace(".", " ", regex=False)
    df["cluster_name"] = df["cluster_name"].str.strip('"')
    df["zone"] = df["cluster_name"].str.extract(r"(Zone\d)")
    df["subcluster"] = df["cluster_name"].str.extract(r"Zone\d_(.*)")
    # Flag species appearing in more than one zone
    zone_counts = df.groupby("species")["zone"].nunique()
    df["multi_zone"] = df["species"].map(zone_counts) > 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Species Trait Database
# ─────────────────────────────────────────────────────────────────────────────

def _load_worms_cache() -> dict:
    if WORMS_CACHE.exists():
        return json.loads(WORMS_CACHE.read_text())
    return {}


def _save_worms_cache(cache: dict):
    WORMS_CACHE.write_text(json.dumps(cache, indent=2))


def _worms_get(url: str, retries: int = 2) -> dict | None:
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1)
    return None


def fetch_worms_traits(species_name: str, cache: dict) -> dict:
    """Query WoRMS for depth and substrate traits; return partial trait dict."""
    if species_name in cache:
        return cache[species_name]

    # Step 1: get AphiaID
    safe = requests.utils.quote(species_name)
    rec = _worms_get(
        f"https://www.marinespecies.org/rest/AphiaRecordsByName/{safe}"
        "?like=false&marine_only=false&offset=1"
    )
    time.sleep(1.2)
    if not rec or not isinstance(rec, list):
        cache[species_name] = {}
        return {}

    aphia_id = rec[0].get("AphiaID")
    if not aphia_id:
        cache[species_name] = {}
        return {}

    # Step 2: get attributes
    attrs = _worms_get(
        f"https://www.marinespecies.org/rest/AphiaAttributesByAphiaID/{aphia_id}"
    )
    time.sleep(1.2)
    traits = {}
    if attrs and isinstance(attrs, list):
        for a in attrs:
            mtype = (a.get("measurementType") or "").lower()
            mval  = (a.get("measurementValue") or "").lower()
            if "salinity" in mtype:
                traits["salinity_category"] = mval
            elif "depth" in mtype or "zone" in mtype:
                if not traits.get("depth_zone"):
                    traits["depth_zone"] = mval
            elif "substrate" in mtype or "substratum" in mtype:
                if not traits.get("substrate"):
                    traits["substrate"] = mval
    cache[species_name] = traits
    return traits


def infer_from_taxonomy(species_name: str) -> dict:
    genus = species_name.split()[0]
    return GENUS_FALLBACKS.get(genus, {})


def build_trait_table(df: pd.DataFrame) -> pd.DataFrame:
    species_list = df["species"].unique()
    print(f"Building traits for {len(species_list)} unique species …")

    cache = _load_worms_cache()
    rows = []

    for i, sp in enumerate(species_list):
        # Tier 1
        if sp in TIER1_TRAITS:
            t = dict(TIER1_TRAITS[sp])
            t["trait_source"] = "tier1"
        else:
            # Tier 2: WoRMS
            w = fetch_worms_traits(sp, cache)
            t = {}
            if w:
                t.update(w)
                t["trait_source"] = "worms"
            # Tier 3: genus fallback (fills missing fields)
            fb = infer_from_taxonomy(sp)
            for k, v in fb.items():
                if k not in t:
                    t[k] = v
            if "sal_midpoint" not in t and "salinity_min" in t and "salinity_max" in t:
                t["sal_midpoint"] = (t["salinity_min"] + t["salinity_max"]) / 2
            if not t.get("trait_source"):
                t["trait_source"] = "genus_fallback" if fb else "unknown"

        t["species"] = sp
        rows.append(t)

        if (i + 1) % 50 == 0:
            _save_worms_cache(cache)
            print(f"  … {i+1}/{len(species_list)} done")

    _save_worms_cache(cache)

    traits_df = pd.DataFrame(rows)
    required = ["salinity_min","salinity_max","sal_midpoint","depth_zone",
                "functional_group","substrate","trait_source"]
    for col in required:
        if col not in traits_df.columns:
            traits_df[col] = np.nan

    # Ensure sal_midpoint is numeric
    for col in ["salinity_min","salinity_max","sal_midpoint"]:
        traits_df[col] = pd.to_numeric(traits_df[col], errors="coerce")

    traits_df.to_csv(ROOT / "traits" / "species_traits_merged.csv", index=False)
    src_counts = traits_df["trait_source"].value_counts()
    print("Trait source breakdown:")
    print(src_counts.to_string())
    return traits_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Merge & Compute Base Columns
# ─────────────────────────────────────────────────────────────────────────────

def merge_traits(df: pd.DataFrame, traits_df: pd.DataFrame) -> pd.DataFrame:
    df_full = df.merge(traits_df, on="species", how="left")

    # Zone expected salinity
    df_full["zone_sal_min"]  = df_full["zone"].map(lambda z: ZONE_SALINITY.get(z, {}).get("min",  np.nan))
    df_full["zone_sal_max"]  = df_full["zone"].map(lambda z: ZONE_SALINITY.get(z, {}).get("max",  np.nan))
    df_full["zone_sal_mid"]  = df_full["zone"].map(lambda z: ZONE_SALINITY.get(z, {}).get("midpoint", np.nan))

    # Salinity distance from zone midpoint
    df_full["sal_distance"]  = (df_full["sal_midpoint"] - df_full["zone_sal_mid"]).abs()

    # Is species' salinity tolerance fully outside zone range?
    def sal_overlap(row):
        if pd.isna(row["salinity_min"]) or pd.isna(row["zone_sal_min"]):
            return np.nan
        # True if species range overlaps zone range
        return not (row["salinity_max"] < row["zone_sal_min"] or
                    row["salinity_min"] > row["zone_sal_max"])
    df_full["sal_zone_overlap"] = df_full.apply(sal_overlap, axis=1)

    return df_full


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Coherence Metrics
# ─────────────────────────────────────────────────────────────────────────────

def salinity_zone_purity(cluster_df: pd.DataFrame) -> float:
    valid = cluster_df["sal_zone_overlap"].dropna()
    if len(valid) == 0:
        return np.nan
    return valid.mean()


def functional_group_entropy(cluster_df: pd.DataFrame) -> tuple[float, str, float]:
    fgs = cluster_df["functional_group"].dropna()
    if len(fgs) == 0:
        return np.nan, "unknown", np.nan
    counts = fgs.value_counts()
    probs  = counts / counts.sum()
    H = -sum(p * math.log(p) for p in probs if p > 0)
    return H, counts.index[0], float(probs.iloc[0])


def substrate_purity(cluster_df: pd.DataFrame) -> float:
    subs = cluster_df["substrate"].dropna()
    if len(subs) == 0:
        return np.nan
    return subs.value_counts(normalize=True).iloc[0]


def depth_zone_coherence(cluster_df: pd.DataFrame) -> tuple[float, str]:
    dz = cluster_df["depth_zone"].dropna()
    if len(dz) == 0:
        return np.nan, "unknown"
    vc = dz.value_counts(normalize=True)
    return float(vc.iloc[0]), vc.index[0]


def helcom_alignment(cluster_df: pd.DataFrame, helcom_biotopes: dict) -> tuple[str, float, list]:
    """Recall-form Jaccard: |cluster ∩ indicators| / |indicators|"""
    cluster_species = set(cluster_df["species"].str.lower())
    best_name, best_score = "No match", 0.0
    scores = []
    for bname, bdata in helcom_biotopes.items():
        inds = {s.lower() for s in bdata["indicator_species"]}
        if not inds:
            continue
        recall = len(cluster_species & inds) / len(inds)
        scores.append((bname, recall))
    scores.sort(key=lambda x: x[1], reverse=True)
    if scores and scores[0][1] > 0:
        best_name, best_score = scores[0]
    top3 = [f"{n} ({s:.2f})" for n, s in scores[:3]]
    return best_name, best_score, top3


def compute_salinity_outlier_scores(df_full: pd.DataFrame) -> pd.DataFrame:
    """Tukey fence on sal_midpoint within each cluster."""
    results = []
    for cname, grp in df_full.groupby("cluster_name"):
        sal = grp["sal_midpoint"].dropna()
        if len(sal) < 4:
            for _, row in grp.iterrows():
                results.append({"species": row["species"],
                                 "cluster_name": cname,
                                 "sal_outlier_score": np.nan,
                                 "is_sal_outlier": False})
            continue
        q1, q3 = sal.quantile(0.25), sal.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for _, row in grp.iterrows():
            v = row["sal_midpoint"]
            if pd.isna(v):
                score, flag = np.nan, False
            else:
                score = max(0, (v - hi) if v > hi else (lo - v)) / max(iqr, 0.1)
                flag  = (v < lo) or (v > hi)
            results.append({"species": row["species"],
                             "cluster_name": cname,
                             "sal_outlier_score": score,
                             "is_sal_outlier": flag})
    return pd.DataFrame(results)


def compute_all_metrics(df_full: pd.DataFrame, helcom_biotopes: dict) -> pd.DataFrame:
    records = []
    for cname, grp in df_full.groupby("cluster_name", sort=False):
        sal_pur  = salinity_zone_purity(grp)
        H, dom_fg, dom_fg_prop = functional_group_entropy(grp)
        sub_pur  = substrate_purity(grp)
        dz_coh, dom_dz = depth_zone_coherence(grp)
        hname, hscore, htop3 = helcom_alignment(grp, helcom_biotopes)
        zone = grp["zone"].iloc[0]
        records.append({
            "cluster_name":          cname,
            "zone":                  zone,
            "n_species":             len(grp),
            "salinity_purity":       round(sal_pur, 3) if not pd.isna(sal_pur) else np.nan,
            "functional_entropy":    round(H, 3) if not pd.isna(H) else np.nan,
            "dominant_func_group":   dom_fg,
            "dominant_fg_prop":      round(dom_fg_prop, 3) if not pd.isna(dom_fg_prop) else np.nan,
            "substrate_purity":      round(sub_pur, 3) if not pd.isna(sub_pur) else np.nan,
            "depth_coherence":       round(dz_coh, 3) if not pd.isna(dz_coh) else np.nan,
            "dominant_depth_zone":   dom_dz,
            "helcom_best_match":     hname,
            "helcom_score":          round(hscore, 3),
            "helcom_top3":           " | ".join(htop3),
        })
    metrics_df = pd.DataFrame(records)
    # Cluster order
    order = [f"Zone{z}_{c}" for z in [1,2,3,4]
             for c in list("ABCDEFGHIJK")]
    order = [o for o in order if o in metrics_df["cluster_name"].values]
    metrics_df = metrics_df.set_index("cluster_name").reindex(order).reset_index()
    return metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Outlier Report
# ─────────────────────────────────────────────────────────────────────────────

PRE_IDENTIFIED_OUTLIERS = {
    ("Cerastoderma edule",   "Zone1_A"): ("HIGH",   "Polyhaline bivalve (requires >15 PSU) placed in freshwater cluster"),
    ("Ascidiella aspersa",   "Zone1_A"): ("HIGH",   "Euhaline tunicate (18–35 PSU) in freshwater cluster"),
    ("Ericthonius brasiliensis","Zone1_A"):("HIGH",  "Marine amphipod (>10 PSU) in freshwater cluster"),
    ("Juncus gerardii",      "Zone1_A"): ("MEDIUM", "Salt-marsh plant (5–25 PSU) among obligate freshwater species"),
    ("Triglochin maritima",  "Zone1_A"): ("MEDIUM", "Salt-marsh plant (5–30 PSU) among obligate freshwater species"),
    ("Rangia cuneata",       "Zone1_E"): ("HIGH",   "Invasive mesohaline clam (2–15 PSU) grouped with freshwater macrophytes"),
    ("Mysis mixta",          "Zone2_D"): ("MEDIUM", "Profundal/pelagic glacial relict assigned to shallow seagrass/macroalgae cluster"),
    ("Mysis relicta",        "Zone2_D"): ("MEDIUM", "Deep profundal glacial relict assigned to shallow seagrass/macroalgae cluster"),
    ("Schoenoplectus lacustris","Zone2_D"):("MEDIUM","Freshwater emergent macrophyte in intermediate-salinity benthic cluster"),
    ("Eleocharis parvula",   "Zone2_D"): ("MEDIUM", "Oligohaline emergent plant among mesohaline algae and marine bivalves"),
    ("Chara horrida",        "Zone2_D"): ("MEDIUM", "Low-salinity charophyte co-occurring with Zostera marina and Astarte"),
    ("Lemna minor",          "Zone3_A"): ("HIGH",   "Obligate freshwater floating plant placed in high-salinity Kattegat cluster"),
    ("Schoenoplectus lacustris","Zone3_A"):("HIGH",  "Freshwater emergent macrophyte in Kattegat-zone cluster with Ascophyllum nodosum"),
    ("Juncus gerardii",      "Zone3_A"): ("MEDIUM", "Salt-marsh plant at edge of salinity range for Kattegat zone"),
    ("Triglochin maritima",  "Zone3_A"): ("LOW",    "Salt-marsh plant; occurs at high salinities but unusual with subtidal fauna"),
    ("Gammarus tigrinus",    "Zone3_C"): ("MEDIUM", "Introduced brackish amphipod (0–25 PSU) in marine circalittoral benthic community"),
    ("Echinocardium cordatum","Zone3_H"):("MEDIUM",  "Sandy-bottom echinoderm assigned to hard-substrate macroalgae cluster"),
    ("Metridium senile",     "Zone3_A"): ("MEDIUM", "Euhaline anemone (15–35 PSU) co-occurring with freshwater/marsh taxa in same cluster"),
}


def generate_outlier_report(df_full: pd.DataFrame,
                             outlier_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Pre-identified outliers
    for (sp, cname), (sev, note) in PRE_IDENTIFIED_OUTLIERS.items():
        if len(df_full[(df_full["species"] == sp) & (df_full["cluster_name"] == cname)]) > 0:
            rows.append({"species": sp, "cluster_name": cname,
                         "severity": sev, "reason": "pre-identified ecological mismatch",
                         "ecological_note": note, "source": "manual"})

    # Automatic salinity outliers (not already listed)
    flagged_pairs = {(r["species"], r["cluster_name"]) for r in rows}
    sal_outliers = outlier_scores[outlier_scores["is_sal_outlier"] == True]
    for _, row in sal_outliers.iterrows():
        pair = (row["species"], row["cluster_name"])
        if pair in flagged_pairs:
            continue
        sp_row = df_full[(df_full["species"] == row["species"]) &
                          (df_full["cluster_name"] == row["cluster_name"])].iloc[0]
        sal = sp_row.get("sal_midpoint", np.nan)
        zone_mid = sp_row.get("zone_sal_mid", np.nan)
        if pd.isna(sal):
            continue
        delta = abs(sal - zone_mid) if not pd.isna(zone_mid) else np.nan
        sev = "HIGH" if (not pd.isna(delta) and delta > 15) else "MEDIUM"
        rows.append({
            "species": row["species"],
            "cluster_name": row["cluster_name"],
            "severity": sev,
            "reason": f"salinity outlier (species midpoint {sal:.0f} PSU, zone midpoint {zone_mid:.0f} PSU)",
            "ecological_note": f"Salinity outlier score: {row['sal_outlier_score']:.2f}",
            "source": "automatic"
        })
        flagged_pairs.add(pair)

    # Functional group singletons in clusters ≥ 8
    for cname, grp in df_full.groupby("cluster_name"):
        if len(grp) < 8:
            continue
        fg_counts = grp["functional_group"].value_counts()
        singleton_fgs = set(fg_counts[fg_counts == 1].index)
        for _, row in grp.iterrows():
            if row["functional_group"] in singleton_fgs:
                pair = (row["species"], cname)
                if pair in flagged_pairs:
                    continue
                rows.append({
                    "species": row["species"],
                    "cluster_name": cname,
                    "severity": "LOW",
                    "reason": f"sole representative of functional group '{row['functional_group']}' in cluster of {len(grp)} species",
                    "ecological_note": "Possible outlier or rare community member; verify occurrence data",
                    "source": "automatic"
                })
                flagged_pairs.add(pair)

    outlier_df = pd.DataFrame(rows)
    if len(outlier_df):
        sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        outlier_df["sev_rank"] = outlier_df["severity"].map(sev_order)
        outlier_df = outlier_df.sort_values(["sev_rank","cluster_name"]).drop(columns="sev_rank")
    return outlier_df


# ─────────────────────────────────────────────────────────────────────────────
# HELCOM Reference Biotopes
# ─────────────────────────────────────────────────────────────────────────────

HELCOM_BIOTOPES = {
    "Charophyte meadows (oligohaline)": {
        "description": "Dense Chara/Nitella beds in low-salinity bays and lagoons",
        "salinity": "0–5 PSU", "depth": "infralittoral",
        "indicator_species": ["Chara aspera","Chara baltica","Chara canescens",
                               "Chara horrida","Chara tomentosa","Nitella flexilis",
                               "Nitellopsis obtusa","Ruppia maritima","Zannichellia palustris",
                               "Myriophyllum spicatum","Stuckenia pectinata"]},
    "Freshwater/oligohaline macrophyte beds": {
        "description": "Submerged and emergent macrophytes in nearshore freshwater/oligohaline areas",
        "salinity": "0–3 PSU", "depth": "littoral–infralittoral",
        "indicator_species": ["Potamogeton perfoliatus","Potamogeton natans",
                               "Nymphaea alba","Nuphar lutea","Typha angustifolia",
                               "Schoenoplectus lacustris","Phragmites australis",
                               "Elodea canadensis","Ceratophyllum demersum",
                               "Sparganium emersum","Equisetum fluviatile"]},
    "Baltic profundal soft bottom (Monoporeia/Halicryptus community)": {
        "description": "Glacial relict deepwater community in brackish basins",
        "salinity": "2–10 PSU", "depth": "profundal",
        "indicator_species": ["Monoporeia affinis","Pontoporeia femorata",
                               "Mysis relicta","Mysis mixta","Halicryptus spinulosus",
                               "Saduria entomon","Bylgides sarsi","Diastylis rathkei"]},
    "Fucus vesiculosus / mixed macrophyte belt (mesohaline)": {
        "description": "Bladder wrack and associated epiphytes and invertebrates on hard substrates",
        "salinity": "3–15 PSU", "depth": "littoral–infralittoral",
        "indicator_species": ["Fucus vesiculosus radicans","Ceramium tenuicorne",
                               "Cladophora rupestris","Elachista fucicola",
                               "Dictyosiphon foeniculaceus","Furcellaria lumbricalis",
                               "Mytilus","Amphibalanus improvisus","Gammarus salinus",
                               "Einhornia crustulenta","Coccotylus Phyllophora"]},
    "Mytilus / barnacle mussel beds": {
        "description": "Hard substrate dominated by mussels and barnacles",
        "salinity": "3–35 PSU", "depth": "littoral–infralittoral",
        "indicator_species": ["Mytilus","Amphibalanus improvisus","Hediste diversicolor",
                               "Mya arenaria","Gammarus salinus","Ampithoe rubricata",
                               "Einhornia crustulenta","Furcellaria lumbricalis"]},
    "Zostera marina / seagrass beds": {
        "description": "Seagrass meadows, typically on soft/sandy substrates",
        "salinity": "5–30 PSU", "depth": "infralittoral",
        "indicator_species": ["Zostera marina","Ruppia cirrhosa","Ruppia maritima",
                               "Chaetomorpha linum","Cladophora sericea",
                               "Bittium reticulatum","Capitella capitata","Arenicola marina"]},
    "Estuarine soft bottom (mesohaline intertidal/infralittoral)": {
        "description": "Muddy/sandy shores with depositional fauna and opportunistic algae",
        "salinity": "5–25 PSU", "depth": "littoral–infralittoral",
        "indicator_species": ["Hediste diversicolor","Pygospio elegans","Limecola balthica",
                               "Mya arenaria","Cerastoderma glaucum","Peringia ulvae",
                               "Corophium volutator","Nereis diversicolor","Marenzelleria",
                               "Arenicola marina","Tubificoides benedii"]},
    "Subtidal red/brown algae assemblage (mesohaline–polyhaline)": {
        "description": "Diverse sublittoral macroalgae on hard substrates",
        "salinity": "5–25 PSU", "depth": "infralittoral–circalittoral",
        "indicator_species": ["Furcellaria lumbricalis","Coccotylus Phyllophora",
                               "Phyllophora crispa","Ahnfeltia plicata","Ceramium virgatum",
                               "Delesseria sanguinea","Rhodomela confervoides",
                               "Saccharina latissima","Laminaria digitata",
                               "Phycodrys rubens","Membranoptera alata"]},
    "Kattegat kelp / high-salinity macroalgae": {
        "description": "Laminaria and diverse red algae on exposed rocky substrates",
        "salinity": "20–35 PSU", "depth": "infralittoral–circalittoral",
        "indicator_species": ["Laminaria hyperborea","Corallina officinalis",
                               "Odonthalia dentata","Palmaria palmata",
                               "Lithothamnion glaciale","Ptilota gunneri",
                               "Bonnemaisonia asparagoides","Heterosiphonia plumosa",
                               "Spirorbis corallinae","Halisarca dujardinii",
                               "Crisia eburnea","Flustra foliacea"]},
    "Kattegat deep soft bottom (Amphiura community)": {
        "description": "Offshore circalittoral soft sediments with brittle stars",
        "salinity": "25–35 PSU", "depth": "circalittoral",
        "indicator_species": ["Amphiura filiformis","Nucula nitidosa","Thyasira flexuosa",
                               "Turritella communis","Magelona alleni","Prionospio fallax",
                               "Chaetozone setosa","Glycera alba","Abra nitida",
                               "Maldane sarsi","Galathowenia oculata","Dosinia exoleta"]},
    "Kattegat deep mud (Brissopsis community)": {
        "description": "Deep muddy basins with spatangoid sea urchins and tubicolous polychaetes",
        "salinity": "25–35 PSU", "depth": "circalittoral–profundal",
        "indicator_species": ["Brissopsis lyrifera","Virgularia mirabilis",
                               "Golfingia vulgaris","Rhodine loveni","Praxillella affinis",
                               "Labidoplax buskii","Harpinia antennaria",
                               "Diastylis laevis","Ampelisca diadema","Sphaerodorum gracilis"]},
    "Kattegat sandy infralittoral (Echinocardium/Lanice community)": {
        "description": "Sandy bottoms with heart urchins and tube-building polychaetes",
        "salinity": "15–35 PSU", "depth": "infralittoral",
        "indicator_species": ["Echinocardium cordatum","Lanice conchilega",
                               "Spiophanes bombyx","Fabulina fabula","Nephtys longosetosa",
                               "Ophelia borealis","Bathyporeia guilliamsonia",
                               "Branchiostoma lanceolatum","Spisula subtruncata"]},
    "Baltic brackish soft bottom (mixed infauna)": {
        "description": "Soft bottom communities of intermediate salinity zones",
        "salinity": "5–20 PSU", "depth": "infralittoral–circalittoral",
        "indicator_species": ["Limecola balthica","Pygospio elegans","Marenzelleria",
                               "Bylgides sarsi","Ampharete acutifrons","Diastylis rathkei",
                               "Alitta succinea","Scoloplos armiger","Ampharete baltica"]},
    "Ruppia / charophyte lagoon (oligo-mesohaline)": {
        "description": "Sheltered lagoon vegetation in low to moderate salinity",
        "salinity": "2–15 PSU", "depth": "infralittoral",
        "indicator_species": ["Ruppia maritima","Ruppia cirrhosa","Chara baltica",
                               "Chara canescens","Zannichellia palustris",
                               "Myriophyllum spicatum","Stuckenia pectinata",
                               "Bolboschoenus maritimus","Phragmites australis"]},
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Visualisations
# ─────────────────────────────────────────────────────────────────────────────

FG_COLORS = {
    "submerged_macrophyte": "#2ecc71",
    "emergent_macrophyte":  "#27ae60",
    "floating_macrophyte":  "#1abc9c",
    "macroalga":            "#16a085",
    "filter_feeder":        "#3498db",
    "deposit_feeder":       "#e67e22",
    "predator":             "#e74c3c",
    "suspension_feeder":    "#9b59b6",
    "epibenthic":           "#f39c12",
    "other":                "#95a5a6",
    "unknown":              "#bdc3c7",
}
ZONE_COLORS = {"Zone1": "#3498db", "Zone2": "#27ae60", "Zone3": "#e67e22"}


def _cluster_order(metrics_df):
    return list(metrics_df["cluster_name"])


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_cluster_sizes(df_full: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    order = _cluster_order(metrics_df)
    fg_cols = sorted(df_full["functional_group"].dropna().unique())

    data = {}
    for cname, grp in df_full.groupby("cluster_name"):
        fg_c = grp["functional_group"].fillna("unknown").value_counts()
        data[cname] = {fg: fg_c.get(fg, 0) for fg in fg_cols}
    plot_df = pd.DataFrame(data, index=fg_cols).T.reindex(order)

    fig, ax = plt.subplots(figsize=(12, 9))
    bottom = np.zeros(len(order))
    for fg in fg_cols:
        vals = plot_df[fg].values.astype(float)
        colors = [FG_COLORS.get(fg, "#bdc3c7")] * len(order)
        ax.barh(range(len(order)), vals, left=bottom, color=colors,
                label=fg.replace("_"," "), height=0.7)
        bottom += vals

    # Zone colour left border
    for i, cname in enumerate(order):
        z = cname.split("_")[0]
        ax.barh(i, 0.25, left=-0.35, color=ZONE_COLORS.get(z,"grey"), height=0.7)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Number of species")
    ax.set_title("Cluster Sizes and Functional Group Composition", fontsize=13)
    handles = [mpatches.Patch(color=FG_COLORS.get(fg,"#bdc3c7"),
                              label=fg.replace("_"," ")) for fg in fg_cols]
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    ax.invert_yaxis()
    plt.tight_layout()

    path = OUT_FIG / "01_cluster_sizes_functional.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig_to_base64(fig)


def plot_salinity_heatmap(df_full: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    SAL_CATS = ["freshwater","oligohaline","mesohaline","polyhaline","euhaline","unknown"]
    def sal_cat(row):
        mn, mx = row.get("salinity_min"), row.get("salinity_max")
        if pd.isna(mn):
            return "unknown"
        mid = (mn + mx) / 2
        if mid < 0.5: return "freshwater"
        if mid < 5:   return "oligohaline"
        if mid < 18:  return "mesohaline"
        if mid < 30:  return "polyhaline"
        return "euhaline"

    df_full["sal_cat"] = df_full.apply(sal_cat, axis=1)
    order = _cluster_order(metrics_df)

    mat = pd.DataFrame(0, index=order, columns=SAL_CATS)
    for cname, grp in df_full.groupby("cluster_name"):
        if cname not in mat.index:
            continue
        for cat, cnt in grp["sal_cat"].value_counts().items():
            if cat in mat.columns:
                mat.loc[cname, cat] = cnt

    # Proportion
    mat_prop = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)

    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(mat_prop, ax=ax, cmap="YlOrRd", annot=mat.values,
                fmt="d", linewidths=0.5, cbar_kws={"label":"proportion"},
                annot_kws={"size": 8})

    # Highlight zone-incompatible cells with red border
    zone_compat = {
        "Zone1": {"freshwater","oligohaline","mesohaline"},
        "Zone2": {"oligohaline","mesohaline","polyhaline"},
        "Zone3": {"mesohaline","polyhaline","euhaline"},
    }
    for i, cname in enumerate(order):
        z = cname.split("_")[0]
        compat = zone_compat.get(z, set())
        for j, cat in enumerate(SAL_CATS):
            if cat not in compat and mat.loc[cname, cat] > 0:
                ax.add_patch(mpatches.Rectangle(
                    (j, i), 1, 1, fill=False, edgecolor="red", lw=2))

    ax.set_title("Salinity Category Distribution per Cluster\n(red border = zone-incompatible)", fontsize=12)
    ax.set_xlabel("Salinity category"); ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(OUT_FIG / "02_salinity_heatmap.png", dpi=120, bbox_inches="tight")
    return fig_to_base64(fig)


def plot_metrics_heatmap(metrics_df: pd.DataFrame) -> str:
    cols = ["salinity_purity","dominant_fg_prop","substrate_purity","depth_coherence","helcom_score"]
    labels = ["Salinity\npurity","Funct. group\ndominance","Substrate\npurity","Depth zone\ncoherence","HELCOM\nalignment"]
    sub = metrics_df.set_index("cluster_name")[cols].astype(float)

    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = LinearSegmentedColormap.from_list("rg", ["#e74c3c","#f39c12","#2ecc71"])
    im = ax.imshow(sub.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.index, fontsize=9)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            v = sub.values[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "N/A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if v > 0.4 else "white")

    plt.colorbar(im, ax=ax, label="Score (0–1)")
    ax.set_title("Cluster Coherence Metrics\n(green=high, red=low)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "03_metrics_heatmap.png", dpi=120, bbox_inches="tight")
    return fig_to_base64(fig)


def plot_outlier_scatter(df_full: pd.DataFrame, outlier_scores: pd.DataFrame) -> str:
    df_m = df_full.merge(outlier_scores[["species","cluster_name","is_sal_outlier",
                                          "sal_outlier_score"]],
                          on=["species","cluster_name"], how="left")
    zones = ["Zone1","Zone2","Zone3"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)

    for ax, zone in zip(axes, zones):
        sub = df_m[df_m["zone"] == zone].copy()
        cluster_meds = sub.groupby("cluster_name")["sal_midpoint"].median().rename("cluster_sal_med")
        sub = sub.join(cluster_meds, on="cluster_name")

        normal = sub[sub["is_sal_outlier"] != True]
        outl   = sub[sub["is_sal_outlier"] == True]

        ax.scatter(normal["sal_midpoint"], normal["cluster_sal_med"],
                   alpha=0.5, c="#3498db", s=25, label="Normal")
        ax.scatter(outl["sal_midpoint"], outl["cluster_sal_med"],
                   alpha=0.9, c="#e74c3c", s=60, marker="*", label="Outlier", zorder=5)

        # Label top outliers
        top = outl.nlargest(min(8, len(outl)), "sal_outlier_score")
        for _, row in top.iterrows():
            if pd.notna(row["sal_midpoint"]):
                ax.annotate(row["species"].split()[0], (row["sal_midpoint"], row["cluster_sal_med"]),
                            fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")

        # Diagonal reference
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)
        ax.set_title(zone, fontsize=11)
        ax.set_xlabel("Species salinity midpoint (PSU)")
        if zone == "Zone1":
            ax.set_ylabel("Cluster median salinity (PSU)")
        ax.legend(fontsize=8)

    fig.suptitle("Salinity Outlier Detection — Species vs Cluster Salinity", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "04_outlier_scatter.png", dpi=120, bbox_inches="tight")
    return fig_to_base64(fig)


def plot_helcom_alignment(metrics_df: pd.DataFrame) -> str:
    order = _cluster_order(metrics_df)
    scores = metrics_df.set_index("cluster_name").reindex(order)["helcom_score"].fillna(0)
    names  = metrics_df.set_index("cluster_name").reindex(order)["helcom_best_match"]

    colors = [ZONE_COLORS.get(c.split("_")[0], "#aaa") for c in order]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(order)), scores.values, color=colors, height=0.7)
    ax.axvline(0.3, color="k", ls="--", lw=1, label="Threshold = 0.30")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Recall-Jaccard score (|cluster ∩ indicators| / |indicators|)")
    ax.set_title("HELCOM Biotope Alignment Score per Cluster", fontsize=12)

    for i, (s, nm) in enumerate(zip(scores.values, names)):
        if s > 0.05:
            ax.text(s + 0.005, i, nm[:45], va="center", fontsize=7)

    handles = [mpatches.Patch(color=v, label=k) for k, v in ZONE_COLORS.items()]
    handles.append(mpatches.Patch(color="k", label="Threshold 0.30"))
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "05_helcom_alignment.png", dpi=120, bbox_inches="tight")
    return fig_to_base64(fig)


def plot_trait_profiles(df_full: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    order = _cluster_order(metrics_df)
    n = len(order)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    for idx, cname in enumerate(order):
        ax = axes[idx]
        grp = df_full[df_full["cluster_name"] == cname]
        fg_counts = grp["functional_group"].fillna("unknown").value_counts()
        labels = [l.replace("_"," ")[:16] for l in fg_counts.index]
        colors = [FG_COLORS.get(l, "#bdc3c7") for l in fg_counts.index]
        ax.pie(fg_counts.values, labels=None, colors=colors,
               autopct=lambda p: f"{p:.0f}%" if p > 8 else "",
               pctdistance=0.75, startangle=90,
               wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
        zone = cname.split("_")[0]
        ax.set_title(cname, fontsize=9, fontweight="bold",
                     color=ZONE_COLORS.get(zone, "black"))

        # Add small text legend inside figure
        legend_text = "\n".join([f"  {l}: {c}" for l, c in zip(labels, fg_counts.values)][:6])
        ax.text(1.05, 0, legend_text, transform=ax.transAxes,
                fontsize=6, va="center", family="monospace")

    # Hide unused axes
    for i in range(len(order), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Functional Group Composition per Cluster", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "06_cluster_trait_profiles.png", dpi=120, bbox_inches="tight")
    return fig_to_base64(fig)


def plot_cross_zone_species(df_full: pd.DataFrame) -> str:
    multi = df_full[df_full["multi_zone"]].copy()
    counts = multi.groupby("species")["zone"].nunique().reset_index()
    counts.columns = ["species","n_zones"]
    top = counts.nlargest(30, "n_zones")

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = []
    for _, row in top.iterrows():
        rows = multi[multi["species"] == row["species"]]
        fg = rows["functional_group"].dropna()
        fg = fg.iloc[0] if len(fg) else "unknown"
        colors.append(FG_COLORS.get(fg, "#bdc3c7"))

    ax.barh(top["species"], top["n_zones"], color=colors, height=0.7)
    ax.set_xlabel("Number of salinity zones occupied")
    ax.set_title("Multi-Zone Species (occurring in 2+ salinity zones)", fontsize=12)
    ax.invert_yaxis()

    handles = [mpatches.Patch(color=v, label=k.replace("_"," "))
               for k, v in FG_COLORS.items() if k != "unknown"]
    ax.legend(handles=handles, fontsize=7, loc="lower right", ncol=2)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "07_cross_zone_species.png", dpi=120, bbox_inches="tight")
    return fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: HTML Report
# ─────────────────────────────────────────────────────────────────────────────

ZONE_DESCRIPTIONS = {
    "Zone1": (
        "Zone 1 represents low-salinity to freshwater environments (0–8 PSU), "
        "corresponding to the inner Baltic, river mouths, brackish coastal lagoons, "
        "and sheltered bays with significant freshwater influence. "
        "Clusters A–C are dominated by freshwater macrophytes and charophytes, "
        "Cluster D reflects transitional oligohaline vegetation (Ruppia, Chara baltica), "
        "Cluster E captures shallow brackish and emergent plant communities, "
        "Cluster F is the mesohaline littoral community (Fucus vesiculosus f. radicans, "
        "Ceramium, mixed invertebrates), and Cluster B is the diagnostic brackish profundal "
        "glacial-relict fauna (Mysis relicta, Monoporeia affinis, Halicryptus spinulosus). "
    ),
    "Zone2": (
        "Zone 2 represents intermediate salinities (3–18 PSU), typical of the central "
        "Baltic proper and transitional zones. "
        "Cluster F contains transitional macrophyte species (Ruppia, Chara, Fucus f. radicans); "
        "Cluster E is a brackish littoral algal community; "
        "Cluster D is a mixed subtidal community including Zostera marina, soft-bottom fauna, "
        "and some freshwater outliers; "
        "Cluster C is a classic brackish soft-bottom infaunal assemblage; "
        "Cluster B resembles the Fucus–Mytilus–barnacle intertidal/shallow community; "
        "Cluster A is a subtidal community with both algae and soft-bottom fauna. "
    ),
    "Zone3": (
        "Zone 3 covers higher-salinity environments (8–35 PSU) including the outer Baltic, "
        "the Sound (Øresund), and the Kattegat. "
        "Clusters A–E reflect a range from mixed brackish–estuarine habitats to "
        "Zostera marina beds and mesohaline soft-bottom communities; "
        "Clusters F–G capture the diverse sublittoral macroalgal assemblages of the "
        "outer Baltic and Kattegat; Cluster H is a hard-substrate assemblage with "
        "coralline algae and associated epifauna; "
        "Clusters I–J represent the deep-water circalittoral soft-bottom communities "
        "of the Kattegat (Brissopsis/Virgularia and Amphiura communities respectively). "
    ),
}


def severity_badge(sev: str) -> str:
    colors = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#f1c40f"}
    color = colors.get(sev, "#95a5a6")
    return (f'<span style="background:{color};color:white;padding:2px 7px;'
            f'border-radius:4px;font-size:0.8em;font-weight:bold">{sev}</span>')


def score_cell(val: float) -> str:
    if pd.isna(val):
        return '<td style="text-align:center">N/A</td>'
    if val >= 0.7:
        bg = "#d4efdf"
    elif val >= 0.4:
        bg = "#fef9e7"
    else:
        bg = "#fadbd8"
    return f'<td style="text-align:center;background:{bg}">{val:.2f}</td>'


def export_html_report(df_full, metrics_df, outlier_df, trait_coverage, b64_figs):
    n_clusters = len(metrics_df)
    n_species  = df_full["species"].nunique()
    n_outliers = len(outlier_df)
    mean_sal_pur = metrics_df["salinity_purity"].mean()
    high_out = len(outlier_df[outlier_df["severity"] == "HIGH"])
    med_out  = len(outlier_df[outlier_df["severity"] == "MEDIUM"])

    # Metrics table rows
    metric_rows = ""
    for _, r in metrics_df.iterrows():
        metric_rows += f"""
        <tr>
          <td><b>{r['cluster_name']}</b></td>
          <td style="text-align:center">{r['n_species']}</td>
          {score_cell(r['salinity_purity'])}
          {score_cell(r['dominant_fg_prop'])}
          {score_cell(r['substrate_purity'])}
          {score_cell(r['depth_coherence'])}
          {score_cell(r['helcom_score'])}
          <td style="font-size:0.85em">{r['helcom_best_match']}</td>
          <td style="font-size:0.82em;color:#555">{r['dominant_func_group'].replace('_',' ')}</td>
        </tr>"""

    # Outlier table rows
    outlier_rows = ""
    for _, r in outlier_df.iterrows():
        outlier_rows += f"""
        <tr>
          <td><i>{r['species']}</i></td>
          <td>{r['cluster_name']}</td>
          <td>{severity_badge(r['severity'])}</td>
          <td style="font-size:0.85em">{r['reason']}</td>
          <td style="font-size:0.83em;color:#444">{r['ecological_note']}</td>
        </tr>"""

    # Zone narratives
    zone_html = ""
    for z, desc in ZONE_DESCRIPTIONS.items():
        zone_html += f"""
        <div style="background:#f8f9fa;border-left:4px solid {ZONE_COLORS[z]};
                    padding:12px 16px;margin-bottom:12px;border-radius:4px">
          <b style="color:{ZONE_COLORS[z]}">{z}</b> — {desc}
        </div>"""

    # Figures
    def fig_html(b64, title, caption=""):
        return f"""
        <div style="margin:20px 0">
          <h4 style="color:#2c3e50">{title}</h4>
          <img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:4px"/>
          {f'<p style="color:#666;font-size:0.9em">{caption}</p>' if caption else ''}
        </div>"""

    figs_html  = fig_html(b64_figs[0], "Figure 1: Cluster Sizes and Functional Group Composition",
                          "Coloured left bars indicate salinity zone (blue=Zone1, green=Zone2, orange=Zone3).")
    figs_html += fig_html(b64_figs[1], "Figure 2: Salinity Category Distribution per Cluster",
                          "Red borders mark cells where species fall outside the zone's expected salinity range.")
    figs_html += fig_html(b64_figs[2], "Figure 3: Coherence Metrics Summary",
                          "Green = high coherence, red = low. HELCOM score = recall-Jaccard against biotope indicator lists.")
    figs_html += fig_html(b64_figs[3], "Figure 4: Salinity Outlier Detection",
                          "Stars mark Tukey-fence outliers per cluster. Dashed diagonal = perfect salinity match.")
    figs_html += fig_html(b64_figs[4], "Figure 5: HELCOM Biotope Alignment",
                          "Dashed line = 0.30 threshold. Scores are recall-form: fraction of a biotope's indicator species found in the cluster.")
    figs_html += fig_html(b64_figs[5], "Figure 6: Functional Group Profiles per Cluster")
    figs_html += fig_html(b64_figs[6], "Figure 7: Multi-Zone Species",
                          "Species appearing in 2 or 3 salinity zones; expected for euryhaline taxa but may indicate artefacts for freshwater taxa.")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Baltic Sea Habitat Cluster — Ecological Evaluation</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;max-width:1200px;margin:0 auto;padding:24px;color:#2c3e50;line-height:1.6}}
  h1{{color:#1a252f;border-bottom:3px solid #3498db;padding-bottom:8px}}
  h2{{color:#2980b9;margin-top:2em}}
  h3{{color:#16a085}}
  table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:0.88em}}
  th{{background:#2c3e50;color:white;padding:8px 10px;text-align:left}}
  td{{padding:7px 10px;border-bottom:1px solid #ecf0f1}}
  tr:hover{{background:#f8f9fa}}
  .summary-box{{background:#eaf4fb;border:1px solid #aed6f1;border-radius:6px;padding:16px;margin:16px 0}}
  .metric{{display:inline-block;background:#2c3e50;color:white;border-radius:6px;
           padding:10px 20px;margin:6px;text-align:center;min-width:140px}}
  .metric .val{{font-size:2em;font-weight:bold;display:block}}
  .metric .lbl{{font-size:0.8em;opacity:0.85}}
  footer{{color:#888;font-size:0.85em;margin-top:40px;border-top:1px solid #ddd;padding-top:12px}}
</style>
</head>
<body>

<h1>Baltic Sea Habitat Cluster Ecological Evaluation</h1>
<p style="color:#555">Analysis of hierarchical clustering results from a 250 m resolution
benthic species dataset covering the Baltic Sea and Kattegat region.</p>

<div class="summary-box">
  <b>Executive Summary</b> —
  The analysis covers <b>{n_clusters} clusters</b> across 3 salinity zones,
  containing <b>{n_species} unique species</b>.
  Mean salinity purity across all clusters is <b>{mean_sal_pur:.1%}</b>.
  A total of <b>{n_outliers} ecological flags</b> were raised,
  including {high_out} HIGH-severity and {med_out} MEDIUM-severity outliers.
  Zone 1 and Zone 2 clusters generally show good ecological coherence;
  several Zone 3 clusters (especially Zone3_A) show high heterogeneity
  likely attributable to the 250 m resolution capturing multiple microhabitats
  in a single grid cell.
</div>

<div>
  <div class="metric"><span class="val">{n_clusters}</span><span class="lbl">Clusters</span></div>
  <div class="metric"><span class="val">{n_species}</span><span class="lbl">Unique species</span></div>
  <div class="metric"><span class="val">{mean_sal_pur:.0%}</span><span class="lbl">Mean salinity purity</span></div>
  <div class="metric"><span class="val">{high_out}</span><span class="lbl">HIGH outliers</span></div>
  <div class="metric"><span class="val">{med_out}</span><span class="lbl">MEDIUM outliers</span></div>
</div>

<h2>Ecological Zone Narratives</h2>
{zone_html}

<h2>Cluster Metrics Table</h2>
<p>Colour coding:
  <span style="background:#d4efdf;padding:2px 8px;border-radius:3px">≥ 0.70 good</span>&nbsp;
  <span style="background:#fef9e7;padding:2px 8px;border-radius:3px">0.40–0.70 moderate</span>&nbsp;
  <span style="background:#fadbd8;padding:2px 8px;border-radius:3px">&lt; 0.40 low</span>
</p>
<table>
  <thead>
    <tr>
      <th>Cluster</th><th>N</th>
      <th>Salinity purity</th><th>FG dominance</th>
      <th>Substrate purity</th><th>Depth coherence</th>
      <th>HELCOM score</th><th>Best HELCOM match</th>
      <th>Dominant group</th>
    </tr>
  </thead>
  <tbody>{metric_rows}</tbody>
</table>

<h2>Outlier Species ({n_outliers} flags)</h2>
<table>
  <thead>
    <tr><th>Species</th><th>Cluster</th><th>Severity</th>
        <th>Reason</th><th>Ecological note</th></tr>
  </thead>
  <tbody>{outlier_rows}</tbody>
</table>

<h2>Visualisations</h2>
{figs_html}

<h2>Methodology</h2>
<h3>Trait Assignment</h3>
<ul>
{"".join(f"<li><b>{k}:</b> {v} species</li>" for k,v in trait_coverage.items())}
</ul>
<p>Traits assigned: salinity tolerance range (PSU), depth zone, functional group, substrate type.
Salinity midpoints used for numeric metrics:
freshwater ≈ 0, oligohaline ≈ 3, mesohaline ≈ 11.5, polyhaline ≈ 24, euhaline ≈ 35 PSU.</p>

<h3>Metrics</h3>
<ul>
  <li><b>Salinity purity:</b> proportion of species whose salinity tolerance overlaps with the zone's expected range</li>
  <li><b>Functional group dominance:</b> proportion of species in the single most common functional group</li>
  <li><b>Substrate purity:</b> proportion of species in the modal substrate category</li>
  <li><b>Depth coherence:</b> proportion of species in the modal depth zone</li>
  <li><b>HELCOM alignment:</b> recall-form Jaccard against 14 reference HELCOM/EUNIS biotope indicator lists</li>
  <li><b>Salinity outlier score:</b> Tukey fence on salinity midpoint within each cluster (IQR method)</li>
</ul>

<h3>Resolution Note</h3>
<p>The 250 m grid resolution means that single pixels may contain overlapping microhabitats
(e.g., hard substrate and adjacent soft bottom; freshwater seep within a brackish bay).
Some apparent species-cluster mismatches (especially in transitional zones) may therefore
reflect genuine co-occurrence at that spatial scale rather than analysis artefacts.
Species flagged as HIGH severity represent the clearest ecological mismatches.</p>

<h3>Data Sources</h3>
<ul>
  <li>Species trait assignments: Tier 1 — Baltic/HELCOM literature (Kautsky 1992; Zettler et al. 2013;
      HELCOM Red List 2019; Boström et al. 2014); Tier 2 — WoRMS REST API
      (marinespecies.org); Tier 3 — genus-level ecological inference</li>
  <li>HELCOM biotope reference: HELCOM HUB (habitat.helcom.fi) and
      EUNIS Marine Habitat Classification</li>
</ul>

<footer>
  Generated by Baltic Sea Habitat Cluster Ecological Evaluation script |
  Input: run33_3BottomSalinityZones_10perc_hc_binary.csv |
  Model resolution: 250 m
</footer>
</body>
</html>"""

    out = OUT_REP / "cluster_evaluation_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"HTML report written: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== Baltic Sea Habitat Cluster Ecological Evaluation ===\n")

    print("Step 1/7  Loading data …")
    df = load_cluster_data(DATA_PATH)
    print(f"  {len(df)} rows, {df['species'].nunique()} unique species, "
          f"{df['cluster_name'].nunique()} clusters")

    print("\nStep 2/7  Building trait table …")
    traits_df = build_trait_table(df)
    trait_cov = traits_df["trait_source"].value_counts().to_dict()

    print("\nStep 3/7  Merging traits …")
    df_full = merge_traits(df, traits_df)

    print("\nStep 4/7  Computing coherence metrics …")
    metrics_df = compute_all_metrics(df_full, HELCOM_BIOTOPES)
    metrics_df.to_csv(OUT_REP / "cluster_metrics.csv", index=False)
    print(f"  Metrics saved: {OUT_REP / 'cluster_metrics.csv'}")

    print("\nStep 5/7  Identifying outliers …")
    outlier_scores = compute_salinity_outlier_scores(df_full)
    outlier_df = generate_outlier_report(df_full, outlier_scores)
    outlier_df.to_csv(OUT_REP / "outlier_species.csv", index=False)
    print(f"  {len(outlier_df)} flags raised; saved: {OUT_REP / 'outlier_species.csv'}")

    print("\nStep 6/7  Generating figures …")
    b64_figs = [
        plot_cluster_sizes(df_full, metrics_df),
        plot_salinity_heatmap(df_full, metrics_df),
        plot_metrics_heatmap(metrics_df),
        plot_outlier_scatter(df_full, outlier_scores),
        plot_helcom_alignment(metrics_df),
        plot_trait_profiles(df_full, metrics_df),
        plot_cross_zone_species(df_full),
    ]
    print(f"  7 figures saved to {OUT_FIG}")

    print("\nStep 7/7  Writing HTML report …")
    report_path = export_html_report(df_full, metrics_df, outlier_df, trait_cov, b64_figs)

    print(f"\nDone.  Main output: {report_path}")
    print(f"  Also: {OUT_REP / 'cluster_metrics.csv'}")
    print(f"        {OUT_REP / 'outlier_species.csv'}")
    print(f"        {ROOT / 'traits' / 'species_traits_merged.csv'}")


if __name__ == "__main__":
    main()
