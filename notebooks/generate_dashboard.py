"""
Baltic Sea Habitat Cluster — Interactive Dashboard Generator
============================================================
Generates outputs/reports/cluster_dashboard.html — a fully self-contained
interactive HTML dashboard with a sidebar cluster browser.

Run from the project root:
    uv run --with pillow --with numpy --with pandas --with matplotlib \\
           notebooks/generate_dashboard.py

Requirements:
    pillow, numpy, pandas, matplotlib
"""

import sys, os, json, base64, io, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA      = ROOT / "data"
SDM_DIR   = DATA / "sdms"
OUT_REP   = ROOT / "outputs" / "reports"
TRAITS_F  = ROOT / "traits" / "species_traits_merged.csv"
OUTLIER_F = DATA / "outlier_species.csv"
METRICS_F = OUT_REP / "cluster_metrics.csv"

# Set at runtime by main() from the CLI argument
RUN_DIR   = None
PNG_DIR   = None
DENDRO_DIR = None
COEFF_F   = None
CLUSTER_ORDER = []


def _set_run_paths(run_dir: Path):
    global RUN_DIR, PNG_DIR, DENDRO_DIR, COEFF_F
    RUN_DIR    = run_dir
    PNG_DIR    = run_dir / "_pngs" / "individ_cluster_binary_png"
    DENDRO_DIR = run_dir / "dendrograms"
    COEFF_F    = run_dir / "species_cluster_coefficients.csv"


def _derive_cluster_order(species_df: "pd.DataFrame") -> list:
    """Sort cluster names by zone number then letter: Zone1_A, Zone1_B, …"""
    names = species_df["cluster_name"].unique()
    def _key(n):
        zone, letter = n.split("_")
        return (int(zone.replace("Zone", "")), letter)
    return sorted(names, key=_key)

ZONE_COLORS   = {"Zone1": "#3498db", "Zone2": "#27ae60", "Zone3": "#e67e22", "Zone4": "#9b59b6"}

# Ecological descriptions derived from species composition, substrate, and depth data
HELCOM_OVERRIDES = {
    "Zone1_A": "Oligohaline mixed macrophyte and stonewort beds",
    "Zone1_B": "Shallow oligohaline stonewort and pondweed meadow",
    "Zone1_C": "Oligohaline emergent and floating-leaved macrophyte community",
    "Zone1_D": "Shallow oligohaline diverse pondweed and pioneer macrophyte beds",
    "Zone1_E": "Oligohaline soft-sediment Elodea and filamentous algae community",
    "Zone2_A": "Oligo-mesohaline emergent and floating-leaved macrophyte beds",
    "Zone2_B": "Shallow oligo-mesohaline mixed brackish macrophyte community",
    "Zone2_C": "Shallow oligo-mesohaline filamentous brown algae and pondweed community",
    "Zone2_D": "Deep oligo-mesohaline soft-sediment amphipod and priapulan community",
    "Zone2_E": "Sublittoral oligo-mesohaline sandy epifaunal and bivalve community",
    "Zone3_A": "Shallow mesohaline mixed algae, filter-feeder and bivalve assemblage",
    "Zone3_B": "Mesohaline red and brown macroalgal belt on mixed substrates",
    "Zone3_C": "Mesohaline sandy-substrate macroalgal and polychaete community",
    "Zone3_D": "Sublittoral mesohaline deposit-feeder and macroalgal community on sandy sediment",
    "Zone3_E": "Deep mesohaline soft-sediment amphipod and isopod community",
    "Zone3_F": "Sublittoral mesohaline diverse benthic invertebrate community",
    "Zone4_A": "Shallow polyhaline epiphytic filamentous algae and encrusting fauna",
    "Zone4_B": "Circalittoral polyhaline subtidal red algae bed",
    "Zone4_C": "Shallow polyhaline shoreline community with salt-tolerant vegetation and green algae",
    "Zone4_D": "Shallow polyhaline mixed brown and green macroalgal beds",
    "Zone4_E": "Sublittoral polyhaline sandy filter-feeder and bryozoan community",
    "Zone4_F": "Sublittoral polyhaline mixed soft-sediment invertebrate community",
    "Zone4_G": "Polyhaline sandy sublittoral community with coralline algae and lancelets",
    "Zone4_H": "Sublittoral polyhaline kelp and red algae community",
    "Zone4_I": "Sublittoral polyhaline diverse red and brown algae community",
    "Zone4_J": "Deep polyhaline soft-sediment polychaete deposit-feeder community",
    "Zone4_K": "Deep polyhaline diverse soft-sediment invertebrate community",
}
ZONE_LABELS   = {"Zone1": "Zone 1 — Low salinity (0–3 PSU)",
                 "Zone2": "Zone 2 — Low-intermediate (3–6 PSU)",
                 "Zone3": "Zone 3 — Intermediate (6–12 PSU)",
                 "Zone4": "Zone 4 — High salinity (12–31 PSU)"}

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

SUBSTRATE_ORDER = ["Hard", "Coarse", "Sand", "Soft"]

SUBSTRATE_COLORS = {
    "Hard":   "#607d8b",
    "Coarse": "#9e9e9e",
    "Sand":   "#ffe082",
    "Soft":   "#a1887f",
}

# Shallow → deep display order (only ranges present in any cluster)
DEPTH_ORDER = [
    "0-10 m", "10-30 m", "30-60 m", "60-100 m",
    "100-200 m", "200-300 m", "> 300 m",
]

DEPTH_COLORS = {
    "0-10 m":    "#a8d8ea",
    "10-30 m":   "#7ec8e3",
    "30-60 m":   "#3b9bbf",
    "60-100 m":  "#2e94c4",
    "100-200 m": "#1a6a9a",
    "200-300 m": "#0a4d82",
    "> 300 m":   "#011944",
}

MOBILE_GROUP_COLORS = {
    "fish":          "#2980b9",
    "invertebrates": "#e67e22",
    "birds":         "#27ae60",
    "mammals":       "#8e44ad",
}

# If you have mobile species names, fill them in here as a list of 98 strings.
# Leave as None to use generic "Species N" labels.
MOBILE_SPECIES_NAMES = None   # e.g. ["Cod", "Herring", ...]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_species_df() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "full_cluster_df.csv")
    df.columns = df.columns.str.strip().str.strip('"')
    df["species"]       = df["species"].str.strip('"').str.replace(".", " ", regex=False)
    df["cluster_name"]  = df["cluster_name"].str.strip('"')
    df["colour"]        = df["colour"].str.strip('"')
    return df


def load_traits() -> pd.DataFrame:
    if TRAITS_F.exists():
        return pd.read_csv(TRAITS_F)
    return pd.DataFrame()


def load_outliers() -> set:
    """Return a set of (species_name, cluster_name) tuples flagged as outliers."""
    if not OUTLIER_F.exists():
        return set()
    df = pd.read_csv(OUTLIER_F, encoding="utf-8-sig", encoding_errors="replace")
    df.columns = df.columns.str.strip()
    # Convert dot-notation to spaces to match how species names are stored internally
    names    = df["scientific_name"].str.strip().str.replace(".", " ", regex=False)
    clusters = df["cluster_name"].str.strip()
    return set(zip(names, clusters))


def load_substrate() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "substrate_overlap.csv")
    df.columns = df.columns.str.strip().str.strip('"')
    df["cluster_names"] = df["cluster_names"].str.strip('"')
    df = df.set_index("cluster_names")
    return df


def load_depth() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "depth_overlap_habitat.csv")
    df.columns = df.columns.str.strip().str.strip('"')
    df["depth_range"] = df["depth_range"].str.strip('"')
    # Rows = depth ranges, cols = clusters → transpose so clusters are the index
    df = df.set_index("depth_range").T
    return df


def load_mobile() -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "mobile_overlap_habitat.csv")
    df.columns = df.columns.str.strip().str.strip('"')
    if "scientific_name" in df.columns:
        df["scientific_name"] = (df["scientific_name"]
                                 .str.strip('"')
                                 .str.replace(".", " ", regex=False))
        df = df.set_index("scientific_name")
    elif MOBILE_SPECIES_NAMES and len(MOBILE_SPECIES_NAMES) == len(df):
        df.index = MOBILE_SPECIES_NAMES
    else:
        df.index = [f"Species {i+1}" for i in range(len(df))]
    return df


def load_metrics() -> pd.DataFrame:
    if METRICS_F.exists():
        return pd.read_csv(METRICS_F).set_index("cluster_name")
    return pd.DataFrame()



def load_coefficients() -> pd.DataFrame:
    if not COEFF_F.exists():
        return pd.DataFrame()
    df = pd.read_csv(COEFF_F)
    df.columns = df.columns.str.strip().str.strip('"')
    df["scientific_name"] = (df["scientific_name"]
                             .str.strip('"')
                             .str.replace(".", " ", regex=False))
    df["cluster_name"] = df["cluster_name"].str.strip('"')
    return df


def load_sdm_images(species_names: list) -> dict:
    """Return {species_name: relative_path} for every species that has an SDM PNG.
    Paths are relative to outputs/reports/ where the HTML is written."""
    categories = ["fish", "invertebrates", "macrophytes"]
    sdm = {}
    for name in species_names:
        dot_name = name.replace(" ", ".")
        for cat in categories:
            p = SDM_DIR / cat / "binary_strict_png" / f"{dot_name}.png"
            if p.exists():
                sdm[name] = f"../../data/sdms/{cat}/binary_strict_png/{dot_name}.png"
                break
    return sdm



def png_to_b64(cluster_name: str, png_dir: Path = None, max_dim: int = 900) -> str:
    """Load a cluster map PNG, downscale to max_dim, and return a base64 data URI."""
    path = (png_dir or PNG_DIR) / f"{cluster_name}.png"
    if not path.exists():
        return ""
    img = Image.open(path)
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Per-cluster substrate mini-chart (SVG)
# ─────────────────────────────────────────────────────────────────────────────

def substrate_svg(cluster_name: str, substrate_df: pd.DataFrame) -> str:
    if cluster_name not in substrate_df.index:
        return "<p style='color:#999'>No substrate data</p>"

    row = substrate_df.loc[cluster_name]
    top = row[row > 0.5].sort_values(ascending=False).head(10)
    if top.empty:
        return "<p style='color:#999'>No significant substrate</p>"

    # Simple SVG horizontal bar chart
    bar_h   = 22
    bar_gap = 6
    label_w = 170
    chart_w = 340
    total_h = (bar_h + bar_gap) * len(top) + 20
    max_val = top.max()

    bars = ""
    for i, (name, val) in enumerate(top.items()):
        y    = i * (bar_h + bar_gap) + 10
        bw   = int((val / 100) * chart_w)
        col  = SUBSTRATE_COLORS.get(name, "#aaa")
        pct  = f"{val:.1f}%"
        bars += (
            f'<text x="{label_w-6}" y="{y+bar_h-6}" '
            f'text-anchor="end" font-size="11" fill="#333">{name}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bw}" height="{bar_h}" '
            f'fill="{col}" rx="3"/>'
            f'<text x="{label_w+bw+4}" y="{y+bar_h-6}" '
            f'font-size="11" fill="#555">{pct}</text>'
        )

    total_w = label_w + chart_w + 60
    return (f'<svg width="{total_w}" height="{total_h}" '
            f'style="max-width:100%;overflow:visible">{bars}</svg>')


# ─────────────────────────────────────────────────────────────────────────────
# Build all cluster data as a JS object
# ─────────────────────────────────────────────────────────────────────────────

def build_cluster_data(species_df, traits_df, substrate_df, depth_df,
                        mobile_df, metrics_df, coeff_df, outlier_set) -> dict:
    data = {}

    # Cluster → colour map
    colour_map = (species_df.groupby("cluster_name")["colour"]
                  .first().to_dict())

    for cname in CLUSTER_ORDER:
        hex_col = colour_map.get(cname, "#888888")
        zone    = cname.split("_")[0]

        # Species in this cluster
        sp_rows = species_df[species_df["cluster_name"] == cname]["species"].tolist()

        # Coefficient lookup for this cluster (normalised 0–100 within cluster)
        fit_pct_map = {}
        fit_val_map = {}
        if not coeff_df.empty:
            c_rows = coeff_df[coeff_df["cluster_name"] == cname]
            if not c_rows.empty:
                vals = c_rows.set_index("scientific_name")["mean_coefficient"]
                cmin, cmax = vals.min(), vals.max()
                span = cmax - cmin if cmax != cmin else 1.0
                fit_pct_map = {sp: round((v - cmin) / span * 100, 1) for sp, v in vals.items()}
                fit_val_map = {sp: round(float(v), 4) for sp, v in vals.items()}

        # Merge with traits
        sp_info = []
        for sp in sp_rows:
            t = {}
            if not traits_df.empty:
                match = traits_df[traits_df["species"] == sp]
                if len(match):
                    t = match.iloc[0].to_dict()
            sp_info.append({
                "name":    sp,
                "fg":      str(t.get("functional_group","unknown")),
                "depth":   str(t.get("depth_zone","?")),
                "sal_min": float(t["salinity_min"]) if pd.notna(t.get("salinity_min")) else None,
                "sal_max": float(t["salinity_max"]) if pd.notna(t.get("salinity_max")) else None,
                "substrate": str(t.get("substrate","?")),
                "source":  str(t.get("trait_source","?")),
                "fit":     fit_pct_map.get(sp, None),
                "fit_val": fit_val_map.get(sp, None),
                "outlier": (sp, cname) in outlier_set,
            })

        # Sort by fit descending so strongest members appear first
        sp_info.sort(key=lambda x: (x["fit"] is None, -(x["fit"] or 0)))

        # Substrate: all types in fixed fine→coarse order
        sub_list = []
        if cname in substrate_df.index:
            row = substrate_df.loc[cname]
            for sname in SUBSTRATE_ORDER:
                val = float(row.get(sname, 0) or 0)
                sub_list.append({"name": sname, "pct": round(val, 1)})

        # Depth: all ranges in fixed shallow→deep order
        depth_list = []
        if not depth_df.empty and cname in depth_df.index:
            row = depth_df.loc[cname]
            for dname in DEPTH_ORDER:
                val = float(row.get(dname, 0) or 0)
                depth_list.append({"name": dname, "pct": round(val, 1)})

        # Mobile species top-20 for this cluster
        mob_list = []
        if cname in mobile_df.columns:
            col = mobile_df[cname]
            top_mob = col[col > 0.5].sort_values(ascending=False).head(20)
            grp_series = mobile_df["species_group"] if "species_group" in mobile_df.columns else None
            mob_list = [
                {
                    "name":  idx,
                    "pct":   round(float(v), 1),
                    "group": str(grp_series.loc[idx]) if grp_series is not None and idx in grp_series.index else "",
                }
                for idx, v in top_mob.items()
            ]

        # Metrics
        m = {}
        if cname in metrics_df.index:
            row = metrics_df.loc[cname]
            m = {
                "n_species":       int(row.get("n_species", len(sp_rows))),
                "sal_purity":      _safe(row, "salinity_purity"),
                "depth_coherence": _safe(row, "depth_coherence"),
                "helcom_match":    HELCOM_OVERRIDES.get(cname, str(row.get("helcom_best_match",""))),
                "helcom_score":    1.0 if cname in HELCOM_OVERRIDES else _safe(row, "helcom_score"),
                "dom_fg":          str(row.get("dominant_func_group","")),
                "dom_fg_prop":     _safe(row, "dominant_fg_prop"),
                "substrate_purity":_safe(row, "substrate_purity"),
            }

        data[cname] = {
            "colour":   hex_col,
            "zone":     zone,
            "species":  sp_info,
            "substrate":sub_list,
            "depth":    depth_list,
            "mobile":   mob_list,
            "metrics":  m,
        }

    return data


def _safe(row, col):
    v = row.get(col)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(float(v), 3)


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def _dendro_b64(zone: str) -> str:
    """Base64-encode a dendrogram JPEG at original quality."""
    path = DENDRO_DIR / zone / f"{zone}_dendrogram.jpg"
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode()


def build_html(cluster_data: dict, map_b64_binary: dict, map_b64_sorensen: dict, map_b64_specprop: dict, sdm_images: dict) -> str:
    # ── Dendrogram images (compressed, base64) ────────────────────────────────
    dendro_b64 = {z: _dendro_b64(z) for z in ["Zone1", "Zone2", "Zone3", "Zone4"]}

    # ── Sidebar HTML ──────────────────────────────────────────────────────────
    sidebar_items = ""
    for zone in ["Zone1", "Zone2", "Zone3", "Zone4"]:
        clusters_in_zone = [c for c in CLUSTER_ORDER if c.startswith(zone)]
        has_dendro = bool(dendro_b64.get(zone))
        sidebar_items += f"""
      <div class="zone-group">
        <div class="zone-label" style="border-left:4px solid {ZONE_COLORS[zone]}">
          {ZONE_LABELS[zone]}
        </div>
        <div class="cluster-buttons">"""
        for cname in clusters_in_zone:
            d = cluster_data.get(cname, {})
            col = d.get("colour", "#888")
            sub = cname.split("_")[1]
            n   = d.get("metrics", {}).get("n_species", "?")
            sidebar_items += f"""
          <button class="cluster-btn" id="btn-{cname}"
                  onclick="showCluster('{cname}')"
                  style="border-left:4px solid {col}">
            <span class="cluster-label">{cname}</span>
            <span class="cluster-meta">{n} spp</span>
          </button>"""
        if has_dendro:
            sidebar_items += f"""
          <button class="dendro-btn" onclick="openDendrogram('{zone}')">
            Dendrogram
          </button>"""
        sidebar_items += "\n        </div>\n      </div>"

    # ── Substrate colour legend ──────────────────────────────────────────────
    sub_legend = "".join(
        f'<span class="sub-legend-item">'
        f'<span style="background:{c};display:inline-block;width:12px;height:12px;'
        f'border-radius:2px;vertical-align:middle;margin-right:4px"></span>{n}</span>'
        for n, c in SUBSTRATE_COLORS.items()
    )

    # ── Functional group colour legend ───────────────────────────────────────
    fg_legend = "".join(
        f'<span class="fg-legend-item">'
        f'<span class="fg-dot" style="background:{c}"></span>{k.replace("_"," ")}</span>'
        for k, c in FG_COLORS.items() if k != "unknown"
    )

    # ── Embed map images and cluster data as JS ───────────────────────────────
    def _maps_js(varname, d):
        js = f"const {varname} = {{\n"
        for cname, b64 in d.items():
            if b64:
                js += f'  "{cname}": "data:image/png;base64,{b64}",\n'
        js += "};\n"
        return js
    maps_js = (_maps_js("MAPS_BINARY", map_b64_binary)
             + _maps_js("MAPS_SORENSEN", map_b64_sorensen)
             + _maps_js("MAPS_SPECPROP", map_b64_specprop))

    data_js = f"const CLUSTER_DATA = {json.dumps(cluster_data, ensure_ascii=False)};\n"

    fg_colors_js    = f"const FG_COLORS = {json.dumps(FG_COLORS)};\n"
    sub_colors_js   = f"const SUB_COLORS = {json.dumps(SUBSTRATE_COLORS)};\n"
    depth_colors_js = f"const DEPTH_COLORS = {json.dumps(DEPTH_COLORS)};\n"
    zone_colors_js  = f"const ZONE_COLORS = {json.dumps(ZONE_COLORS)};\n"
    mob_group_colors_js = f"const MOB_GROUP_COLORS = {json.dumps(MOBILE_GROUP_COLORS)};\n"

    # Dendrogram images
    dendro_js = "const DENDROGRAMS = {\n"
    for zone, b64 in dendro_b64.items():
        if b64:
            dendro_js += f'  "{zone}": "data:image/jpeg;base64,{b64}",\n'
    dendro_js += "};\n"

    # Species SDM images keyed by display name
    sdm_js = "const SPECIES_SDM = {\n"
    for name, uri in sdm_images.items():
        sdm_js += f'  {json.dumps(name)}: {json.dumps(uri)},\n'
    sdm_js += "};\n"

    mobile_note = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Baltic Sea Habitat Clusters — Dashboard</title>
<style>
/* ── Reset & base ─────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
html, body {{ height:100%; font-family:'Segoe UI',Arial,sans-serif;
              font-size:14px; background:#f0f4f8; color:#2c3e50; }}

/* ── Layout ──────────────────────────────────────────────────────────── */
#app {{ display:flex; height:100vh; }}

#sidebar {{
  width:230px; min-width:230px; background:#1a2535;
  overflow-y:auto; display:flex; flex-direction:column;
  border-right:1px solid #0d1520;
}}

#main {{
  flex:1; overflow-y:auto; display:flex; flex-direction:column; min-width:0;
}}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
#sidebar-header {{
  padding:16px 14px 10px; color:#ecf0f1;
  font-weight:700; font-size:1em; letter-spacing:.03em;
  border-bottom:1px solid #2c3e50; flex-shrink:0;
}}
#sidebar-header small {{ display:block; font-weight:400; color:#7f8c8d;
                          font-size:.82em; margin-top:3px; }}

.zone-group {{ padding:8px 0; border-bottom:1px solid #2c3e50; }}
.zone-label {{
  padding:6px 14px; color:#95a5a6; font-size:.78em;
  font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  margin-bottom:4px;
}}
.cluster-buttons {{ display:flex; flex-direction:column; gap:1px; padding:0 6px; }}
.cluster-btn {{
  background:#243447; border:none; border-radius:5px; cursor:pointer;
  padding:7px 10px; text-align:left; color:#ecf0f1;
  display:flex; justify-content:space-between; align-items:center;
  transition:background .15s; font-size:.87em;
}}
.cluster-btn:hover {{ background:#2e4461; }}
.cluster-btn.active {{ background:#2980b9 !important; color:#fff; }}
.cluster-label {{ font-weight:600; }}
.cluster-meta {{ font-size:.82em; color:#7f8c8d; }}
.cluster-btn.active .cluster-meta {{ color:#bee3f8; }}
.dendro-btn {{
  background:none; border:1px solid #2c3e50; border-radius:4px;
  color:#7f8c8d; font-size:.75em; cursor:pointer;
  padding:4px 10px; margin:4px 6px 6px; text-align:center;
  width:calc(100% - 12px); transition:all .15s; letter-spacing:.03em;
}}
.dendro-btn:hover {{ background:#2c3e50; color:#ecf0f1; }}

/* ── Top header bar ──────────────────────────────────────────────────── */
#cluster-header {{
  padding:14px 24px; background:#fff; border-bottom:2px solid #e0e8f0;
  display:flex; align-items:center; gap:16px; flex-shrink:0; min-height:60px;
}}
#cluster-dot {{
  width:18px; height:18px; border-radius:50%; flex-shrink:0;
}}
#cluster-title {{ font-size:1.3em; font-weight:700; }}
#cluster-subtitle {{ font-size:.85em; color:#7f8c8d; margin-top:2px; }}

/* ── Content panels ──────────────────────────────────────────────────── */
#content {{ flex:1; padding:0; overflow-y:auto; }}

/* Top row: map column (fixed) + info column (fills remaining, wraps internally) */
.top-row {{
  display:flex; flex-direction:row; flex-wrap:nowrap;
  border-bottom:1px solid #e0e8f0; background:#fff; align-items:stretch;
}}
.map-wrap {{
  flex:0 0 auto; width:720px; min-width:400px; max-width:800px;
  padding:12px 10px 12px 16px;
  display:flex; flex-direction:column; gap:8px;
}}
.map-toggle {{
  display:flex; gap:4px;
}}
.map-toggle-btn {{
  padding:3px 10px; font-size:.75em; border:1px solid #ccc;
  background:#f0f4f8; border-radius:4px; cursor:pointer; color:#555;
  transition:all .15s;
}}
.map-toggle-btn.active {{
  background:#2c3e50; color:#fff; border-color:#2c3e50;
}}
#map-img {{
  width:100%; border-radius:5px;
  border:1px solid #ccc; display:block;
  image-rendering: crisp-edges;
}}

/* Metrics row below map — never hidden */
.metrics-panel {{
  display:flex; gap:6px; flex-wrap:wrap;
}}
.metric-card {{
  background:#f7fafd; border-radius:6px; padding:7px 10px;
  border:1px solid #e0eaf6; flex:1 1 100px; min-width:100px;
}}
.metric-val {{
  font-size:1.05em; font-weight:700; line-height:1.3;
}}
.metric-lbl {{
  font-size:.72em; color:#7f8c8d; margin-top:2px;
}}

/* Info column: wraps substrate+depth beside mobile when space allows */
.info-col {{
  flex:1 1 0; display:flex; flex-wrap:nowrap; align-items:stretch;
  border-left:1px solid #e0e8f0; min-width:0;
}}

/* Sub-depth stack (substrate + depth stacked vertically, travel together) */
.sub-depth-stack {{
  flex:1 1 160px; display:flex; flex-direction:column;
  border-right:1px solid #e0e8f0;
}}
.sub-depth-stack .panel + .panel {{
  border-top:1px solid #eef2f7;
}}

.panel {{
  padding:14px 18px; background:#fff;
  flex:1 1 auto;
}}
.panel-title {{
  font-size:.83em; font-weight:700; color:#2c3e50;
  margin-bottom:10px; text-transform:uppercase; letter-spacing:.04em;
  border-bottom:2px solid #e0e8f0; padding-bottom:6px;
}}

/* Mobile panel */
.mobile-panel {{
  flex:1 1 160px; padding:14px 18px; background:#fff;
}}

/* Substrate bars */
.sub-bar-row {{ display:flex; align-items:center; gap:8px;
                margin-bottom:6px; font-size:.84em; }}
.sub-label {{ flex:0 0 120px; text-align:right; color:#555;
              white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis; }}
.sub-track {{ flex:0 0 120px; height:14px; background:#eef2f7; border-radius:3px;
              overflow:hidden; }}
.sub-fill {{ height:100%; border-radius:3px; }}
.sub-pct {{ flex:0 0 42px; font-weight:600; color:#555; font-size:.87em; }}

/* Mobile species table */
.mob-table {{ width:100%; border-collapse:collapse; font-size:.84em; }}
.mob-table th {{
  text-align:left; padding:5px 6px; background:#f0f4f8;
  font-weight:600; color:#555; border-bottom:2px solid #e0e8f0;
}}
.mob-table td {{ padding:4px 6px; border-bottom:1px solid #f0f4f8; }}
.mob-table tr:hover td {{ background:#f7fbff; }}
.mob-pct-cell {{ font-weight:600; white-space:nowrap; }}
.mob-bar-cell {{ width:80px; }}
.mob-bar {{ height:8px; border-radius:3px; background:#c0392b; }}

/* Species list */
.species-section {{ padding:16px 24px; background:#fff; }}
.species-grid {{
  display:grid;
  grid-template-columns:repeat(auto-fill, minmax(250px, 1fr));
  gap:8px;
}}
.sp-card {{
  border-radius:6px; padding:8px 12px;
  border:1px solid #e8f0f8; background:#fafcff;
  font-size:.86em; position:relative;
}}

.sp-header {{ display:flex; justify-content:space-between; align-items:baseline; gap:6px; }}
.sp-name {{ font-style:italic; font-weight:600; color:#2c3e50; }}
.sp-fit {{ font-size:.78em; font-weight:600; color:#7f8c8d; white-space:nowrap; }}
.sp-tags {{ display:flex; gap:4px; flex-wrap:wrap; margin-top:5px; }}
.sp-tag {{
  padding:1px 7px; border-radius:10px; font-size:.78em;
  font-weight:600; color:#fff; white-space:nowrap;
}}
.sp-sal {{ background:#7f8c8d; color:#fff; padding:1px 7px;
           border-radius:10px; font-size:.78em; }}


/* Legend area */
.legend-bar {{
  padding:10px 24px; background:#fafcff; border-top:1px solid #e0e8f0;
  font-size:.8em;
}}
.fg-dot {{ width:10px; height:10px; border-radius:50%;
           display:inline-block; vertical-align:middle; margin-right:4px; }}
.fg-legend-item, .sub-legend-item {{
  display:inline-flex; align-items:center; margin:2px 10px 2px 0;
}}

/* Map image clickable */
#map-img {{ cursor:zoom-in; }}

/* ── Map lightbox ────────────────────────────────────────────────────── */
#map-lightbox {{
  display:none; position:fixed; inset:0;
  background:rgba(0,0,0,.82); z-index:1000;
  align-items:center; justify-content:center; cursor:zoom-out;
}}
#map-lightbox.open {{ display:flex; }}
#map-lightbox img {{
  max-width:94vw; max-height:94vh; display:block;
  border-radius:4px; box-shadow:0 8px 40px rgba(0,0,0,.6);
  cursor:default;
}}
#map-lightbox-close {{
  position:absolute; top:14px; right:18px;
  background:rgba(255,255,255,.15); border:none; color:#fff;
  font-size:1.4em; cursor:pointer; border-radius:4px;
  padding:2px 8px; line-height:1.4;
}}
#map-lightbox-close:hover {{ background:rgba(255,255,255,.3); }}

/* ── SDM image modal ─────────────────────────────────────────────────── */
#sdm-modal {{
  display:none; position:fixed; inset:0;
  background:rgba(0,0,0,.75); z-index:1000;
  align-items:center; justify-content:center;
}}
#sdm-modal.open {{ display:flex; }}
#sdm-modal-box {{
  background:#fff; border-radius:8px; box-shadow:0 8px 40px rgba(0,0,0,.4);
  max-width:92vw; max-height:92vh;
  display:flex; flex-direction:column; overflow:hidden;
}}
#sdm-modal-header {{
  display:flex; align-items:center; gap:12px;
  padding:10px 16px; border-bottom:1px solid #e0e8f0; flex-shrink:0;
  background:#f7fafd;
}}
#sdm-modal-title {{
  font-style:italic; font-weight:700; font-size:1.1em; flex:1; color:#2c3e50;
}}
#sdm-modal-close {{
  background:none; border:none; font-size:1.3em; cursor:pointer;
  color:#7f8c8d; line-height:1; padding:2px 6px; border-radius:4px;
}}
#sdm-modal-close:hover {{ background:#e0e8f0; color:#2c3e50; }}
#sdm-modal-body {{ overflow:auto; display:flex; align-items:center; justify-content:center; }}
#sdm-img {{ max-width:90vw; max-height:calc(90vh - 56px); display:block; }}

/* Dendrogram lightbox */
#dendro-modal {{
  display:none; position:fixed; inset:0;
  background:rgba(0,0,0,.75); z-index:1001;
  align-items:center; justify-content:center;
}}
#dendro-modal.open {{ display:flex; }}
#dendro-modal-box {{
  background:#fff; border-radius:8px; box-shadow:0 8px 40px rgba(0,0,0,.5);
  width:92vw; height:92vh;
  display:flex; flex-direction:column; overflow:hidden;
}}
#dendro-modal-header {{
  display:flex; align-items:center; gap:12px;
  padding:8px 14px; border-bottom:1px solid #e0e8f0;
  background:#f7fafd; flex-shrink:0;
}}
#dendro-modal-title {{
  font-weight:700; font-size:1em; flex:1; color:#2c3e50;
}}
#dendro-hint {{
  font-size:.75em; color:#aaa; white-space:nowrap;
}}
#dendro-modal-close {{
  background:none; border:none; font-size:1.3em; cursor:pointer;
  color:#7f8c8d; line-height:1; padding:2px 6px; border-radius:4px;
}}
#dendro-modal-close:hover {{ background:#e0e8f0; color:#2c3e50; }}
#dendro-viewport {{
  flex:1; overflow:hidden; position:relative;
  cursor:grab; background:#f0f4f8;
}}
#dendro-viewport.dragging {{ cursor:grabbing; }}
#dendro-img {{
  position:absolute; top:0; left:0;
  transform-origin:0 0;
  max-width:none; max-height:none;
  display:block; user-select:none;
  -webkit-user-drag:none;
}}

/* Species / mobile rows that have an SDM */
.sp-card.has-sdm {{ cursor:pointer; }}
.sp-card.has-sdm:hover {{ border-color:#3498db; background:#f0f8ff; }}
.sp-card.is-outlier {{ border-color:#e74c3c; background:#fff5f5; }}
.sp-card.is-outlier:hover {{ border-color:#c0392b; background:#ffe8e8; }}
.mob-has-sdm {{ cursor:pointer; }}
.mob-has-sdm:hover td {{ background:#f0f8ff !important; }}

/* Placeholder when no cluster selected */
#placeholder {{
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; height:100%; color:#aaa; gap:12px;
  font-size:1.1em;
}}
#placeholder svg {{ opacity:.25; }}

/* Scrollbar styling */
#sidebar::-webkit-scrollbar {{ width:5px; }}
#sidebar::-webkit-scrollbar-thumb {{ background:#2c3e50; border-radius:3px; }}

/* Responsive: narrow sidebar on small screens */
@media (max-width:700px) {{
  #sidebar {{ width:180px; min-width:180px; }}
}}
</style>
</head>
<body>
<div id="app">

  <!-- ── SIDEBAR ─────────────────────────────────────────────────────── -->
  <nav id="sidebar">
    <div id="sidebar-header">
      Baltic Habitats
      <small>{len(cluster_data)} clusters · Click to explore</small>
    </div>
    {sidebar_items}
  </nav>

  <!-- ── MAIN ────────────────────────────────────────────────────────── -->
  <div id="main">

    <!-- Cluster header -->
    <div id="cluster-header" style="display:none">
      <div id="cluster-dot"></div>
      <div>
        <div id="cluster-title"></div>
        <div id="cluster-subtitle"></div>
      </div>
    </div>

    <!-- Placeholder (shown before any selection) -->
    <div id="placeholder">
      <svg width="80" height="80" viewBox="0 0 24 24" fill="none"
           stroke="#2c3e50" stroke-width="1">
        <circle cx="12" cy="12" r="10"/>
        <path d="M2 12h20M12 2a15 15 0 010 20M12 2a15 15 0 000 20"/>
      </svg>
      Select a habitat cluster from the sidebar
    </div>

    <!-- Content (hidden until a cluster is selected) -->
    <div id="content" style="display:none">

      <!-- Map | Substrate+Depth | Mobile -->
      <div class="top-row">
        <div class="map-wrap">
          <div class="map-toggle">
            <button class="map-toggle-btn active" id="map-btn-sorensen"
                    onclick="switchMapType('sorensen')">Sørensen</button>
            <button class="map-toggle-btn" id="map-btn-binary"
                    onclick="switchMapType('binary')">Binary</button>
            <button class="map-toggle-btn" id="map-btn-specprop"
                    onclick="switchMapType('specprop')">Species proportion</button>
          </div>
          <img id="map-img" src="" alt="Cluster distribution map"
               title="Click to enlarge  ·  Grey = outside study area  ·  White = absence  ·  Red = presence"
               onclick="openMapLightbox()"/>
          <div class="metrics-panel" id="metrics-panel"></div>
        </div>
        <div class="info-col">
          <div class="sub-depth-stack">
            <div class="panel">
              <div class="panel-title">Substrate composition</div>
              <div id="substrate-chart"></div>
            </div>
            <div class="panel">
              <div class="panel-title">Depth distribution</div>
              <div id="depth-chart"></div>
            </div>
          </div>
          <div class="mobile-panel">
            <div class="panel-title">Mobile species
              <span style="font-weight:400;color:#aaa;font-size:.85em">(top 20 by % overlap)</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:6px 12px;margin-bottom:8px;font-size:.75em">
              {"".join(
                f'<span style="display:inline-flex;align-items:center;gap:4px">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{c};display:inline-block"></span>'
                f'{g.capitalize()}</span>'
                for g, c in MOBILE_GROUP_COLORS.items()
              )}
            </div>
            <div id="mobile-table"></div>
          </div>
        </div>
      </div>

      <!-- Species list -->
      <div class="species-section">
        <div class="panel-title" style="border-bottom:2px solid #e0e8f0;
             padding-bottom:8px;margin-bottom:14px">
          Benthic species in cluster
          <span id="sp-count" style="font-weight:400;color:#aaa;font-size:.85em"></span>
        </div>
        <div id="species-list" class="species-grid"></div>
      </div>

      <!-- Legend -->
      <div class="legend-bar">
        <b>Functional groups:</b>&nbsp;{fg_legend}
        <span style="margin-left:18px;border-left:1px solid #dde;padding-left:18px;">
          <span style="display:inline-block;width:12px;height:12px;border-radius:3px;
                       background:#fff5f5;border:1.5px solid #e74c3c;
                       vertical-align:middle;margin-right:5px;"></span>Species flagged as an outlier through expert evaluation
        </span>
      </div>

    </div><!-- /#content -->
  </div><!-- /#main -->
</div><!-- /#app -->

<!-- Dendrogram lightbox -->
<div id="dendro-modal">
  <div id="dendro-modal-box">
    <div id="dendro-modal-header">
      <div id="dendro-modal-title"></div>
      <span id="dendro-hint">Scroll to zoom · Drag to pan · Double-click to reset</span>
      <button id="dendro-modal-close" onclick="closeDendrogram()" title="Close">✕</button>
    </div>
    <div id="dendro-viewport">
      <img id="dendro-img" src="" alt="Dendrogram" draggable="false"/>
    </div>
  </div>
</div>

<!-- Map lightbox -->
<div id="map-lightbox" onclick="closeMapLightbox()">
  <button id="map-lightbox-close" onclick="closeMapLightbox()" title="Close">✕</button>
  <img src="" alt="Cluster distribution map enlarged" onclick="event.stopPropagation()"/>
</div>

<!-- SDM image modal -->
<div id="sdm-modal">
  <div id="sdm-modal-box">
    <div id="sdm-modal-header">
      <div id="sdm-modal-title"></div>
      <button id="sdm-modal-close" onclick="closeSdmModal()" title="Close">✕</button>
    </div>
    <div id="sdm-modal-body">
      <img id="sdm-img" src="" alt="Species distribution model"/>
    </div>
  </div>
</div>

<script>
{maps_js}
{data_js}
{fg_colors_js}
{sub_colors_js}
{depth_colors_js}
{zone_colors_js}
{mob_group_colors_js}
{dendro_js}
{sdm_js}

const ZONE_LABELS = {{
  "Zone1": "Zone 1 — Low salinity (0–3 PSU)",
  "Zone2": "Zone 2 — Low-intermediate (3–6 PSU)",
  "Zone3": "Zone 3 — Intermediate (6–12 PSU)",
  "Zone4": "Zone 4 — High salinity (12–31 PSU)"
}};

let currentCluster = null;
let activeMapType = 'sorensen';

function switchMapType(type) {{
  activeMapType = type;
  document.querySelectorAll('.map-toggle-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('map-btn-' + type).classList.add('active');
  if (currentCluster) {{
    const maps = activeMapType === 'binary' ? MAPS_BINARY : activeMapType === 'specprop' ? MAPS_SPECPROP : MAPS_SORENSEN;
    const mapImg = document.getElementById('map-img');
    if (maps[currentCluster]) {{
      mapImg.src = maps[currentCluster];
      mapImg.style.display = 'block';
    }} else {{
      mapImg.style.display = 'none';
    }}
  }}
}}

function showCluster(name) {{
  if (currentCluster) {{
    document.getElementById('btn-' + currentCluster)?.classList.remove('active');
  }}
  currentCluster = name;
  document.getElementById('btn-' + name)?.classList.add('active');
  document.getElementById('btn-' + name)?.scrollIntoView({{block:'nearest'}});

  const d = CLUSTER_DATA[name];
  if (!d) return;

  // Show panels
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('content').style.display = 'block';
  document.getElementById('cluster-header').style.display = 'flex';

  // Header
  document.getElementById('cluster-dot').style.background = d.colour;
  document.getElementById('cluster-title').textContent = name;
  const m = d.metrics || {{}};
  document.getElementById('cluster-subtitle').textContent =
    ZONE_LABELS[d.zone] + ' · ' + (m.dom_fg || '').replace(/_/g,' ');

  // Map
  const mapImg = document.getElementById('map-img');
  const activeMaps = activeMapType === 'binary' ? MAPS_BINARY : activeMapType === 'specprop' ? MAPS_SPECPROP : MAPS_SORENSEN;
  if (activeMaps[name]) {{
    mapImg.src = activeMaps[name];
    mapImg.style.display = 'block';
  }} else {{
    mapImg.style.display = 'none';
  }}

  // Metrics panel
  renderMetrics(d.metrics, d.colour);

  // Substrate chart
  renderSubstrate(d.substrate);

  // Mobile species
  renderDepth(d.depth);
  renderMobile(d.mobile, d.colour);

  // Species
  renderSpecies(d.species, d.colour);

  // Scroll main to top
  document.getElementById('main').scrollTop = 0;
}}

function renderMetrics(m, col) {{
  const el = document.getElementById('metrics-panel');
  if (!m) {{ el.innerHTML = ''; return; }}

  function card(label, val) {{
    return `<div class="metric-card">
      <div class="metric-val" style="color:${{col}}">${{val ?? 'N/A'}}</div>
      <div class="metric-lbl">${{label}}</div></div>`;
  }}

  const fg = (m.dom_fg || '').replace(/_/g, ' ');
  const fgPct = m.dom_fg_prop !== null && m.dom_fg_prop !== undefined
    ? ' (' + (m.dom_fg_prop * 100).toFixed(0) + '%)' : '';

  el.innerHTML =
    card('Species in cluster', m.n_species) +
    (fg ? card('Dominant group', fg + fgPct) : '');
}}

// Fixed absolute scale: 100% = MAX_BAR_PX so charts are comparable across clusters
const MAX_BAR_PX = 120;

function renderBarChart(items, elId, colorMap) {{
  const el = document.getElementById(elId);
  if (!items || items.length === 0) {{
    el.innerHTML = '<p style="color:#aaa">No data</p>';
    return;
  }}
  el.innerHTML = items.map(s => {{
    const w = Math.round((s.pct / 100) * MAX_BAR_PX);
    const col = colorMap[s.name] || '#999';
    return `<div class="sub-bar-row">
      <div class="sub-label" title="${{s.name}}">${{s.name}}</div>
      <div class="sub-track" style="width:${{MAX_BAR_PX}}px">
        <div class="sub-fill" style="width:${{w}}px;background:${{col}}"></div>
      </div>
      <div class="sub-pct">${{s.pct > 0 ? s.pct.toFixed(1) + '%' : ''}}</div>
    </div>`;
  }}).join('');
}}

function renderSubstrate(subList) {{ renderBarChart(subList, 'substrate-chart', SUB_COLORS); }}
function renderDepth(depthList)   {{ renderBarChart(depthList, 'depth-chart',     DEPTH_COLORS); }}

function renderMobile(mobList, col) {{
  const el = document.getElementById('mobile-table');
  if (!mobList || mobList.length === 0) {{
    el.innerHTML = '<p style="color:#aaa">No mobile species with &gt;0.5% overlap</p>';
    return;
  }}
  const maxPct = Math.max(...mobList.map(s => s.pct));
  el.innerHTML = `<table class="mob-table">
    <thead><tr><th>Species</th><th style="text-align:right">Overlap %</th>
    <th class="mob-bar-cell"></th></tr></thead>
    <tbody>` +
    mobList.map(s => {{
      const bw = Math.round((s.pct / Math.max(maxPct,1)) * 75);
      const grpCol = (s.group && MOB_GROUP_COLORS[s.group]) ? MOB_GROUP_COLORS[s.group] : col;
      const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{grpCol}};margin-right:5px;vertical-align:middle;flex-shrink:0"></span>`;
      const hasSdm = !!SPECIES_SDM[s.name];
      const cls = hasSdm ? ' class="mob-has-sdm"' : '';
      const click = hasSdm ? `onclick="openSpeciesMap('${{s.name}}')"` : '';
      return `<tr${{cls}} ${{click}}>
        <td>${{dot}}<i>${{s.name}}</i></td>
        <td class="mob-pct-cell" style="text-align:right;color:${{grpCol}}">${{s.pct.toFixed(1)}}%</td>
        <td class="mob-bar-cell">
          <div class="mob-bar" style="width:${{bw}}px;background:${{grpCol}}"></div>
        </td></tr>`;
    }}).join('') + `</tbody></table>`;
}}

function renderSpecies(spList, col) {{
  const el = document.getElementById('species-list');
  const count = document.getElementById('sp-count');
  count.textContent = `— ${{spList.length}} species`;

  el.innerHTML = spList.map(sp => {{
    const fg   = sp.fg || 'unknown';
    const fgCol= FG_COLORS[fg] || '#bdc3c7';
    let cls = 'sp-card';
    const salText = (sp.sal_min !== null && sp.sal_max !== null)
      ? `${{sp.sal_min}}–${{sp.sal_max}} PSU` : '';
    const hasSdm = !!SPECIES_SDM[sp.name];
    if (hasSdm) cls += ' has-sdm';
    if (sp.outlier) cls += ' is-outlier';
    const click = hasSdm ? `onclick="openSpeciesMap('${{sp.name}}')"` : '';
    const fitPct = sp.fit_val != null ? Math.round(sp.fit_val * 100) : null;
    const fitColor = fitPct !== null && fitPct < 10 ? '#e74c3c' : '#7f8c8d';
    const fitTxt = fitPct !== null
      ? `<span class="sp-fit" style="color:${{fitColor}}">${{fitPct}}%</span>`
      : '';
    return `<div class="${{cls}}" ${{click}}>
      <div class="sp-header">
        <div class="sp-name">${{sp.name}}</div>
        ${{fitTxt}}
      </div>
      <div class="sp-tags">
        <span class="sp-tag" style="background:${{fgCol}}">${{fg.replace(/_/g,' ')}}</span>
        ${{sp.depth && sp.depth !== '?' ? `<span class="sp-tag" style="background:#7f8c8d">${{sp.depth}}</span>` : ''}}
        ${{salText ? `<span class="sp-sal">${{salText}}</span>` : ''}}
      </div>
    </div>`;
  }}).join('');
}}

// ── Dendrogram lightbox with zoom + pan ────────────────────────────────────
let dendroScale = 1, dendroX = 0, dendroY = 0;
let _dragging = false, _dx = 0, _dy = 0, _dtx = 0, _dty = 0;

function _dendroTransform() {{
  document.getElementById('dendro-img').style.transform =
    `translate(${{dendroX}}px,${{dendroY}}px) scale(${{dendroScale}})`;
}}

function _resetDendroView() {{
  const vp  = document.getElementById('dendro-viewport');
  const img = document.getElementById('dendro-img');
  // Fit image to viewport on open
  const scaleW = vp.clientWidth  / img.naturalWidth;
  const scaleH = vp.clientHeight / img.naturalHeight;
  dendroScale  = Math.min(scaleW, scaleH, 1);
  dendroX = (vp.clientWidth  - img.naturalWidth  * dendroScale) / 2;
  dendroY = (vp.clientHeight - img.naturalHeight * dendroScale) / 2;
  _dendroTransform();
}}

function openDendrogram(zone) {{
  const src = DENDROGRAMS[zone];
  if (!src) return;
  document.getElementById('dendro-modal-title').textContent =
    zone.replace('Zone', 'Zone ') + ' — Dendrogram';
  const img = document.getElementById('dendro-img');
  img.onload = _resetDendroView;
  img.src = src;
  document.getElementById('dendro-modal').classList.add('open');
}}
function closeDendrogram() {{
  document.getElementById('dendro-modal').classList.remove('open');
}}
document.getElementById('dendro-modal').addEventListener('click', function(e) {{
  if (e.target === this) closeDendrogram();
}});

// Wheel zoom around cursor
const _vp = document.getElementById('dendro-viewport');
_vp.addEventListener('wheel', e => {{
  e.preventDefault();
  const r  = _vp.getBoundingClientRect();
  const cx = e.clientX - r.left;
  const cy = e.clientY - r.top;
  const factor   = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newScale = Math.max(0.1, Math.min(20, dendroScale * factor));
  dendroX = cx - (cx - dendroX) * (newScale / dendroScale);
  dendroY = cy - (cy - dendroY) * (newScale / dendroScale);
  dendroScale = newScale;
  _dendroTransform();
}}, {{ passive: false }});

// Drag to pan
_vp.addEventListener('mousedown', e => {{
  _dragging = true; _dx = e.clientX; _dy = e.clientY;
  _dtx = dendroX; _dty = dendroY;
  _vp.classList.add('dragging');
  e.preventDefault();
}});
document.addEventListener('mousemove', e => {{
  if (!_dragging) return;
  dendroX = _dtx + (e.clientX - _dx);
  dendroY = _dty + (e.clientY - _dy);
  _dendroTransform();
}});
document.addEventListener('mouseup', () => {{
  _dragging = false;
  document.getElementById('dendro-viewport').classList.remove('dragging');
}});

// Double-click to reset
_vp.addEventListener('dblclick', _resetDendroView);

// ── Map lightbox ───────────────────────────────────────────────────────────
function openMapLightbox() {{
  const src = document.getElementById('map-img').src;
  if (!src) return;
  document.querySelector('#map-lightbox img').src = src;
  document.getElementById('map-lightbox').classList.add('open');
}}
function closeMapLightbox() {{
  document.getElementById('map-lightbox').classList.remove('open');
}}

// ── SDM image modal ────────────────────────────────────────────────────────
function openSpeciesMap(name) {{
  const uri = SPECIES_SDM[name];
  if (!uri) return;
  document.getElementById('sdm-modal-title').textContent = name;
  document.getElementById('sdm-img').src = uri;
  document.getElementById('sdm-modal').classList.add('open');
}}
function closeSdmModal() {{
  document.getElementById('sdm-modal').classList.remove('open');
  document.getElementById('sdm-img').src = '';
}}
document.getElementById('sdm-modal').addEventListener('click', function(e) {{
  if (e.target === this) closeSdmModal();
}});

// Keyboard: up/down arrows to navigate clusters, Escape to close modal
const ALL_CLUSTERS = { json.dumps(CLUSTER_ORDER) };
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{ closeMapLightbox(); closeSdmModal(); closeDendrogram(); return; }}
  if (!currentCluster) return;
  const idx = ALL_CLUSTERS.indexOf(currentCluster);
  if (e.key === 'ArrowDown' && idx < ALL_CLUSTERS.length - 1)
    showCluster(ALL_CLUSTERS[idx + 1]);
  else if (e.key === 'ArrowUp' && idx > 0)
    showCluster(ALL_CLUSTERS[idx - 1]);
}});

window.addEventListener('load', () => showCluster(ALL_CLUSTERS[0]));
</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Resolve run directory from CLI argument ────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: generate_dashboard.py <run_folder_name>")
        print("  e.g. generate_dashboard.py run1_4manSalinityZones_10perc_binStrictLVL")
        sys.exit(1)

    run_arg = sys.argv[1]
    run_dir = Path(run_arg) if Path(run_arg).is_absolute() else DATA / run_arg
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    _set_run_paths(run_dir)
    run_name = run_dir.name

    print(f"=== Baltic Sea Habitat Cluster Dashboard — {run_name} ===\n")

    print("Loading data …")
    species_df  = load_species_df()
    global CLUSTER_ORDER
    CLUSTER_ORDER = _derive_cluster_order(species_df)
    print(f"  {len(CLUSTER_ORDER)} clusters detected across {len(set(c.split('_')[0] for c in CLUSTER_ORDER))} zones")

    traits_df   = load_traits()
    substrate_df= load_substrate()
    depth_df    = load_depth()
    mobile_df   = load_mobile()
    metrics_df  = load_metrics()
    coeff_df    = load_coefficients()
    outlier_set = load_outliers()

    # Cluster → colour
    colour_map = species_df.groupby("cluster_name")["colour"].first().to_dict()

    print("Loading binary PNG maps …")
    map_b64_binary = {}
    for cname in CLUSTER_ORDER:
        b64 = png_to_b64(cname)
        map_b64_binary[cname] = b64
        print(f"  {cname}: {'ok' if b64 else 'MISSING'}")

    print("Loading Sørensen PNG maps …")
    sorensen_dir = RUN_DIR / "_pngs" / "individ_cluster_sorensen_png"
    map_b64_sorensen = {}
    for cname in CLUSTER_ORDER:
        b64 = png_to_b64(cname, sorensen_dir)
        map_b64_sorensen[cname] = b64
        print(f"  {cname}: {'ok' if b64 else 'MISSING'}")

    print("Loading species proportion PNG maps …")
    specprop_dir = RUN_DIR / "_pngs" / "individ_cluster_specprop_png"
    map_b64_specprop = {}
    for cname in CLUSTER_ORDER:
        b64 = png_to_b64(cname, specprop_dir)
        map_b64_specprop[cname] = b64
        print(f"  {cname}: {'ok' if b64 else 'MISSING'}")

    print("\nLoading SDM images …")
    all_species = species_df["species"].unique().tolist()
    sdm_images = load_sdm_images(all_species)
    print(f"  {len(sdm_images)} / {len(all_species)} species have SDMs")

    print("Building cluster data …")
    cluster_data = build_cluster_data(
        species_df, traits_df, substrate_df, depth_df, mobile_df, metrics_df, coeff_df, outlier_set
    )

    print("Generating HTML …")
    html = build_html(cluster_data, map_b64_binary, map_b64_sorensen, map_b64_specprop, sdm_images)

    out = OUT_REP / f"{run_name}.html"
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"\nDone.  Output: {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
