"""
Urban House Sparrow Distribution Model — Albuquerque
====================================================
A domain-specific MECHANISTIC species-distribution model (SDM), built to
demonstrate: training a model on real ecological data, grounded in the
mechanism of how an urban-adapted bird responds to environmental drivers.

Pipeline (the whole loop, end to end):
  1. FETCH real House Sparrow (Passer domesticus) occurrences from GBIF
     for the Albuquerque area  (open API, no key needed).
  2. GENERATE pseudo-absence points (standard SDM practice — where the
     bird was NOT recorded) so the model has something to contrast against.
  3. BUILD environmental drivers at each point. These are the MECHANISM —
     the ecological literature says urban sparrows track:
        - vegetation greenness (NDVI proxy)
        - impervious surface / built density
        - distance to buildings/structures (they nest in human structures)
        - temperature
     Here we synthesize plausible driver surfaces from geography so the
     pipeline runs anywhere; swap in real rasters (NDVI, NLCD, PRISM) to
     make it publication-grade.
  4. TRAIN a model to predict presence from drivers, and REPORT which
     drivers matter and in which direction — the mechanistic interpretation,
     not just a prediction.
  5. RENDER a habitat-suitability map across Albuquerque (Folium HTML).

Run:
    pip3 install geopandas scikit-learn folium requests rasterio numpy pandas
    python3 sparrow_model.py
    open sparrow_suitability_map.html

Swap-in points for real data are marked  # REAL DATA:
"""

import io
import json
import numpy as np
import pandas as pd
import requests

# Albuquerque bounding box (roughly the metro area)
ABQ = {"min_lat": 34.95, "max_lat": 35.25, "min_lon": -106.75, "max_lon": -106.45}
CITY_CENTER = (35.0844, -106.6504)   # downtown ABQ
RANDOM_SEED = 42

# Target-group background sampling (Phillips et al. 2009).
# True  = draw absence points from where OTHER BIRDS were recorded, so that
#         observer effort appears in both classes and cancels out.
# False = old behaviour, uniform random points across the bbox.
# Flip this to compare the two; the direction table will show how much moved.
USE_TARGET_GROUP = True
TARGET_GROUP_TAXON = 212          # GBIF key for class Aves (all birds)
HOUSE_SPARROW_TAXON = 5231190     # excluded from the target group


# ---------------------------------------------------------------- 1. FETCH
# --- seasons: month -> season label. ABQ has strong seasonality. ---
SEASONS = {"winter": [12, 1, 2], "spring": [3, 4, 5],
           "summer": [6, 7, 8], "fall": [9, 10, 11]}
MONTH_TO_SEASON = {m: s for s, months in SEASONS.items() for m in months}

# ---- Seasonal color palettes: each season tells its own visual story ----
# Each has 3 suitability classes (marginal -> suitable -> prime) + a dot color.
SEASON_PALETTE = {
    "winter": {"marginal": "#cfe3f2", "suitable": "#7fb2e0", "prime": "#2c6fb5",
               "dot": "#0b1d3a", "accent": "#2c6fb5", "name": "Winter"},
    "spring": {"marginal": "#e5a83b", "suitable": "#7cc96a", "prime": "#159c3c",
               "dot": "#123", "accent": "#159c3c", "name": "Spring"},
    "summer": {"marginal": "#ffe08a", "suitable": "#ffab4d", "prime": "#e8600f",
               "dot": "#5a1a00", "accent": "#e8600f", "name": "Summer"},
    "fall":   {"marginal": "#e8c39a", "suitable": "#c8863b", "prime": "#8a3b1e",
               "dot": "#2b1005", "accent": "#8a3b1e", "name": "Fall"},
}
MONTH_TO_SEASON = MONTH_TO_SEASON if "MONTH_TO_SEASON" in dir() else None

def _season_of_month(m):
    for s, months in SEASONS.items():
        if m in months:
            return s
    return "spring"




def fetch_sparrow_occurrences(per_month=300):
    """Real House Sparrow occurrences from GBIF, sampled MONTH BY MONTH so the
    whole calendar year is represented.

    Returns (points, year_span) where:
      points    = list of (lat, lon, month, year)
      year_span = (min_year, max_year) across all returned records, so we can
                  honestly label which years of sightings are on the map.
    """
    url = "https://api.gbif.org/v1/occurrence/search"
    print("Fetching House Sparrow occurrences from GBIF, month by month "
          "(spreads sampling across the whole year)...")
    all_pts = []
    years = []
    try:
        for mo in range(1, 13):
            params = {
                "taxonKey": 5231190,
                "decimalLatitude": f"{ABQ['min_lat']},{ABQ['max_lat']}",
                "decimalLongitude": f"{ABQ['min_lon']},{ABQ['max_lon']}",
                "hasCoordinate": "true",
                "month": mo,
                "limit": per_month,
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            for rec in data.get("results", []):
                la, lo = rec.get("decimalLatitude"), rec.get("decimalLongitude")
                rmo, ryr = rec.get("month"), rec.get("year")
                if la and lo and rmo:
                    yr = int(ryr) if ryr else None
                    all_pts.append((la, lo, int(rmo), yr))
                    if yr:
                        years.append(yr)
        all_pts = list({p for p in all_pts})
        year_span = (min(years), max(years)) if years else (None, None)
        by_mo = {m: 0 for m in range(1, 13)}
        for _, _, m, _ in all_pts:
            by_mo[m] += 1
        spread = " ".join(f"{m:02d}:{by_mo[m]}" for m in range(1, 13))
        yr_txt = (f"{year_span[0]}\u2013{year_span[1]}" if year_span[0] else "n/a")
        print(f"  got {len(all_pts)} dated points, sighting years {yr_txt}")
        print(f"  monthly spread  {spread}")
        return all_pts, year_span
    except Exception as e:
        print(f"  GBIF fetch failed ({e}); synthetic dated points.")
        rng = np.random.default_rng(RANDOM_SEED)
        base = _synthetic_presence(600)
        pts = [(la, lo, (i % 12) + 1, 2020 + (i % 5))
               for i, (la, lo) in enumerate(base)]
        return pts, (2020, 2024)


def fetch_ebird_occurrences(days_back=30, radius_km=50):
    """Recent House Sparrow observations from the eBird API 2.0.
    Reads the key from the EBIRD_API_KEY environment variable (or a local
    .env file) so the key NEVER lives in this shareable code.

    eBird species code for House Sparrow is 'houspa'.
    Returns list of (lat, lon). Empty list if no key or fetch fails —
    the model still runs on GBIF alone.
    """
    import os
    key = os.environ.get("EBIRD_API_KEY")
    if not key:
        # try a local .env file (KEY=value lines)
        try:
            with open(".env") as f:
                for line in f:
                    if line.startswith("EBIRD_API_KEY="):
                        key = line.strip().split("=", 1)[1]
        except FileNotFoundError:
            pass
    if not key:
        print("  no EBIRD_API_KEY found (.env or env var) — skipping eBird layer.")
        return []

    # eBird 'recent nearby observations' endpoint, centered on ABQ
    url = "https://api.ebird.org/v2/data/obs/geo/recent/houspa"
    params = {"lat": CITY_CENTER[0], "lng": CITY_CENTER[1],
              "dist": radius_km, "back": days_back, "maxResults": 1000}
    print("Fetching House Sparrow observations from eBird...")
    try:
        r = requests.get(url, params=params,
                         headers={"X-eBirdApiToken": key}, timeout=30)
        r.raise_for_status()
        recs = r.json()
        import datetime
        # eBird 'recent' obs carry an obsDt like '2026-07-20 08:15'; parse month.
        pts = []
        for rec in recs:
            if rec.get("lat") and rec.get("lng"):
                mo = None
                dt = rec.get("obsDt", "")
                try:
                    mo = int(dt.split("-")[1])
                except Exception:
                    mo = datetime.date.today().month
                pts.append((rec["lat"], rec["lng"], mo))
        pts = list({p for p in pts})
        if not pts:
            print(f"  eBird returned 0 House Sparrow reports in the last "
                  f"{days_back}d / {radius_km}km. House Sparrows are common "
                  f"enough that birders often skip logging them \u2014 this is "
                  f"expected, not an error. GBIF layer covers it.")
        else:
            print(f"  got {len(pts)} eBird observation points "
                  f"(last {days_back} days, {radius_km}km radius)")
        return pts
    except Exception as e:
        print(f"  eBird fetch failed ({e}); continuing with GBIF only.")
        return []


def fetch_target_group(per_month=300):
    """Occurrence records for ALL OTHER BIRDS in the same box, month by month.

    This is the heart of target-group background sampling. The problem with
    random background points is that they represent places nobody ever looked,
    so the model can learn "birders go here" and call it "sparrows live here."

    Records of other bird species mark places where somebody WAS looking and
    did not report a house sparrow. Using those as the background means
    observer effort is present in both classes and largely divides out,
    leaving habitat preference behind.

    Returns {month: [(lat, lon), ...]} with house sparrow records removed.
    """
    url = "https://api.gbif.org/v1/occurrence/search"
    print("\nFetching TARGET GROUP (all other birds) from GBIF, month by month...")
    print("  these become the background points, replacing random ones")
    pool = {m: [] for m in range(1, 13)}
    try:
        for mo in range(1, 13):
            params = {
                "taxonKey": TARGET_GROUP_TAXON,
                "decimalLatitude": f"{ABQ['min_lat']},{ABQ['max_lat']}",
                "decimalLongitude": f"{ABQ['min_lon']},{ABQ['max_lon']}",
                "hasCoordinate": "true",
                "month": mo,
                "limit": per_month,
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            for rec in r.json().get("results", []):
                la, lo = rec.get("decimalLatitude"), rec.get("decimalLongitude")
                # drop house sparrows -- they are the presence class, not background
                if rec.get("speciesKey") == HOUSE_SPARROW_TAXON:
                    continue
                if "Passer domesticus" in str(rec.get("scientificName", "")):
                    continue
                if la and lo:
                    pool[mo].append((la, lo))
            pool[mo] = list({p for p in pool[mo]})
        spread = " ".join(f"{m:02d}:{len(pool[m])}" for m in range(1, 13))
        total = sum(len(v) for v in pool.values())
        print(f"  got {total} target-group points")
        print(f"  monthly spread  {spread}")
        return pool
    except Exception as e:
        print(f"  target-group fetch failed ({e}); falling back to random background.")
        return {m: [] for m in range(1, 13)}


def _synthetic_presence(n=300):
    """Fallback if offline: sparrows cluster near the urban center."""
    rng = np.random.default_rng(RANDOM_SEED)
    lat = rng.normal(CITY_CENTER[0], 0.05, n).clip(ABQ["min_lat"], ABQ["max_lat"])
    lon = rng.normal(CITY_CENTER[1], 0.06, n).clip(ABQ["min_lon"], ABQ["max_lon"])
    return list(zip(lat, lon))


# ------------------------------------------------------- 2. PSEUDO-ABSENCE
def make_pseudo_absence(presence, n=None, pool=None, label=""):
    """Background points -- the 'available but not occupied' class.

    TWO MODES:

    1. TARGET GROUP (preferred, used when `pool` has enough points).
       Background is drawn from recorded sightings of OTHER bird species.
       Those are places a person demonstrably visited and looked at birds
       without reporting a house sparrow. Because both classes now carry the
       same observer-effort signal, the model can no longer win by learning
       where birders go -- that cancels -- and what is left is habitat.
       Phillips et al. 2009, Ecological Applications 19(1).

    2. UNIFORM RANDOM (fallback). Points scattered across the bounding box,
       including places nobody has ever surveyed. Simple, standard, and
       silently conflates "no sparrow here" with "nobody looked here."

    Points coinciding with presence records are removed from the pool first.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = n or len(presence)

    if USE_TARGET_GROUP and pool:
        pres_keys = {(round(la, 4), round(lo, 4)) for la, lo in presence}
        cand = [p for p in pool
                if (round(p[0], 4), round(p[1], 4)) not in pres_keys]
        # need a usable pool; too few and the background is degenerate
        if len(cand) >= max(30, n // 4):
            take = min(n, len(cand))
            idx = rng.choice(len(cand), size=take, replace=False)
            picked = [cand[i] for i in idx]
            if label:
                print(f"    background [{label}]: {len(picked)} target-group pts "
                      f"(from {len(cand)} available)")
            return picked
        if label:
            print(f"    background [{label}]: only {len(cand)} target-group pts "
                  f"-- falling back to random")

    lat = rng.uniform(ABQ["min_lat"], ABQ["max_lat"], n)
    lon = rng.uniform(ABQ["min_lon"], ABQ["max_lon"], n)
    return list(zip(lat, lon))


# ------------------------------------------------------------ 3. DRIVERS
def _dist(lat, lon, ref):
    return np.sqrt((lat - ref[0]) ** 2 + (lon - ref[1]) ** 2)


def environmental_drivers(lat, lon):
    """The MECHANISM. Return the driver values at a point.

    Grounded in urban-sparrow ecology:
      - built_density  : sparrows are human commensals; nest in structures.
                         Peaks near the urban core, falls toward desert edge.
      - ndvi           : they need some vegetation (seeds, insects) but not
                         dense wildland — an intermediate optimum.
      - impervious     : too much pavement (pure downtown) reduces foraging.
      - temperature    : warmer near the built core (urban heat island).

    # REAL DATA: replace each of these with a raster sample:
    #   built_density <- NLCD impervious / building footprints
    #   ndvi          <- Sentinel-2 / Landsat NDVI
    #   temperature   <- PRISM or Landsat thermal
    """
    d_core = _dist(lat, lon, CITY_CENTER)
    # built density: high near center, decays outward (0..1)
    built = np.exp(-(d_core / 0.10) ** 2)
    # NDVI: modest near the river corridor (the Rio Grande runs ~ -106.68),
    # low in pure downtown and in far desert — intermediate optimum
    river_lon = -106.68
    d_river = np.abs(lon - river_lon)
    ndvi = 0.25 + 0.5 * np.exp(-(d_river / 0.04) ** 2)
    # impervious: very high only at the dense core
    impervious = np.exp(-(d_core / 0.05) ** 2)
    # temperature: urban heat island — warmer near built core
    temp = 20 + 8 * built + np.random.default_rng(int((lat*1000+lon*1000)) % 2**31).normal(0, 0.5)
    ndvi_v = float(np.atleast_1d(ndvi)[0])
    built_v = float(np.atleast_1d(built)[0])
    imperv_v = float(np.atleast_1d(impervious)[0])
    temp_v = float(np.atleast_1d(temp)[0])
    # synthetic proxies for the extra indices (real Earth Engine overrides these)
    river_lon = -106.68
    d_river = abs(lon - river_lon)
    ndwi_v = float(0.4 * np.exp(-((d_river) / 0.03) ** 2) - 0.1)
    ndbi_v = float(imperv_v * 0.8 - 0.1)
    ndre_v = ndvi_v * 0.8
    savi_v = ndvi_v * 1.2
    return {"built_density": built_v, "impervious": imperv_v,
            "temperature": temp_v, "ndvi": ndvi_v, "ndwi": ndwi_v,
            "ndbi": ndbi_v, "ndre": ndre_v, "savi": savi_v}



# ============================ EARTH ENGINE REAL DRIVERS =====================
# Set USE_EARTH_ENGINE = True to pull REAL satellite data instead of the
# synthetic driver surfaces. Requires: earthengine-api, an authenticated
# session (python3 -c "import ee; ee.Authenticate()"), and your EE project.
USE_EARTH_ENGINE = True
EE_PROJECT = "sparrow-sdm"
DRIVER_YEAR = 2024   # satellite imagery year for the animation frames

_EE_READY = False
_EE_SEASON_IMAGES = {}     # season -> ee.Image (cached)
_EE_STATIC = None          # built_density + impervious (season-invariant)

# Sentinel-2 / Landsat date ranges per season (2024)
SEASON_DATES = {
    "winter": ("2023-12-01", "2024-03-01"),
    "spring": ("2024-03-01", "2024-06-01"),
    "summer": ("2024-06-01", "2024-09-01"),
    "fall":   ("2024-09-01", "2024-12-01"),
}


def _init_earth_engine():
    global _EE_READY, _EE_STATIC
    if _EE_READY:
        return True
    try:
        import ee
        ee.Initialize(project=EE_PROJECT)
        # season-invariant layers: built density + impervious from NLCD
        nlcd_img = (ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD")
                    .filter(ee.Filter.eq("system:index", "2021")).first())
        impervious = nlcd_img.select("impervious").rename("impervious").divide(100)
        landcover = nlcd_img.select("landcover")
        built = (landcover.gte(21).And(landcover.lte(24))
                 .rename("built_density").toFloat())
        _EE_STATIC = ee.Image.cat([built, impervious])
        _EE_READY = True
        print("  Earth Engine initialised — seasonal real drivers "
              "(S2 NDVI + Landsat LST per season, NLCD static).")
        return True
    except Exception as e:
        print(f"  Earth Engine init failed ({e}); falling back to synthetic drivers.")
        return False


def _spectral_indices(img):
    """Compute five spectral indices from a Sentinel-2 median image. Each is a
    different physical 'lens' on the surface. Sentinel-2 bands used:
      B3=green, B4=red, B5=red-edge, B8=NIR, B11=shortwave-IR.
    """
    import ee
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")   # greenness
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("ndwi")   # water/moisture
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("ndbi")  # built-up
    ndre = img.normalizedDifference(["B8", "B5"]).rename("ndre")   # veg health (red-edge)
    # SAVI: soil-adjusted vegetation (L=0.5 for intermediate cover)
    savi = img.expression(
        "1.5 * (NIR - RED) / (NIR + RED + 0.5)",
        {"NIR": img.select("B8"), "RED": img.select("B4")}).rename("savi")
    return ee.Image.cat([ndvi, ndwi, ndbi, ndre, savi])


def _month_image(month):
    """Driver image for a specific calendar MONTH of DRIVER_YEAR: that month's
    Sentinel-2 NDVI + Landsat LST, plus the static NLCD built/impervious.
    Cached in _EE_SEASON_IMAGES under key 'm{month}'."""
    import ee
    key = f"m{month}"
    if key in _EE_SEASON_IMAGES:
        return _EE_SEASON_IMAGES[key]
    region = ee.Geometry.Rectangle(
        [ABQ["min_lon"], ABQ["min_lat"], ABQ["max_lon"], ABQ["max_lat"]])
    # a ~2-month window centered on the target month, to get enough clear scenes
    start = ee.Date.fromYMD(DRIVER_YEAR, month, 1).advance(-15, "day")
    end = ee.Date.fromYMD(DRIVER_YEAR, month, 1).advance(45, "day")

    def mask_s2(img):
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(cloud).divide(10000)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)).map(mask_s2))
    s2med = s2.median()
    indices = _spectral_indices(s2med)      # the five-lens feature stack

    def lst_c(img):
        return (img.select("ST_B10").multiply(0.00341802).add(149.0)
                .subtract(273.15).rename("temperature"))
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(region).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUD_COVER", 50)))
    temp = l8.map(lst_c).select("temperature").median()

    built = _EE_STATIC.select("built_density")
    imperv = _EE_STATIC.select("impervious")
    img = ee.Image.cat([built, imperv, temp, indices])
    _EE_SEASON_IMAGES[key] = img
    return img


def _season_image(season):
    """Build (and cache) the full driver image for one season: seasonal NDVI +
    seasonal temperature + the static built/impervious layers."""
    import ee
    if season in _EE_SEASON_IMAGES:
        return _EE_SEASON_IMAGES[season]
    region = ee.Geometry.Rectangle(
        [ABQ["min_lon"], ABQ["min_lat"], ABQ["max_lon"], ABQ["max_lat"]])
    d0, d1 = SEASON_DATES[season]

    def mask_s2(img):
        qa = img.select("QA60")
        cloud = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(cloud).divide(10000)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(d0, d1)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask_s2))
    indices = _spectral_indices(s2.median())

    def lst_c(img):
        return (img.select("ST_B10").multiply(0.00341802).add(149.0)
                .subtract(273.15).rename("temperature"))
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(region).filterDate(d0, d1)
          .filter(ee.Filter.lt("CLOUD_COVER", 40)))
    temp = l8.map(lst_c).select("temperature").median()

    built = _EE_STATIC.select("built_density")
    imperv = _EE_STATIC.select("impervious")
    img = ee.Image.cat([built, imperv, temp, indices])
    _EE_SEASON_IMAGES[season] = img
    return img



def ee_sample_points_month(latlons, month):
    """Sample the given MONTH's real-driver image at each (lat,lon)."""
    import ee
    if not _init_earth_engine():
        return [environmental_drivers(la, lo) for la, lo in latlons]
    image = _month_image(month)
    feats = [ee.Feature(ee.Geometry.Point([lo, la]), {"idx": i})
             for i, (la, lo) in enumerate(latlons)]
    fc = ee.FeatureCollection(feats)
    sampled = image.reduceRegions(
        collection=fc, reducer=ee.Reducer.first(), scale=30)
    rows = sampled.getInfo()["features"]
    out = [None] * len(latlons)
    for f in rows:
        p = f["properties"]
        i = int(p.get("idx", 0))
        out[i] = {k: float(p.get(k) if p.get(k) is not None
                            else (20.0 if k == "temperature" else 0.0))
                  for k in DRIVERS}
    for i, (la, lo) in enumerate(latlons):
        if out[i] is None:
            out[i] = environmental_drivers(la, lo)
    return out


def ee_sample_points(latlons, season):
    """Sample the given SEASON's real-driver image at each (lat,lon).
    Falls back to synthetic if EE unavailable."""
    import ee
    if not _init_earth_engine():
        return [environmental_drivers(la, lo) for la, lo in latlons]
    image = _season_image(season)
    feats = [ee.Feature(ee.Geometry.Point([lo, la]), {"idx": i})
             for i, (la, lo) in enumerate(latlons)]
    fc = ee.FeatureCollection(feats)
    sampled = image.reduceRegions(
        collection=fc, reducer=ee.Reducer.first(), scale=30)
    rows = sampled.getInfo()["features"]
    out = [None] * len(latlons)
    for f in rows:
        p = f["properties"]
        i = int(p.get("idx", 0))
        out[i] = {k: float(p.get(k) if p.get(k) is not None
                            else (20.0 if k == "temperature" else 0.0))
                  for k in DRIVERS}
    for i, (la, lo) in enumerate(latlons):
        if out[i] is None:
            out[i] = environmental_drivers(la, lo)
    return out


def build_frame_for_season(presence, absence, season):
    """presence/absence are (lat,lon) lists already filtered to this season."""
    if USE_EARTH_ENGINE:
        pres_d = ee_sample_points(presence, season)
        abs_d = ee_sample_points(absence, season)
    else:
        pres_d = [environmental_drivers(la, lo) for la, lo in presence]
        abs_d = [environmental_drivers(la, lo) for la, lo in absence]
    rows = []
    for (lat, lon), d in zip(presence, pres_d):
        d = dict(d); d.update(lat=lat, lon=lon, present=1); rows.append(d)
    for (lat, lon), d in zip(absence, abs_d):
        d = dict(d); d.update(lat=lat, lon=lon, present=0); rows.append(d)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- 4. TRAIN
# Full feature set: NLCD structure + Landsat temp + FIVE Sentinel-2 spectral
# indices, each a different "lens" on the landscape. The model's feature
# importance will tell us which lens actually predicts sparrow habitat.
DRIVERS = ["built_density", "impervious", "temperature",
           "ndvi",   # greenness (NIR-Red)      -> vegetation presence
           "ndwi",   # water/moisture (Green-NIR) -> river, ponds, irrigation
           "ndbi",   # built-up (SWIR-NIR)       -> impervious from spectra
           "ndre",   # red-edge (NIR-RedEdge)    -> vegetation health/stress
           "savi"]   # soil-adjusted vegetation  -> better in sparse desert


def direction_report(df, label="", verbose=True):
    """Which WAY does each driver push? Importance says how much a variable
    matters; this says whether the birds sit at HIGHER or LOWER values of it.

    Compares each driver at presence points vs. pseudo-absence points and
    returns a signed standardized effect (Cohen's d):
        d > 0  -> sparrows are at HIGHER values than background
        d < 0  -> sparrows are at LOWER values than background
        |d| < 0.2 negligible | ~0.5 moderate | > 0.8 large
    """
    pres = df[df["present"] == 1]
    absn = df[df["present"] == 0]
    rows = {}
    for k in DRIVERS:
        p, a = pres[k], absn[k]
        sp = np.sqrt(((len(p) - 1) * p.var(ddof=1) +
                      (len(a) - 1) * a.var(ddof=1)) /
                     max(len(p) + len(a) - 2, 1))
        d = float((p.mean() - a.mean()) / sp) if sp > 0 else 0.0
        rows[k] = {"presence": float(p.mean()),
                   "background": float(a.mean()), "d": d}
    if verbose:
        top = sorted(rows.items(), key=lambda r: -abs(r[1]["d"]))[:3]
        bits = ", ".join(f"{k} {'+' if v['d'] > 0 else '-'}{abs(v['d']):.2f}"
                         for k, v in top)
        print(f"    direction [{label}]: {bits}")
    return rows


def train(df, label=""):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X, y = df[DRIVERS].values, df["present"].values
    if len(set(y)) < 2 or len(df) < 20:
        print(f"  [{label}] too few points to train ({len(df)}); skipping.")
        return None, None, None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                          random_state=RANDOM_SEED, stratify=y)
    model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   random_state=RANDOM_SEED)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    importances = dict(zip(DRIVERS, model.feature_importances_))
    ranked = sorted(importances.items(), key=lambda x: -x[1])
    top3 = ", ".join(f"{k} {v:.2f}" for k, v in ranked[:3])
    print(f"  [{label}] {len(df)} pts · AUC {auc:.3f} · top: {top3}")
    return model, auc, importances


# -------------------------------------------------------------- 5. MAP
def render_seasonal_map(season_models, season_presence, grid_n=45):
    """One map, four toggleable seasonal suitability layers + observations.
    Each season's grid is scored with that season's model and drivers."""
    import folium
    lats = np.linspace(ABQ["min_lat"], ABQ["max_lat"], grid_n)
    lons = np.linspace(ABQ["min_lon"], ABQ["max_lon"], grid_n)
    grid_pts = [(la, lo) for la in lats for lo in lons]
    dlat = (ABQ["max_lat"] - ABQ["min_lat"]) / grid_n
    dlon = (ABQ["max_lon"] - ABQ["min_lon"]) / grid_n

    m = folium.Map(location=CITY_CENTER, zoom_start=11, tiles="CartoDB positron")

    season_order = ["winter", "spring", "summer", "fall"]
    for si, season in enumerate(season_order):
        model = season_models.get(season)
        if model is None:
            continue
        # sample this season's drivers across the grid, score suitability
        if USE_EARTH_ENGINE:
            print(f"  scoring {season} grid ({len(grid_pts)} cells)...")
            grid_d = ee_sample_points(grid_pts, season)
        else:
            grid_d = [environmental_drivers(la, lo) for la, lo in grid_pts]
        probs = model.predict_proba(
            [[d[k] for k in DRIVERS] for d in grid_d])[:, 1]

        # only first season shown by default; others toggle on
        layer = folium.FeatureGroup(name=f"{season.title()} suitability",
                                    show=(si == 0))
        for (la, lo), p in zip(grid_pts, probs):
            # discrete, high-contrast classes for legibility (your ask)
            if p < 0.35:      col, op = "#3b1f2b", 0.0     # unsuitable: transparent
            elif p < 0.55:    col, op = "#e5a83b", 0.35    # marginal: amber
            elif p < 0.75:    col, op = "#7cc96a", 0.5     # suitable: light green
            else:             col, op = "#159c3c", 0.62    # prime: strong green
            if op == 0:
                continue
            folium.Rectangle(
                bounds=[[la, lo], [la + dlat, lo + dlon]],
                color=None, fill=True, fill_color=col, fill_opacity=op,
            ).add_to(layer)
        layer.add_to(m)

    # observation points per season as their own toggleable layers
    colors = {"winter": "#3d6fb0", "spring": "#2fbf8f",
              "summer": "#e4574f", "fall": "#c8863b"}
    for season in season_order:
        pts = season_presence.get(season, [])
        if not pts:
            continue
        olayer = folium.FeatureGroup(name=f"{season.title()} sightings",
                                     show=False)
        for la, lo in pts:
            folium.CircleMarker([la, lo], radius=2, color=colors[season],
                                fill=True, fill_opacity=0.7,
                                popup=f"House Sparrow · {season}").add_to(olayer)
        olayer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend = ('<div style="position:fixed;top:150px;left:50px;z-index:9999;'
              'background:white;padding:10px 14px;border-radius:6px;'
              'font-family:sans-serif;font-size:12px;box-shadow:0 1px 4px #0003">'
              '<b>Predicted suitability</b><br>'
              '<span style="color:#159c3c">\u25a0</span> prime &nbsp;'
              '<span style="color:#7cc96a">\u25a0</span> suitable &nbsp;'
              '<span style="color:#e5a83b">\u25a0</span> marginal<br>'
              '<span style="font-size:10px;color:#666">unsuitable = transparent</span>'
              '</div>')
    m.get_root().html.add_child(folium.Element(legend))





    title = ('<div style="position:fixed;top:10px;left:50px;z-index:9999;'
             'background:white;padding:8px 14px;border-radius:6px;'
             'font-family:sans-serif;font-size:13px;box-shadow:0 1px 4px #0003">'
             '<b>Urban House Sparrow \u2014 Seasonal Habitat Suitability</b><br>'
             'Albuquerque \u00b7 toggle seasons top-right to see the shift<br>'
             '<span style="font-size:11px;color:#555">Drivers: seasonal Sentinel-2 '
             'NDVI + Landsat LST + NLCD. Data: GBIF + eBird/Cornell Lab.</span></div>')
    m.get_root().html.add_child(folium.Element(title))
    m.save("sparrow_seasonal_map.html")
    print("\nWrote sparrow_seasonal_map.html")


# --------------------------------------------------------------- MAIN


def _left_rail_html(season_importances, note="", initial_season="spring"):
    """Return ONE left-rail overlay containing the suitability key and the
    feature-importance chart, stacked in a flex column.

    Because both cards are children of a single positioned container, their
    left edges align by construction and the chart can never overlap the key,
    no matter how tall either card gets.

    The key is seasonal: SEASON_PALETTE changes the map colors per season, so
    a fixed green/amber key would be wrong for winter (blue) and summer
    (orange). window.updateImpChart(season) repaints BOTH cards, so the
    existing animation watcher and layer-toggle hook drive it unchanged.
    """
    import json
    labels = {"built_density": "built density (NLCD)",
              "impervious": "impervious (NLCD)",
              "temperature": "land surface temp",
              "ndvi": "NDVI \u2014 greenness",
              "ndwi": "NDWI \u2014 water",
              "ndbi": "NDBI \u2014 built-up",
              "ndre": "NDRE \u2014 veg health",
              "savi": "SAVI \u2014 soil-adj veg"}
    order = list(DRIVERS)
    data = {s: {k: float(imp.get(k, 0)) for k in order}
            for s, imp in season_importances.items()}
    payload = json.dumps({"data": data,
                          "labels": {k: labels[k] for k in order},
                          "order": order,
                          "palette": SEASON_PALETTE})

    html = """
    <div id="sdmRail" style="position:fixed;top:150px;left:50px;z-index:9998;
         width:250px;max-height:calc(100vh - 250px);overflow-y:auto;
         display:flex;flex-direction:column;gap:10px;
         font-family:sans-serif;">

      <div style="background:rgba(255,255,255,.96);padding:10px 12px;
           border-radius:8px;box-shadow:0 1px 5px #0003;">
        <div style="font-size:12px;font-weight:700;margin-bottom:7px;">
          Predicted suitability &mdash; <span id="keySeason">Spring</span></div>
        <div id="keyRows"></div>
        <div style="font-size:9px;color:#888;margin-top:7px;line-height:1.35;">
          __NOTE__</div>
      </div>

      <div style="background:rgba(255,255,255,.96);padding:10px 12px;
           border-radius:8px;box-shadow:0 1px 5px #0003;">
        <div id="impTitle" style="font-size:12px;font-weight:700;margin-bottom:6px;">
          Feature importance</div>
        <div id="impBars"></div>
        <div style="font-size:9px;color:#888;margin-top:6px;line-height:1.35;">
          Which satellite lens the model leans on. Bars shift by season &mdash;
          correlated features share importance, so read broad patterns.</div>
      </div>

    </div>
    <script>
    (function(){
      var IMP = __PAYLOAD__;

      function cap(s){ return s.charAt(0).toUpperCase() + s.slice(1); }

      function paintKey(season){
        var pal = IMP.palette[season];
        if(!pal) return;
        var rows = [["prime","prime"],["suitable","suitable"],
                    ["marginal","marginal"]];
        var html = "";
        rows.forEach(function(r){
          html += '<div style="display:flex;align-items:center;gap:7px;' +
                  'margin:3px 0;font-size:11px;">' +
                  '<span style="width:13px;height:13px;border-radius:3px;' +
                  'display:inline-block;background:' + pal[r[0]] + ';"></span>' +
                  '<span>' + r[1] + '</span></div>';
        });
        html += '<div style="display:flex;align-items:center;gap:7px;' +
                'margin:5px 0 0;font-size:11px;">' +
                '<span style="width:9px;height:9px;border-radius:50%;' +
                'margin:0 2px;display:inline-block;background:' + pal.dot +
                ';"></span><span>sightings</span></div>';
        document.getElementById("keyRows").innerHTML = html;
        document.getElementById("keySeason").textContent = cap(season);
      }

      function paintBars(season){
        var d = IMP.data[season];
        var title = document.getElementById("impTitle");
        title.textContent = "Feature importance \u2014 " + cap(season);
        if(!d){
          document.getElementById("impBars").innerHTML =
            '<div style="font-size:10px;color:#999;padding:4px 0;">' +
            'Too few sightings this season to fit a model.</div>';
          return;
        }
        var order = IMP.order.slice().sort(function(a,b){ return d[b]-d[a]; });
        var maxv = Math.max.apply(null, order.map(function(k){return d[k];})) || 1;
        var accent = (IMP.palette[season] || {}).accent || "#159c3c";
        var html = "";
        order.forEach(function(k){
          var w = Math.round(100 * d[k] / maxv);
          html += '<div style="margin:3px 0;font-size:10px;">' +
                  '<div style="display:flex;justify-content:space-between;">' +
                  '<span>' + IMP.labels[k] + '</span>' +
                  '<span style="color:#666;">' + d[k].toFixed(3) + '</span></div>' +
                  '<div style="background:#eee;border-radius:3px;height:7px;">' +
                  '<div style="width:' + w + '%;height:7px;border-radius:3px;' +
                  'transition:width .25s ease;background:' + accent +
                  ';"></div></div></div>';
        });
        document.getElementById("impBars").innerHTML = html;
      }

      // same name the animation watcher and the layer-toggle hook already call
      window.updateImpChart = function(season){
        paintKey(season);
        paintBars(season);
      };

      window.addEventListener("load", function(){
        setTimeout(function(){ window.updateImpChart("__INIT__"); }, 300);
      });
    })();
    </script>
    """
    return (html.replace("__PAYLOAD__", payload)
                .replace("__INIT__", initial_season)
                .replace("__NOTE__", note))


def render_monthly_animation(month_models, month_presence, year_span,
                             season_importances, grid_n=38):
    """A TRUE monthly time-series animation via TimestampedGeoJson.
    Each frame = one calendar month of DRIVER_YEAR: that month's sightings and
    that month's satellite-derived suitability, stamped with a real date so the
    play-slider shows exactly which month is displayed.

    year_span = (min_year, max_year) of the SIGHTINGS, stated on the map so the
    provenance is explicit (sightings pooled across these years by month;
    suitability drivers from DRIVER_YEAR imagery).
    """
    import folium
    from folium.plugins import TimestampedGeoJson

    lats = np.linspace(ABQ["min_lat"], ABQ["max_lat"], grid_n)
    lons = np.linspace(ABQ["min_lon"], ABQ["max_lon"], grid_n)
    grid_pts = [(la, lo) for la in lats for lo in lons]
    dlat = (ABQ["max_lat"] - ABQ["min_lat"]) / grid_n
    dlon = (ABQ["max_lon"] - ABQ["min_lon"]) / grid_n

    features = []
    for month in range(1, 13):
        model = month_models.get(month)
        if model is None:
            continue
        stamp = f"{DRIVER_YEAR}-{month:02d}-01"

        # suitability cells for this month
        if USE_EARTH_ENGINE:
            print(f"  scoring {DRIVER_YEAR}-{month:02d} grid...")
            grid_d = ee_sample_points_month(grid_pts, month)
        else:
            grid_d = [environmental_drivers(la, lo) for la, lo in grid_pts]
        probs = model.predict_proba(
            [[d[k] for k in DRIVERS] for d in grid_d])[:, 1]

        pal = SEASON_PALETTE[_season_of_month(month)]
        for (la, lo), p in zip(grid_pts, probs):
            if p < 0.45:
                continue                       # unsuitable: omit for clarity
            if p < 0.6:    col = pal["marginal"]
            elif p < 0.78: col = pal["suitable"]
            else:          col = pal["prime"]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lo, la], [lo + dlon, la], [lo + dlon, la + dlat],
                    [lo, la + dlat], [lo, la]]]},
                "properties": {
                    "time": stamp,
                    "style": {"color": col, "fillColor": col,
                              "fillOpacity": 0.55, "weight": 0},
                },
            })

        # sightings for this month (as small points, same timestamp)
        for la, lo in month_presence.get(month, []):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lo, la]},
                "properties": {
                    "time": stamp,
                    "icon": "circle",
                    "iconstyle": {"fillColor": pal["dot"], "fillOpacity": 0.85,
                                  "stroke": "false", "radius": 3},
                },
            })

    m = folium.Map(location=CITY_CENTER, zoom_start=11, tiles="CartoDB positron")

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="P1M", duration="P1M", transition_time=600,
        auto_play=False, loop=True, add_last_point=False,
        date_options="YYYY-MM",
    ).add_to(m)

    yr_txt = (f"{year_span[0]}\u2013{year_span[1]}"
              if year_span[0] else "multiple years")
    # legend now lives inside the left rail (added below)




    title = (f'<div style="position:fixed;top:10px;left:50px;z-index:9999;'
             f'background:white;padding:8px 14px;border-radius:6px;'
             f'font-family:sans-serif;font-size:13px;box-shadow:0 1px 4px #0003">'
             f'<b>Urban House Sparrow \u2014 Monthly Habitat Suitability '
             f'({DRIVER_YEAR})</b><br>'
             f'Albuquerque \u00b7 press play to animate through the year<br>'
             f'<span style="font-size:11px;color:#555">'
             f'Suitability drivers: {DRIVER_YEAR} monthly Sentinel-2 NDVI + '
             f'Landsat LST + NLCD. &nbsp;'
             f'Sightings: GBIF + eBird, pooled by month across {yr_txt}.'
             f'</span></div>')
    m.get_root().html.add_child(folium.Element(title))
    # left rail: suitability key + feature-importance chart, stacked.
    # Both repaint per-season as the animation plays.
    m.get_root().html.add_child(folium.Element(_left_rail_html(
        season_importances,
        note="Dark dots = sightings that month &nbsp;\u00b7&nbsp; "
             "press play at the bottom of the map.",
        initial_season="winter")))
    watch = """
    <script>
    (function(){
      function monthToSeason(mo){
        if([12,1,2].indexOf(mo)>=0) return "winter";
        if([3,4,5].indexOf(mo)>=0) return "spring";
        if([6,7,8].indexOf(mo)>=0) return "summer";
        return "fall";
      }
      var last = null;
      function poll(){
        // the TimeDimension current date shows in the timecontrol display
        var el = document.querySelector('.leaflet-control-timecontrol.timecontrol-date');
        if(el){
          var t = el.textContent || el.innerText || "";
          var m = t.match(/-(\\d{2})$/) || t.match(/-(\\d{2})-/);
          if(m){
            var mo = parseInt(m[1],10);
            var s = monthToSeason(mo);
            if(s !== last && window.updateImpChart){ window.updateImpChart(s); last = s; }
          }
        }
        requestAnimationFrame(poll);
      }
      window.addEventListener("load", function(){ setTimeout(poll, 800); });
    })();
    </script>
    """
    m.get_root().html.add_child(folium.Element(watch))
    m.save("sparrow_animation.html")
    print("\nWrote sparrow_animation.html")




def render_seasonal_compare(season_models, season_presence, season_importances, grid_n=45):
    """SEPARATE, pure seasonal map: four toggleable static seasonal suitability
    layers + per-season sightings. No animation — definitely-works checkboxes."""
    import folium
    lats = np.linspace(ABQ["min_lat"], ABQ["max_lat"], grid_n)
    lons = np.linspace(ABQ["min_lon"], ABQ["max_lon"], grid_n)
    grid_pts = [(la, lo) for la in lats for lo in lons]
    dlat = (ABQ["max_lat"] - ABQ["min_lat"]) / grid_n
    dlon = (ABQ["max_lon"] - ABQ["min_lon"]) / grid_n

    m = folium.Map(location=CITY_CENTER, zoom_start=11, tiles="CartoDB positron")
    order = ["winter", "spring", "summer", "fall"]
    for si, season in enumerate(order):
        model = season_models.get(season)
        if model is None:
            continue
        if USE_EARTH_ENGINE:
            print(f"  scoring {season} layer...")
            gd = ee_sample_points(grid_pts, season)
        else:
            gd = [environmental_drivers(la, lo) for la, lo in grid_pts]
        probs = model.predict_proba([[d[k] for k in DRIVERS] for d in gd])[:, 1]
        pal = SEASON_PALETTE[season]
        layer = folium.FeatureGroup(name=f"{pal['name']} suitability",
                                    show=(si == 0))
        for (la, lo), p in zip(grid_pts, probs):
            if p < 0.45:
                continue
            if p < 0.6:    col = pal["marginal"]
            elif p < 0.78: col = pal["suitable"]
            else:          col = pal["prime"]
            folium.Rectangle(bounds=[[la, lo], [la + dlat, lo + dlon]],
                             color=None, fill=True, fill_color=col,
                             fill_opacity=0.55).add_to(layer)
        layer.add_to(m)

    for season in order:
        pts = season_presence.get(season, [])
        if not pts:
            continue
        pal = SEASON_PALETTE[season]
        ol = folium.FeatureGroup(name=f"{pal['name']} sightings", show=False)
        for la, lo in pts:
            folium.CircleMarker([la, lo], radius=2, color=pal["dot"],
                                fill=True, fill_color=pal["dot"],
                                fill_opacity=0.8).add_to(ol)
        ol.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    # legend now lives inside the left rail (added below)
    title = ('<div style="position:fixed;top:10px;left:50px;z-index:9999;'
             'background:white;padding:8px 14px;border-radius:6px;'
             'font-family:sans-serif;font-size:13px;box-shadow:0 1px 4px #0003">'
             '<b>Urban House Sparrow \u2014 Seasonal Comparison</b><br>'
             'Albuquerque \u00b7 toggle seasons (top-right) to compare<br>'
             '<span style="font-size:11px;color:#555">Drivers: seasonal Sentinel-2 '
             'NDVI + Landsat LST + NLCD. Data: GBIF + eBird.</span></div>')
    m.get_root().html.add_child(folium.Element(title))
    # updating feature-importance chart (starts on the default-shown season)
    first_season = next((s for s in ["winter","spring","summer","fall"]
                         if s in season_models), "spring")
    m.get_root().html.add_child(folium.Element(_left_rail_html(
        season_importances,
        note="Check one season at a time (top-right) to compare.",
        initial_season=first_season)))
    # JS: when a season layer checkbox is toggled on, update the chart to it
    hook = """
    <script>
    (function(){
      function wire(){
        var names = {"Winter suitability":"winter","Spring suitability":"spring",
                     "Summer suitability":"summer","Fall suitability":"fall"};
        document.querySelectorAll('.leaflet-control-layers-overlays label')
          .forEach(function(lab){
            var txt = (lab.textContent||"").trim();
            if(names[txt]){
              var cb = lab.querySelector('input');
              if(cb){ cb.addEventListener('change', function(){
                if(cb.checked && window.updateImpChart) window.updateImpChart(names[txt]);
              }); }
            }
          });
      }
      window.addEventListener("load", function(){ setTimeout(wire, 500); });
    })();
    </script>
    """
    m.get_root().html.add_child(folium.Element(hook))
    m.save("sparrow_seasonal.html")
    print("Wrote sparrow_seasonal.html")


def main():
    raw, year_span = fetch_sparrow_occurrences()   # (lat,lon,month,year), span
    ebird = fetch_ebird_occurrences()              # (lat,lon,month)
    tg_pool = (fetch_target_group() if USE_TARGET_GROUP
               else {m: [] for m in range(1, 13)})
    print(f"\n{len(raw)} dated GBIF points + {len(ebird)} eBird points")
    if year_span[0]:
        print(f"Sighting years span {year_span[0]}\u2013{year_span[1]}; "
              f"suitability imagery from {DRIVER_YEAR}.")

    # group presence by calendar MONTH (pooled across sighting years)
    month_presence = {m: [] for m in range(1, 13)}
    for la, lo, mo, yr in raw:
        month_presence[mo].append((la, lo))
    for la, lo, mo in ebird:
        month_presence[mo].append((la, lo))
    for m in range(1, 13):
        month_presence[m] = list(set(month_presence[m]))

    print("\nMonthly sighting counts:")
    counts = " ".join(f"{m:02d}:{len(month_presence[m])}" for m in range(1, 13))
    print(f"  {counts}")

    # train one model per month on that month's real drivers
    print(f"\nTraining per-month models on {DRIVER_YEAR} monthly drivers...")
    month_models = {}
    _all_importances = []
    _all_directions = []
    for m in range(1, 13):
        pres = month_presence[m]
        if len(pres) < 12:
            print(f"  [{DRIVER_YEAR}-{m:02d}] only {len(pres)} sightings — skip.")
            continue
        absence = make_pseudo_absence(pres, pool=tg_pool.get(m),
                                      label=f"{DRIVER_YEAR}-{m:02d}")
        if USE_EARTH_ENGINE:
            pres_d = ee_sample_points_month(pres, m)
            abs_d = ee_sample_points_month(absence, m)
        else:
            pres_d = [environmental_drivers(la, lo) for la, lo in pres]
            abs_d = [environmental_drivers(la, lo) for la, lo in absence]
        rows = []
        for (la, lo), d in zip(pres, pres_d):
            d = dict(d); d.update(present=1); rows.append(d)
        for (la, lo), d in zip(absence, abs_d):
            d = dict(d); d.update(present=0); rows.append(d)
        df = pd.DataFrame(rows)
        model, auc, imp = train(df, label=f"{DRIVER_YEAR}-{m:02d}")
        if model is not None:
            month_models[m] = model
            _all_importances.append(imp)
            _all_directions.append((m, direction_report(df, label=f"{DRIVER_YEAR}-{m:02d}")))

    if not month_models:
        print("No month had enough sightings to model.")
        return

    # === WHICH SPECTRAL LENS WINS? aggregate importance across all months ===
    if _all_importances:
        agg = {k: np.mean([imp[k] for imp in _all_importances]) for k in DRIVERS}
        print("\n" + "=" * 56)
        print("WHICH FEATURE PREDICTS SPARROW HABITAT? (avg across months)")
        print("=" * 56)
        labels = {"built_density": "built density (NLCD)",
                  "impervious": "impervious surface (NLCD)",
                  "temperature": "land surface temp (Landsat)",
                  "ndvi": "NDVI  \u2014 greenness",
                  "ndwi": "NDWI  \u2014 water/moisture",
                  "ndbi": "NDBI  \u2014 built-up (spectral)",
                  "ndre": "NDRE  \u2014 vegetation health",
                  "savi": "SAVI  \u2014 soil-adjusted veg"}
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            bar = "\u2588" * int(v * 60)
            print(f"  {labels.get(k, k):<32} {v:.3f} {bar}")
        winner = max(agg, key=agg.get)
        print(f"\n  \u2192 Strongest predictor: {labels.get(winner, winner)}")

    # === AND IN WHICH DIRECTION? averaged across months ===
    _bg_mode = ("TARGET-GROUP (other birds' records)" if USE_TARGET_GROUP
                else "UNIFORM RANDOM across the bbox")
    print(f"\n  [background points: {_bg_mode}]")
    if _all_directions:
        print("\n" + "=" * 66)
        print("DO SPARROWS SIT AT HIGH OR LOW VALUES? (avg across months)")
        print("=" * 66)
        print(f"  {'driver':<26} {'at sparrows':>12} {'background':>12} {'effect':>8}")
        agg_d = {}
        for k in DRIVERS:
            agg_d[k] = {
                "presence":   np.mean([r[k]["presence"] for _, r in _all_directions]),
                "background": np.mean([r[k]["background"] for _, r in _all_directions]),
                "d":          np.mean([r[k]["d"] for _, r in _all_directions]),
            }
        for k, v in sorted(agg_d.items(), key=lambda x: -abs(x[1]["d"])):
            d = v["d"]
            word = "HIGHER" if d > 0 else "LOWER "
            mag = ("large " if abs(d) > 0.8 else
                   "moderate" if abs(d) > 0.5 else
                   "small " if abs(d) > 0.2 else "none  ")
            print(f"  {labels.get(k, k):<26} {v['presence']:>12.3f} "
                  f"{v['background']:>12.3f} {d:>+8.2f}   {word} ({mag})")
        print("\n  Read this as: sparrow locations sit at HIGHER/LOWER values of each")
        print("  driver than the background points do. This is the DIRECTION that")
        print("  feature importance alone cannot tell you.")
        print("  NOTE: a flat effect can still hide a preferred BAND (suitable in the")
        print("  middle, poor at both ends). Partial dependence plots show that shape.")

    # === DOES THE DIRECTION FLIP BY SEASON? ===
    # An annual average hides seasonal reversal: a bird that seeks warmth in
    # January and shade in July averages out to roughly nothing. This splits
    # the same numbers by season so a reversal becomes visible.
    if _all_directions:
        print("\n" + "=" * 66)
        print("DOES THE DIRECTION FLIP BY SEASON?  (effect size per season)")
        print("=" * 66)
        order = ["winter", "spring", "summer", "fall"]
        by_season = {s: [] for s in order}
        for mo, rep in _all_directions:
            by_season[MONTH_TO_SEASON[mo]].append(rep)

        head = "  {:<26}".format("driver") + "".join(f"{s:>9}" for s in order)
        print(head)
        for k in sorted(DRIVERS,
                        key=lambda kk: -max(
                            abs(np.mean([r[kk]["d"] for r in by_season[s]]))
                            for s in order if by_season[s])):
            cells = ""
            vals = []
            for s in order:
                if by_season[s]:
                    v = float(np.mean([r[k]["d"] for r in by_season[s]]))
                    vals.append(v)
                    cells += f"{v:>+9.2f}"
                else:
                    cells += f"{'--':>9}"
            flip = "   <-- FLIPS" if vals and min(vals) < -0.15 < 0.15 < max(vals) else ""
            print(f"  {labels.get(k, k):<26}{cells}{flip}")
        print("\n  A row with both + and - values reverses across the year.")
        print("  That is a seasonal behaviour change, not a fixed preference,")
        print("  and the annual average above will have hidden most of it.")
        print("    (This is the ecological finding: the spectral 'lens' the")
        print("     sparrows respond to most, learned from the data.)")

    # also build 4 SEASONAL models (pooled months) for the toggle layers
    print("\nTraining seasonal models (for the static toggle layers)...")
    season_presence = {s: [] for s in SEASONS}
    for m in range(1, 13):
        season_presence[MONTH_TO_SEASON[m]].extend(month_presence[m])
    season_models = {}
    season_importances = {}
    for s in ["winter", "spring", "summer", "fall"]:
        pres = list(set(season_presence[s]))
        season_presence[s] = pres
        if len(pres) < 15:
            continue
        season_pool = []
        for _m in SEASONS[s]:
            season_pool.extend(tg_pool.get(_m, []))
        season_pool = list({p for p in season_pool})
        absence = make_pseudo_absence(pres, pool=season_pool, label=s)
        df = build_frame_for_season(pres, absence, s)
        model, auc, imp = train(df, label=s)
        if model is not None:
            season_models[s] = model
            season_importances[s] = imp

    render_monthly_animation(month_models, month_presence, year_span,
                             season_importances)
    render_seasonal_compare(season_models, season_presence, season_importances)
    print("\nDone. Two maps written:")
    print("  open sparrow_animation.html   \u2014 press play, animate through the year")
    print("  open sparrow_seasonal.html    \u2014 toggle seasons to compare (no animation)")


if __name__ == "__main__":
    main()
