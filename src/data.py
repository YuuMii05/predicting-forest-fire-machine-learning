"""Build the Tunisian forest-fire dataset.

Two public sources are combined, for 7 forest zones of northern and central
Tunisia, every day from 1 June to 30 September, 2013-2025:

* Weather: Open-Meteo historical weather API (ERA5 reanalysis), daily
  maximum temperature, minimum relative humidity, maximum wind speed and
  total rainfall  -> data/raw/tunisie/meteo_tunisie.csv
* Fires: NASA FIRMS active-fire detections from the VIIRS S-NPP satellite
  (375 m resolution), one CSV per year for Tunisia
  -> data/raw/tunisie/viirs-snpp_<year>_Tunisia.csv

A zone-day is labelled "fire" (1) when the satellite detected at least one
vegetation fire (type 0, nominal or high confidence) within ~25 km of the
zone centre that day. The 6 indices of the Canadian Fire Weather Index
system are computed from the weather with src/fwi.py.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .fwi import compute_fwi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "tunisie"
CLEAN_PATH = ROOT / "data" / "processed" / "tunisie_feux.csv"

# Forest zones, in the same order as the Open-Meteo request (location_id 0..6)
ZONES = [
    # name,          governorate, lat,   lon
    ("Aïn Draham",   "Jendouba",  36.78, 8.69),
    ("Nefza",        "Béja",      36.95, 9.08),
    ("Sejnane",      "Bizerte",   37.06, 9.24),
    ("Kesra",        "Siliana",   35.85, 9.30),
    ("Le Kef",       "Le Kef",    36.20, 8.70),
    ("Cap Bon",      "Nabeul",    36.95, 10.95),
    ("Chaambi",      "Kasserine", 35.20, 8.65),
]
RADIUS_DEG = 0.25           # ~25 km around each zone centre
SEASON_MONTHS = [6, 7, 8, 9]
YEARS = range(2013, 2026)

WEATHER_FEATURES = ["Temperature", "RH", "Ws", "Rain"]
FWI_FEATURES = ["FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
ALL_FEATURES = WEATHER_FEATURES + FWI_FEATURES
TARGET = "fire"


def load_weather(path: Path = RAW / "meteo_tunisie.csv") -> pd.DataFrame:
    """The Open-Meteo CSV has a metadata block, a blank line, then the data."""
    text = path.read_text(encoding="utf-8")
    data_part = text.split("\n\n", 1)[1]
    w = pd.read_csv(io.StringIO(data_part))
    w.columns = ["location_id", "date", "Temperature", "RH", "Ws", "Rain"]
    w["date"] = pd.to_datetime(w["date"])
    w["zone"] = w["location_id"].map({i: z[0] for i, z in enumerate(ZONES)})
    assert w[WEATHER_FEATURES].notna().all().all(), "missing weather values"
    return w.drop(columns="location_id")


def load_fires() -> pd.DataFrame:
    files = sorted(RAW.glob("viirs-snpp_*_Tunisia.csv"))
    assert len(files) == len(YEARS), f"expected {len(YEARS)} FIRMS files, found {len(files)}"
    f = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    f["date"] = pd.to_datetime(f["acq_date"])
    # type 0 = presumed vegetation fire (2 = other static land source, e.g. gas flares; 3 = offshore)
    # confidence: l = low, n = nominal, h = high
    return f[(f["type"] == 0) & (f["confidence"].isin(["n", "h"]))]


def assign_zone(fires: pd.DataFrame) -> pd.DataFrame:
    """Each detection goes to the nearest zone centre, if within RADIUS_DEG."""
    lat = fires["latitude"].to_numpy()[:, None]
    lon = fires["longitude"].to_numpy()[:, None]
    zlat = np.array([z[2] for z in ZONES])[None, :]
    zlon = np.array([z[3] for z in ZONES])[None, :]
    # degrees of longitude are shorter than degrees of latitude at 36°N
    dist = np.sqrt((lat - zlat) ** 2 + ((lon - zlon) * np.cos(np.radians(36.5))) ** 2)
    nearest = dist.argmin(axis=1)
    ok = dist.min(axis=1) <= RADIUS_DEG
    out = fires.loc[ok].copy()
    out["zone"] = [ZONES[i][0] for i in nearest[ok]]
    return out


def build() -> pd.DataFrame:
    weather = load_weather()
    # FWI codes need the days before the season: computed on the full series, per zone
    weather = pd.concat([compute_fwi(g.sort_values("date")) for _, g in weather.groupby("zone", sort=False)])

    fires = assign_zone(load_fires())
    fire_days = (fires.groupby(["zone", "date"]).agg(n_detections=("frp", "size"), frp_total=("frp", "sum"))
                 .reset_index())

    df = weather.merge(fire_days, on=["zone", "date"], how="left")
    df = df[df["date"].dt.month.isin(SEASON_MONTHS) & df["date"].dt.year.isin(YEARS)].copy()
    df["n_detections"] = df["n_detections"].fillna(0).astype(int)
    df["frp_total"] = df["frp_total"].fillna(0.0).round(1)
    df[TARGET] = (df["n_detections"] > 0).astype(int)
    df["governorate"] = df["zone"].map({z[0]: z[1] for z in ZONES})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    cols = ["date", "year", "month", "zone", "governorate"] + ALL_FEATURES + ["n_detections", "frp_total", TARGET]
    df = df[cols].sort_values(["zone", "date"]).reset_index(drop=True)

    expected = len(ZONES) * len(YEARS) * 122
    assert len(df) == expected, (len(df), expected)
    assert not df[ALL_FEATURES].isna().any().any()
    return df


def load_clean(save: bool = True) -> pd.DataFrame:
    if CLEAN_PATH.exists() and not save:
        return pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    df = build()
    if save:
        CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_PATH, index=False)
    return df


if __name__ == "__main__":
    d = load_clean()
    print(d.shape, f"{d[TARGET].mean():.1%} de jours avec feu")
    print(d.groupby("zone")[TARGET].agg(["sum", "mean"]))
    print(d.groupby("year")[TARGET].sum())
