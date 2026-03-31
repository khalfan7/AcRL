"""
EPA PRZM Meteorological Hourly Data Loader
===========================================
Parses .hNN fixed-width files (EPA/HYDRO-PRZM format, verified column offsets):

    Date         cols  2-11  (yyyy-mm-dd)            Python [1:11]
    Hour         cols 12-14  (1 = 00:xx … 24 = 23:xx; 25 = daily total → skip)
    GHI          cols 30-34  (Global Horizontal Radiation, Wh/m²)   [29:34]
    T_out (dry)  cols 65-69  (Dry-bulb temperature, °C)             [64:69]
    wind_speed   cols 96-100 (Wind speed @ 10 m, m/s)               [95:100]

Missing values are represented as '---' in the source file; these are
forward-filled then back-filled to remove gaps.

Usage
-----
    from envs.weather import load_hnn, load_hnn_multi

    df = load_hnn("w14771.h89")
    df = load_hnn_multi(["w14771.h89", "w14771.h90"])
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Regex: strips trailing flag characters (letters, ?, !, *) from a field value
_FLAG_RE = re.compile(r"[A-Za-z?!*]+$")


def _field(line: str, start: int, end: int) -> float:
    """
    Extract a numeric value from the fixed-width field line[start:end] (0-based).
    Returns np.nan for missing ('---') or unparseable values.
    """
    raw = line[start:end].strip()
    if not raw or "---" in raw:
        return np.nan
    raw = _FLAG_RE.sub("", raw).strip()
    if not raw:
        return np.nan
    try:
        return float(raw)
    except ValueError:
        return np.nan


def load_hnn(path: str | Path) -> pd.DataFrame:
    """
    Parse one EPA .hNN hourly file.

    Returns a DataFrame sorted by datetime with columns:
        datetime    — Python datetime object (EPA hour 1 → 00:00 local)
        T_out       — Dry-bulb temperature (°C)
        GHI         — Global Horizontal Radiation (Wh/m²), clamped ≥ 0
        wind_speed  — Wind speed at 10 m (m/s), clamped ≥ 0
        month       — 1–12
        doy         — Day-of-year 1–366
        hour_of_day — 0–23
    """
    path = Path(path)
    records: list[dict] = []
    first_data = True  # first non-comment line is the station header

    with path.open("r", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.lstrip()

            # Skip blank lines and comment lines
            if not stripped or stripped.startswith("!"):
                continue

            # Station header — skip once
            if first_data:
                first_data = False
                continue

            # Need at least 100 characters to reach the wind-speed field
            if len(line) < 100:
                continue

            # Hour: cols 12-14 (1-indexed) → Python [11:14]
            try:
                hour_epa = int(line[11:14].strip())
            except ValueError:
                continue

            if hour_epa == 25:       # daily summary row — skip
                continue
            if not (1 <= hour_epa <= 24):
                continue

            # Date: cols 2-11 → Python [1:11]  format yyyy-mm-dd
            try:
                d = datetime.date.fromisoformat(line[1:11].strip())
            except ValueError:
                continue

            # EPA hour 1 → 00:00, hour 24 → 23:00
            hour_0 = hour_epa - 1
            ts = datetime.datetime(d.year, d.month, d.day, hour_0)

            ghi   = _field(line, 29, 34)   # Global Horizontal Radiation
            t_out = _field(line, 64, 69)   # Dry-bulb temperature
            wind  = _field(line, 95, 100)  # Wind speed

            records.append({
                "datetime":    ts,
                "T_out":       t_out,
                "GHI":         ghi,
                "wind_speed":  wind,
                "month":       d.month,
                "doy":         d.timetuple().tm_yday,
                "hour_of_day": hour_0,
            })

    if not records:
        raise ValueError(f"No valid hourly records parsed from {path}")

    df = (
        pd.DataFrame(records)
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Forward-fill then back-fill small gaps from missing-value records
    for col in ("T_out", "GHI", "wind_speed"):
        df[col] = df[col].ffill().bfill()

    # Physical bounds
    df["GHI"]        = df["GHI"].clip(lower=0.0)
    df["wind_speed"] = df["wind_speed"].clip(lower=0.0)

    return df


def load_hnn_multi(paths: Iterable[str | Path]) -> pd.DataFrame:
    """
    Load and concatenate multiple .hNN files into a single sorted DataFrame.

    Example
    -------
        df = load_hnn_multi(["w14771.h89", "w14771.h90"])
    """
    frames = [load_hnn(p) for p in paths]
    if not frames:
        raise ValueError("No files provided to load_hnn_multi")
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
