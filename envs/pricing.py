"""
Residential electricity price loader
=====================================
Parses the EPA/EIA monthly residential rate table (newyork_monthly.txt format):

    <title line>
    <units line>
    <blank line(s)>
    <year header:  2025   2024   2023 ...>
    January        25.3   23.5   23.6 ...
    February       26.2   24.3   24.2 ...
    ...

Returns a 12-element NumPy array (¢/kWh) indexed 0 = January … 11 = December.

Notes
-----
Some months in the source file span two physical lines when older years are
present (the table wraps at column 80).  Because 2024 is always in column 1
(i.e., the second value after the month name on the first line), this parser
correctly extracts 2024 values for all 12 months.

Falls back to the most-recent available year if the requested year is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def load_monthly_prices(path: str | Path, year: int = 2025) -> np.ndarray:
    """
    Load 12 monthly electricity prices (¢/kWh) for the requested year.

    Parameters
    ----------
    path : str | Path
        Path to the price table (e.g. Data/Pricing/newyork_monthly.txt).
    year : int
        Calendar year to extract.  Falls back to the most-recent year present
        if ``year`` is not found.

    Returns
    -------
    prices : np.ndarray, shape (12,)
        prices[0] = January price, prices[11] = December price.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # --- Locate the header row (a line whose tokens are all 4-digit years) ---
    header_idx: int | None = None
    years_avail: list[int] = []

    for i, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) < 2:
            continue
        try:
            candidates = [int(t) for t in tokens if len(t) == 4]
            if len(candidates) >= 2 and all(1990 < y < 2040 for y in candidates):
                header_idx = i
                years_avail = candidates
                break
        except ValueError:
            continue

    if header_idx is None or not years_avail:
        raise ValueError(f"Could not locate year header in {path}")

    # --- Choose the column index within the data rows ---
    year_col = years_avail.index(year) if year in years_avail else 0

    # --- Parse monthly value rows ---
    prices = np.full(12, np.nan)
    month_to_idx = {m: i for i, m in enumerate(_MONTHS)}

    for line in lines[header_idx + 1 :]:
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0].lower()
        if key not in month_to_idx:
            continue
        data = tokens[1:]           # strip month name
        if year_col >= len(data):
            continue
        try:
            prices[month_to_idx[key]] = float(data[year_col])
        except ValueError:
            pass

    # Fill any NaN with the column mean (e.g. missing rows)
    if np.any(np.isnan(prices)):
        prices[np.isnan(prices)] = np.nanmean(prices)

    return prices   # ¢/kWh, shape (12,)
