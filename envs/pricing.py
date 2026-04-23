from __future__ import annotations
from pathlib import Path
import numpy as np
_MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

def load_monthly_prices(path: str | Path, year: int=2025) -> np.ndarray:
    path = Path(path)
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    header_idx: int | None = None
    years_avail: list[int] = []
    for i, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) < 2:
            continue
        try:
            candidates = [int(t) for t in tokens if len(t) == 4]
            if len(candidates) >= 2 and all((1990 < y < 2040 for y in candidates)):
                header_idx = i
                years_avail = candidates
                break
        except ValueError:
            continue
    if header_idx is None or not years_avail:
        raise ValueError(f'Could not locate year header in {path}')
    year_col = years_avail.index(year) if year in years_avail else 0
    prices = np.full(12, np.nan)
    month_to_idx = {m: i for i, m in enumerate(_MONTHS)}
    for line in lines[header_idx + 1:]:
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0].lower()
        if key not in month_to_idx:
            continue
        data = tokens[1:]
        if year_col >= len(data):
            continue
        try:
            prices[month_to_idx[key]] = float(data[year_col])
        except ValueError:
            pass
    if np.any(np.isnan(prices)):
        prices[np.isnan(prices)] = np.nanmean(prices)
    return prices
