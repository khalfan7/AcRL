from __future__ import annotations
import datetime
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
_FLAG_RE = re.compile('[A-Za-z?!*]+$')

def _field(line: str, start: int, end: int) -> float:
    raw = line[start:end].strip()
    if not raw or '---' in raw:
        return np.nan
    raw = _FLAG_RE.sub('', raw).strip()
    if not raw:
        return np.nan
    try:
        return float(raw)
    except ValueError:
        return np.nan

def load_hnn(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    records: list[dict] = []
    first_data = True
    with path.open('r', errors='replace') as fh:
        for raw in fh:
            line = raw.rstrip('\n')
            stripped = line.lstrip()
            if not stripped or stripped.startswith('!'):
                continue
            if first_data:
                first_data = False
                continue
            if len(line) < 100:
                continue
            try:
                hour_epa = int(line[11:14].strip())
            except ValueError:
                continue
            if hour_epa == 25:
                continue
            if not 1 <= hour_epa <= 24:
                continue
            try:
                d = datetime.date.fromisoformat(line[1:11].strip())
            except ValueError:
                continue
            hour_0 = hour_epa - 1
            ts = datetime.datetime(d.year, d.month, d.day, hour_0)
            ghi = _field(line, 29, 34)
            t_out = _field(line, 64, 69)
            wind = _field(line, 95, 100)
            records.append({'datetime': ts, 'T_out': t_out, 'GHI': ghi, 'wind_speed': wind, 'month': d.month, 'doy': d.timetuple().tm_yday, 'hour_of_day': hour_0})
    if not records:
        raise ValueError(f'No valid hourly records parsed from {path}')
    df = pd.DataFrame(records).sort_values('datetime').reset_index(drop=True)
    for col in ('T_out', 'GHI', 'wind_speed'):
        df[col] = df[col].ffill().bfill()
    df['GHI'] = df['GHI'].clip(lower=0.0)
    df['wind_speed'] = df['wind_speed'].clip(lower=0.0)
    return df

def load_hnn_multi(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [load_hnn(p) for p in paths]
    if not frames:
        raise ValueError('No files provided to load_hnn_multi')
    df = pd.concat(frames, ignore_index=True)
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
