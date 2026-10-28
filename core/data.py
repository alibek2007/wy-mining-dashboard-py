"""Loads the generator spreadsheet through the public Google Visualization (gviz)
endpoint, which returns raw (unformatted) values, so locale number formats in the
sheet don't matter."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import requests

DEFAULT_SHEET_ID = "1Eo5jdZx9pzSfgGoJA_qfhPfO9xB-Fw4a"
_DATE_RE = re.compile(r"Date\((\d+),(\d+),(\d+)")


def parse_sheet_id(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", text)
    return m.group(1) if m else text


def _gviz(sheet_id: str, **params) -> dict:
    params = {"tqx": "out:json", "_": str(time.time()), **params}
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq"
    for attempt in range(3):  # Google occasionally drops the connection; retry briefly
        try:
            r = requests.get(url, params=params, timeout=20, headers={"User-Agent": "wy-mining-dashboard/1.0"})
            break
        except requests.ConnectionError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    r.raise_for_status()
    text = r.text
    payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
    if payload.get("status") != "ok":
        msgs = "; ".join(e.get("detailed_message") or e.get("message", "") for e in payload.get("errors", []))
        raise RuntimeError(msgs or "Spreadsheet query failed (is it shared publicly?)")
    return payload["table"]


def _cell_date(cell) -> datetime | None:
    if not cell or not isinstance(cell.get("v"), str):
        return None
    m = _DATE_RE.match(cell["v"])
    return datetime(int(m[1]), int(m[2]) + 1, int(m[3])) if m else None


@dataclass
class Event:
    name: str
    start: datetime
    duration: int
    magnitude: float

    @property
    def end(self) -> datetime:
        return self.start + timedelta(days=self.duration - 1)


@dataclass
class Dataset:
    df: pd.DataFrame                      # index = date, one column per mine
    events: list[Event] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def mines(self) -> list[str]:
        return list(self.df.columns)


def load_dataset(sheet_id: str) -> pd.DataFrame:
    """Data sheet: column A = date, then one numeric column per mine, then 'Total'.
    Inactive mine columns have no header and are skipped; Total is recomputed."""
    table = _gviz(sheet_id, sheet="Data", headers="1")
    cols = [
        (i, (c.get("label") or "").strip())
        for i, c in enumerate(table["cols"])
        if i > 0 and (c.get("label") or "").strip() and c.get("type") == "number"
    ]
    mine_cols = [(i, lbl) for i, lbl in cols if not lbl.lower().startswith("total")]
    rows, dates = [], []
    for row in table["rows"]:
        cells = row["c"]
        d = _cell_date(cells[0])
        if d is None:
            continue
        vals = [(cells[i] or {}).get("v") for i, _ in mine_cols]
        if any(not isinstance(v, (int, float)) for v in vals):
            continue
        dates.append(d)
        rows.append(vals)
    if not rows:
        raise RuntimeError('No rows found on the "Data" sheet.')
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="date"), columns=[lbl for _, lbl in mine_cols], dtype=float)


def load_events(sheet_id: str) -> list[Event]:
    """Generator events from the Settings sheet (used only as context)."""
    try:
        table = _gviz(sheet_id, sheet="Settings", range="B50:G70", headers="0")
    except Exception:
        return []
    events = []
    for row in table["rows"]:
        c = row["c"] + [None] * 6
        name = (c[0] or {}).get("v")
        start = _cell_date(c[1])
        dur = (c[2] or {}).get("v")
        mag = (c[3] or {}).get("v")
        if name and start and isinstance(dur, (int, float)) and isinstance(mag, (int, float)):
            events.append(Event(str(name), start, int(dur), float(mag)))
    return events
