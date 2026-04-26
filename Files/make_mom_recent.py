"""
Generate Files/mom_recent.png: cumulative log return of the WML (Mom) factor
from Ken French's Data Library, with the 2009 and 2020 crashes annotated.

Usage:
    python make_mom_recent.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Momentum_Factor_CSV.zip"
)

OUT = Path(__file__).resolve().parent / "mom_recent.png"


def download_mom() -> pd.Series:
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        raw = z.read(name).decode("latin-1")

    # The monthly block ends before the annual block ("Annual Factors").
    monthly_block = raw.split("Annual Factors")[0]
    lines = [ln for ln in monthly_block.splitlines() if ln.strip()]

    # Find the header row (contains "Mom") then keep numeric date rows.
    header_idx = next(i for i, ln in enumerate(lines) if "Mom" in ln)
    data_lines = []
    for ln in lines[header_idx + 1 :]:
        token = ln.split(",")[0].strip()
        if len(token) == 6 and token.isdigit():
            data_lines.append(ln)

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["date", "mom"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df["mom"] = pd.to_numeric(df["mom"], errors="coerce") / 100.0
    df = df.dropna().set_index("date").sort_index()
    return df["mom"]


def plot(mom: pd.Series) -> None:
    cum_log = np.log1p(mom).cumsum()

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    ax.plot(cum_log.index, cum_log.values, color="#1f4e79", lw=1.4,
            label="WML (Mom) cumulative log return")

    crashes = [
        ("2009-03", "2009-05", "2009 crash\n($\\approx-73\\%$ in 2mo)"),
        ("2020-04", "2020-11", "2020 COVID rotation\n($\\approx-45\\%$)"),
    ]
    for start, end, label in crashes:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                   color="#c0392b", alpha=0.18)
        mid = pd.to_datetime(start) + (pd.to_datetime(end) - pd.to_datetime(start)) / 2
        ax.annotate(
            label, xy=(mid, cum_log.loc[:end].iloc[-1]),
            xytext=(0, -45), textcoords="offset points",
            ha="center", fontsize=8, color="#7b241c",
            arrowprops=dict(arrowstyle="-", color="#7b241c", lw=0.6),
        )

    start_year = cum_log.index.min().year
    end_year = cum_log.index.max().year
    ax.set_title(
        f"Momentum (WML) Cumulative Log Return, {start_year}–{end_year}",
        fontsize=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative log return")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(
        0.99, 0.02,
        "Source: Ken French Data Library",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="gray",
    )

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Wrote {OUT}  ({cum_log.index.min().date()} → {cum_log.index.max().date()})")


if __name__ == "__main__":
    plot(download_mom())
