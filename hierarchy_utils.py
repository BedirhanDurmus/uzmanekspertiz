"""Marka → seri → model (motor + paket) hiyerarşisi — ham CSV'den."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def split_model_motor_paket(model_str: str) -> tuple[str, str]:
    s = (model_str or "").strip()
    if not s:
        return "", ""
    parts = s.rsplit(" ", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0].strip(), parts[1].strip()


def load_marka_seri_model_hierarchy(*csv_paths: Path) -> pd.DataFrame:
    cols = ["marka", "seri", "model"]
    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        if not p.exists():
            continue
        try:
            dfs.append(pd.read_csv(p, usecols=cols, encoding="utf-8", low_memory=False))
        except (ValueError, OSError, KeyError):
            continue
    if not dfs:
        return pd.DataFrame(columns=cols)
    df = pd.concat(dfs, ignore_index=True)
    for c in cols:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df = df[df["marka"] != ""]
    return df.drop_duplicates()


def seri_for_marka(h: pd.DataFrame, marka: str) -> list[str]:
    if h.empty or not str(marka).strip():
        return []
    sub = h[h["marka"] == str(marka).strip()]
    return sorted({s for s in sub["seri"] if s})


def models_for_marka_seri(h: pd.DataFrame, marka: str, seri: str) -> list[str]:
    if h.empty or not str(marka).strip():
        return []
    m = str(marka).strip()
    s = str(seri).strip()
    if s:
        sub = h[(h["marka"] == m) & (h["seri"] == s)]
        out = sorted({x for x in sub["model"] if x})
        if out:
            return out
    sub_m = h[h["marka"] == m]
    return sorted({x for x in sub_m["model"] if x})


def motor_prefix_to_paket_map(models: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {}
    for mod in models:
        pre, suf = split_model_motor_paket(mod)
        buckets.setdefault(pre, [])
        if suf and suf not in buckets[pre]:
            buckets[pre].append(suf)
    for pre in buckets:
        buckets[pre] = sorted(buckets[pre], key=str.lower)
    return buckets


def resolve_model_from_motor_paket(motor_key: str, paket: str, all_models: list[str]) -> str:
    composed = f"{motor_key} {paket}".strip() if motor_key else paket
    if composed in all_models:
        return composed
    for m in all_models:
        if m.replace("  ", " ").strip() == composed.replace("  ", " ").strip():
            return m
    return composed


MOTOR_SINGLE = "__single__"


def branch_payload_for_marka_seri(h: pd.DataFrame, marka: str, seri: str) -> dict:
    """API için: tek model / motor+paket dalları / boş."""
    mod_opts = models_for_marka_seri(h, marka, seri)
    if not mod_opts:
        return {"mode": "empty", "models": []}

    if len(mod_opts) == 1:
        return {"mode": "single", "model_full": mod_opts[0], "models": mod_opts}

    buckets = motor_prefix_to_paket_map(mod_opts)
    non_empty = sorted([p for p in buckets if p], key=str.lower)
    has_single = "" in buckets and buckets[""]

    motors: list[dict] = []
    for p in non_empty:
        motors.append(
            {
                "key": p,
                "label": f"{p} …",
                "paketler": buckets[p],
            }
        )
    if has_single:
        motors.append(
            {
                "key": "",
                "label": "(tek parça — paket adı)",
                "paketler": buckets[""],
            }
        )

    if not motors:
        return {"mode": "list", "models": mod_opts}

    return {
        "mode": "motor_paket",
        "models": mod_opts,
        "motors": motors,
    }
