"""Column type detection and dataset profiling."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

ID_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|ticket_no|order_no)", re.I)


@dataclass
class Profile:
    rows: int
    cols: int
    types: dict[str, str] = field(default_factory=dict)  # col -> datetime|numeric|categorical|id|text|boolean

    def of(self, kind: str) -> list[str]:
        return [c for c, t in self.types.items() if t == kind]

    @property
    def numeric(self) -> list[str]:
        return self.of("numeric")

    @property
    def categorical(self) -> list[str]:
        return self.of("categorical") + self.of("boolean")

    @property
    def datetime(self) -> list[str]:
        return self.of("datetime")


def _looks_datetime(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if sample.empty or not sample.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.8:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _numeric_share(s: pd.Series) -> float:
    non_null = s.dropna()
    if non_null.empty:
        return 0.0
    cleaned = non_null.astype(str).str.replace(r"[,₹$\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").notna().mean()


def detect_type(name: str, s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nunique = s.nunique(dropna=True)
    if ID_PATTERN.search(name) and nunique > 0.9 * max(len(s.dropna()), 1):
        return "id"
    if pd.api.types.is_numeric_dtype(s):
        return "categorical" if nunique <= 2 and set(s.dropna().unique()) <= {0, 1} else "numeric"
    if _looks_datetime(s):
        return "datetime"
    if _numeric_share(s) > 0.9:
        return "numeric"
    if nunique <= max(50, int(0.05 * len(s))):
        return "categorical"
    return "text"


def coerce(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    """Return a typed copy. Unparseable values become NaN/NaT (counted by quality checks)."""
    out = df.copy()
    for c, t in types.items():
        if t == "datetime" and not pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = pd.to_datetime(out[c], errors="coerce", format="mixed")
        elif t == "numeric" and not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(r"[,₹$\s]", "", regex=True), errors="coerce")
    return out


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_").lower() or f"col_{i}" for i, c in enumerate(df.columns)]
    seen: dict[str, int] = {}
    for i, c in enumerate(cols):
        if c in seen:
            seen[c] += 1
            cols[i] = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
    df.columns = cols
    return df


def profile(df: pd.DataFrame) -> Profile:
    return Profile(rows=len(df), cols=df.shape[1], types={c: detect_type(c, df[c]) for c in df.columns})


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, Profile]:
    """Clean names, detect types, coerce. Returns (typed_df, profile)."""
    df = clean_columns(raw)
    prof = profile(df)
    return coerce(df, prof.types), prof


def load_file(file, name: str) -> pd.DataFrame:
    lower = name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(file)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")
