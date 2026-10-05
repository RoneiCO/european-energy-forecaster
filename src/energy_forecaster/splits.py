from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import polars as pl

DEV_START = date(2021, 1, 1)
TEST_START = date(2026, 1, 1)
DATA_END = date(2027, 1, 1)  # exclusive upper bound, safely after the last observation


@dataclass(frozen=True)
class Split:
    """One train/evaluation split, defined by local calendar dates.

    Train covers [train_start, train_end); evaluation covers [eval_start, eval_end).
    """

    name: str
    train_start: date
    train_end: date
    eval_start: date
    eval_end: date


def expanding_year_splits(eval_years: Iterable[int]) -> list[Split]:
    """One split per evaluation year, each training on everything from DEV_START up to it."""
    return [
        Split(f"cv_{year}", DEV_START, date(year, 1, 1), date(year, 1, 1), date(year + 1, 1, 1))
        for year in eval_years
    ]


CV_SPLITS = expanding_year_splits([2023, 2024, 2025])
TEST_SPLIT = Split("test_2026", DEV_START, TEST_START, TEST_START, DATA_END)


def _within(start: date, end: date) -> pl.Expr:
    local_date = pl.col("timestamp").dt.date()
    return (local_date >= start) & (local_date < end)


def apply_split(df: pl.DataFrame, split: Split) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train, evaluation) frames for a split."""
    train = df.filter(_within(split.train_start, split.train_end))
    evaluation = df.filter(_within(split.eval_start, split.eval_end))
    return train, evaluation
