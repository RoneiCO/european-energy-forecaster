from datetime import date

import polars as pl

from energy_forecaster.splits import CV_SPLITS, TEST_SPLIT, TEST_START, apply_split


def test_every_split_trains_only_on_the_past() -> None:
    for split in [*CV_SPLITS, TEST_SPLIT]:
        assert split.train_start < split.train_end <= split.eval_start < split.eval_end


def test_cross_validation_never_touches_the_test_period() -> None:
    for split in CV_SPLITS:
        assert split.eval_end <= TEST_START


def test_the_test_split_trains_on_everything_before_the_test_year() -> None:
    assert TEST_SPLIT.train_end == TEST_START == TEST_SPLIT.eval_start


def test_the_new_year_boundary_follows_the_local_calendar() -> None:
    # hourly rows from 00:00 on 31 December to 00:00 on 1 January, local time (25 rows)
    timestamps = pl.datetime_range(
        date(2023, 12, 31),
        date(2024, 1, 1),
        interval="1h",
        time_zone="Europe/Stockholm",
        eager=True,
    )
    frame = pl.DataFrame({"timestamp": timestamps})
    cv_2024 = next(split for split in CV_SPLITS if split.name == "cv_2024")
    train, evaluation = apply_split(frame, cv_2024)
    assert train.height == 24  # all of 31 December
    assert evaluation.height == 1  # only 00:00 on 1 January
