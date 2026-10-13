"""Day-ahead price explorer: 2026 backtest with what-if sliders."""

from datetime import date
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import polars as pl
import streamlit as st

from energy_forecaster.config import REFERENCE_TIMEZONE
from energy_forecaster.dashboard import (
    complete_days,
    day_summary,
    predict,
    select_day,
    zone_context,
)
from energy_forecaster.features import load_feature_table
from energy_forecaster.models import BASELINE_COLUMN
from energy_forecaster.preprocessing import TARGET_COLUMN
from energy_forecaster.scenarios import apply_scenario
from energy_forecaster.splits import TEST_SPLIT, apply_split

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "xgb_full.joblib"

st.set_page_config(page_title="Nordic day-ahead price explorer", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_table() -> pl.DataFrame:
    _, test = apply_split(load_feature_table(), TEST_SPLIT)
    return test


@st.cache_data
def load_context_table() -> pl.DataFrame:
    rows = load_test_table().drop_nulls([TARGET_COLUMN, BASELINE_COLUMN])
    return rows.select(
        "timestamp",
        "zone",
        pl.col(TARGET_COLUMN).alias("actual"),
        pl.col(BASELINE_COLUMN).alias("yesterday"),
    ).with_columns(pl.Series("predicted", predict(load_model(), rows)))


model = load_model()
test = load_test_table()
days = complete_days(test)
if not days:
    st.error("No complete days in the test table.")
    st.stop()

st.title("Nordic day-ahead price explorer")
st.info(
    "**Backtest, not a live forecast.** The model was trained on 2021-2025 and has never seen 2026. "
    "Over 280 test days its error was **0.776x** the 'yesterday' baseline (95% interval 0.737-0.815); "
    "without wind/solar forecasts, 0.877x. It loses to the baseline in NO_3 and NO_5, and 2026 was a "
    "harder, higher-priced year than the training years. The sliders show how *the model's* forecast "
    "responds to changed inputs. They are not a market simulation, and yesterday's prices stay fixed."
)

PALETTE = {
    "light": {"Actual": "#2a78d6", "Model": "#eb6834", "What-if": "#1baf7a"},
    "dark": {"Actual": "#3987e5", "Model": "#d95926", "What-if": "#199e70"},
}
NEUTRAL = "#7c7b77"  # the 'yesterday' reference line


def theme_is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:  # older Streamlit: no theme info, assume light
        return False


with st.sidebar:
    st.header("Pick a day")
    zone = st.selectbox("Bidding zone", sorted(test["zone"].unique().to_list()), index=0)
    if zone is None:
        st.stop()
    day = st.date_input("Date", value=days[-1], min_value=days[0], max_value=days[-1])
    if not isinstance(day, date):
        st.warning("Pick a single date.")
        st.stop()
    if day not in days:
        st.caption(f"{day} has incomplete data, so the metrics use the hours available.")
    st.header("What-if (x the TSO forecast)")
    wind = st.slider("Nordic wind", 0.5, 1.5, 1.0, 0.05)
    solar = st.slider("Nordic solar", 0.5, 1.5, 1.0, 0.05)
    load = st.slider("Nordic load", 0.8, 1.2, 1.0, 0.05)
    german = st.slider("German wind + solar", 0.5, 1.5, 1.0, 0.05)

day_all = select_day(test, day)
if day_all.is_empty():
    st.error("No data for that day.")
    st.stop()

base = predict(model, day_all)
changed = apply_scenario(day_all, wind=wind, solar=solar, load=load, german_renewables=german)
scenario = predict(model, changed)

view = (
    day_all.select(
        pl.col("timestamp")
        .dt.convert_time_zone(REFERENCE_TIMEZONE)
        .dt.replace_time_zone(None)
        .alias("time"),
        "zone",
        pl.col(TARGET_COLUMN).alias("Actual"),
        pl.col(BASELINE_COLUMN).alias("Yesterday"),
    )
    .with_columns(pl.Series("Model", base), pl.Series("What-if", scenario))
    .filter(pl.col("zone") == zone)
)

context = zone_context(load_context_table(), zone)
if context["pooled_relative_mae"] > 1:
    st.warning(f"{zone} is a zone where the model loses to the 'yesterday' baseline over 2026.")
st.caption(
    f"{zone} over {int(context['days'])} test days: the model's error is {context['pooled_relative_mae']:.2f}x "
    f"yesterday's, and it beats 'yesterday' on {context['win_rate']:.0%} of days."
)

summary = day_summary(view["Actual"], view["Yesterday"], view["Model"])
shift = float(np.mean(view["What-if"].to_numpy() - view["Model"].to_numpy()))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Yesterday MAE (EUR/MWh)", f"{summary['mae_yesterday']:.1f}")
c2.metric(
    "Model MAE (EUR/MWh)",
    f"{summary['mae_model']:.1f}",
    delta=f"{summary['mae_model'] - summary['mae_yesterday']:+.1f}",
    delta_color="inverse",
)
c3.metric("Relative MAE, this day", f"{summary['relative_mae']:.2f}")
c4.metric("What-if shift, mean (EUR/MWh)", f"{shift:+.1f}")

scenario_active = any(f != 1.0 for f in (wind, solar, load, german))
series = ["Actual", "Yesterday", "Model", *(["What-if"] if scenario_active else [])]
long = view.to_pandas().melt(
    id_vars="time", value_vars=series, var_name="Series", value_name="price"
)

colors = PALETTE["dark" if theme_is_dark() else "light"]
palette = {
    "Actual": colors["Actual"],
    "Model": colors["Model"],
    "What-if": colors["What-if"],
    "Yesterday": NEUTRAL,
}
shown = [name for name in palette if name in series]
chart = (
    alt.Chart(long)
    .mark_line(strokeWidth=2)
    .encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y("price:Q", title="EUR/MWh"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=shown, range=[palette[name] for name in shown]),
            legend=alt.Legend(orient="top", title=None),
        ),
        strokeDash=alt.condition(
            alt.datum.Series == "Yesterday", alt.value([4, 3]), alt.value([1, 0])
        ),
        tooltip=[
            alt.Tooltip("time:T", format="%H:%M"),
            "Series:N",
            alt.Tooltip("price:Q", format=".1f"),
        ],
    )
    .properties(height=380)
)
st.altair_chart(chart, width="stretch")

with st.expander("Show the numbers"):
    st.dataframe(view.drop("zone").with_columns(pl.col(pl.Float64).round(1)), hide_index=True)
st.caption(
    "Hours are Stockholm wall-clock time. A single day's relative MAE is noisy: the 0.776 figure is a "
    "280-day aggregate. The sliders rescale the TSO day-ahead forecasts for every Nordic zone at once."
)
