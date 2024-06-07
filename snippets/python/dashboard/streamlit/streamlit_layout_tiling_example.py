"""
TAGS: dashboard|data visualisation|data visualization|datavis|dataviz|frontend|grid|gui|html|streamlit|tile|tiled|tiles|ui|user interface|visualisation|visualization|web|website
DESCRIPTION: An example of how to tile elements (i.e. a grid-layout) in a streamlit app
NOTES: The only lines worth reading in this script are from line 98 til the end
NOTES: Please excuse the badly-written function code - the emphasis of this example is streamlit layout

Deploy streamlit app from terminal:
    $ python -m venv venv && \
            source venv/bin/activate && \
            pip install altair numpy streamlit watchdog
    $ streamlit run streamlit_layout_tiling_example.py 

Layout:
    +------+---------------------------------+
    | text |                                 |
    +------+                                 |
    | text |          histogram              |
    +------+                                 |
    | text |                                 |
    +------+---------------------------------+
    |                                        |
    |                                        |
    |              line graph                |
    |                                        |
    |                                        |
    +----------------------------------------+
"""

import itertools
import random
import statistics

import altair as alt
import numpy as np
import streamlit as st


@st.cache_data
def simulate_data(n_transactions: int) -> dict[str, list]:
    """Simulates user transactions"""
    time_vec: list[int] = []
    amount_vec: list[int] = []
    time: int = 0
    while len(amount_vec) < n_transactions:
        time += 1
        if random.uniform(0, 1) < 0.2:
            time_vec.append(time)
            amount_vec.append(round(random.uniform(-50, 50), 2))
    return time_vec, amount_vec


def hist_altair(vec: list[int | float]) -> alt.vegalite.v5.api.Chart:
    """Turns a list of numbers into an altair histogram plot object"""
    pass
    counts, bin_edges = np.histogram(amounts, bins=10)
    hist_data = [
        {
            "Bin Start": bin_edges[idx],
            "Bin End": bin_edges[idx + 1],
            "Count": count,
        }
        for idx, count in enumerate(counts)
    ]
    plot_obj = (
        alt.Chart(alt.Data(values=hist_data))
        .mark_bar()
        .encode(
            x=alt.X(
                "Bin Start:Q",
                bin=alt.Bin(step=(bin_edges[1] - bin_edges[0])),
                title="Amount",
            ),
            y=alt.Y("Count:Q", title="Frequency"),
        )
        .properties(title="Histogram of Transaction Amounts")
    )
    return plot_obj


def lineplot_altair(
    x: list[int | float], y: list[int | float]
) -> alt.vegalite.v5.api.Chart:
    """Turns a list of numbers into an altair line chart object"""
    data = [{"Time": xx, "Cumulative Amount": yy} for xx, yy in zip(x, y)]
    line_chart = (
        alt.Chart(alt.Data(values=data))
        .mark_line()
        .encode(
            x=alt.X("Time:Q", title="Time"),
            y=alt.Y("Cumulative Amount:Q", title="Cumulative Amount"),
        )
        .properties(title="Cumulative Amount over Time")
    )
    return line_chart


times, amounts = simulate_data(50)

with st.container(border=True):
    col1, col2 = st.columns([3, 6])
    with col1:
        with st.container(border=True):
            st.text(f"n transactions: {len(amounts)}")
        with st.container(border=True):
            st.text(f"avg amount: {statistics.mean(amounts):.1f}")
        with st.container(border=True):
            st.text(f"max amount: {max(amounts):.2f}")
    with col2:
        with st.container(border=True):
            st.altair_chart(hist_altair(amounts), use_container_width=True)
with st.container(border=True):
    st.altair_chart(
        lineplot_altair(x=times, y=itertools.accumulate(amounts)),
        use_container_width=True,
    )
