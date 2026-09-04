"""Plotly figures for the Gradio UI."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from web_app.config import N_MECHANISMS


def plot_learning_progress(learner) -> go.Figure:
    if not learner.history_metrics:
        return go.Figure()
    metrics = pd.DataFrame(learner.history_metrics)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics["step"],
            y=metrics["entropy"],
            mode="lines+markers",
            name="Ensemble entropy",
            line=dict(color="orange", width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="Ensemble entropy",
        xaxis_title="Loop",
        yaxis_title="Entropy (nats)",
        height=300,
        template="plotly_white",
        margin=dict(t=40, b=20, l=40, r=20),
    )
    return fig


def plot_mechanism_probs(probs) -> go.Figure:
    if probs is None:
        return go.Figure()
    labels = [f"M{i + 1}" for i in range(N_MECHANISMS)]
    fig = go.Figure(data=[go.Bar(x=labels, y=probs, marker_color="#2ca02c")])
    fig.update_layout(
        title="Mechanism probabilities",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        height=300,
        template="plotly_white",
        margin=dict(t=40, b=20, l=40, r=20),
    )
    return fig
