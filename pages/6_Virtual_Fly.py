
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Virtual Fly Navigation", layout="wide")
st.title("Phase 6 — Virtual Fly Navigation")
st.caption("Interactive fly heading demonstration")

heading = st.slider("Turn the fly (degrees)", 0, 359, 0)
landmarks = st.checkbox("Visual landmarks available", value=True)

theta = np.deg2rad(heading)

fig = go.Figure()

# Arena
fig.add_shape(
    type="circle",
    x0=-1, y0=-1, x1=1, y1=1,
    line=dict(color="gray", width=2)
)

# Fly body direction
dx = 0.55 * np.sin(theta)
dy = 0.55 * np.cos(theta)

fig.add_trace(go.Scatter(
    x=[0, dx],
    y=[0, dy],
    mode="lines+markers",
    line=dict(color="orange", width=8),
    marker=dict(size=[16, 12]),
    name="Fly heading"
))

if landmarks:
    fig.add_trace(go.Scatter(
        x=[0.85], y=[0.0],
        mode="markers",
        marker=dict(size=22, color="green", symbol="square"),
        name="Visual landmark"
    ))

fig.update_layout(
    xaxis=dict(range=[-1.2, 1.2], visible=False),
    yaxis=dict(range=[-1.2, 1.2], visible=False, scaleanchor="x"),
    height=500,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)
st.metric("Fly heading", f"{heading}°")

st.info(
    "This is a visual prototype. "
    "The FlyWire neural activity model will be connected next."
)
