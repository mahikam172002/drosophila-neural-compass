
import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Virtual Fly Navigation",
    layout="wide"
)

st.title("Phase 6 — Virtual Fly Navigation")
st.caption("Interactive Drosophila navigation demonstration")

heading = st.slider("Turn the fly (degrees)", 0, 359, 0)
landmarks = st.checkbox("Visual landmarks available", value=True)

theta = np.deg2rad(heading)

# Rotate points according to the fly's heading.
# Zero degrees means the fly faces upward.
def rotate(x, y):
    return (
        x * np.cos(theta) + y * np.sin(theta),
        -x * np.sin(theta) + y * np.cos(theta)
    )

fig = go.Figure()

# Circular navigation arena
angle = np.linspace(0, 2 * np.pi, 200)
fig.add_trace(go.Scatter(
    x=np.cos(angle),
    y=np.sin(angle),
    mode="lines",
    line=dict(color="gray", width=3),
    showlegend=False
))

# Draw a rotated ellipse for each body part
def ellipse(cx, cy, rx, ry, color, opacity=1):
    t = np.linspace(0, 2 * np.pi, 80)
    x = cx + rx * np.cos(t)
    y = cy + ry * np.sin(t)

    xr, yr = rotate(x, y)

    fig.add_trace(go.Scatter(
        x=xr, y=yr,
        mode="lines",
        fill="toself",
        fillcolor=color,
        opacity=opacity,
        line=dict(color=color, width=1),
        showlegend=False,
        hoverinfo="skip"
    ))

# Wings (drawn behind the body)
ellipse(-0.22, 0.08, 0.22, 0.10, "lightblue", 0.65)
ellipse(0.22, 0.08, 0.22, 0.10, "lightblue", 0.65)

# Abdomen, thorax and head
ellipse(0, -0.17, 0.12, 0.23, "peru")
ellipse(0, 0.07, 0.13, 0.15, "saddlebrown")
ellipse(0, 0.25, 0.12, 0.11, "goldenrod")

# Eyes
ellipse(-0.09, 0.28, 0.045, 0.055, "darkred")
ellipse(0.09, 0.28, 0.045, 0.055, "darkred")

# Legs and antennae
def line(x1, y1, x2, y2, color="saddlebrown"):
    xr, yr = rotate(
        np.array([x1, x2]),
        np.array([y1, y2])
    )
    fig.add_trace(go.Scatter(
        x=xr, y=yr,
        mode="lines",
        line=dict(color=color, width=3),
        showlegend=False,
        hoverinfo="skip"
    ))

for side in [-1, 1]:
    for y in [-0.12, 0.0, 0.12]:
        line(side * 0.1, y, side * 0.28, y - 0.12)

    line(side * 0.04, 0.34, side * 0.10, 0.44)

# Direction indicator
xr, yr = rotate(
    np.array([0, 0]),
    np.array([0.34, 0.58])
)
fig.add_trace(go.Scatter(
    x=xr, y=yr,
    mode="lines",
    line=dict(color="limegreen", width=4),
    name="Heading direction"
))

# Fixed visual landmark
if landmarks:
    fig.add_trace(go.Scatter(
        x=[0.85],
        y=[0],
        mode="markers",
        marker=dict(
            color="limegreen",
            size=24,
            symbol="square"
        ),
        name="Visual landmark"
    ))

fig.update_layout(
    height=570,
    xaxis=dict(
        range=[-1.15, 1.15],
        visible=False
    ),
    yaxis=dict(
        range=[-1.15, 1.15],
        visible=False,
        scaleanchor="x"
    ),
    margin=dict(l=10, r=10, t=20, b=10),
    legend=dict(orientation="h"),
    dragmode=False
)

st.plotly_chart(fig, use_container_width=True)

st.metric("Fly heading", f"{heading}°")

st.info(
    "This is a visual fly model. "
    "The neural activity will be connected to "
    "the FlyWire network in the next step."
)
