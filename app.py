
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from model import RingAttractor

st.set_page_config(
    page_title="Drosophila Neural Compass",
    layout="wide"
)

st.title("The Fly's Internal GPS")
st.write("Phase 2: Ring Attractor Neural Network")

heading = st.slider(
    "Turn the virtual fly (degrees)",
    0, 359, 90
)

landmarks = st.checkbox(
    "Visual landmarks available",
    value=True
)

# Create a neural network for this simulation.
brain = RingAttractor(n=16)

# Simulate a gradual turn from 0 degrees.
steps = 30
turn_per_step = heading / steps

history = [brain.activity.copy()]

for i in range(steps):
    current_heading = (i + 1) * turn_per_step

    brain.step(
        turn_degrees=turn_per_step,
        visual_heading=current_heading if landmarks else None
    )

    history.append(brain.activity.copy())

estimated = brain.estimated_heading()

left, right = st.columns(2)

with left:
    st.subheader("Virtual fruit fly")

    theta = np.deg2rad(heading)

    fly = go.Figure()

    fly.add_trace(go.Scatter(
        x=[0, np.sin(theta)],
        y=[0, np.cos(theta)],
        mode="lines+markers",
        line=dict(width=6),
        marker=dict(size=[16, 10]),
        name="Fly direction"
    ))

    fly.update_layout(
        xaxis=dict(
            range=[-1.3, 1.3],
            scaleanchor="y"
        ),
        yaxis=dict(range=[-1.3, 1.3]),
        height=350,
        showlegend=False
    )

    st.plotly_chart(fly, use_container_width=True)

with right:
    st.subheader("Neural compass activity")

    compass = go.Figure()

    compass.add_trace(go.Scatterpolar(
        r=brain.activity,
        theta=np.rad2deg(brain.angles),
        mode="lines+markers",
        fill="toself",
        name="Neural activity"
    ))

    compass.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1]),
            angularaxis=dict(
                rotation=90,
                direction="clockwise"
            )
        ),
        height=350
    )

    st.plotly_chart(compass, use_container_width=True)

st.metric(
    "Actual fly heading",
    f"{heading} degrees"
)

st.metric(
    "Neural network estimated heading",
    f"{estimated:.1f} degrees"
)

st.subheader("Neural activity over time")

activity_history = np.array(history)

heatmap = go.Figure(data=go.Heatmap(
    z=activity_history.T,
    x=list(range(len(history))),
    y=list(range(16)),
    colorscale="Viridis"
))

heatmap.update_layout(
    xaxis_title="Simulation step",
    yaxis_title="Neural unit",
    height=350
)

st.plotly_chart(heatmap, use_container_width=True)

st.caption(
    "This is a simplified rate-based neural network. "
    "Actual FlyWire connections will be incorporated "
    "in a later phase."
)




