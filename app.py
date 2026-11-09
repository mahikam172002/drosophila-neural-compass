
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Drosophila Neural Compass",
    layout="wide"
)

st.title("The Fly's Internal GPS")
st.write("Phase 1: A simplified neural compass simulation")

# Set the virtual fly's direction
heading = st.slider("Turn the fruit fly (degrees)", 0, 359, 90)

# Create 16 heading-representing neural units
n = 16
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
theta = np.deg2rad(heading)

# Calculate the simulated activity bump
difference = np.angle(np.exp(1j * (angles - theta)))
activity = np.exp(3 * (np.cos(difference) - 1))

left, right = st.columns(2)

with left:
    st.subheader("Virtual fruit fly")

    fly = go.Figure()
    fly.add_trace(go.Scatter(
        x=[0, np.sin(theta)],
        y=[0, np.cos(theta)],
        mode="lines+markers",
        line=dict(width=6),
        marker=dict(size=[16, 10]),
        name="Heading"
    ))

    fly.update_layout(
        xaxis=dict(range=[-1.3, 1.3], scaleanchor="y"),
        yaxis=dict(range=[-1.3, 1.3]),
        height=350,
        showlegend=False
    )

    st.plotly_chart(fly, use_container_width=True)
    st.write(f"Fly heading: {heading} degrees")

with right:
    st.subheader("Simulated E-PG neural activity")

    compass = go.Figure()
    compass.add_trace(go.Scatterpolar(
        r=activity,
        theta=np.rad2deg(angles),
        mode="lines+markers",
        fill="toself",
        name="Neural activity"
    ))

    compass.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1]),
            angularaxis=dict(rotation=90, direction="clockwise")
        ),
        height=350
    )

    st.plotly_chart(compass, use_container_width=True)

st.info(
    "Move the slider and observe how the neural activity "
    "bump follows the fly's heading."
)

st.caption(
    "Phase 1 is an illustrative model. FlyWire connectivity "
    "and neural dynamics will be added in later phases."
)
  
