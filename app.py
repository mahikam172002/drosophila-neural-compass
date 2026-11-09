
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

# ==========================================
# PHASE 3: REAL FLYWIRE EPG CONNECTIVITY
# ==========================================

import pandas as pd

st.divider()
st.header("Phase 3: Real FlyWire EPG Neural Network")

# Read the two CSV files uploaded to GitHub
neurons = pd.read_csv(
    "search_results_cell_type_EPG.csv",
    dtype={"root_id": str}
)

connections = pd.read_csv(
    "connections.csv",
    dtype={"From": str, "To": str}
)

# Convert synapse counts into numbers
connections["Synapses"] = pd.to_numeric(
    connections["Synapses"], errors="coerce"
).fillna(0)

# Aggregate synapses across brain regions
edges = connections.groupby(
    ["From", "To"], as_index=False
)["Synapses"].sum()

ids = neurons["root_id"].dropna().unique().tolist()
id_set = set(ids)

# Keep connections between selected EPG neurons
edges = edges[
    edges["From"].isin(id_set)
    & edges["To"].isin(id_set)
]

st.write("EPG neurons:", len(ids))
st.write("Connection records:", len(connections))
st.write("Total annotated synapses:",
         int(connections["Synapses"].sum()))

# Interactive synapse strength filter
minimum = st.slider(
    "Minimum synapses per neuron pair",
    1, 30, 1
)

visible = edges[edges["Synapses"] >= minimum]

# Circular arrangement for visual clarity
angles = np.linspace(
    0, 2 * np.pi, len(ids), endpoint=False
)

positions = {
    neuron_id: (
        float(np.cos(angle)),
        float(np.sin(angle))
    )
    for neuron_id, angle in zip(ids, angles)
}

# Draw connectivity edges
edge_x = []
edge_y = []

for _, edge in visible.iterrows():
    x0, y0 = positions[edge["From"]]
    x1, y1 = positions[edge["To"]]

    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig_network = go.Figure()

fig_network.add_trace(go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(color="rgba(80,150,210,0.3)", width=1),
    hoverinfo="skip",
    name="Connections"
))

# Draw actual EPG neuron nodes
node_x = [positions[i][0] for i in ids]
node_y = [positions[i][1] for i in ids]

fig_network.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    marker=dict(
        size=11,
        color="#e4a04c",
        line=dict(width=1, color="white")
    ),
    text=ids,
    hovertemplate="Neuron ID: %{text}<extra></extra>",
    name="EPG neurons"
))

fig_network.update_layout(
    title="FlyWire EPG Anatomical Connectivity",
    xaxis=dict(visible=False, scaleanchor="y"),
    yaxis=dict(visible=False),
    height=600,
    showlegend=False
)

st.plotly_chart(fig_network, use_container_width=True)

st.subheader("Connection Data")

st.dataframe(
    visible.sort_values("Synapses", ascending=False),
    use_container_width=True
)

st.caption(
    "Neuron identities and connections come from FlyWire "
    "FAFB v783. Circular node positions are illustrative, "
    "not anatomical coordinates. The graph shows structural "
    "connectivity, not measured electrical activity."
)




