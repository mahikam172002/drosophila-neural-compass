"""Phase 4: exploratory EPG activity model informed by FlyWire synapse counts.

Place this file at pages/4_Neural_Activity.py in the Streamlit repository.
Requires pandas, numpy, plotly, streamlit and the two CSVs at repo root.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FlyWire EPG Neural Activity", layout="wide")
st.title("Phase 4 · FlyWire EPG Neural Activity")
st.info("This is a simplified computational experiment, not recorded activity or a validated fly heading model.")

root = Path(__file__).resolve().parent.parent
metadata_file = root / "search_results_cell_type_EPG.csv"
connections_file = root / "connections.csv"
if not metadata_file.exists() or not connections_file.exists():
    st.error("Missing CSV files. Put search_results_cell_type_EPG.csv and connections.csv in the repository root (beside app.py).")
    st.stop()

neurons = pd.read_csv(metadata_file, dtype={"root_id": str})
connections = pd.read_csv(connections_file, dtype={"From": str, "To": str})
required_neurons = {"root_id"}
required_edges = {"From", "To", "Synapses"}
if not required_neurons.issubset(neurons.columns) or not required_edges.issubset(connections.columns):
    st.error("Unexpected CSV columns. The neuron file needs root_id; the connections file needs From, To and Synapses.")
    st.stop()

ids = neurons["root_id"].dropna().astype(str).drop_duplicates().tolist()
n = len(ids)
if n == 0:
    st.error("No neuron IDs found in the metadata file.")
    st.stop()
index = {neuron_id: i for i, neuron_id in enumerate(ids)}
connections["Synapses"] = pd.to_numeric(connections["Synapses"], errors="coerce").fillna(0).clip(lower=0)
W = np.zeros((n, n), dtype=float)  # W[postsynaptic, presynaptic]
for src, dst, count in connections[["From", "To", "Synapses"]].itertuples(index=False, name=None):
    if src in index and dst in index:
        W[index[dst], index[src]] += count

incoming = W.sum(axis=1, keepdims=True)
W_normalized = np.divide(W, incoming, out=np.zeros_like(W), where=incoming > 0)

c1, c2, c3 = st.columns(3)
c1.metric("EPG neurons", n)
c2.metric("Connection records", len(connections))
c3.metric("Synapses in selected network", int(W.sum()))

st.subheader("Simulation controls")
steps = st.slider("Simulation steps", 20, 200, 80)
gain = st.slider("Recurrent coupling strength", 0.0, 2.0, 1.0, 0.1)
noise = st.slider("Noise strength", 0.0, 0.30, 0.03, 0.01)
visual = st.checkbox("Include an illustrative visual landmark input", value=True)
turn = st.slider("Illustrative turning rate (degrees per step)", -5.0, 5.0, 1.0, 0.5)

# IMPORTANT: The neuron ordering below is arbitrary. FlyWire root IDs do not
# encode angular position. Angular decoding is illustrative, not biological.
angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)

def bump(theta):
    return np.exp(3.0 * (np.cos(angles - theta) - 1.0))

def rotate_activity(values, degrees):
    sampling = (angles - np.deg2rad(degrees)) % (2 * np.pi)
    return np.interp(sampling, np.r_[angles, 2 * np.pi], np.r_[values, values[0]])

rng = np.random.default_rng(42)
activity = bump(0.0)
history = [activity.copy()]
errors = []
alpha = 0.25
for t in range(steps):
    true_angle = np.deg2rad((t + 1) * turn)
    shifted = rotate_activity(activity, turn)
    recurrent = W_normalized @ shifted
    visual_drive = 0.6 * bump(true_angle) if visual else np.zeros(n)
    drive = 0.45 * shifted + gain * recurrent + visual_drive + rng.normal(0, noise, n)
    target = np.maximum(drive, 0.0)
    if target.max() > 0:
        target /= target.max()
    activity = (1 - alpha) * shifted + alpha * target
    history.append(activity.copy())
    resultant = np.sum(activity * np.exp(1j * angles))
    if abs(resultant) > 1e-9:
        estimate = np.angle(resultant)
        errors.append(np.rad2deg(np.angle(np.exp(1j * (estimate - true_angle)))))
    else:
        errors.append(np.nan)

st.subheader("Simulated activity across time")
fig = go.Figure(go.Heatmap(z=np.array(history).T, x=list(range(steps + 1)), y=list(range(n)), colorscale="Viridis", colorbar=dict(title="Activity")))
fig.update_layout(xaxis_title="Simulation step", yaxis_title="Neuron index (arbitrary ordering)", height=480)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Actual FlyWire anatomical connectivity matrix")
fig2 = go.Figure(go.Heatmap(z=W, colorscale="Blues", colorbar=dict(title="Synapses")))
fig2.update_layout(xaxis_title="Presynaptic neuron index", yaxis_title="Postsynaptic neuron index", height=480)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Final simulated neuron activity")
fig3 = go.Figure(go.Bar(x=list(range(n)), y=activity))
fig3.update_layout(xaxis_title="Neuron index", yaxis_title="Activity (arbitrary units)", height=320)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Illustrative heading-decoding error")
st.metric("Mean absolute circular error", f"{np.nanmean(np.abs(errors)):.1f}°")
fig4 = go.Figure(go.Scatter(x=list(range(1, steps + 1)), y=errors, mode="lines"))
fig4.update_layout(xaxis_title="Simulation step", yaxis_title="Heading error (degrees)", height=320)
st.plotly_chart(fig4, use_container_width=True)

with st.expander("How to interpret these results"):
    st.write("FlyWire supplies neuron identities and synapse counts. The network influences the simulated activity via a normalized incoming-connectivity matrix. The assumed neuron ordering, turning-related shift, visual input, noise, and activity update are illustrative. The heading error is not a scientifically validated measure until anatomical angular assignments and additional compass neuron types are incorporated.")
