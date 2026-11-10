from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Phase 5 | Combined FlyWire Network', layout='wide')
st.title('Phase 5 — Combined FlyWire Neural Network')
st.caption('FAFB v783 | Anatomical connectivity is real; activity is simulated, not recorded.')
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'combined_connections.csv'
if not DATA.exists():
    st.error('Missing combined_connections.csv in the repository root. Upload the combined FlyWire CSV with this exact filename.')
    st.stop()

edges = pd.read_csv(DATA, dtype={'From': str, 'To': str})
required = {'From', 'To', 'Synapses'}
if not required.issubset(edges.columns):
    st.error(f'CSV must include {sorted(required)}; found {list(edges.columns)}')
    st.stop()
edges['Synapses'] = pd.to_numeric(edges['Synapses'], errors='coerce').fillna(0)
edges = edges.loc[edges['Synapses'] > 0].copy()
ids = sorted(set(edges['From']) | set(edges['To']))
index = {neuron_id: i for i, neuron_id in enumerate(ids)}
n = len(ids)
W = np.zeros((n, n), dtype=float)
for (src, dst), synapses in edges.groupby(['From', 'To'])['Synapses'].sum().items():
    W[index[dst], index[src]] += float(synapses)

# Metadata is optional, but needed to label cell types and assign model signs.
metadata_files = {
    'EPG': 'search_results_cell_type_EPG.csv',
    'P-EN1': 'search_results_cell_type_PEN_a_PEN1.csv',
    'Delta7': 'search_results_cell_type_Delta7.csv',
}
cell_type = {neuron_id: 'Unclassified' for neuron_id in ids}
missing = []
for group, filename in metadata_files.items():
    path = ROOT / filename
    if not path.exists():
        missing.append(filename)
        continue
    table = pd.read_csv(path, dtype={'root_id': str})
    if 'root_id' not in table.columns:
        st.warning(f'{filename} has no root_id column.')
        continue
    for neuron_id in table['root_id'].dropna():
        if neuron_id in cell_type:
            cell_type[neuron_id] = group
if missing:
    st.warning('Upload the three neuron metadata CSVs to the repository root to label all populations: ' + ', '.join(missing))

groups = pd.Series([cell_type[x] for x in ids], name='Population')
unique_pairs = edges[['From', 'To']].drop_duplicates().shape[0]
a, b, c, d = st.columns(4)
a.metric('Neurons', n)
b.metric('CSV rows', len(edges))
c.metric('Directed pairs', unique_pairs)
d.metric('Synapses', f'{int(edges.Synapses.sum()):,}')
st.write('Population counts:', groups.value_counts().to_dict())

# Show actual anatomical counts, without assigning excitatory/inhibitory sign.
ordered = sorted(range(n), key=lambda i: (cell_type[ids[i]], ids[i]))
labels = [f'{cell_type[ids[i]]} {ids[i][-6:]}' for i in ordered]
st.subheader('1. FlyWire anatomical connectivity matrix')
fig = px.imshow(W[np.ix_(ordered, ordered)], color_continuous_scale='Blues', aspect='auto', labels={'x':'Presynaptic neuron index','y':'Postsynaptic neuron index','color':'Synapses'})
fig.update_layout(height=560)
st.plotly_chart(fig, use_container_width=True)

st.subheader('2. Activity simulation (illustrative rate model)')
st.write('A stimulus activates selected neurons, and activity propagates through the measured synapse-count matrix. This is a network demonstration, not a validated fly heading model.')
col1, col2, col3 = st.columns(3)
with col1:
    stimulate = st.selectbox('Stimulated population', sorted(set(groups)))
    steps = st.slider('Simulation steps', 20, 200, 80)
with col2:
    coupling = st.slider('Connectivity coupling', 0.0, 2.0, 0.8, 0.1)
    noise = st.slider('Noise', 0.0, 0.3, 0.02, 0.01)
with col3:
    disruption = st.selectbox('Reduce outgoing connectivity from', ['None'] + sorted(set(groups)))
    reduction = st.slider('Reduction (%)', 0, 100, 75, 5, disabled=disruption == 'None')

# Each target neuron gets a weighted mean of incoming firing rates.
W_test = W.copy()
if disruption != 'None':
    cols = [i for i, neuron_id in enumerate(ids) if cell_type[neuron_id] == disruption]
    W_test[:, cols] *= (1 - reduction / 100)
row_sum = W_test.sum(axis=1, keepdims=True)
W_norm = np.divide(W_test, row_sum, out=np.zeros_like(W_test), where=row_sum > 0)
mask = np.array([cell_type[neuron_id] == stimulate for neuron_id in ids], dtype=float)
if not mask.any():
    st.error('No neurons in selected population; please upload metadata files.')
    st.stop()
input_drive = mask * 0.7
rng = np.random.default_rng(2026)
activity = input_drive.copy()
history = [activity.copy()]
for t in range(steps):
    recurrent = W_norm @ activity
    proposed = np.clip(input_drive + coupling * recurrent + rng.normal(0, noise, n), 0, 1)
    activity = 0.8 * activity + 0.2 * proposed
    history.append(activity.copy())
history = np.asarray(history)
fig2 = px.imshow(history[:, ordered].T, color_continuous_scale='Viridis', aspect='auto', labels={'x':'Simulation step','y':'Neuron index (grouped by population)','color':'Activity'})
fig2.update_layout(height=550)
st.plotly_chart(fig2, use_container_width=True)

summary = pd.DataFrame({'Population': [cell_type[ids[i]] for i in range(n)], 'Activity': activity})
summary = summary.groupby('Population', as_index=False)['Activity'].mean()
st.subheader('3. Mean final simulated activity by population')
st.plotly_chart(px.bar(summary, x='Population', y='Activity', range_y=[0, 1]), use_container_width=True)
st.info('Important: FlyWire provides synapse counts and predicted transmitter annotations, not physiological response strengths, synaptic signs, or recorded activity. This simulation uses unsigned, normalized weights. Reducing connectivity is a hypothetical manipulation, not evidence of a biological lesion effect. No anatomical heading angles are assigned yet.')
