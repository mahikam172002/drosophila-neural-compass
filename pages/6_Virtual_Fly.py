from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Virtual Fly + FlyWire', layout='wide')
st.title('Phase 6 — Virtual Fly and Neural Compass')
st.caption('FlyWire FAFB v783 connectivity is anatomical; all activity and heading tuning below are simulated assumptions.')
ROOT = Path(__file__).resolve().parent.parent
path = ROOT / 'combined_connections.csv'
if not path.exists():
    st.error('combined_connections.csv is missing from the repository root.')
    st.stop()

@st.cache_data
def load_network(filepath):
    df = pd.read_csv(filepath, dtype={'From': str, 'To': str})
    if not {'From','To','Synapses'}.issubset(df.columns):
        raise ValueError('CSV requires From, To, Synapses columns')
    df['Synapses'] = pd.to_numeric(df['Synapses'], errors='coerce').fillna(0)
    df = df[df['Synapses'] > 0]
    ids = sorted(set(df['From']) | set(df['To']))
    idx = {k:i for i,k in enumerate(ids)}
    w = np.zeros((len(ids),len(ids)))
    for (src,dst),count in df.groupby(['From','To'])['Synapses'].sum().items():
        w[idx[dst],idx[src]] = count
    types = {k:'Unknown' for k in ids}
    for label,name in [('EPG','search_results_cell_type_EPG.csv'),('P-EN1','search_results_cell_type_PEN_a_PEN1.csv'),('Delta7','search_results_cell_type_Delta7.csv')]:
        file = filepath.parent / name
        if file.exists():
            meta = pd.read_csv(file, dtype={'root_id':str})
            if 'root_id' in meta.columns:
                for k in meta['root_id'].dropna():
                    if k in types: types[k] = label
    return ids,w,types

try:
    ids, W, types = load_network(path)
except Exception as e:
    st.error(f'Could not load FlyWire network: {e}')
    st.stop()

if 'EPG' not in types.values():
    st.error('EPG metadata is required: upload search_results_cell_type_EPG.csv to the repository root.')
    st.stop()

heading = st.slider('Turn the fly (degrees)', 0, 359, 0)
landmarks = st.checkbox('Visual landmarks available', value=True)
coupling = st.slider('Network coupling (illustrative)', 0.0, 1.5, 0.65, 0.05)
steps = st.slider('Simulation steps', 10, 100, 45, 5)

angle = np.deg2rad(heading)
def rot(x,y):
    x,y = np.asarray(x),np.asarray(y)
    return x*np.cos(angle)+y*np.sin(angle), -x*np.sin(angle)+y*np.cos(angle)

fig = go.Figure()
t = np.linspace(0,2*np.pi,240)
fig.add_trace(go.Scatter(x=np.cos(t),y=np.sin(t),mode='lines',line=dict(color='#89929b',width=3),showlegend=False))
def ellipse(cx,cy,rx,ry,color,opacity=1):
    a=np.linspace(0,2*np.pi,90)
    x,y=rot(cx+rx*np.cos(a),cy+ry*np.sin(a))
    fig.add_trace(go.Scatter(x=x,y=y,mode='lines',fill='toself',fillcolor=color,opacity=opacity,line=dict(color=color,width=1),showlegend=False,hoverinfo='skip'))
def segment(x1,y1,x2,y2,color='#94663e',width=3):
    x,y=rot([x1,x2],[y1,y2]); fig.add_trace(go.Scatter(x=x,y=y,mode='lines',line=dict(color=color,width=width),showlegend=False,hoverinfo='skip'))
ellipse(-.22,.08,.22,.10,'#b9e1f4',.65)
ellipse(.22,.08,.22,.10,'#b9e1f4',.65)
ellipse(0,-.17,.12,.23,'#b87943')
ellipse(0,.07,.13,.15,'#81512f')
ellipse(0,.25,.12,.11,'#d9a044')
ellipse(-.09,.28,.045,.055,'#b02e39')
ellipse(.09,.28,.045,.055,'#b02e39')
for side in (-1,1):
    for yy in (-.12,0,.12): segment(side*.1,yy,side*.28,yy-.12)
    segment(side*.04,.34,side*.10,.44)
segment(0,.35,0,.57,'#47bd78',4)
if landmarks:
    fig.add_trace(go.Scatter(x=[.85],y=[0],mode='markers',marker=dict(color='#42cc78',size=22,symbol='square'),name='Visual landmark'))
fig.update_layout(height=480,xaxis=dict(range=[-1.15,1.15],visible=False),yaxis=dict(range=[-1.15,1.15],visible=False,scaleanchor='x'),margin=dict(l=5,r=5,t=15,b=5),dragmode=False)

# Source columns, destination rows. Counts are normalized by target input.
row_sums=W.sum(axis=1,keepdims=True)
Wnorm=np.divide(W,row_sums,out=np.zeros_like(W),where=row_sums>0)
epg=np.array([i for i,k in enumerate(ids) if types[k]=='EPG'])
# Preferred angles are evenly spaced placeholders, NOT anatomical FlyWire positions.
preferred=np.linspace(0,2*np.pi,len(epg),endpoint=False)
diff=np.angle(np.exp(1j*(preferred-angle)))
tuning=np.exp(3*(np.cos(diff)-1))
drive=np.zeros(len(ids))
# External orientation drive is weaker in darkness but never claims actual path integration.
drive[epg]=tuning*(0.55 if landmarks else 0.12)
state=np.zeros(len(ids))
history=[]
for _ in range(steps):
    target=np.clip(drive+coupling*(Wnorm@state),0,1)
    state=.78*state+.22*target
    history.append(state.copy())
history=np.array(history)

left,right=st.columns([1,1])
with left:
    st.subheader('Virtual fruit fly')
    st.plotly_chart(fig,use_container_width=True)
    st.metric('Actual fly heading',f'{heading}°')
with right:
    st.subheader('Simulated EPG activity')
    polar=go.Figure(go.Barpolar(theta=np.rad2deg(preferred),r=state[epg],width=360/len(epg),marker_color=state[epg],marker_colorscale='Viridis',showlegend=False))
    polar.update_layout(height=480,polar=dict(radialaxis=dict(range=[0,1]),angularaxis=dict(direction='clockwise',rotation=90)),margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(polar,use_container_width=True)

st.subheader('Simulated activity through the FlyWire network')
order=sorted(range(len(ids)),key=lambda i:(types[ids[i]],ids[i]))
heat=px.imshow(history[:,order].T,aspect='auto',color_continuous_scale='Viridis',labels={'x':'Simulation step','y':'Neuron (grouped by population)','color':'Activity'})
heat.update_layout(height=450)
st.plotly_chart(heat,use_container_width=True)
st.write('Neurons:',len(ids),'| Populations:',pd.Series([types[k] for k in ids]).value_counts().to_dict())
st.info('Scientific limitation: this is a static-heading rate-network demonstration. Heading tuning is assigned arbitrarily to EPG cells; synapse counts are normalized and unsigned. Darkness reduces external heading drive but does not yet model turning-based path integration. We must map EPG neurons to anatomical compass positions and introduce velocity-driven P-EN1 dynamics before interpreting heading tracking or memory.')
