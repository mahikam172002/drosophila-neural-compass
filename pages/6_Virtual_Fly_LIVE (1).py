import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Live Virtual Fruit Fly", layout="wide")
st.title("The Fly's Internal GPS — Live Simulation")
st.caption("Watch a virtual fruit fly move while its simulated EPG, P-EN1 and Delta7 circuit activates. Anatomical positions and neural activity are illustrative, not recordings.")

html = r'''<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#0c1525;color:#e8f3ff;font:15px system-ui,sans-serif} .wrap{padding:12px} .toolbar{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin-bottom:12px}button{border:1px solid #4a6787;background:#203a58;color:white;padding:9px 13px;border-radius:9px;cursor:pointer;font-weight:650}button:hover{background:#315b82}label{display:flex;gap:7px;align-items:center}input{accent-color:#68e1ff} .panels{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{background:#142239;border:1px solid #2b4867;border-radius:14px;padding:12px;min-width:0}h3{margin:0 0 7px;font-size:17px}canvas{width:100%;height:auto;display:block;border-radius:9px;background:#0b192b}.key{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.status{color:#b9d4ea;margin:9px 0 0;font-size:13px}@media(max-width:660px){.panels{grid-template-columns:1fr}}
</style></head><body><div class="wrap"><div class="toolbar"><button id="toggle">⏸ Pause</button><button id="left">↶ Turn left</button><button id="right">↷ Turn right</button><button id="reset">Reset</button><label><input id="landmark" type="checkbox" checked>Visual landmark</label><label><input id="auto" type="checkbox" checked>Auto-walk</label><label>Speed <input id="speed" type="range" min="1" max="6" value="3"></label></div><div class="panels"><div class="panel"><h3>🐝 Virtual fruit fly in an arena</h3><canvas id="arena" width="560" height="480"></canvas></div><div class="panel"><h3>🧠 Live simulated neural compass</h3><canvas id="brain" width="560" height="480"></canvas></div></div><div class="key"><span><i class="dot" style="background:#f8c75c"></i>EPG: heading representation</span><span><i class="dot" style="background:#6de6ef"></i>P-EN1: turn-related activity</span><span><i class="dot" style="background:#bd8dff"></i>Delta7: circuit regulation (illustrative)</span></div><p class="status" id="status">Running · The activity bump follows the simulated heading. The circuit is schematic, not a measured connectome animation.</p></div>
<script>
const a=document.getElementById('arena'),ac=a.getContext('2d'),b=document.getElementById('brain'),bc=b.getContext('2d');
let running=true, heading=0, x=280,y=240, t=0, turn=0, trail=[], last=performance.now();
const PI=Math.PI*2, clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
function ellipse(c,x,y,rx,ry,fill,stroke='#463024'){c.beginPath();c.ellipse(x,y,rx,ry,0,0,PI);c.fillStyle=fill;c.fill();if(stroke){c.strokeStyle=stroke;c.lineWidth=1.5;c.stroke();}}
function line(c,x1,y1,x2,y2,col,w=2){c.beginPath();c.moveTo(x1,y1);c.lineTo(x2,y2);c.strokeStyle=col;c.lineWidth=w;c.stroke();}
function drawFly(c){c.save();c.translate(x,y);c.rotate(heading);let wing=Math.sin(t*19)*.12;
// forward points up in local coordinates
c.save();c.translate(-10,-2);c.rotate(-.36+wing);ellipse(c,-16,-14,29,11,'rgba(171,222,255,.55)','#9cd7f7');c.restore();
c.save();c.translate(10,-2);c.rotate(.36-wing);ellipse(c,16,-14,29,11,'rgba(171,222,255,.55)','#9cd7f7');c.restore();
for(let s of [-1,1])for(let k=0;k<3;k++){let yy=-8+k*11;line(c,s*10,yy,s*(21+k*3),yy+11,'#d9a774',2.5);line(c,s*(21+k*3),yy+11,s*(30+k*3),yy+17,'#d9a774',2);}
ellipse(c,0,19,13,27,'#b97b42');for(let j=0;j<3;j++)line(c,-10+j*.6,10+j*9,10-j*.6,10+j*9,'#6c432b',2);
ellipse(c,0,-9,15,18,'#a56b36');ellipse(c,0,-32,15,13,'#e0ad52');ellipse(c,-12,-34,6,9,'#e94251');ellipse(c,12,-34,6,9,'#e94251');
line(c,-5,-43,-10,-55,'#caa06a',2);line(c,5,-43,10,-55,'#caa06a',2);
line(c,0,-50,0,-75,'#5af2ac',3);c.beginPath();c.moveTo(-6,-68);c.lineTo(0,-79);c.lineTo(6,-68);c.strokeStyle='#5af2ac';c.lineWidth=3;c.stroke();c.restore();}
function drawArena(){ac.clearRect(0,0,560,480);let g=ac.createRadialGradient(280,240,20,280,240,225);g.addColorStop(0,'#193651');g.addColorStop(1,'#0b192b');ac.fillStyle=g;ac.fillRect(0,0,560,480);ac.beginPath();ac.arc(280,240,205,0,PI);ac.strokeStyle='#7797b3';ac.lineWidth=4;ac.stroke();
if(document.getElementById('landmark').checked){ac.fillStyle='#60ef95';ac.fillRect(466,213,18,54);ac.fillStyle='#b2ffd1';ac.font='13px system-ui';ac.fillText('Landmark',428,198);}
ac.beginPath();trail.forEach((p,i)=>{if(!i)ac.moveTo(p[0],p[1]);else ac.lineTo(p[0],p[1]);});ac.strokeStyle='rgba(114,226,248,.40)';ac.lineWidth=2;ac.stroke();drawFly(ac);ac.fillStyle='#d6ebff';ac.font='14px system-ui';ac.fillText('Heading: '+((heading*180/Math.PI+360)%360).toFixed(0)+'°',18,28);ac.fillText('Green arrow = forward',18,48);}
function neuron(c,cx,cy,r,color,act){let glow=c.createRadialGradient(cx,cy,1,cx,cy,r*3.4);glow.addColorStop(0,color.replace('ALPHA',String(.14+.6*act)));glow.addColorStop(1,'rgba(0,0,0,0)');c.fillStyle=glow;c.beginPath();c.arc(cx,cy,r*3.4,0,PI);c.fill();c.beginPath();c.arc(cx,cy,r*(.65+.45*act),0,PI);c.fillStyle=color.replace('ALPHA',String(.20+.8*act));c.fill();}
function drawBrain(){bc.clearRect(0,0,560,480);bc.fillStyle='#0b192b';bc.fillRect(0,0,560,480);
// schematic central-complex outline
bc.beginPath();bc.ellipse(280,240,184,162,0,0,PI);bc.fillStyle='rgba(53,83,120,.18)';bc.fill();bc.strokeStyle='#50718f';bc.lineWidth=2;bc.stroke();bc.textAlign='center';bc.fillStyle='#c9e3fa';bc.font='14px system-ui';bc.fillText('Schematic circuit — not anatomical coordinates',280,28);
let nodes=[];let phi=(heading+PI/2+PI)%PI*2; // normalized heading around ring
phi=((heading%(PI)+PI)%PI); // overwritten below for full circle
phi=((heading%PI)+PI)%PI; // use full turn below
phi=((heading% (Math.PI*2)) + Math.PI*2)%(Math.PI*2);
// 47 EPG neurons arranged as a conceptual ring
for(let i=0;i<47;i++){let q=i/47*PI-PI/2,px=280+112*Math.cos(q),py=235+112*Math.sin(q),diff=Math.atan2(Math.sin(q-phi+PI/2),Math.cos(q-phi+PI/2)),act=Math.exp(-.5*(diff/.38)**2);nodes.push({x:px,y:py,a:act});}
// schematic radial connections to P-EN1, animated signal
for(let i=0;i<20;i++){let q=i/20*PI-PI/2,xx=280+164*Math.cos(q),yy=235+164*Math.sin(q);let j=Math.floor(i*47/20);line(bc,xx,yy,nodes[j].x,nodes[j].y,'rgba(100,202,229,.14)',1);}
for(let i=0;i<47;i++){let n=nodes[i],next=nodes[(i+1)%47];line(bc,n.x,n.y,next.x,next.y,'rgba(248,199,92,.15)',1.2);neuron(bc,n.x,n.y,4.5,'rgba(255,203,91,ALPHA)',n.a);}
let v=Math.abs(turn);for(let i=0;i<20;i++){let q=i/20*PI-PI/2,xx=280+164*Math.cos(q),yy=235+164*Math.sin(q);let diff=Math.atan2(Math.sin(q-phi+PI/2),Math.cos(q-phi+PI/2));let act=Math.exp(-.5*(diff/.55)**2)*(.20+.8*clamp(v,0,1));neuron(bc,xx,yy,4.5,'rgba(103,233,245,ALPHA)',act);}
// Delta7 as central inhibitory regulatory population; layout illustrative
for(let i=0;i<42;i++){let q=i/42*PI-PI/2,xx=280+58*Math.cos(q),yy=235+58*Math.sin(q);let diff=Math.atan2(Math.sin(q-phi+PI/2),Math.cos(q-phi+PI/2));let act=.15+.45*(1-Math.exp(-.5*(diff/.7)**2));neuron(bc,xx,yy,3.3,'rgba(189,141,255,ALPHA)',act);}
bc.fillStyle='#d7eaff';bc.font='15px system-ui';bc.fillText('47 EPG • 20 P-EN1 • 42 Delta7 = 109 neurons',280,440);bc.font='12px system-ui';bc.fillStyle='#8eafcb';bc.fillText('Activity generated by a teaching model, not measured signals',280,460);bc.textAlign='start';}
function tick(now){let dt=Math.min((now-last)/1000,.06);last=now;if(running){let sp=+document.getElementById('speed').value;let auto=document.getElementById('auto').checked;turn=auto?(.36*Math.sin(t*.65)+.22*Math.sin(t*.23)):0;heading+=turn*dt;let step=auto?(18+sp*13)*dt:0;let nx=x+Math.sin(heading)*step,ny=y-Math.cos(heading)*step;if(Math.hypot(nx-280,ny-240)>172){heading+=Math.PI*.85;nx=x+Math.sin(heading)*step;ny=y-Math.cos(heading)*step;}x=nx;y=ny;t+=dt;trail.push([x,y]);if(trail.length>250)trail.shift();}drawArena();drawBrain();requestAnimationFrame(tick);}
document.getElementById('toggle').onclick=()=>{running=!running;document.getElementById('toggle').textContent=running?'⏸ Pause':'▶ Play';};document.getElementById('left').onclick=()=>{heading-=Math.PI/8;turn=-1;drawArena();drawBrain();};document.getElementById('right').onclick=()=>{heading+=Math.PI/8;turn=1;drawArena();drawBrain();};document.getElementById('reset').onclick=()=>{x=280;y=240;heading=0;t=0;trail=[];turn=0;};requestAnimationFrame(tick);
</script></body></html>'''
components.html(html, height=770, scrolling=True)
st.info("This is a continuously animated teaching prototype. The 109 dots represent the selected cell-type counts, but their positions, connections and activity are schematic. It does not yet read combined_connections.csv or reproduce FlyWire's exact synaptic wiring.")
