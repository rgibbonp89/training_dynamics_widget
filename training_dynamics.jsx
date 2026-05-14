import React, { useState, useEffect, useRef, useMemo } from ‘react’;
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid, Legend, ReferenceLine } from ‘recharts’;
import { Play, Pause, RotateCcw, Zap, Lock } from ‘lucide-react’;

// ─────────────────────────────────────────────────────────────────────────────
// Typography & color
// ─────────────────────────────────────────────────────────────────────────────
const FONT_IMPORT = `@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap'); .serif { font-family: 'Crimson Pro', Georgia, serif; } .mono  { font-family: 'JetBrains Mono', monospace; }`;

const C = {
bg:        ‘#f5f0e6’,
bgAlt:     ‘#ebe4d4’,
ink:       ‘#1a1814’,
inkSoft:   ‘#4a463d’,
inkFaint:  ‘#8a8475’,
rule:      ‘#c9c0a8’,
v1:        ‘#2d5a5a’,    // teal — fast eigendirection (range, when lam1 large)
v2:        ‘#7a2424’,    // burgundy — slow / kernel eigendirection
gWant:     ‘#b87333’,    // copper — what loss wants (-g)
uVel:      ‘#1a1814’,    // ink — actual velocity
kernel:    ‘#7a2424’,    // line through origin in slow direction
reach:     ‘#2d5a5a’,    // line through u(0) along range
grid:      ‘#dcd4bd’,
highlight: ‘#d4a76a’,    // honey — escape event
};

// ─────────────────────────────────────────────────────────────────────────────
// 2x2 utilities
// ─────────────────────────────────────────────────────────────────────────────
const rot = (t) => [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];
const tT  = (M) => [M[0], M[2], M[1], M[3]];
const mm  = (A, B) => [
A[0]*B[0]+A[1]*B[2], A[0]*B[1]+A[1]*B[3],
A[2]*B[0]+A[3]*B[2], A[2]*B[1]+A[3]*B[3],
];
const mv = (M, v) => [M[0]*v[0]+M[1]*v[1], M[2]*v[0]+M[3]*v[1]];
const dg = (a, b) => [a, 0, 0, b];
const sub = (a, b) => [a[0]-b[0], a[1]-b[1]];
const add = (a, b) => [a[0]+b[0], a[1]+b[1]];
const sc  = (a, s) => [a[0]*s, a[1]*s];
const norm = (v) => Math.hypot(v[0], v[1]);
const buildK = (l1, l2, th) => mm(mm(rot(th), dg(l1, l2)), tT(rot(th)));

// ─────────────────────────────────────────────────────────────────────────────
// Kernel-schedule profiles
// ─────────────────────────────────────────────────────────────────────────────
// Each profile maps t ∈ [0, tMax] → (lam1, lam2, theta).
function profileLazy(t, p) {
return { lam1: p.lam1_0, lam2: p.lam2_0, theta: p.theta_0 };
}

function profileRotate(t, p) {
// Linear rotation from theta_0 to theta_0 + p.delta_theta over [0, tMax]
const frac = Math.min(1, t / p.tMax);
return { lam1: p.lam1_0, lam2: p.lam2_0, theta: p.theta_0 + p.delta_theta * frac };
}

function profileAwaken(t, p) {
// lam2 starts near 0, then ramps up after t = p.wake_t.
const frac = Math.min(1, Math.max(0, (t - p.wake_t) / Math.max(0.01, p.wake_dur)));
// smooth sigmoid
const s = frac <= 0 ? 0 : frac >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * frac);
const lam2 = p.lam2_0 + (p.lam2_final - p.lam2_0) * s;
return { lam1: p.lam1_0, lam2, theta: p.theta_0 };
}

const PROFILES = {
lazy:    { name: ‘Lazy (frozen NTK)’, fn: profileLazy },
rotate:  { name: ‘Feature-learning (rotating eigenbasis)’, fn: profileRotate },
awaken:  { name: ‘Feature-learning (eigenvalue awakens)’,  fn: profileAwaken },
};

// ─────────────────────────────────────────────────────────────────────────────
// Numerical integration (RK4 in output space)
// r = u - y, dr/dt = -K(t) r,  but track u directly so we can drag y/u(0).
// ─────────────────────────────────────────────────────────────────────────────
function integrate({ u0, y, profile, params, tMax, steps = 600 }) {
const dt = tMax / steps;
const xs = new Float64Array(steps + 1);
const ys = new Float64Array(steps + 1);
const lams1 = new Float64Array(steps + 1);
const lams2 = new Float64Array(steps + 1);
const thetas = new Float64Array(steps + 1);
const losses = new Float64Array(steps + 1);
// Eigenbasis coordinates of r(t) wrt the *instantaneous* K(t):
const e1 = new Float64Array(steps + 1);
const e2 = new Float64Array(steps + 1);

let u = [u0[0], u0[1]];
for (let i = 0; i <= steps; i++) {
const t = i * dt;
const { lam1, lam2, theta } = profile.fn(t, params);
lams1[i] = lam1; lams2[i] = lam2; thetas[i] = theta;
xs[i] = u[0]; ys[i] = u[1];
const r = sub(u, y);
losses[i] = 0.5 * (r[0]*r[0] + r[1]*r[1]);
const Vt = tT(rot(theta));
const rT = mv(Vt, r);
e1[i] = rT[0]; e2[i] = rT[1];

```
if (i === steps) break;
// RK4 step on du/dt = -K(t) (u - y)
const f = (uu, tt) => {
  const pr = profile.fn(tt, params);
  const K = buildK(pr.lam1, pr.lam2, pr.theta);
  return sc(mv(K, sub(uu, y)), -1);
};
const k1 = f(u, t);
const k2 = f(add(u, sc(k1, dt/2)), t + dt/2);
const k3 = f(add(u, sc(k2, dt/2)), t + dt/2);
const k4 = f(add(u, sc(k3, dt)),   t + dt);
u = add(u, sc(add(add(k1, sc(k2, 2)), add(sc(k3, 2), k4)), dt/6));
```

}
return { xs, ys, lams1, lams2, thetas, losses, e1, e2, dt, steps };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function useDrag(svgRef, fromScreen) {
const [dragging, setDragging] = useState(null);
const onDown = (key) => (e) => { e.preventDefault(); setDragging(key); };
const onUp = () => setDragging(null);
const makeMove = (handler) => (e) => {
if (!dragging) return;
const svg = svgRef.current;
if (!svg) return;
const pt = svg.createSVGPoint();
const evt = e.touches ? e.touches[0] : e;
pt.x = evt.clientX; pt.y = evt.clientY;
const m = svg.getScreenCTM();
if (!m) return;
const lp = pt.matrixTransform(m.inverse());
handler(dragging, fromScreen(lp.x, lp.y));
};
return { dragging, onDown, onUp, makeMove };
}

function Slider({ label, value, setValue, min, max, step, accent, fmt, disabled }) {
const f = fmt || ((v) => v.toFixed(2));
return (
<div className=“mb-2” style={{ opacity: disabled ? 0.4 : 1 }}>
<div className="flex justify-between items-baseline">
<span className=“mono text-xs” style={{ color: C.inkSoft }}>{label}</span>
<span className=“mono text-xs” style={{ color: accent || C.ink }}>{f(value)}</span>
</div>
<input type=“range” min={min} max={max} step={step} value={value} disabled={disabled}
onChange={(e) => setValue(parseFloat(e.target.value))}
className=“w-full” style={{ accentColor: accent || C.ink }} />
</div>
);
}

// ─────────────────────────────────────────────────────────────────────────────
// Output-space view
// ─────────────────────────────────────────────────────────────────────────────
function OutputView({ u0, setU0, y, setY, traj, tIdx, showKer, showReach }) {
const W = 460, H = 400, S = 65;
const cx = W/2, cy = H/2;
const toScreen = (v) => ({ x: cx + v[0]*S, y: cy - v[1]*S });
const fromScreen = (px, py) => [(px - cx)/S, -(py - cy)/S];

const svgRef = useRef(null);
const { onDown, onUp, makeMove } = useDrag(svgRef, fromScreen);
const onMove = makeMove((key, val) => {
const clamped = [Math.max(-3, Math.min(3, val[0])), Math.max(-3, Math.min(3, val[1]))];
if (key === ‘u0’) setU0(clamped);
else if (key === ‘y’) setY(clamped);
});

const idx = Math.min(tIdx, traj.steps);
const u = [traj.xs[idx], traj.ys[idx]];
const r = sub(u, y);
const lam1 = traj.lams1[idx], lam2 = traj.lams2[idx], theta = traj.thetas[idx];
const K = buildK(lam1, lam2, theta);
const gWant = sc(r, -1); // -g points from u toward y
const udot = sc(mv(K, r), -1); // u̇ = -K r

// eigenvectors (instantaneous)
const v1 = [Math.cos(theta), Math.sin(theta)];
const v2 = [-Math.sin(theta), Math.cos(theta)];

// ker K_SS: the eigendirection with the smaller eigenvalue, through origin
const kerVec = lam2 < lam1 ? v2 : v1;
const rangeVec = lam2 < lam1 ? v1 : v2;
const kerLam = Math.min(lam1, lam2);

// Trajectory points
const pts = [];
for (let i = 0; i <= idx; i++) pts.push({ x: traj.xs[i], y: traj.ys[i] });

// future ghost
const ghost = [];
for (let i = idx; i <= traj.steps; i++) ghost.push({ x: traj.xs[i], y: traj.ys[i] });

// Long lines for kernel (through origin) and reach (through u(0))
const longLine = (point, dir, len = 5) => {
const a = add(point, sc(dir, len));
const b = add(point, sc(dir, -len));
return { a: toScreen(a), b: toScreen(b) };
};
const kerLine = longLine([0,0], kerVec);
const reachLine = longLine(u0, rangeVec);

const uS = toScreen(u);
const u0S = toScreen(u0);
const yS = toScreen(y);
const origin = { x: cx, y: cy };
const gEnd = toScreen(add(u, sc(gWant, 0.5)));
const udotEnd = toScreen(add(u, sc(udot, 0.5)));

// Eigendirection dashes (instantaneous, faint)
const e1a = toScreen(sc(v1, 3.5));
const e1b = toScreen(sc(v1, -3.5));
const e2a = toScreen(sc(v2, 3.5));
const e2b = toScreen(sc(v2, -3.5));

return (
<svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className=“w-full h-auto select-none touch-none”
style={{ maxHeight: 460 }}
onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
onTouchMove={onMove} onTouchEnd={onUp}>
<rect x="0" y="0" width={W} height={H} fill={C.bg} />

```
  {/* grid */}
  {[...Array(7)].map((_, i) => {
    const k = i - 3;
    return (
      <g key={i}>
        <line x1={cx + k*S} y1={0} x2={cx + k*S} y2={H} stroke={C.grid} strokeWidth="0.5" opacity={k === 0 ? 0.6 : 0.18} />
        <line x1={0} y1={cy + k*S} x2={W} y2={cy + k*S} stroke={C.grid} strokeWidth="0.5" opacity={k === 0 ? 0.6 : 0.18} />
      </g>
    );
  })}

  <defs>
    <marker id="arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill={C.gWant} />
    </marker>
    <marker id="arr-u" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill={C.uVel} />
    </marker>
  </defs>

  {/* Eigendirections (instantaneous), faint */}
  <line x1={e1a.x} y1={e1a.y} x2={e1b.x} y2={e1b.y} stroke={C.v1}
        strokeWidth="0.9" strokeDasharray="2 4" opacity="0.55" />
  <line x1={e2a.x} y1={e2a.y} x2={e2b.x} y2={e2b.y} stroke={C.v2}
        strokeWidth="0.9" strokeDasharray="2 4" opacity="0.55" />

  {/* ker K_SS line through origin */}
  {showKer && (
    <>
      <line x1={kerLine.a.x} y1={kerLine.a.y} x2={kerLine.b.x} y2={kerLine.b.y}
            stroke={C.kernel} strokeWidth="2" opacity={kerLam < 0.15 ? 0.85 : 0.35} />
      <text x={kerLine.a.x + 6} y={kerLine.a.y - 4} fontSize="11" className="mono"
            fill={C.kernel} opacity={kerLam < 0.15 ? 1 : 0.4}>
        <tspan fontStyle="italic">ker</tspan> K_SS · through origin
      </text>
    </>
  )}

  {/* reachable set from u(0): line through u(0) along range */}
  {showReach && (
    <>
      <line x1={reachLine.a.x} y1={reachLine.a.y} x2={reachLine.b.x} y2={reachLine.b.y}
            stroke={C.reach} strokeWidth="2" strokeDasharray="6 3" opacity={kerLam < 0.15 ? 0.85 : 0.35} />
      <text x={reachLine.a.x + 6} y={reachLine.a.y - 4} fontSize="11" className="mono"
            fill={C.reach} opacity={kerLam < 0.15 ? 1 : 0.4}>
        u(0) + range K_SS
      </text>
    </>
  )}

  {/* Trajectory: past (solid) and future (ghost) */}
  <polyline
    points={ghost.map(p => { const s = toScreen([p.x, p.y]); return `${s.x},${s.y}`; }).join(' ')}
    stroke={C.inkFaint} strokeWidth="0.8" fill="none" strokeDasharray="1 4" opacity="0.5" />
  <polyline
    points={pts.map(p => { const s = toScreen([p.x, p.y]); return `${s.x},${s.y}`; }).join(' ')}
    stroke={C.inkSoft} strokeWidth="1.6" fill="none" />

  {/* -g and u̇ arrows */}
  <line x1={uS.x} y1={uS.y} x2={gEnd.x} y2={gEnd.y} stroke={C.gWant}
        strokeWidth="1.7" markerEnd="url(#arr-g)" opacity="0.9" />
  <line x1={uS.x} y1={uS.y} x2={udotEnd.x} y2={udotEnd.y} stroke={C.uVel}
        strokeWidth="2.4" markerEnd="url(#arr-u)" />

  {/* y (target) */}
  <g onMouseDown={onDown('y')} onTouchStart={onDown('y')} style={{ cursor: 'grab' }}>
    <circle cx={yS.x} cy={yS.y} r="14" fill="transparent" />
    <path d={`M ${yS.x-7} ${yS.y} L ${yS.x+7} ${yS.y} M ${yS.x} ${yS.y-7} L ${yS.x} ${yS.y+7}`} stroke={C.ink} strokeWidth="1.8" />
    <circle cx={yS.x} cy={yS.y} r="5" fill="none" stroke={C.ink} strokeWidth="1.8" />
    <text x={yS.x + 11} y={yS.y + 4} fontSize="14" className="serif" fontStyle="italic" fill={C.ink}>y</text>
  </g>

  {/* u(0) */}
  <g onMouseDown={onDown('u0')} onTouchStart={onDown('u0')} style={{ cursor: 'grab' }}>
    <circle cx={u0S.x} cy={u0S.y} r="14" fill="transparent" />
    <circle cx={u0S.x} cy={u0S.y} r="5" fill="none" stroke={C.inkSoft} strokeWidth="1.8" />
    <text x={u0S.x + 10} y={u0S.y - 8} fontSize="11" className="serif" fontStyle="italic" fill={C.inkSoft}>u(0)</text>
  </g>

  {/* u(t) */}
  <circle cx={uS.x} cy={uS.y} r="6.5" fill={C.ink} />
  <text x={uS.x + 11} y={uS.y - 9} fontSize="13" className="serif" fontStyle="italic" fill={C.ink}>u(t)</text>

  {/* origin marker */}
  <circle cx={origin.x} cy={origin.y} r="2" fill={C.inkFaint} />
</svg>
```

);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel evolution view (small inset showing how K(t) changes)
// ─────────────────────────────────────────────────────────────────────────────
function KernelEvolution({ traj, tIdx }) {
const N = traj.steps;
const data = useMemo(() => {
const arr = [];
const stride = Math.max(1, Math.floor(N / 100));
for (let i = 0; i <= N; i += stride) {
arr.push({
t: i * traj.dt,
lam1: traj.lams1[i],
lam2: traj.lams2[i],
theta_deg: traj.thetas[i] * 180 / Math.PI,
});
}
return arr;
}, [traj]);
const currentT = tIdx * traj.dt;
const tickStyle = { fill: C.inkSoft, fontSize: 9, fontFamily: ‘JetBrains Mono, monospace’ };

return (
<div className="grid grid-cols-1 gap-2">
<div>
<div className=“mono text-xs” style={{ color: C.inkFaint }}>Eigenvalues of K_SS(t)</div>
<div style={{ width: ‘100%’, height: 110 }}>
<ResponsiveContainer>
<LineChart data={data} margin={{ top: 4, right: 6, left: 0, bottom: 0 }}>
<CartesianGrid stroke={C.rule} strokeDasharray="2 4" />
<XAxis dataKey=“t” type=“number” domain={[0, traj.steps * traj.dt]}
tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(1)} />
<YAxis tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(1)} />
<ReferenceLine x={currentT} stroke={C.ink} strokeDasharray="3 3" strokeWidth={1} />
<Line type="monotone" dataKey="lam1" stroke={C.v1} strokeWidth={1.6} dot={false} isAnimationActive={false} />
<Line type="monotone" dataKey="lam2" stroke={C.v2} strokeWidth={1.6} dot={false} isAnimationActive={false} />
</LineChart>
</ResponsiveContainer>
</div>
</div>
<div>
<div className=“mono text-xs” style={{ color: C.inkFaint }}>Eigenbasis angle θ(t)</div>
<div style={{ width: ‘100%’, height: 90 }}>
<ResponsiveContainer>
<LineChart data={data} margin={{ top: 4, right: 6, left: 0, bottom: 0 }}>
<CartesianGrid stroke={C.rule} strokeDasharray="2 4" />
<XAxis dataKey=“t” type=“number” domain={[0, traj.steps * traj.dt]}
tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(1)} />
<YAxis tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(0)} />
<ReferenceLine x={currentT} stroke={C.ink} strokeDasharray="3 3" strokeWidth={1} />
<Line type="monotone" dataKey="theta_deg" stroke={C.ink} strokeWidth={1.6} dot={false} isAnimationActive={false} />
</LineChart>
</ResponsiveContainer>
</div>
</div>
</div>
);
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss curve
// ─────────────────────────────────────────────────────────────────────────────
function LossView({ traj, tIdx }) {
const N = traj.steps;
const data = useMemo(() => {
const arr = [];
const stride = Math.max(1, Math.floor(N / 100));
for (let i = 0; i <= N; i += stride) {
arr.push({ t: i * traj.dt, loss: traj.losses[i] });
}
return arr;
}, [traj]);
const currentT = tIdx * traj.dt;
const tickStyle = { fill: C.inkSoft, fontSize: 9, fontFamily: ‘JetBrains Mono, monospace’ };
return (
<div>
<div className=“mono text-xs” style={{ color: C.inkFaint }}>Loss Φ_S(u(t))</div>
<div style={{ width: ‘100%’, height: 120 }}>
<ResponsiveContainer>
<LineChart data={data} margin={{ top: 4, right: 6, left: 0, bottom: 0 }}>
<CartesianGrid stroke={C.rule} strokeDasharray="2 4" />
<XAxis dataKey=“t” type=“number” domain={[0, traj.steps * traj.dt]}
tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(1)} />
<YAxis tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(2)} />
<ReferenceLine x={currentT} stroke={C.ink} strokeDasharray="3 3" strokeWidth={1} />
<Tooltip contentStyle={{ background: C.bg, border: `1px solid ${C.rule}`,
fontFamily: ‘JetBrains Mono, monospace’, fontSize: 11 }}
labelFormatter={(v) => `t = ${Number(v).toFixed(2)}`}
formatter={(v) => Number(v).toFixed(4)} />
<Line type="monotone" dataKey="loss" stroke={C.uVel} strokeWidth={1.8} dot={false} isAnimationActive={false} />
</LineChart>
</ResponsiveContainer>
</div>
</div>
);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
const tMax = 10;
const [mode, setMode] = useState(‘awaken’); // ‘lazy’ | ‘rotate’ | ‘awaken’
const [u0, setU0] = useState([2.0, 1.4]);
const [y, setY] = useState([-0.6, -0.4]);

// shared / mode-specific params
const [lam1_0, setLam1_0]   = useState(1.5);
const [lam2_0, setLam2_0]   = useState(0.03);
const [lam2_final, setLam2Final] = useState(1.2);
const [theta_0, setTheta0]  = useState(0.5);
const [delta_theta, setDeltaTheta] = useState(1.2);
const [wake_t, setWakeT]    = useState(4.0);
const [wake_dur, setWakeDur] = useState(0.8);

const [tIdx, setTIdx] = useState(0);
const [playing, setPlaying] = useState(false);
const lastTs = useRef(null);
const [showKer, setShowKer] = useState(true);
const [showReach, setShowReach] = useState(true);

const params = {
lam1_0, lam2_0, lam2_final, theta_0, delta_theta, wake_t, wake_dur, tMax,
};
const profile = PROFILES[mode];

const traj = useMemo(
() => integrate({ u0, y, profile, params, tMax, steps: 600 }),
[u0, y, mode, lam1_0, lam2_0, lam2_final, theta_0, delta_theta, wake_t, wake_dur]
);

// playback
useEffect(() => {
if (!playing) { lastTs.current = null; return; }
let raf;
const step = (ts) => {
if (lastTs.current == null) lastTs.current = ts;
const dt = (ts - lastTs.current) / 1000;
lastTs.current = ts;
setTIdx((cur) => {
const nxt = cur + dt * traj.steps / tMax * 0.5; // half speed
if (nxt >= traj.steps) { setPlaying(false); return traj.steps; }
return nxt;
});
raf = requestAnimationFrame(step);
};
raf = requestAnimationFrame(step);
return () => cancelAnimationFrame(raf);
}, [playing, traj.steps]);

// reset when mode changes
useEffect(() => { setTIdx(0); setPlaying(false); }, [mode]);

const reset = () => { setTIdx(0); setPlaying(false); };
const currentT = tIdx * traj.dt;

// current instantaneous values
const idx = Math.min(Math.floor(tIdx), traj.steps);
const curL1 = traj.lams1[idx], curL2 = traj.lams2[idx], curTheta = traj.thetas[idx];
const curLoss = traj.losses[idx];
const finalLoss = traj.losses[traj.steps];

const modeBlurb = {
lazy: ‘NTK is constant. ker K_SS is a fixed line through the origin; the reachable set from u(0) is a fixed line. If y is not on that line, u(∞) stalls at the closest point — permanently.’,
rotate: ‘NTK eigenvectors rotate over time. ker K_SS is the same line through origin but it ROTATES with t. So the unreachable direction at t = 0 becomes reachable later — u escapes a previously-frozen subspace.’,
awaken: ‘NTK eigenvectors are fixed but a small eigenvalue grows. ker K_SS shrinks to {0} once λ_2 lifts off zero. The previously-frozen direction comes online and the remaining residual finally drains.’,
};

return (
<div className=“min-h-screen w-full” style={{ background: C.bg, color: C.ink }}>
<style>{FONT_IMPORT}</style>

```
  {/* Header */}
  <div className="max-w-6xl mx-auto px-5 pt-6 pb-3">
    <div className="mono text-xs uppercase tracking-widest" style={{ color: C.inkFaint }}>
      Litman & Guo · feature-learning extension
    </div>
    <h1 className="serif text-3xl md:text-4xl mt-1 mb-1" style={{ color: C.ink, fontWeight: 600 }}>
      When the kernel <em>moves</em>: ker K<sub>SS</sub> vs reachable set
    </h1>
    <p className="serif text-base md:text-lg italic" style={{ color: C.inkSoft, lineHeight: 1.4 }}>
      The lazy regime versus two flavours of feature learning. Watch the previously-frozen
      subspace come unstuck.
    </p>
    <div className="mt-3" style={{ height: 1, background: C.rule }} />
  </div>

  <div className="max-w-6xl mx-auto px-5 pb-12">

    {/* mode selector */}
    <div className="mb-4 grid grid-cols-1 md:grid-cols-3 gap-2">
      {Object.entries(PROFILES).map(([key, p]) => (
        <button key={key} onClick={() => setMode(key)}
                className="text-left p-3 rounded transition-all"
                style={{
                  background: mode === key ? C.ink : C.bgAlt,
                  color: mode === key ? C.bg : C.ink,
                  border: `1px solid ${mode === key ? C.ink : C.rule}`,
                }}>
          <div className="flex items-center gap-2 mono text-xs uppercase tracking-wider mb-1"
               style={{ color: mode === key ? C.bgAlt : C.inkFaint }}>
            {key === 'lazy' ? <Lock size={12} /> : <Zap size={12} />}
            {key}
          </div>
          <div className="serif text-sm" style={{ fontWeight: mode === key ? 600 : 400 }}>{p.name}</div>
        </button>
      ))}
    </div>

    {/* mode description */}
    <div className="mb-5 p-3 rounded serif italic text-sm"
         style={{ background: C.bgAlt, border: `1px solid ${C.rule}`, color: C.inkSoft }}>
      {modeBlurb[mode]}
    </div>

    <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
      {/* Output-space panel */}
      <div className="lg:col-span-3">
        <div className="rounded p-3" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
          <OutputView u0={u0} setU0={setU0} y={y} setY={setY}
                      traj={traj} tIdx={Math.floor(tIdx)}
                      showKer={showKer} showReach={showReach} />

          {/* toggle row */}
          <div className="mt-2 flex flex-wrap gap-2 items-center">
            <button onClick={() => setShowKer(s => !s)}
                    className="px-2 py-1 mono text-xs uppercase tracking-wider rounded"
                    style={{ background: showKer ? C.kernel : C.bg,
                             color: showKer ? C.bg : C.kernel,
                             border: `1px solid ${C.kernel}` }}>
              ker K_SS  {showKer ? '✓' : ''}
            </button>
            <button onClick={() => setShowReach(s => !s)}
                    className="px-2 py-1 mono text-xs uppercase tracking-wider rounded"
                    style={{ background: showReach ? C.reach : C.bg,
                             color: showReach ? C.bg : C.reach,
                             border: `1px solid ${C.reach}` }}>
              u(0) + range K_SS  {showReach ? '✓' : ''}
            </button>
          </div>
          <div className="mono text-xs mt-2" style={{ color: C.inkFaint }}>
            Drag u(0) and y to reposition. Solid trail = past, dotted = future.
          </div>
        </div>

        {/* playback */}
        <div className="mt-3 p-3 rounded" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
          <div className="flex items-center gap-3">
            <button onClick={() => setPlaying(p => !p)} className="p-2 rounded"
                    style={{ background: C.ink, color: C.bg }}>
              {playing ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button onClick={reset} className="p-2 rounded"
                    style={{ background: C.bg, color: C.ink, border: `1px solid ${C.rule}` }}>
              <RotateCcw size={16} />
            </button>
            <div className="flex-1">
              <input type="range" min={0} max={traj.steps} step={1} value={tIdx}
                     onChange={(e) => { setTIdx(parseInt(e.target.value)); setPlaying(false); }}
                     className="w-full" style={{ accentColor: C.ink }} />
              <div className="flex justify-between mono text-xs mt-1" style={{ color: C.inkSoft }}>
                <span>t = {currentT.toFixed(2)}</span>
                <span>loss = {curLoss.toFixed(4)}</span>
                <span>t_max = {tMax.toFixed(1)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* legend */}
        <div className="mt-3 p-3 rounded grid grid-cols-2 gap-2 mono text-xs"
             style={{ background: C.bgAlt, border: `1px solid ${C.rule}`, color: C.inkSoft }}>
          <div className="flex items-center gap-2">
            <span className="inline-block" style={{ width: 18, height: 2, background: C.gWant }} />
            <span>−g (loss's desired direction)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block" style={{ width: 18, height: 2.4, background: C.uVel }} />
            <span>u̇ = −K_SS g (actual)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block" style={{ width: 18, height: 2, background: C.kernel }} />
            <span>ker K_SS — frozen direction</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-block" style={{ width: 18, height: 2, background: C.reach, opacity: 0.85 }} />
            <span>u(0) + range — reachable line</span>
          </div>
        </div>
      </div>

      {/* Plots + controls panel */}
      <div className="lg:col-span-2 space-y-3">
        <div className="rounded p-3" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
          <KernelEvolution traj={traj} tIdx={Math.floor(tIdx)} />
        </div>
        <div className="rounded p-3" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
          <LossView traj={traj} tIdx={Math.floor(tIdx)} />
        </div>

        {/* current instantaneous K display */}
        <div className="rounded p-3" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
          <div className="mono text-xs uppercase tracking-widest mb-2" style={{ color: C.inkFaint }}>
            K_SS at t = {currentT.toFixed(2)}
          </div>
          <div className="grid grid-cols-3 gap-2 mono text-xs">
            <div><span style={{ color: C.v1 }}>λ₁</span> = {curL1.toFixed(3)}</div>
            <div><span style={{ color: C.v2 }}>λ₂</span> = {curL2.toFixed(3)}</div>
            <div><span style={{ color: C.ink }}>θ</span>  = {(curTheta*180/Math.PI).toFixed(1)}°</div>
          </div>
          {Math.min(curL1, curL2) < 0.05 && (
            <div className="mt-2 mono text-xs" style={{ color: C.kernel }}>
              ⚠ near-degenerate: ker K_SS is essentially a line
            </div>
          )}
        </div>
      </div>
    </div>

    {/* Controls */}
    <div className="mt-5 grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="rounded p-4" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
        <div className="serif italic text-sm mb-2" style={{ color: C.ink }}>NTK at t = 0</div>
        <Slider label="λ₁ (fast eigenvalue, initial)" value={lam1_0} setValue={setLam1_0}
                min={0.05} max={2.5} step={0.01} accent={C.v1} />
        <Slider label="λ₂ (slow eigenvalue, initial)" value={lam2_0} setValue={setLam2_0}
                min={0.0} max={2.5} step={0.01} accent={C.v2} />
        <Slider label="θ₀ (initial eigenbasis angle)" value={theta_0} setValue={setTheta0}
                min={-Math.PI/2} max={Math.PI/2} step={0.01} accent={C.ink}
                fmt={(v) => `${(v*180/Math.PI).toFixed(0)}°`} />
      </div>

      <div className="rounded p-4" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
        <div className="serif italic text-sm mb-2" style={{ color: C.ink }}>Schedule (mode-specific)</div>
        <Slider label="Δθ (total eigenbasis rotation over training)"
                value={delta_theta} setValue={setDeltaTheta}
                min={-Math.PI} max={Math.PI} step={0.01}
                accent={C.ink} disabled={mode !== 'rotate'}
                fmt={(v) => `${(v*180/Math.PI).toFixed(0)}°`} />
        <Slider label="t_wake (when λ₂ starts ramping)" value={wake_t} setValue={setWakeT}
                min={0} max={tMax} step={0.01} accent={C.v2} disabled={mode !== 'awaken'} />
        <Slider label="λ₂ final (end-of-training value)" value={lam2_final} setValue={setLam2Final}
                min={0} max={2.5} step={0.01} accent={C.v2} disabled={mode !== 'awaken'} />
      </div>
    </div>

    {/* Reading guide */}
    <div style={{ height: 1, background: C.rule }} className="my-6" />
    <div className="serif text-xs uppercase tracking-widest mb-3" style={{ color: C.inkFaint }}>Reading the three modes</div>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-5 serif text-sm leading-snug" style={{ color: C.inkSoft }}>
      <div>
        <div className="mono text-xs uppercase tracking-widest mb-1 flex items-center gap-1" style={{ color: C.ink }}>
          <Lock size={11} /> lazy
        </div>
        ker K_SS and u(0)+range stay put for all t. The trajectory hits the dashed reachable line and slides along it. If y isn't on that line, the gap is permanent — loss plateaus at a nonzero value. The matrix exponential e^(−tK_SS) describes everything; no propagator needed.
      </div>
      <div>
        <div className="mono text-xs uppercase tracking-widest mb-1 flex items-center gap-1" style={{ color: C.ink }}>
          <Zap size={11} /> rotate
        </div>
        Eigenvalues stay the same, but θ(t) drifts. ker K_SS sweeps through the plane; the "frozen" direction is different at every moment. The trajectory escapes via the rotation: a residual along the kernel at t=0 stops being in the kernel later, and gets fitted. K_SS(τ₁) and K_SS(τ₂) no longer commute — this is where the propagator becomes necessary.
      </div>
      <div>
        <div className="mono text-xs uppercase tracking-widest mb-1 flex items-center gap-1" style={{ color: C.ink }}>
          <Zap size={11} /> awaken
        </div>
        Eigenvectors fixed but λ₂ ramps from 0 to a positive value at t = t_wake. Before the wake, the v₂ direction is frozen and the loss curve plateaus. After the wake, v₂ unfreezes and the residual drains. Compare to "grokking"-like late-training jumps: a previously-inactive mode comes online and finishes the job.
      </div>
    </div>

    {/* Suggested experiments */}
    <div className="mt-6 rounded p-4" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
      <div className="serif text-xs uppercase tracking-widest mb-2" style={{ color: C.inkFaint }}>Experiments worth running</div>
      <ul className="serif text-sm leading-snug space-y-2" style={{ color: C.inkSoft }}>
        <li>
          <span className="mono text-xs" style={{ color: C.ink }}>1.</span> In <em>lazy</em> mode with λ₂ ≈ 0, place y off the dashed reachable line. Note where u(t) stalls. Now switch to <em>rotate</em> with the same initial conditions — the trajectory now wanders off the original reachable line as θ drifts, and eventually reaches y.
        </li>
        <li>
          <span className="mono text-xs" style={{ color: C.ink }}>2.</span> In <em>awaken</em> mode, watch the loss curve. Before t_wake, it plateaus. After t_wake, a sharp second drop kicks in. Slide t_wake left/right — you're choosing when feature learning happens.
        </li>
        <li>
          <span className="mono text-xs" style={{ color: C.ink }}>3.</span> In <em>rotate</em> mode, set Δθ near 0. Trajectory looks almost lazy. Crank Δθ to 180° and watch the kernel basis flip completely — the early "frozen" direction becomes the "fast" one. This is the picture the propagator is solving.
        </li>
      </ul>
    </div>
  </div>
</div>
```

);
}