import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid, Legend } from 'recharts';
import { Play, Pause, RotateCcw } from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────
// Typography & color tokens
// ─────────────────────────────────────────────────────────────────────────────
const FONT_IMPORT = `
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');
.serif { font-family: 'Crimson Pro', Georgia, serif; }
.mono  { font-family: 'JetBrains Mono', monospace; }
.kc { font-feature-settings: "ss01"; }
`;

const C = {
  bg:        '#f5f0e6',   // warm paper
  bgAlt:     '#ebe4d4',
  ink:       '#1a1814',
  inkSoft:   '#4a463d',
  inkFaint:  '#8a8475',
  rule:      '#c9c0a8',
  accent:    '#7a2424',   // burgundy
  accent2:   '#2d5a5a',   // teal
  accent3:   '#b87333',   // copper
  upstream:  '#2d5a5a',
  downstream:'#7a2424',
  selected:  '#1a1814',
};

// ─────────────────────────────────────────────────────────────────────────────
// Dependency graph spec
// ─────────────────────────────────────────────────────────────────────────────
// Layout uses (col, row) with col ∈ [0..3], row ∈ [0..8].
const NODES = [
  { id: 'os',  label: 'ℝⁿᵖ',  name: 'Output space',         col: 0, row: 0, parents: [],
    formula: 'ℝ^{np}',
    blurb: 'The ambient space where stacked predictions live. p output dims × n training points.' },
  { id: 'ws',  label: 'ℝᵈ',   name: 'Weight space',         col: 3, row: 0, parents: [],
    formula: 'ℝ^d',
    blurb: 'Where the model parameters w live.' },
  { id: 'u',   label: 'u',     name: 'Stacked outputs',      col: 0, row: 1, parents: ['os','ws'],
    formula: 'u = U_S(w)  ∈ ℝ^{np}',
    blurb: 'Predictions on the training set, treated as a single point in output space. Determined by w.' },
  { id: 'J',   label: 'J_S',   name: 'Jacobian',             col: 3, row: 1, parents: ['u','ws'],
    formula: 'J_S = D_w U_S  ∈ ℝ^{np×d}',
    blurb: 'How outputs respond to weight perturbations. The only formal bridge between weight space and output space.' },
  { id: 'phi', label: 'Φ_S',   name: 'Output-space loss',    col: 0, row: 2, parents: ['u'],
    formula: 'Φ_S(u),  convex C²',
    blurb: 'Scalar loss on output space. For squared loss, Φ_S(u) = (1/2n)‖u−y‖².' },
  { id: 'L',   label: 'L_S',   name: 'Weight-space loss',    col: 3, row: 2, parents: ['phi','u'],
    formula: 'L_S(w) = Φ_S(U_S(w))',
    blurb: 'What gradient descent actually sees. Pullback of Φ_S to weight space via U_S.' },
  { id: 'g',   label: 'g',     name: 'Output gradient',      col: 0, row: 3, parents: ['phi'],
    formula: 'g = ∇_u Φ_S',
    blurb: 'Where the loss wants outputs to go. For squared loss g = (u−y)/n — the residual.' },
  { id: 'B',   label: 'B',     name: 'Output Hessian',       col: 1, row: 3, parents: ['phi'],
    formula: 'B = ∇²_u Φ_S',
    blurb: 'Curvature of the loss in output space. For squared loss, B = I/n.' },
  { id: 'gL',  label: '∇L',    name: 'Weight gradient',      col: 3, row: 3, parents: ['g','J','L'],
    formula: '∇_w L_S = J_S^⊤ g',
    blurb: 'Chain rule: routes output gradient back to weight space through J_S^⊤.' },
  { id: 'K',   label: 'K_SS',  name: 'NTK',                  col: 2, row: 4, parents: ['J'],
    formula: 'K_{SS} = J_S J_S^⊤  ⪰ 0',
    blurb: 'Reachable subspace in output space. Built from J_S alone — no loss info. ker K_{SS} is frozen.' },
  { id: 'wd',  label: 'ẇ',     name: 'Gradient flow',        col: 3, row: 4, parents: ['gL'],
    formula: '∂_t w = −J_S^⊤ g',
    blurb: 'The training rule, in continuous time. Choosing to descend in weight space is what makes the NTK relevant.' },
  { id: 'ud',  label: 'u̇',     name: 'Output velocity',      col: 0, row: 5, parents: ['K','g','wd'],
    formula: '∂_t u = −K_{SS} g   (Eq. 4)',
    blurb: 'Chain rule: J_S(−J_S^⊤g) = −K_{SS}g. The loss\'s desired direction filtered through the reachable subspace.' },
  { id: 'dis', label: 'Φ̇_S',   name: 'Loss dissipation',    col: 1, row: 6, parents: ['ud','g'],
    formula: 'Φ̇_S = −g^⊤ K_{SS} g = −‖J_S^⊤g‖²   (Eq. 6)',
    blurb: 'Inner product of g and u̇. Loss only decreases where J_S^⊤g ≠ 0.' },
  { id: 'gd',  label: 'ġ',     name: 'Output-gradient ODE', col: 2, row: 6, parents: ['B','K','g','ud'],
    formula: '∂_t g = −B K_{SS} g   (Eq. 5)',
    blurb: 'Chain rule: B·u̇. Linear in g, but with a time-varying coefficient matrix B(t)K_{SS}(t).' },
  { id: 'P',   label: 'P_g',   name: 'Propagator',           col: 2, row: 7, parents: ['gd'],
    formula: '∂_t P_g = −B K_{SS} P_g,  P_g(s,s)=I   (Eq. 7)',
    blurb: 'Solution operator: g(t) = P_g(t,s)g(s). The right object when K_{SS} eigenvectors rotate.' },
];

const NODE_BY_ID = Object.fromEntries(NODES.map(n => [n.id, n]));

function buildLineage(rootId) {
  const upstream = new Set();
  const downstream = new Set();
  function up(id) {
    const n = NODE_BY_ID[id];
    if (!n) return;
    for (const p of n.parents) {
      if (!upstream.has(p)) {
        upstream.add(p);
        up(p);
      }
    }
  }
  function down(id) {
    for (const n of NODES) {
      if (n.parents.includes(id) && !downstream.has(n.id)) {
        downstream.add(n.id);
        down(n.id);
      }
    }
  }
  up(rootId);
  down(rootId);
  return { upstream, downstream };
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear algebra utilities (2x2 only — that's all we need)
// ─────────────────────────────────────────────────────────────────────────────
function mat2(a, b, c, d) { return [a, b, c, d]; } // row-major [a b; c d]
function rot(theta) { const c = Math.cos(theta), s = Math.sin(theta); return [c, -s, s, c]; }
function transp(M) { return [M[0], M[2], M[1], M[3]]; }
function matmul(A, B) {
  return [
    A[0]*B[0]+A[1]*B[2], A[0]*B[1]+A[1]*B[3],
    A[2]*B[0]+A[3]*B[2], A[2]*B[1]+A[3]*B[3],
  ];
}
function matvec(M, v) { return [M[0]*v[0]+M[1]*v[1], M[2]*v[0]+M[3]*v[1]]; }
function diagMat(d1, d2) { return [d1, 0, 0, d2]; }
function sub(a, b) { return [a[0]-b[0], a[1]-b[1]]; }
function add(a, b) { return [a[0]+b[0], a[1]+b[1]]; }
function scale(a, s) { return [a[0]*s, a[1]*s]; }
function norm(v) { return Math.hypot(v[0], v[1]); }

// K_SS = V * diag(lam1, lam2) * V^T,  V = R(theta)
function buildK(lam1, lam2, theta) {
  const V = rot(theta);
  return matmul(matmul(V, diagMat(lam1, lam2)), transp(V));
}

// exp(-t * K) for K = V Λ V^T
function expNegTK(t, lam1, lam2, theta) {
  const V = rot(theta);
  const E = diagMat(Math.exp(-t*lam1), Math.exp(-t*lam2));
  return matmul(matmul(V, E), transp(V));
}

// u(t) = y + exp(-tK)(u0 - y).   With squared loss Φ(u)=½‖u−y‖², so g = u−y, B = I.
function solveU(t, u0, y, lam1, lam2, theta) {
  const E = expNegTK(t, lam1, lam2, theta);
  return add(y, matvec(E, sub(u0, y)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Dependency graph component
// ─────────────────────────────────────────────────────────────────────────────
function DepGraph({ selected, setSelected }) {
  const COL_W = 90;
  const ROW_H = 60;
  const PAD_X = 30;
  const PAD_Y = 28;
  const W = PAD_X*2 + COL_W*3;
  const H = PAD_Y*2 + ROW_H*7;

  const xy = (n) => ({ x: PAD_X + n.col * COL_W, y: PAD_Y + n.row * ROW_H });

  const { upstream, downstream } = useMemo(
    () => selected ? buildLineage(selected) : { upstream: new Set(), downstream: new Set() },
    [selected]
  );

  // edges: parent -> child
  const edges = [];
  for (const n of NODES) {
    for (const pid of n.parents) {
      edges.push({ from: pid, to: n.id });
    }
  }

  function edgeColor(e) {
    if (!selected) return C.rule;
    const onUp = (e.to === selected || upstream.has(e.to)) && (e.from === selected || upstream.has(e.from));
    const onDown = (e.from === selected || downstream.has(e.from)) && (e.to === selected || downstream.has(e.to));
    if (onUp) return C.upstream;
    if (onDown) return C.downstream;
    return '#dcd4bd';
  }
  function edgeWidth(e) {
    if (!selected) return 1;
    const onUp = (e.to === selected || upstream.has(e.to)) && (e.from === selected || upstream.has(e.from));
    const onDown = (e.from === selected || downstream.has(e.from)) && (e.to === selected || downstream.has(e.to));
    return (onUp || onDown) ? 1.8 : 0.7;
  }
  function nodeRole(id) {
    if (!selected) return 'idle';
    if (id === selected) return 'selected';
    if (upstream.has(id)) return 'upstream';
    if (downstream.has(id)) return 'downstream';
    return 'dim';
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" style={{ maxHeight: 480 }}>
      <defs>
        <marker id="arr-rule" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={C.rule} />
        </marker>
        <marker id="arr-up" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={C.upstream} />
        </marker>
        <marker id="arr-down" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={C.downstream} />
        </marker>
      </defs>

      {edges.map((e, i) => {
        const from = xy(NODE_BY_ID[e.from]);
        const to = xy(NODE_BY_ID[e.to]);
        const col = edgeColor(e);
        const mid = { x: (from.x+to.x)/2, y: (from.y+to.y)/2 - 6 };
        let marker = 'url(#arr-rule)';
        if (selected) {
          const onUp = (e.to === selected || upstream.has(e.to)) && (e.from === selected || upstream.has(e.from));
          const onDown = (e.from === selected || downstream.has(e.from)) && (e.to === selected || downstream.has(e.to));
          if (onUp) marker = 'url(#arr-up)';
          else if (onDown) marker = 'url(#arr-down)';
        }
        return (
          <path
            key={i}
            d={`M ${from.x} ${from.y+12} Q ${mid.x} ${mid.y} ${to.x} ${to.y-12}`}
            stroke={col}
            strokeWidth={edgeWidth(e)}
            fill="none"
            markerEnd={marker}
            opacity={selected && edgeColor(e) === '#dcd4bd' ? 0.4 : 1}
          />
        );
      })}

      {NODES.map((n) => {
        const { x, y } = xy(n);
        const role = nodeRole(n.id);
        let fill = C.bg, stroke = C.ink, txt = C.ink, op = 1;
        if (role === 'selected') { fill = C.ink; stroke = C.ink; txt = C.bg; }
        else if (role === 'upstream') { stroke = C.upstream; txt = C.upstream; }
        else if (role === 'downstream') { stroke = C.downstream; txt = C.downstream; }
        else if (role === 'dim') { op = 0.35; }
        return (
          <g key={n.id} onClick={() => setSelected(n.id === selected ? null : n.id)} style={{ cursor: 'pointer' }} opacity={op}>
            <ellipse cx={x} cy={y} rx={26} ry={11} fill={fill} stroke={stroke} strokeWidth={role === 'selected' ? 1.6 : 1} />
            <text x={x} y={y+4} fontSize={13} textAnchor="middle" className="serif kc" fill={txt} fontStyle="italic">
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D output-space simulation
// ─────────────────────────────────────────────────────────────────────────────
function OutputSpaceView({ u0, setU0, y, setY, lam1, lam2, theta, t, tMax }) {
  const W = 360;
  const H = 360;
  const SCALE = 60; // units: 1 = 60px
  const cx = W/2, cy = H/2;
  const toScreen = (v) => ({ x: cx + v[0]*SCALE, y: cy - v[1]*SCALE });
  const fromScreen = (px, py) => [(px - cx)/SCALE, -(py - cy)/SCALE];

  const u = solveU(t, u0, y, lam1, lam2, theta);
  const g = sub(u, y); // squared loss: g = u - y
  const K = buildK(lam1, lam2, theta);
  const udot = scale(matvec(K, g), -1); // u̇ = -K g

  // trajectory
  const traj = [];
  const N = 80;
  for (let i = 0; i <= N; i++) {
    const tt = (i/N) * tMax;
    traj.push(solveU(tt, u0, y, lam1, lam2, theta));
  }

  // eigenvector axes
  const v1 = matvec(rot(theta), [1, 0]);
  const v2 = matvec(rot(theta), [0, 1]);
  const axLen = 2.4;

  // dragging
  const [drag, setDrag] = useState(null); // 'u0' | 'y' | null
  const svgRef = useRef(null);
  const onDown = (which) => (e) => { e.preventDefault(); setDrag(which); };
  const onMove = (e) => {
    if (!drag) return;
    const svg = svgRef.current;
    if (!svg) return;
    const pt = svg.createSVGPoint();
    const evt = e.touches ? e.touches[0] : e;
    pt.x = evt.clientX; pt.y = evt.clientY;
    const m = svg.getScreenCTM();
    if (!m) return;
    const lp = pt.matrixTransform(m.inverse());
    const v = fromScreen(lp.x, lp.y);
    const clamped = [Math.max(-2.6, Math.min(2.6, v[0])), Math.max(-2.6, Math.min(2.6, v[1]))];
    if (drag === 'u0') setU0(clamped);
    else if (drag === 'y') setY(clamped);
  };
  const onUp = () => setDrag(null);

  const uS = toScreen(u);
  const u0S = toScreen(u0);
  const yS = toScreen(y);

  // velocity arrow endpoint
  const udotEnd = toScreen(add(u, scale(udot, 0.3)));
  const gNegEnd = toScreen(add(u, scale(scale(g, -1), 0.4))); // -g points toward y

  // eigenvector lines
  const ev1a = toScreen(scale(v1, axLen));
  const ev1b = toScreen(scale(v1, -axLen));
  const ev2a = toScreen(scale(v2, axLen));
  const ev2b = toScreen(scale(v2, -axLen));

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="w-full h-auto select-none touch-none"
      style={{ maxHeight: 420 }}
      onMouseMove={onMove}
      onMouseUp={onUp}
      onMouseLeave={onUp}
      onTouchMove={onMove}
      onTouchEnd={onUp}
    >
      {/* background */}
      <rect x="0" y="0" width={W} height={H} fill={C.bg} />

      {/* grid */}
      {[...Array(7)].map((_, i) => {
        const k = i - 3;
        return (
          <g key={i}>
            <line x1={cx + k*SCALE} y1={0} x2={cx + k*SCALE} y2={H} stroke={C.rule} strokeWidth="0.5" opacity={k === 0 ? 0.7 : 0.25} />
            <line x1={0} y1={cy + k*SCALE} x2={W} y2={cy + k*SCALE} stroke={C.rule} strokeWidth="0.5" opacity={k === 0 ? 0.7 : 0.25} />
          </g>
        );
      })}

      {/* eigenvector axes of K_SS */}
      <line x1={ev1a.x} y1={ev1a.y} x2={ev1b.x} y2={ev1b.y} stroke={C.accent2} strokeWidth="1" strokeDasharray="3 3" opacity="0.7" />
      <line x1={ev2a.x} y1={ev2a.y} x2={ev2b.x} y2={ev2b.y} stroke={C.accent2} strokeWidth="1" strokeDasharray="3 3" opacity="0.7" />
      <text x={ev1a.x + 6} y={ev1a.y - 4} fontSize="10" fill={C.accent2} className="mono">v₁ (λ₁={lam1.toFixed(2)})</text>
      <text x={ev2a.x + 6} y={ev2a.y - 4} fontSize="10" fill={C.accent2} className="mono">v₂ (λ₂={lam2.toFixed(2)})</text>

      {/* trajectory */}
      <polyline
        points={traj.map(p => { const s = toScreen(p); return `${s.x},${s.y}`; }).join(' ')}
        stroke={C.inkFaint}
        strokeWidth="1"
        fill="none"
        strokeDasharray="2 3"
      />

      {/* -g vector (direction loss wants u to go: toward y) */}
      <defs>
        <marker id="arrh-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={C.accent3} />
        </marker>
        <marker id="arrh-ud" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M 0 0 L 10 5 L 0 10 z" fill={C.accent} />
        </marker>
      </defs>
      <line x1={uS.x} y1={uS.y} x2={gNegEnd.x} y2={gNegEnd.y} stroke={C.accent3} strokeWidth="1.5" markerEnd="url(#arrh-g)" opacity="0.85" />
      <line x1={uS.x} y1={uS.y} x2={udotEnd.x} y2={udotEnd.y} stroke={C.accent} strokeWidth="2.2" markerEnd="url(#arrh-ud)" />

      {/* points */}
      {/* y (target) */}
      <g onMouseDown={onDown('y')} onTouchStart={onDown('y')} style={{ cursor: 'grab' }}>
        <circle cx={yS.x} cy={yS.y} r="14" fill="transparent" />
        <path d={`M ${yS.x-6} ${yS.y} L ${yS.x+6} ${yS.y} M ${yS.x} ${yS.y-6} L ${yS.x} ${yS.y+6}`} stroke={C.ink} strokeWidth="1.5" />
        <circle cx={yS.x} cy={yS.y} r="4" fill="none" stroke={C.ink} strokeWidth="1.5" />
        <text x={yS.x + 10} y={yS.y + 4} fontSize="13" className="serif" fill={C.ink} fontStyle="italic">y</text>
      </g>

      {/* u(0) (initial) */}
      <g onMouseDown={onDown('u0')} onTouchStart={onDown('u0')} style={{ cursor: 'grab' }}>
        <circle cx={u0S.x} cy={u0S.y} r="14" fill="transparent" />
        <circle cx={u0S.x} cy={u0S.y} r="5" fill="none" stroke={C.inkSoft} strokeWidth="1.5" />
        <text x={u0S.x + 9} y={u0S.y - 7} fontSize="11" className="serif" fill={C.inkSoft} fontStyle="italic">u(0)</text>
      </g>

      {/* u(t) (current) */}
      <circle cx={uS.x} cy={uS.y} r="6" fill={C.ink} />
      <text x={uS.x + 10} y={uS.y - 8} fontSize="13" className="serif" fill={C.ink} fontStyle="italic">u(t)</text>

      {/* legend */}
      <g transform={`translate(10, ${H - 64})`} fontSize="10" className="mono" fill={C.inkSoft}>
        <rect x="-4" y="-12" width="170" height="62" fill={C.bgAlt} opacity="0.8" stroke={C.rule} strokeWidth="0.5" />
        <line x1="0" y1="0" x2="20" y2="0" stroke={C.accent3} strokeWidth="1.5" markerEnd="url(#arrh-g)" />
        <text x="26" y="3">−g  (where loss wants u)</text>
        <line x1="0" y1="16" x2="20" y2="16" stroke={C.accent} strokeWidth="2.2" markerEnd="url(#arrh-ud)" />
        <text x="26" y="19">u̇ = −K_SS g  (actual)</text>
        <line x1="0" y1="32" x2="20" y2="32" stroke={C.accent2} strokeWidth="1" strokeDasharray="3 3" />
        <text x="26" y="35">eigendirections of K_SS</text>
      </g>
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Plots: loss + gradient norm + mode-wise decay
// ─────────────────────────────────────────────────────────────────────────────
function Plots({ u0, y, lam1, lam2, theta, tMax, t }) {
  const data = useMemo(() => {
    const N = 100;
    const arr = [];
    for (let i = 0; i <= N; i++) {
      const tt = (i/N) * tMax;
      const u = solveU(tt, u0, y, lam1, lam2, theta);
      const g = sub(u, y);
      const phi = 0.5 * (g[0]*g[0] + g[1]*g[1]);
      const gn = norm(g);
      // project g onto eigenbasis of K
      const Vt = transp(rot(theta));
      const gE = matvec(Vt, g);
      arr.push({
        t: tt,
        loss: phi,
        gnorm: gn,
        m1: Math.abs(gE[0]),
        m2: Math.abs(gE[1]),
      });
    }
    return arr;
  }, [u0, y, lam1, lam2, theta, tMax]);

  const tickStyle = { fill: C.inkSoft, fontSize: 10, fontFamily: 'JetBrains Mono, monospace' };

  return (
    <div className="grid grid-cols-1 gap-3">
      <div>
        <div className="serif kc text-sm italic mb-1" style={{ color: C.ink }}>Loss Φ_S(u(t))</div>
        <div style={{ width: '100%', height: 130 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid stroke={C.rule} strokeDasharray="2 4" />
              <XAxis dataKey="t" type="number" domain={[0, tMax]} tick={tickStyle} stroke={C.rule}
                     tickFormatter={(v) => v.toFixed(1)} />
              <YAxis tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={{ background: C.bg, border: `1px solid ${C.rule}`, fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}
                       labelFormatter={(v) => `t = ${Number(v).toFixed(2)}`}
                       formatter={(v) => Number(v).toFixed(4)} />
              <Line type="monotone" dataKey="loss" stroke={C.accent} strokeWidth={1.8} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div>
        <div className="serif kc text-sm italic mb-1" style={{ color: C.ink }}>Output-gradient components along eigenbasis of K_SS</div>
        <div style={{ width: '100%', height: 150 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <CartesianGrid stroke={C.rule} strokeDasharray="2 4" />
              <XAxis dataKey="t" type="number" domain={[0, tMax]} tick={tickStyle} stroke={C.rule}
                     tickFormatter={(v) => v.toFixed(1)} />
              <YAxis tick={tickStyle} stroke={C.rule} tickFormatter={(v) => v.toFixed(2)} />
              <Tooltip contentStyle={{ background: C.bg, border: `1px solid ${C.rule}`, fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}
                       labelFormatter={(v) => `t = ${Number(v).toFixed(2)}`}
                       formatter={(v) => Number(v).toFixed(4)} />
              <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10 }} />
              <Line type="monotone" dataKey="m1" name={`|⟨g, v₁⟩|  decays ∝ e^(−λ₁t)`} stroke={C.accent2} strokeWidth={1.6} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="m2" name={`|⟨g, v₂⟩|  decays ∝ e^(−λ₂t)`} stroke={C.accent3} strokeWidth={1.6} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [selected, setSelected] = useState('ud'); // start with the central bridge highlighted
  const [u0, setU0] = useState([1.6, 1.2]);
  const [y, setY]   = useState([-0.4, -0.6]);
  const [lam1, setLam1] = useState(1.6);
  const [lam2, setLam2] = useState(0.25);
  const [theta, setTheta] = useState(0.6);
  const [tMax] = useState(6);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const lastTs = useRef(null);

  // animation
  useEffect(() => {
    if (!playing) { lastTs.current = null; return; }
    let raf;
    const step = (ts) => {
      if (lastTs.current == null) lastTs.current = ts;
      const dt = (ts - lastTs.current) / 1000;
      lastTs.current = ts;
      setT((cur) => {
        const nxt = cur + dt * 0.8; // speed
        if (nxt >= tMax) { setPlaying(false); return tMax; }
        return nxt;
      });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing, tMax]);

  const reset = () => { setT(0); setPlaying(false); };

  const sel = selected ? NODE_BY_ID[selected] : null;

  return (
    <div className="min-h-screen w-full" style={{ background: C.bg, color: C.ink }}>
      <style>{FONT_IMPORT}</style>

      {/* Header */}
      <div className="max-w-5xl mx-auto px-5 pt-6 pb-3">
        <div className="mono text-xs uppercase tracking-widest" style={{ color: C.inkFaint }}>Litman & Guo · Eqs. 1–7 · supplementary widget</div>
        <h1 className="serif text-3xl md:text-4xl mt-1 mb-1" style={{ color: C.ink, fontWeight: 600 }}>
          The output-space dynamics, <em>at a glance</em>
        </h1>
        <p className="serif text-base md:text-lg italic" style={{ color: C.inkSoft, lineHeight: 1.4 }}>
          A dependency graph of the section's objects, plus a live 2D simulation of gradient flow under a chosen NTK.
        </p>
        <div className="mt-3" style={{ height: 1, background: C.rule }} />
      </div>

      <div className="max-w-5xl mx-auto px-5 pb-12 grid grid-cols-1 lg:grid-cols-5 gap-6">

        {/* ── Dependency graph ───────────────────────────────────────────── */}
        <section className="lg:col-span-2">
          <div className="serif text-xs uppercase tracking-widest mb-2" style={{ color: C.inkFaint }}>I. Object lattice</div>
          <h2 className="serif text-xl mb-3" style={{ color: C.ink, fontWeight: 600 }}>
            Click any node to trace its lineage
          </h2>
          <div className="rounded p-2" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
            <DepGraph selected={selected} setSelected={setSelected} />
          </div>

          {/* legend */}
          <div className="flex flex-wrap gap-3 mt-3 text-xs mono" style={{ color: C.inkSoft }}>
            <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{ background: C.upstream }}/> upstream (depends on)</div>
            <div className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full" style={{ background: C.downstream }}/> downstream (depends on it)</div>
          </div>

          {/* selected node detail */}
          <div className="mt-4 p-4 rounded" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
            {sel ? (
              <>
                <div className="mono text-xs uppercase tracking-widest mb-1" style={{ color: C.inkFaint }}>Selected</div>
                <div className="serif text-xl mb-1 italic" style={{ color: C.ink, fontWeight: 600 }}>{sel.name}</div>
                <div className="mono text-sm mb-3" style={{ color: C.accent }}>{sel.formula}</div>
                <div className="serif text-base leading-snug" style={{ color: C.inkSoft }}>{sel.blurb}</div>
                {sel.parents.length > 0 && (
                  <div className="mt-3 text-xs mono" style={{ color: C.inkFaint }}>
                    depends on:&nbsp;
                    {sel.parents.map((p, i) => (
                      <span key={p}>
                        <button className="underline" style={{ color: C.upstream }} onClick={() => setSelected(p)}>
                          {NODE_BY_ID[p].label}
                        </button>
                        {i < sel.parents.length - 1 ? ', ' : ''}
                      </span>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="serif italic" style={{ color: C.inkFaint }}>Nothing selected. Click a node above.</div>
            )}
          </div>
        </section>

        {/* ── Simulation ─────────────────────────────────────────────────── */}
        <section className="lg:col-span-3">
          <div className="serif text-xs uppercase tracking-widest mb-2" style={{ color: C.inkFaint }}>II. Live trajectory · n = 2, p = 1, squared loss</div>
          <h2 className="serif text-xl mb-3" style={{ color: C.ink, fontWeight: 600 }}>
            Watch the dynamics unfold in output space
          </h2>

          <div className="rounded p-3" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
            <OutputSpaceView
              u0={u0} setU0={setU0}
              y={y} setY={setY}
              lam1={lam1} lam2={lam2} theta={theta}
              t={t} tMax={tMax}
            />
            <div className="mono text-xs mt-2" style={{ color: C.inkSoft }}>
              Drag <em className="serif not-italic" style={{ color: C.inkSoft }}>u(0)</em> and <em className="serif not-italic">y</em> to reposition.
            </div>
          </div>

          {/* time controls */}
          <div className="mt-3 p-3 rounded" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setPlaying((p) => !p)}
                className="p-2 rounded"
                style={{ background: C.ink, color: C.bg }}
                aria-label={playing ? 'Pause' : 'Play'}
              >
                {playing ? <Pause size={16} /> : <Play size={16} />}
              </button>
              <button onClick={reset} className="p-2 rounded" style={{ background: C.bg, color: C.ink, border: `1px solid ${C.rule}` }} aria-label="Reset">
                <RotateCcw size={16} />
              </button>
              <div className="flex-1">
                <input
                  type="range"
                  min={0}
                  max={tMax}
                  step={0.01}
                  value={t}
                  onChange={(e) => { setT(parseFloat(e.target.value)); setPlaying(false); }}
                  className="w-full accent-stone-800"
                  style={{ accentColor: C.ink }}
                />
                <div className="flex justify-between mono text-xs mt-1" style={{ color: C.inkSoft }}>
                  <span>t = {t.toFixed(2)}</span>
                  <span>t_max = {tMax.toFixed(1)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* NTK controls */}
          <div className="mt-3 p-4 rounded" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
            <div className="serif italic text-sm mb-2" style={{ color: C.ink }}>NTK shape&nbsp;
              <span className="mono not-italic text-xs" style={{ color: C.inkFaint }}>K_SS = V diag(λ₁,λ₂) Vᵀ,&nbsp; V = R(θ)</span>
            </div>
            <Slider label="λ₁  (rate along v₁)" value={lam1} setValue={setLam1} min={0.02} max={2.5} step={0.01} accent={C.accent2} />
            <Slider label="λ₂  (rate along v₂)" value={lam2} setValue={setLam2} min={0.02} max={2.5} step={0.01} accent={C.accent3} />
            <Slider label="θ   (eigenbasis rotation)" value={theta} setValue={setTheta} min={-Math.PI/2} max={Math.PI/2} step={0.01} accent={C.ink}
                    fmt={(v) => `${(v*180/Math.PI).toFixed(0)}°`} />
            <div className="mono text-xs mt-2 leading-relaxed" style={{ color: C.inkSoft }}>
              Try λ₂ ≈ 0 — the corresponding eigendirection is in <span style={{ color: C.accent }}>ker K_SS</span>: any component of <em className="serif not-italic">g</em> along it
              cannot move <em className="serif not-italic">u</em> at all. Watch the trajectory get stuck.
            </div>
          </div>

          {/* plots */}
          <div className="mt-4 p-4 rounded" style={{ background: C.bgAlt, border: `1px solid ${C.rule}` }}>
            <Plots u0={u0} y={y} lam1={lam1} lam2={lam2} theta={theta} tMax={tMax} t={t} />
          </div>
        </section>
      </div>

      {/* footer narrative */}
      <div className="max-w-5xl mx-auto px-5 pb-16">
        <div style={{ height: 1, background: C.rule }} className="mb-5" />
        <div className="serif text-xs uppercase tracking-widest mb-2" style={{ color: C.inkFaint }}>III. Reading the picture</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 serif text-base leading-snug" style={{ color: C.inkSoft }}>
          <div>
            <div className="mono text-xs uppercase tracking-widest mb-1" style={{ color: C.accent3 }}>The copper arrow</div>
            shows <em>−g</em> — the direction the loss wants outputs to travel. It always points from <em>u(t)</em> toward the target <em>y</em>, because that's what minimizing squared loss demands.
          </div>
          <div>
            <div className="mono text-xs uppercase tracking-widest mb-1" style={{ color: C.accent }}>The burgundy arrow</div>
            shows the actual velocity <em>u̇ = −K_SS g</em>. It can be much shorter than the copper one, and rotated away from it — that's the NTK distorting the loss's wish into what the parameter space can deliver.
          </div>
          <div>
            <div className="mono text-xs uppercase tracking-widest mb-1" style={{ color: C.accent2 }}>The dashed lines</div>
            are the eigendirections of <em>K_SS</em>. In the right panel, the gradient component along each axis decays at its own rate <em>e^(−λᵢ t)</em> — the matrix-exponential picture (Case 3 of the ODE notes), exact here because <em>K_SS</em> is constant.
          </div>
        </div>
      </div>
    </div>
  );
}

function Slider({ label, value, setValue, min, max, step, accent, fmt }) {
  const f = fmt || ((v) => v.toFixed(2));
  return (
    <div className="mb-2">
      <div className="flex justify-between items-baseline">
        <span className="mono text-xs" style={{ color: C.inkSoft }}>{label}</span>
        <span className="mono text-xs" style={{ color: accent || C.ink }}>{f(value)}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => setValue(parseFloat(e.target.value))}
        className="w-full"
        style={{ accentColor: accent || C.ink }}
      />
    </div>
  );
}