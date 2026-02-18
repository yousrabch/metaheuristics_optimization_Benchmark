import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="TP – Metaheuristics", layout="wide")

# ── Purple theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
    background-color: #1a0a2e !important;
    color: #e8d5ff !important;
}
.stApp { background: linear-gradient(135deg, #1a0a2e 0%, #0d0520 50%, #1a0a2e 100%) !important; }
.block-container { padding-top: 2rem !important; }

h1 { font-size: 2.4rem !important; font-weight: 700 !important;
     color: #d8a4ff !important; letter-spacing: -0.5px !important; }
h2, h3 { color: #c084fc !important; font-weight: 600 !important; }
p, label, .stMarkdown { color: #e8d5ff !important; }
            


.card {
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(139, 92, 246, 0.15);
}

input[type="number"], input[type="text"], textarea {
    background: rgba(88, 28, 135, 0.3) !important;
    border: 1px solid rgba(168, 85, 247, 0.4) !important;
    border-radius: 8px !important;
    color: #f3e8ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
input[type="number"]:focus, input[type="text"]:focus, textarea:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.25) !important;
}

.stSelectbox > div > div {
    background: rgba(88, 28, 135, 0.3) !important;
    border: 1px solid rgba(168, 85, 247, 0.4) !important;
    border-radius: 8px !important;
    color: #f3e8ff !important;
}
            
.stSelectbox svg { fill: #c084fc !important; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 11px 22px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #9333ea) !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

.mono-box {
    background: rgba(49, 10, 82, 0.6);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 10px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #d8b4fe;
    word-break: break-all;
    min-height: 46px;
    line-height: 1.6;
}
.fitness-box {
    background: rgba(49, 10, 82, 0.6);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 10px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    color: #e9d5ff;
    min-height: 46px;
    font-weight: 500;
}
.badge-uni   { background:#4c1d95; color:#c4b5fd; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
.badge-multi { background:#1e1b4b; color:#a5b4fc; border-radius:6px; padding:3px 10px; font-size:0.78rem; font-weight:600; }
.label-bold  { font-weight: 700; font-size: 1rem; color: #e8d5ff; }
.range-badge { background:rgba(168,85,247,0.2); border:1px solid rgba(168,85,247,0.4);
               border-radius:6px; padding:2px 10px; font-size:0.82rem;
               color:#d8b4fe; font-family:'JetBrains Mono',monospace; }

hr { border: none; border-top: 1px solid rgba(168,85,247,0.25) !important; }
.streamlit-expanderHeader { color: #c084fc !important; font-weight: 600 !important; }
.stCaption, small { color: #9d7ec7 !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0520; }
::-webkit-scrollbar-thumb { background: #7c3aed; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Function definitions ──────────────────────────────────────────────────────
FUNCTIONS = {
    "F1  –  Sphere": {
        "code": "F1", "type": "Unimodal",
        "latex": r"f(x) = \sum_{i=1}^{D} x_i^{\,2}",
        "range": (-100, 100),
        "global_min": "0  at  x = (0, …, 0)",
        "fn":   lambda x: float(np.sum(x**2)),
        "fn2d": lambda X, Y: X**2 + Y**2,
    },
    "F2  –  Schwefel 2.22": {
        "code": "F2", "type": "Unimodal",
        "latex": r"f(x) = \sum_{i=1}^{D}|x_i| + \prod_{i=1}^{D}|x_i|",
        "range": (-10, 10),
        "global_min": "0  at  x = (0, …, 0)",
        "fn":   lambda x: float(np.sum(np.abs(x)) + np.prod(np.abs(x))),
        "fn2d": lambda X, Y: np.abs(X) + np.abs(Y) + np.abs(X)*np.abs(Y),
    },
    "F5  –  Rosenbrock": {
        "code": "F5", "type": "Unimodal",
        "latex": r"f(x) = \sum_{i=1}^{D-1}\!\left[100(x_i^2 - x_{i+1})^2 + (1-x_i)^2\right]",
        "range": (-30, 30),
        "global_min": "0  at  x = (1, …, 1)",
        "fn":   lambda x: float(np.sum(100*(x[:-1]**2 - x[1:])**2 + (1 - x[:-1])**2)),
        "fn2d": lambda X, Y: 100*(X**2 - Y)**2 + (1 - X)**2,
    },
    "F7  –  Quartic + Noise": {
        "code": "F7", "type": "Unimodal",
        "latex": r"f(x) = \sum_{i=1}^{D} i\,x_i^4 + \mathrm{rand}(0,1)",
        "range": (-1.28, 1.28),
        "global_min": "≈ 0  (stochastic noise)",
        "fn":   lambda x: float(np.sum(np.arange(1, len(x)+1)*x**4) + np.random.rand()),
        "fn2d": lambda X, Y: 1*X**4 + 2*Y**4,
    },
    "F8  –  Schwefel 2.26": {
        "code": "F8", "type": "Multimodal",
        "latex": r"f(x) = \sum_{i=1}^{D}\!\left(-x_i\sin\!\left(\sqrt{|x_i|}\right)\right)",
        "range": (-500, 500),
        "global_min": "≈ −418.98·D  at  x ≈ (420.97, …)",
        "fn":   lambda x: float(np.sum(-x*np.sin(np.sqrt(np.abs(x))))),
        "fn2d": lambda X, Y: -X*np.sin(np.sqrt(np.abs(X))) - Y*np.sin(np.sqrt(np.abs(Y))),
    },
    "F9  –  Rastrigin": {
        "code": "F9", "type": "Multimodal",
        "latex": r"f(x) = \sum_{i=1}^{D}\!\left[x_i^2 - 10\cos(2\pi x_i) + 10\right]",
        "range": (-5.12, 5.12),
        "global_min": "0  at  x = (0, …, 0)",
        "fn":   lambda x: float(np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10)),
        "fn2d": lambda X, Y: (X**2 - 10*np.cos(2*np.pi*X)+10) + (Y**2 - 10*np.cos(2*np.pi*Y)+10),
    },
    "F11  –  Griewank": {
        "code": "F11", "type": "Multimodal",
        "latex": r"f(x) = 1 + \frac{1}{4000}\sum_{i=1}^{D}x_i^2 - \prod_{i=1}^{D}\cos\!\left(\frac{x_i}{\sqrt{i}}\right)",
        "range": (-600, 600),
        "global_min": "0  at  x = (0, …, 0)",
        "fn":   lambda x: float(1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1))))),
        "fn2d": lambda X, Y: 1 + (X**2+Y**2)/4000 - np.cos(X)*np.cos(Y/np.sqrt(2)),
    },
}

COLORSCALES = {
    "F1":"Blues","F2":"Teal","F5":"Oranges",
    "F7":"Purples","F8":"RdBu","F9":"Viridis","F11":"Plasma",
}

def make_surface(info, n=80):
    lo, hi = info["range"]
    X, Y = np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n))
    Z = info["fn2d"](X, Y)
    cs = COLORSCALES.get(info["code"], "Viridis")
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z, colorscale=cs, showscale=True, opacity=0.92,
        contours={"z":{"show":True,"usecolormap":True,"project_z":False}},
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x)",
            camera=dict(eye=dict(x=1.6,y=1.6,z=1.0)),
            bgcolor="rgba(13,5,32,0)",
            xaxis=dict(backgroundcolor="rgba(49,10,82,0.5)", gridcolor="#4c1d95", showbackground=True, tickfont=dict(color="#c084fc")),
            yaxis=dict(backgroundcolor="rgba(49,10,82,0.5)", gridcolor="#4c1d95", showbackground=True, tickfont=dict(color="#c084fc")),
            zaxis=dict(backgroundcolor="rgba(49,10,82,0.5)", gridcolor="#4c1d95", showbackground=True, tickfont=dict(color="#c084fc")),
        ),
        margin=dict(l=0,r=0,t=40,b=0), height=500,
        paper_bgcolor="rgba(26,10,46,0)", plot_bgcolor="rgba(26,10,46,0)",
        font=dict(family="Outfit, sans-serif", color="#c084fc"),
        title=dict(text=f"{info['code']} — 2D cross-section (x₃…xD = 0)",
                   font=dict(size=13, color="#a78bfa"), x=0.5),
    )
    return fig

# ── Session state ─────────────────────────────────────────────────────────────
if "solution" not in st.session_state: st.session_state.solution = None
if "fitness"  not in st.session_state: st.session_state.fitness  = None

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# TP – Metaheuristics")
st.markdown("---")
st.markdown("## Part 1 \\ Optimization Problem Initialization")
st.markdown("#### Standard Continuous Optimization Benchmark Problems in Metaheuristics")
st.write("")
st.write("")
# ═════════════════════════════════════════════════════════════════════════════
# MAIN CARD  (mirrors the original layout order exactly)
# ═════════════════════════════════════════════════════════════════════════════
#st.markdown('<div class="card">', unsafe_allow_html=True)

# ── ROW 1 : Solution label | D | Range min | Range max | (spacer) | Generate ──
col_label, col_D, col_low, col_high, col_sp, col_gen = st.columns([1, 1.4, 1.1, 1.1, 2, 2])

with col_label:
    st.markdown('<p class="label-bold" style="margin-top:34px">Solution:</p>', unsafe_allow_html=True)
with col_D:
    st.markdown("**Dimension (D)**")
    D = st.number_input("dim", min_value=2, max_value=200, value=30, label_visibility="collapsed")

# Pick the currently selected function first (needed for default range values)
# We read it from session state so changing the selectbox below updates ranges live
fn_key = st.session_state.get("fn_select", list(FUNCTIONS.keys())[0])
info_preview = FUNCTIONS[fn_key]

with col_low:
    st.markdown("**Range min**")
    low = st.number_input("low", value=float(info_preview["range"][0]), step=1.0, label_visibility="collapsed")
with col_high:
    st.markdown("**Range max**")
    high = st.number_input("high", value=float(info_preview["range"][1]), step=1.0, label_visibility="collapsed")
with col_gen:
    st.markdown('<div style="margin-top:26px">', unsafe_allow_html=True)
    if st.button("🎲  Generate solution"):
        st.session_state.solution = np.random.uniform(low, high, int(D))
        st.session_state.fitness  = None
    st.markdown('</div>', unsafe_allow_html=True)

# ── ROW 2 : Candidate solution display | Fitness ──────────────────────────────
col_sol, col_fit = st.columns([5, 1.5])
with col_sol:
    st.markdown("**Candidate solution example**")
    sol_str = ("  ".join([f"{v:.4f}" for v in st.session_state.solution])
               if st.session_state.solution is not None else "— no solution yet —")
    st.markdown(f'<div class="mono-box">{sol_str}</div>', unsafe_allow_html=True)
with col_fit:
    st.markdown("**Fitness**")
    fit_str = f"{st.session_state.fitness:.6e}" if st.session_state.fitness is not None else "—"
    st.markdown(f'<div class="fitness-box">{fit_str}</div>', unsafe_allow_html=True)

# ── ROW 2b : Manual entry (collapsible, below candidate) ─────────────────────
with st.expander("✏️  Enter your own solution manually"):
    st.caption(f"Provide exactly D = {int(D)} numbers, separated by spaces or commas.")
    manual_input = st.text_area("manual", placeholder="e.g.  1.5  -3.2  0.8  …",
                                height=80, label_visibility="collapsed")
    col_apply, col_hint = st.columns([2, 5])
    with col_apply:
        if st.button("✅  Apply"):
            try:
                vals = [float(v) for v in manual_input.replace(",", " ").split()]
                if len(vals) != int(D):
                    st.error(f"Expected {int(D)} values, got {len(vals)}.")
                else:
                    st.session_state.solution = np.array(vals)
                    st.session_state.fitness  = None
                    st.success("Solution applied!")
            except ValueError:
                st.error("Parse error — use numbers separated by spaces or commas.")

st.markdown("---")

# ── ROW 3 : Function label | Dropdown | Formula | Evaluate ────────────────────
col_fl, col_fn, col_formula, col_ev = st.columns([0.8, 2.2, 3.8, 2])
with col_fl:
    st.markdown('<p class="label-bold" style="margin-top:30px">Function:</p>', unsafe_allow_html=True)
with col_fn:
    st.write("")
    fn_name = st.selectbox("fn", list(FUNCTIONS.keys()),
                           label_visibility="collapsed", key="fn_select")
info = FUNCTIONS[fn_name]
with col_formula:
    st.latex(info["latex"])
with col_ev:
    st.markdown('<div style="margin-top:26px">', unsafe_allow_html=True)
    if st.button("⚡  Evaluate solution"):
        if st.session_state.solution is None:
            st.warning("Generate or enter a solution first.")
        elif len(st.session_state.solution) != int(D):
            st.warning(f"Solution has {len(st.session_state.solution)} values but D = {int(D)}.")
        else:
            st.session_state.fitness = info["fn"](st.session_state.solution)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

badge_class = "badge-uni" if info["type"] == "Unimodal" else "badge-multi"
st.markdown(
    f'<span class="{badge_class}">{"🔵 Unimodal" if info["type"]=="Unimodal" else "🔴 Multimodal"}</span>'
    f'&nbsp;&nbsp;<span class="range-badge">Default range [{info["range"][0]}, {info["range"][1]}]</span>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# 3D SURFACE
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("### 3D Landscape")
st.caption("Interactive surface (2D cross-section, x₃…xD fixed at 0) · Drag to rotate · Scroll to zoom")

with st.spinner("Rendering surface…"):
    st.plotly_chart(make_surface(info), use_container_width=True)


st.markdown("---")