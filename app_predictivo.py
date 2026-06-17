#!/usr/bin/env python3
"""
App interactiva — Predictivo Unfullfilment · UNICEF Argentina
Diseño premium: bento box UI · electric blue accent · microanimaciones
"""
import os, math, unicodedata

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictivo UF · UNICEF ARG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))

UMBRAL_ALTO  = 0.15
UMBRAL_MEDIO = 0.08

NOMBRE_FEAT = {
    "log_monto":    "Monto de donación (log)",
    "email":        "Tiene email",
    "celular":      "Tiene celular",
    "domicilio":    "Tiene domicilio",
    "tel_fijo":     "Tiene teléfono fijo",
    "edad_25_34":   "Edad 25–34",
    "edad_menos25": "Edad < 25",
    "edad_35_44":   "Edad 35–44",
    "edad_45_54":   "Edad 45–54",
    "edad_55_64":   "Edad 55–64",
    "edad_65_74":   "Edad 65–74",
    "tc_debito":    "Tarjeta débito",
    "tc_credito":   "Tarjeta crédito",
    "marca_mc":     "Mastercard",
    "marca_visa":   "VISA",
    "marca_naranja":"T. Naranja",
    "marca_cabal":  "Cabal",
    "marca_amex":   "Amex",
    "aac_acepta":   "Acepta AAC",
    "gen_masc":     "Género masculino",
    "quincena_1ra": "1ra quincena",
    "cuarenta":     "40+ años",
}

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
DESIGN = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg:       #05060f;
  --surface:  #080b18;
  --card:     #0c1022;
  --card2:    #0f1429;
  --border:   #151c35;
  --border2:  #1d2645;
  --accent:   #4361EE;
  --accent2:  #7B8FF5;
  --glow:     rgba(67,97,238,.18);
  --glow2:    rgba(67,97,238,.08);
  --text:     #EEF2FF;
  --muted:    #5E6A8A;
  --muted2:   #8896B3;
  --alto:     #F5385A;
  --alto-bg:  rgba(245,56,90,.10);
  --medio:    #F5A623;
  --medio-bg: rgba(245,166,35,.10);
  --bajo:     #0DD8A0;
  --bajo-bg:  rgba(13,216,160,.10);
}

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }
.stApp { font-family: 'Inter', sans-serif !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius:10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.5rem !important; }
[data-testid="stSidebarNavItems"] { display:none; }

/* ── TITLE ── */
.app-title {
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #fff 30%, var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
  margin-bottom: 4px;
}
.app-subtitle {
  font-size: 13px;
  color: var(--muted2);
  font-weight: 400;
  letter-spacing: 0.01em;
  margin-bottom: 2rem;
}

/* ── BENTO GRID ── */
.bento-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr 1fr 1fr 1.1fr;
  gap: 14px;
  margin: 0 0 28px;
}
.bento-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 22px 22px 18px;
  position: relative;
  overflow: hidden;
  transition: transform 0.25s cubic-bezier(.4,0,.2,1), border-color 0.25s, box-shadow 0.25s;
  cursor: default;
}
.bento-card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(255,255,255,.025) 0%, transparent 60%);
  pointer-events: none;
}
.bento-card:hover {
  transform: translateY(-4px);
  border-color: var(--border2);
  box-shadow: 0 12px 40px rgba(0,0,0,.4), 0 0 0 1px var(--border2);
}
.bento-card.accent:hover  { border-color: var(--accent);  box-shadow: 0 12px 40px var(--glow); }
.bento-card.alto:hover    { border-color: var(--alto);    box-shadow: 0 12px 40px var(--alto-bg); }
.bento-card.medio:hover   { border-color: var(--medio);   box-shadow: 0 12px 40px var(--medio-bg); }
.bento-card.bajo:hover    { border-color: var(--bajo);    box-shadow: 0 12px 40px var(--bajo-bg); }

.bento-tag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted2);
  margin-bottom: 14px;
}
.bento-tag .dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--muted2);
}
.bento-card.accent .bento-tag .dot { background: var(--accent2); }
.bento-card.alto   .bento-tag .dot { background: var(--alto); }
.bento-card.medio  .bento-tag .dot { background: var(--medio); }
.bento-card.bajo   .bento-tag .dot { background: var(--bajo); }
.bento-card.accent .bento-tag { color: var(--accent2); }
.bento-card.alto   .bento-tag { color: var(--alto); }
.bento-card.medio  .bento-tag { color: var(--medio); }
.bento-card.bajo   .bento-tag { color: var(--bajo); }

.bento-val {
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  line-height: 1;
  color: var(--text);
  margin-bottom: 6px;
}
.bento-card.accent .bento-val { color: var(--accent2); }
.bento-card.alto   .bento-val { color: var(--alto); }
.bento-card.medio  .bento-val { color: var(--medio); }
.bento-card.bajo   .bento-val { color: var(--bajo); }

.bento-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 4px;
}
.bento-sub {
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 14px;
}
.bento-divider {
  height: 1px;
  background: var(--border);
  margin: 12px 0;
}
.bento-crit {
  font-size: 10.5px;
  color: var(--muted2);
  line-height: 1.7;
}
.bento-crit b { color: var(--text); font-weight: 600; }
.bento-pill {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 20px;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  margin: 2px 2px 0 0;
}
.pill-risk { background: var(--alto-bg);  color: var(--alto); }
.pill-safe { background: var(--bajo-bg);  color: var(--bajo); }
.pill-neu  { background: rgba(123,143,245,.1); color: var(--accent2); }

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"] {
  gap: 2px;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0 !important;
  background: transparent !important;
}
[data-testid="stTabs"] [role="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
  border-radius: 0 !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  padding: 10px 18px !important;
  transition: color .2s !important;
  position: relative;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--text) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--text) !important;
  font-weight: 600 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0; right: 0;
  height: 2px;
  background: var(--accent);
  border-radius: 2px 2px 0 0;
}
[data-testid="stTabsContent"] {
  padding-top: 24px !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 1.5px dashed var(--border2) !important;
  border-radius: 14px !important;
  transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
[data-testid="stFileUploaderDropzoneInput"] { cursor: pointer; }

/* ── BUTTONS ── */
/* st.button → sort chip (pill, outlined) */
.stButton > button {
  font-family: 'Inter', sans-serif !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  background: var(--card) !important;
  color: var(--muted2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 20px !important;
  padding: 6px 16px !important;
  transition: all .18s cubic-bezier(.4,0,.2,1) !important;
  cursor: pointer !important;
  white-space: nowrap !important;
}
.stButton > button:hover {
  background: var(--glow2) !important;
  color: var(--text) !important;
  border-color: var(--accent) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px var(--glow) !important;
}
.stButton > button:active { transform: scale(.97) !important; }

/* st.download_button → accent outline */
.stDownloadButton > button {
  font-family: 'Inter', sans-serif !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  background: transparent !important;
  color: var(--accent2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 10px !important;
  padding: 8px 20px !important;
  transition: all .2s !important;
  cursor: pointer !important;
}
.stDownloadButton > button:hover {
  background: var(--glow2) !important;
  border-color: var(--accent) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 4px 16px var(--glow) !important;
}

/* ── INPUTS / SELECTS ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  transition: border-color .2s !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--glow2) !important; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] iframe { border-radius: 14px !important; }

/* ── SIDEBAR CONTENT ── */
.sidebar-logo {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 24px;
}
.sidebar-icon {
  width: 38px; height: 38px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; font-weight: 800; color: #fff;
}
.sidebar-title { font-size: 14px; font-weight: 700; color: var(--text); }
.sidebar-sub   { font-size: 10px; color: var(--muted); margin-top: 1px; }

.model-badge {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  margin-top: 14px;
}
.model-badge .badge-row {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: baseline;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  line-height: 1.3;
}
.model-badge .badge-row:last-child { border-bottom: none; }
.badge-key {
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.badge-val   { color: var(--text);    font-weight: 600; text-align: right; white-space: nowrap; }
.badge-auc   { color: var(--accent2); font-weight: 700; text-align: right; white-space: nowrap; }
.badge-mar   { color: var(--bajo);    font-weight: 600; text-align: right; white-space: nowrap; font-size: 10px; }

.upload-hint {
  font-size: 11px; color: var(--muted);
  text-align: center; margin-top: 10px; line-height: 1.6;
}

/* ── SECTION HEADERS ── */
.section-header {
  display: flex; align-items: baseline; gap: 10px;
  margin-bottom: 18px; margin-top: 8px;
}
.section-title { font-size: 16px; font-weight: 700; color: var(--text); letter-spacing: -.02em; }
.section-badge {
  font-size: 10px; font-weight: 600; color: var(--muted2);
  background: var(--card); border: 1px solid var(--border);
  border-radius: 20px; padding: 2px 9px; letter-spacing: .04em;
}

/* ── INFO BANNER ── */
.info-banner {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 12px;
  padding: 16px 20px;
  font-size: 13px; color: var(--muted2);
  margin-bottom: 24px;
}
.info-banner b { color: var(--text); }

/* ── NIVEL BADGE ── */
.nivel-alto  { background:var(--alto-bg);  color:var(--alto);  padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
.nivel-medio { background:var(--medio-bg); color:var(--medio); padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
.nivel-bajo  { background:var(--bajo-bg);  color:var(--bajo);  padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
header[data-testid="stHeader"] { background: transparent !important; }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── CAPTION / SMALL TEXT ── */
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; font-size: 11px !important; }

/* ── LABELS INPUTS ── */
[data-testid="stWidgetLabel"] p { font-size: 11px !important; color: var(--muted2) !important; font-weight: 500 !important; }

/* ── NUMBER INPUT ── */
[data-testid="stNumberInput"] input {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
}
[data-testid="stNumberInput"] button {
  background: var(--card) !important;
  border-color: var(--border2) !important;
  color: var(--muted2) !important;
  border-radius: 6px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] details {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  margin-bottom: 18px !important;
}
[data-testid="stExpander"] summary {
  font-size: 12px !important;
  color: var(--muted2) !important;
  font-weight: 500 !important;
  padding: 12px 16px !important;
  border-radius: 12px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

/* ── SORT CHIP ACTIVE STATE — chips con ↑ o ↓ en su texto ── */
/* No hay forma nativa de distinguirlos en CSS puro sin JS,
   pero el color visual cambia por hover; el ·N indica el orden */
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODELO — HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()


def encode_row(r) -> list:
    edad  = str(r.get("Rango etario", ""))
    tc    = strip_accents(str(r.get("tc_norm", strip_accents(str(r.get("Tipo de Tarjeta", ""))))))
    marca = str(r.get("Marca de Tarjeta", ""))
    aac   = str(r.get("AAC", ""))
    mn    = float(r.get("monto_num", 0) or 0)
    tel   = str(r.get("Telefono Fijo - Laboral - Otro - Alternativo", "")).upper()
    return [
        math.log1p(mn),
        1 if str(r.get("Email",     "")).upper() == "SI" else 0,
        1 if str(r.get("Celular",   "")).upper() == "SI" else 0,
        1 if str(r.get("Domicilio", "")).upper() == "SI" else 0,
        1 if tel == "SI" else 0,
        1 if edad == "25 a 34"  else 0,
        1 if edad == "< 25"     else 0,
        1 if edad == "35 a 44"  else 0,
        1 if edad == "45 a 54"  else 0,
        1 if edad == "55 a 64"  else 0,
        1 if edad == "65 a 74"  else 0,
        1 if tc   == "Debito"   else 0,
        1 if tc   == "Credito"  else 0,
        1 if marca == "MASTERCARD"      else 0,
        1 if marca == "VISA"            else 0,
        1 if marca == "TARJETA NARANJA" else 0,
        1 if marca == "CABAL"           else 0,
        1 if marca == "AMEX"            else 0,
        1 if "Acepta" in aac    else 0,
        1 if str(r.get("Genero del donante", "")) == "Masculino" else 0,
        1 if "1ra" in str(r.get("QUINCENA", "")) else 0,
        1 if str(r.get("40  ANOS", "")).upper() == "SI" else 0,
    ]


def risk_factors(r) -> list:
    facts = []
    tc    = strip_accents(str(r.get("Tipo de Tarjeta", "")))
    marca = str(r.get("Marca de Tarjeta", ""))
    if tc == "Debito": facts.append("Tarjeta débito")
    for k, v in {"MASTERCARD": "Mastercard", "AMEX": "Amex", "CABAL": "Cabal"}.items():
        if marca == k: facts.append(v)
    edad = str(r.get("Rango etario", ""))
    if edad == "25 a 34": facts.append("Edad 25–34")
    if edad == "< 25":    facts.append("< 25 años")
    if str(r.get("Email",   "")).upper() != "SI": facts.append("Sin email")
    if str(r.get("Celular", "")).upper() != "SI": facts.append("Sin celular")
    return facts


@st.cache_resource(show_spinner=False)
def load_model():
    data = joblib.load(os.path.join(BASE, "modelo_uf.pkl"))
    pipe = data["pipe"]
    meta = data["meta"]
    # imps viene como lista de [key, coef] desde JSON — normalizar
    meta["imps"] = [(k, c) for k, c in meta["imps"]]
    return pipe, meta


def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [strip_accents(c.strip()) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    df["monto_num"] = pd.to_numeric(
        df.get("Monto de donacion", pd.Series(dtype=str)).str.replace(r"[^\d]", "", regex=True),
        errors="coerce").fillna(0)
    df["tc_norm"] = df.get("Tipo de Tarjeta", pd.Series(dtype=str)).fillna("").apply(strip_accents)
    return df


def score(df: pd.DataFrame, pipe) -> pd.DataFrame:
    X     = np.array([encode_row(r) for _, r in df.iterrows()])
    probs = pipe.predict_proba(X)[:, 1]
    df    = df.copy()
    df["score"]   = (probs * 100).round(1)
    df["nivel"]   = pd.cut(probs, bins=[-np.inf, UMBRAL_MEDIO, UMBRAL_ALTO, np.inf],
                            labels=["BAJO","MEDIO","ALTO"]).astype(str)
    df["factores"] = [", ".join(risk_factors(r)) for _, r in df.iterrows()]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#08090e",
    plot_bgcolor="#0c1022",
    font=dict(family="Inter", color="#8896B3", size=11),
    margin=dict(t=40, b=30, l=10, r=10),
    coloraxis_colorbar=dict(bgcolor="#0c1022", tickfont=dict(color="#8896B3")),
    xaxis=dict(gridcolor="#0f1429", zerolinecolor="#151c35", tickfont=dict(color="#5E6A8A")),
    yaxis=dict(gridcolor="#0f1429", zerolinecolor="#151c35", tickfont=dict(color="#5E6A8A")),
)

def apply_layout(fig, title=""):
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(color="#EEF2FF", size=13, weight=600)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BENTO CARDS
# ─────────────────────────────────────────────────────────────────────────────
def bento_grid(total, n_alto, n_medio, n_bajo, proj, risk_top, safe_top):
    risk_pills = "".join(f'<span class="bento-pill pill-risk">{f}</span>' for f in risk_top)
    safe_pills = "".join(f'<span class="bento-pill pill-safe">{f}</span>' for f in safe_top)
    pct_a = f"{n_alto/total*100:.1f}%" if total else "—"
    pct_m = f"{n_medio/total*100:.1f}%" if total else "—"
    pct_b = f"{n_bajo/total*100:.1f}%" if total else "—"

    st.markdown(f"""
<div class="bento-grid">

  <div class="bento-card">
    <div class="bento-tag"><span class="dot"></span>Universo analizado</div>
    <div class="bento-val">{total:,}</div>
    <div class="bento-label">Donantes activos</div>
    <div class="bento-sub">UNFULLFILMENT = NO</div>
    <div class="bento-divider"></div>
    <div class="bento-crit">
      <b>Modelo:</b> DIC-2025 · ENE-2026 · MAR-2026<br>
      <b>AUC CV:</b> incluido en sidebar
    </div>
  </div>

  <div class="bento-card alto">
    <div class="bento-tag"><span class="dot"></span>Alto riesgo</div>
    <div class="bento-val">{n_alto:,}</div>
    <div class="bento-label">Probabilidad ≥ 15%</div>
    <div class="bento-sub">{pct_a} del total</div>
    <div class="bento-divider"></div>
    <div class="bento-crit">
      <b>Factores clave</b><br>{risk_pills}
    </div>
  </div>

  <div class="bento-card medio">
    <div class="bento-tag"><span class="dot"></span>Medio riesgo</div>
    <div class="bento-val">{n_medio:,}</div>
    <div class="bento-label">Probabilidad 8–15%</div>
    <div class="bento-sub">{pct_m} del total</div>
    <div class="bento-divider"></div>
    <div class="bento-crit">Riesgo moderado · monitoreo preventivo recomendado</div>
  </div>

  <div class="bento-card bajo">
    <div class="bento-tag"><span class="dot"></span>Bajo riesgo</div>
    <div class="bento-val">{n_bajo:,}</div>
    <div class="bento-label">Probabilidad &lt; 8%</div>
    <div class="bento-sub">{pct_b} del total</div>
    <div class="bento-divider"></div>
    <div class="bento-crit">
      <b>Factores protectores</b><br>{safe_pills}
    </div>
  </div>

  <div class="bento-card accent">
    <div class="bento-tag"><span class="dot"></span>Proyección 3 meses</div>
    <div class="bento-val">~{proj:,}</div>
    <div class="bento-label">Bajas estimadas</div>
    <div class="bento-sub">suma de probabilidades</div>
    <div class="bento-divider"></div>
    <div class="bento-crit">Cada donante contribuye con su probabilidad individual al total proyectado</div>
  </div>

</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(DESIGN, unsafe_allow_html=True)

    with st.spinner("Cargando modelo..."):
        pipe, meta = load_model()

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
          <div class="sidebar-icon">◈</div>
          <div>
            <div class="sidebar-title">Predictivo UF</div>
            <div class="sidebar-sub">UNICEF Argentina · 2026</div>
          </div>
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "", type=["csv","xlsx"],
            label_visibility="collapsed",
            help="CSV o Excel en formato Centro de Comando",
        )
        st.markdown('<div class="upload-hint">Arrastrá o seleccioná un CSV/Excel<br>de donantes (formato Centro de Comando)</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="model-badge">
          <div class="badge-row"><span class="badge-key">Entrenado</span><span class="badge-val">DIC · ENE · MAR</span></div>
          <div class="badge-row"><span class="badge-key">Registros</span><span class="badge-val">{meta['train_size']:,}</span></div>
          <div class="badge-row"><span class="badge-key">UF positivos</span><span class="badge-val">{meta['train_pos']} &nbsp;({meta['uf_rate']}%)</span></div>
          <div class="badge-row"><span class="badge-key">AUC CV</span><span class="badge-auc">{meta['cv_auc']}</span></div>
          <div class="badge-row"><span class="badge-key">Bajas MAR-26</span><span class="badge-mar">{meta['mar26_ids']} incorporadas</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-title">Predictivo Unfullfilment</div>
    <div class="app-subtitle">Scoring de riesgo de baja · UNICEF Argentina 2026 · Horizonte 3 meses</div>
    """, unsafe_allow_html=True)

    # ── SIN ARCHIVO ───────────────────────────────────────────────────────────
    if uploaded is None:
        st.markdown("""
        <div class="info-banner">
          👈 <b>Subí un archivo de donantes</b> desde el panel lateral para iniciar el análisis de riesgo.
          El modelo está entrenado y listo — solo necesita el CSV o Excel con las altas a scorear.
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header"><div class="section-title">Variables del modelo</div><div class="section-badge">Importancia por coeficiente logístico</div></div>', unsafe_allow_html=True)

        imp_df = pd.DataFrame(meta["imps"][:14], columns=["feat","coef"])
        imp_df["nombre"] = imp_df["feat"].map(NOMBRE_FEAT)
        imp_df["dir"]    = imp_df["coef"].apply(lambda x: "Eleva riesgo" if x > 0 else "Factor protector")
        fig = px.bar(imp_df, x="coef", y="nombre", orientation="h", color="dir",
                     color_discrete_map={"Eleva riesgo":"#F5385A","Factor protector":"#0DD8A0"})
        fig.update_layout(**PLOT_LAYOUT, yaxis=dict(autorange="reversed", gridcolor="#0f1429"),
                          showlegend=True, legend=dict(orientation="h", y=1.06, font=dict(color="#8896B3")),
                          title=dict(text="", font=dict(color="#EEF2FF")))
        st.plotly_chart(fig, use_container_width=True)
        return

    # ── CARGAR ────────────────────────────────────────────────────────────────
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded, encoding="utf-8-sig", dtype=str).fillna("")
        else:
            df_raw = pd.read_excel(uploaded, dtype=str).fillna("")
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return

    df = score(prepare(df_raw), pipe)

    total  = len(df)
    n_alto = int((df["nivel"]=="ALTO").sum())
    n_med  = int((df["nivel"]=="MEDIO").sum())
    n_bajo = int((df["nivel"]=="BAJO").sum())
    proj   = int(round(df["score"].sum() / 100))

    risk_top = [NOMBRE_FEAT[n] for n,c in meta["imps"] if c > 0][:3]
    safe_top = [NOMBRE_FEAT[n] for n,c in meta["imps"] if c < 0][:3]

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_r, tab_d, tab_f, tab_a = st.tabs([
        "  Resumen  ","  Donantes  ","  Facers  ","  Análisis  "
    ])

    # ─────────────────────── TAB RESUMEN ─────────────────────────────────────
    with tab_r:
        bento_grid(total, n_alto, n_med, n_bajo, proj, risk_top, safe_top)

        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(
                values=[n_alto, n_med, n_bajo],
                names=["Alto","Medio","Bajo"],
                color=["Alto","Medio","Bajo"],
                color_discrete_map={"Alto":"#F5385A","Medio":"#F5A623","Bajo":"#0DD8A0"},
                hole=0.55,
            )
            fig_pie.update_traces(textinfo="percent+label", textfont=dict(family="Inter", size=11))
            fig_pie.update_layout(**PLOT_LAYOUT,
                                  title=dict(text="Distribución por nivel de riesgo", font=dict(color="#EEF2FF", size=13)))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            fig_hist = px.histogram(df, x="score", nbins=40, color_discrete_sequence=["#4361EE"])
            fig_hist.add_vline(x=15, line_color="#F5385A", line_dash="dash",
                               annotation=dict(text="ALTO ≥15%", font=dict(color="#F5385A", size=10)))
            fig_hist.add_vline(x=8, line_color="#F5A623", line_dash="dash",
                               annotation=dict(text="MEDIO ≥8%", font=dict(color="#F5A623", size=10), yshift=-18))
            fig_hist.update_layout(**PLOT_LAYOUT,
                                   title=dict(text="Distribución de scores de riesgo", font=dict(color="#EEF2FF", size=13)),
                                   bargap=0.08)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ─────────────────────── TAB DONANTES ────────────────────────────────────
    with tab_d:
        st.markdown('<div class="section-header"><div class="section-title">Donantes en riesgo</div><div class="section-badge">Ordenado por score descendente</div></div>', unsafe_allow_html=True)

        cf1, cf2, cf3 = st.columns([2.5, 1, 1])
        with cf1:
            buscar = st.text_input("", placeholder="Buscar donante o facer...", label_visibility="collapsed")
        with cf2:
            f_nivel = st.selectbox("", ["Todos los niveles","ALTO","MEDIO","BAJO"], label_visibility="collapsed")
        with cf3:
            regs = ["Todas las regiones"] + (sorted(df["IMPUTACION REGION"].dropna().unique().tolist()) if "IMPUTACION REGION" in df.columns else [])
            f_reg = st.selectbox("", regs, label_visibility="collapsed")

        mask = pd.Series([True]*total, index=df.index)
        if buscar:
            q = buscar.lower()
            mask &= (df.get("Nombre donante", pd.Series(dtype=str)).fillna("").str.lower().str.contains(q) |
                     df.get("Apellido donante", pd.Series(dtype=str)).fillna("").str.lower().str.contains(q) |
                     df.get("Facer", pd.Series(dtype=str)).fillna("").str.lower().str.contains(q))
        if f_nivel != "Todos los niveles":
            mask &= df["nivel"] == f_nivel
        if f_reg != "Todas las regiones" and "IMPUTACION REGION" in df.columns:
            mask &= df["IMPUTACION REGION"] == f_reg

        df_show = df[mask].sort_values("score", ascending=False)

        COLS = [c for c in ["Nombre donante","Apellido donante","Facer","IMPUTACION REGION",
                             "Rango etario","Tipo de Tarjeta","Marca de Tarjeta",
                             "Monto de donacion","MES-ANO","score","nivel","factores"] if c in df_show.columns]

        st.caption(f"{len(df_show):,} donantes · {total:,} total")
        st.dataframe(df_show[COLS].rename(columns={"score":"Score (%)","nivel":"Nivel","factores":"Factores"}),
                     use_container_width=True, height=480, hide_index=True)

        st.download_button("↓ Exportar CSV",
                           df_show[COLS].to_csv(index=False).encode("utf-8-sig"),
                           "donantes_riesgo.csv", "text/csv")

    # ─────────────────────── TAB FACERS ──────────────────────────────────────
    with tab_f:
        grp = next((c for c in ["Facer","Usuario"] if c in df.columns), None)
        if not grp:
            st.warning("No se encontró columna Facer o Usuario en el archivo.")
        else:
            # Enriquecer con factores más frecuentes de cada facer (para propuestas)
            def top_factor(facer_name):
                sub = df[df[grp] == facer_name]
                all_facts = [f for fl in sub["factores"].str.split(", ") for f in fl if f]
                if not all_facts: return ""
                from collections import Counter
                return Counter(all_facts).most_common(1)[0][0]

            fdf = df.groupby(grp).agg(
                donantes  =("score","count"),
                score_prom=("score","mean"),
                alto      =("nivel", lambda x: (x=="ALTO").sum()),
                medio     =("nivel", lambda x: (x=="MEDIO").sum()),
                bajo      =("nivel", lambda x: (x=="BAJO").sum()),
                proj_bajas=("score", lambda x: round(x.sum()/100, 1)),
            ).reset_index()
            fdf["pct_alto"]    = (fdf["alto"]/fdf["donantes"]*100).round(1)
            fdf["score_prom"]  = fdf["score_prom"].round(1)
            fdf["factor_top"]  = fdf[grp].apply(top_factor)

            def get_banda(p):
                if p >= 20: return "🔴  Crítico  >20%"
                if p >= 10: return "🟡  Atención  10–20%"
                return "🟢  Normal  <10%"

            fdf["banda"] = fdf["pct_alto"].apply(get_banda)

            # ── HEADER + EXPANDER ─────────────────────────────────────────────
            st.markdown('<div class="section-header"><div class="section-title">Ranking de facers por riesgo</div></div>', unsafe_allow_html=True)

            with st.expander("ℹ️  ¿De dónde salen estos números?"):
                st.markdown("""
**Score prom (%)** — Promedio de las probabilidades individuales de baja de todos los donantes del facer.
Un facer con score 20% tiene donantes que en promedio tienen 20% de probabilidad de darse de baja en los próximos 3 meses.

**% Alto** — Porcentaje de donantes cuya probabilidad supera el umbral de 15%.
Un facer con 8 donantes y 6 en riesgo alto tiene 75% alto.

**Proy. bajas 3m** — Suma de las probabilidades individuales de cada donante.
Si un facer tiene 10 donantes con 15% de probabilidad cada uno → ~1.5 bajas esperadas.
Es una estimación estadística, no una certeza.

**ALTO / MEDIO / BAJO** — Conteo absoluto de donantes en cada nivel de riesgo
(ALTO ≥ 15%, MEDIO 8–15%, BAJO < 8%).

---
**Modelo:** Regresión Logística entrenada con DIC-2025, ENE-2026 y MAR-2026.
Variables usadas: monto de donación, tipo y marca de tarjeta, rango etario, email, celular, domicilio, teléfono, género, quincena, AAC, 40+ años.
**AUC CV ≈ 0.71** (poder discriminativo moderado-alto para este tipo de problema).
La tasa histórica del facer fue **excluida deliberadamente** para evaluar al donante por sus propias características, no por quién lo captó.
                """)

            # ── FILTROS (AND) ─────────────────────────────────────────────────
            fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
            with fc1:
                buscar_f = st.text_input("", placeholder="Buscar facer...",
                                         label_visibility="collapsed", key="buscar_facer")
            with fc2:
                min_don = st.number_input("Donantes mín.", min_value=1,
                                          max_value=int(fdf["donantes"].max()),
                                          value=10, step=1, key="min_don")
            with fc3:
                min_pct = st.number_input("% Alto mín.", min_value=0.0,
                                          max_value=100.0, value=0.0,
                                          step=5.0, format="%.0f", key="min_pct")
            with fc4:
                min_score = st.number_input("Score prom mín.", min_value=0.0,
                                             max_value=float(fdf["score_prom"].max()),
                                             value=0.0, step=1.0,
                                             format="%.1f", key="min_score")

            # ── SORT CHIPS ACUMULATIVOS ───────────────────────────────────────
            # Estado: lista de (col, asc) en orden de prioridad
            CHIP_DEFS = [
                ("% Alto s/altas", "pct_alto"),
                ("Score prom",     "score_prom"),
                ("Donantes",       "donantes"),
                ("Proy. bajas",    "proj_bajas"),
                ("ALTO abs.",      "alto"),
            ]
            if "facer_sort" not in st.session_state:
                st.session_state["facer_sort"] = [("pct_alto", False)]

            sort_order = list(st.session_state["facer_sort"])

            st.markdown('<div style="font-size:10px;color:var(--muted);margin:14px 0 8px;font-weight:600;letter-spacing:.08em;text-transform:uppercase">Ordenar por — click para agregar · doble para invertir · triple para quitar</div>', unsafe_allow_html=True)

            chip_cols = st.columns(len(CHIP_DEFS))
            rerun_needed = False
            for i, (label, col) in enumerate(CHIP_DEFS):
                idx = next((j for j, (c, _) in enumerate(sort_order) if c == col), -1)
                with chip_cols[i]:
                    if idx >= 0:
                        _, asc = sort_order[idx]
                        btn_lbl = f"{'↑' if asc else '↓'} {label}  ·{idx+1}"
                    else:
                        btn_lbl = label
                    if st.button(btn_lbl, key=f"chip_{col}", use_container_width=True):
                        if idx == -1:
                            sort_order.append((col, False))      # añadir desc
                        elif not sort_order[idx][1]:
                            sort_order[idx] = (col, True)        # desc → asc
                        else:
                            sort_order.pop(idx)                  # asc → quitar
                        st.session_state["facer_sort"] = sort_order
                        rerun_needed = True
            if rerun_needed:
                st.rerun()

            # ── APLICAR FILTROS ───────────────────────────────────────────────
            fmask = (
                (fdf["donantes"]   >= min_don) &
                (fdf["pct_alto"]   >= min_pct) &
                (fdf["score_prom"] >= min_score)
            )
            if buscar_f:
                fmask &= fdf[grp].str.lower().str.contains(buscar_f.lower(), na=False)
            fdf_filtered = fdf[fmask].copy()

            # ── APLICAR SORT ACUMULATIVO ──────────────────────────────────────
            if sort_order:
                s_cols = [c for c, _ in sort_order]
                s_asc  = [a for _, a in sort_order]
                fdf_filtered = fdf_filtered.sort_values(s_cols, ascending=s_asc)

            n_fil    = len(fdf_filtered)
            proj_fil = int(round(fdf_filtered["proj_bajas"].sum()))

            # ── RESUMEN DE BANDAS ─────────────────────────────────────────────
            n_crit   = (fdf_filtered["pct_alto"] >= 20).sum()
            n_aten   = ((fdf_filtered["pct_alto"] >= 10) & (fdf_filtered["pct_alto"] < 20)).sum()
            n_norm   = (fdf_filtered["pct_alto"] < 10).sum()
            don_crit = int(fdf_filtered.loc[fdf_filtered["pct_alto"] >= 20, "alto"].sum())
            don_aten = int(fdf_filtered.loc[(fdf_filtered["pct_alto"] >= 10) & (fdf_filtered["pct_alto"] < 20), "alto"].sum())

            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown(f"""<div style="background:#0c1022;border:1px solid #F5385A;border-left:4px solid #F5385A;
                    border-radius:12px;padding:14px 16px">
                  <div style="font-size:9px;font-weight:700;color:#F5385A;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">🔴 Crítico · &gt;20% en riesgo</div>
                  <div style="font-size:26px;font-weight:800;color:#F5385A;line-height:1">{n_crit}</div>
                  <div style="font-size:11px;color:#5E6A8A;margin-top:4px">facers · {don_crit} donantes ALTO</div>
                  <div style="font-size:10px;color:#5E6A8A;margin-top:2px">Más de 1 de cada 5 altas en riesgo</div>
                </div>""", unsafe_allow_html=True)
            with b2:
                st.markdown(f"""<div style="background:#0c1022;border:1px solid #F5A623;border-left:4px solid #F5A623;
                    border-radius:12px;padding:14px 16px">
                  <div style="font-size:9px;font-weight:700;color:#F5A623;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">🟡 Atención · 10–20% en riesgo</div>
                  <div style="font-size:26px;font-weight:800;color:#F5A623;line-height:1">{n_aten}</div>
                  <div style="font-size:11px;color:#5E6A8A;margin-top:4px">facers · {don_aten} donantes ALTO</div>
                  <div style="font-size:10px;color:#5E6A8A;margin-top:2px">Entre 1 de cada 5 y 1 de cada 10</div>
                </div>""", unsafe_allow_html=True)
            with b3:
                st.markdown(f"""<div style="background:#0c1022;border:1px solid #151c35;
                    border-radius:12px;padding:14px 16px">
                  <div style="font-size:9px;font-weight:700;color:#0DD8A0;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">🟢 Normal · &lt;10% en riesgo</div>
                  <div style="font-size:26px;font-weight:800;color:#0DD8A0;line-height:1">{n_norm}</div>
                  <div style="font-size:11px;color:#5E6A8A;margin-top:4px">facers · menos de 1 de cada 10</div>
                  <div style="font-size:10px;color:#5E6A8A;margin-top:2px">Monitoreo estándar</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── TABLA ─────────────────────────────────────────────────────────
            DISPLAY_COLS = [grp, "banda", "donantes", "pct_alto", "alto", "score_prom", "medio", "bajo", "proj_bajas"]
            RENAME = {grp:"Facer", "banda":"Franja", "donantes":"Altas activas",
                      "pct_alto":"% en riesgo", "alto":"ALTO", "score_prom":"Score prom (%)",
                      "medio":"MEDIO", "bajo":"BAJO", "proj_bajas":"Proy. bajas 3m"}
            st.dataframe(
                fdf_filtered[DISPLAY_COLS].rename(columns=RENAME),
                use_container_width=True, height=420,
                hide_index=True,
                column_config={
                    "Facer":          st.column_config.TextColumn("Facer",           width="medium"),
                    "Franja":         st.column_config.TextColumn("Franja",          width="medium"),
                    "Altas activas":  st.column_config.NumberColumn("Altas activas", format="%d"),
                    "% en riesgo":    st.column_config.NumberColumn("% en riesgo",   format="%.1f%%"),
                    "ALTO":           st.column_config.NumberColumn("ALTO",          format="%d"),
                    "Score prom (%)": st.column_config.NumberColumn("Score prom (%)",format="%.1f"),
                    "MEDIO":          st.column_config.NumberColumn("MEDIO",         format="%d"),
                    "BAJO":           st.column_config.NumberColumn("BAJO",          format="%d"),
                    "Proy. bajas 3m": st.column_config.NumberColumn("Proy. bajas 3m",format="%.1f"),
                },
            )

            # Mini resumen + export
            sr1, sr2, sr3, _, dl = st.columns([1.2, 1.2, 1.2, 2, 1.2])
            with sr1:
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 16px;font-size:12px;color:var(--muted2)"><b style="color:var(--text);font-size:18px;font-weight:700">{n_fil}</b>&nbsp;facers</div>', unsafe_allow_html=True)
            with sr2:
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:10px 16px;font-size:12px;color:var(--muted2)"><b style="color:var(--alto);font-size:18px;font-weight:700">{int(fdf_filtered["alto"].sum()):,}</b>&nbsp;donantes ALTO</div>', unsafe_allow_html=True)
            with sr3:
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border2);border-left:3px solid var(--accent);border-radius:10px;padding:10px 16px;font-size:12px;color:var(--muted2)"><b style="color:var(--accent2);font-size:18px;font-weight:700">~{proj_fil}</b>&nbsp;bajas proy. 3m</div>', unsafe_allow_html=True)
            with dl:
                st.download_button("↓ Exportar selección",
                                   fdf_filtered[DISPLAY_COLS].rename(columns=RENAME)
                                     .to_csv(index=False).encode("utf-8-sig"),
                                   "facers_riesgo.csv", "text/csv")

            # ── GRÁFICOS ──────────────────────────────────────────────────────
            if len(fdf_filtered):
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
                cb_l, cb_r = st.columns(2)
                with cb_l:
                    top15 = fdf_filtered.head(15)
                    fig_s = go.Figure()
                    for level, color, name in [
                        ("bajo","#0DD8A0","Bajo"),("medio","#F5A623","Medio"),("alto","#F5385A","Alto")
                    ]:
                        fig_s.add_bar(x=top15[grp], y=top15[level], name=name,
                                      marker_color=color, marker_line_width=0)
                    fig_s.update_layout(**PLOT_LAYOUT, barmode="stack",
                        title=dict(text="Composición de riesgo (top 15)", font=dict(color="#EEF2FF", size=13)),
                        xaxis_tickangle=-40,
                        legend=dict(orientation="h", y=1.08, font=dict(color="#8896B3")),
                        bargap=0.25)
                    st.plotly_chart(fig_s, use_container_width=True)

                with cb_r:
                    # fix: no pasar coloraxis_colorbar duplicado desde PLOT_LAYOUT
                    plot_no_cb = {k:v for k,v in PLOT_LAYOUT.items() if k != "coloraxis_colorbar"}
                    fig_sc = px.scatter(
                        fdf_filtered, x="donantes", y="score_prom",
                        size="proj_bajas", color="pct_alto",
                        hover_name=grp,
                        hover_data={"donantes":True,"score_prom":True,"pct_alto":True,"proj_bajas":True},
                        color_continuous_scale=["#0DD8A0","#4361EE","#F5385A"],
                        size_max=30,
                        labels={"donantes":"Donantes","score_prom":"Score prom (%)","pct_alto":"% Alto"},
                    )
                    fig_sc.add_vline(x=fdf_filtered["donantes"].median(), line_dash="dot",
                                     line_color="#1d2645",
                                     annotation=dict(text="mediana vol.", font=dict(color="#5E6A8A", size=9)))
                    fig_sc.add_hline(y=fdf_filtered["score_prom"].median(), line_dash="dot",
                                     line_color="#1d2645",
                                     annotation=dict(text="mediana score", font=dict(color="#5E6A8A", size=9)))
                    fig_sc.update_layout(**plot_no_cb,
                        title=dict(text="Volumen vs Score  ·  tamaño = proy. bajas", font=dict(color="#EEF2FF", size=13)),
                        coloraxis_colorbar=dict(title="% Alto", tickfont=dict(color="#5E6A8A", size=9), bgcolor="#0c1022"))
                    st.plotly_chart(fig_sc, use_container_width=True)

            # ── PROPUESTAS DE TRABAJO — TOP 10 ────────────────────────────────
            st.markdown('<div class="section-header" style="margin-top:36px"><div class="section-title">Plan de acción</div><div class="section-badge">Top 20 · según % en riesgo sobre altas activas</div></div>', unsafe_allow_html=True)

            if len(fdf_filtered) >= 1:
                top20 = fdf_filtered.head(20)

                def propuesta(r):
                    pct   = r["pct_alto"]
                    don   = int(r["donantes"])
                    alto  = int(r["alto"])
                    proj  = r["proj_bajas"]
                    ft    = str(r.get("factor_top", ""))
                    banda = r.get("banda", "")

                    if pct >= 20:
                        accion_urgencia = ("🔴 Intervención inmediata — reunión 1:1",
                            f"El {pct:.0f}% de sus {don} altas activas está en zona crítica ({alto} donantes ALTO). "
                            f"Solicitar reunión individual urgente esta semana para revisar su estrategia de captación y retención.")
                    else:
                        accion_urgencia = ("🟡 Reunión grupal con otros facers en Atención",
                            f"El {pct:.0f}% de sus {don} altas ({alto} ALTO) muestra señales de alerta moderadas. "
                            f"Estos facers comparten un perfil de riesgo similar — convocarlos a una reunión grupal "
                            f"es más eficiente que reuniones individuales. Objetivo: revisar casos en conjunto y definir acción coordinada.")

                    accion_cartera = (
                        ("📋 Auditoría completa de cartera",
                         f"Con {don} altas activas y {alto} en riesgo ALTO, elaborar listado priorizado de donantes "
                         f"a contactar esta semana. Los {int(r['medio'])} en riesgo MEDIO requieren seguimiento en los próximos 15 días.")
                        if don >= 20 else
                        ("📞 Contacto preventivo",
                         f"Priorizar contacto con los {alto} donantes ALTO de su cartera. "
                         f"Objetivo: validar estado del débito/crédito y reforzar el vínculo con la causa.")
                    )

                    accion_causa = None
                    ft_lower = ft.lower()
                    if "débito" in ft_lower:
                        accion_causa = ("💳 Gestión de medios de pago",
                            "El factor de riesgo más frecuente en su cartera es tarjeta débito. "
                            "Trabajar con cada donante ALTO para migrar a crédito o débito automático por CBU "
                            "— reduce la tasa de rechazo por falta de fondos puntuales.")
                    elif "email" in ft_lower or "celular" in ft_lower:
                        accion_causa = ("📧 Completar datos de contacto",
                            "Gran parte de su cartera no tiene email o celular registrado, lo que impide "
                            "la retención proactiva. Priorizar actualización de datos en el próximo contacto.")
                    elif "25-34" in ft or "< 25" in ft or "25" in ft:
                        accion_causa = ("👥 Estrategia para donantes jóvenes",
                            "Perfil etario joven (18–34) concentra el mayor riesgo en su cartera. "
                            "Reforzar el relato emocional y el impacto concreto de UNICEF en cada interacción.")
                    elif ft:
                        accion_causa = ("⚠️ Factor dominante en su cartera",
                            f"El factor más recurrente entre sus donantes ALTO es: {ft}. "
                            "Analizar si su perfil de captación está atrayendo este segmento sistemáticamente.")

                    accion_proy = ("📉 Proyección si no se interviene",
                        f"Se estiman ~{proj:.1f} bajas en los próximos 3 meses sobre su cartera actual. "
                        f"Cada baja evitada representa un donante retenido y el costo de reposición evitado.")

                    acciones = [accion_urgencia, accion_cartera]
                    if accion_causa:
                        acciones.append(accion_causa)
                    acciones.append(accion_proy)
                    return acciones

                def banda_color(pct):
                    if pct >= 20: return "#F5385A"
                    if pct >= 10: return "#F5A623"
                    return "#0DD8A0"

                # Banner de transición entre bandas
                shown_atencion_banner = False
                for i, (_, row) in enumerate(top20.iterrows()):
                    # Separador al entrar en franja Atención
                    if row["pct_alto"] < 20 and not shown_atencion_banner:
                        shown_atencion_banner = True
                        n_aten_top = sum(1 for _, r in top20.iterrows() if 10 <= r["pct_alto"] < 20)
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:14px;
                             margin:28px 0 16px;padding:14px 18px;
                             background:#0c1022;border:1px solid #F5A623;
                             border-radius:12px">
                          <span style="font-size:20px">🟡</span>
                          <div>
                            <div style="font-size:13px;font-weight:700;color:#F5A623">
                              Franja Atención · {n_aten_top} facers con 10–20% en riesgo</div>
                            <div style="font-size:11px;color:#8896B3;margin-top:3px">
                              Estos facers comparten un perfil de riesgo moderado similar.
                              Se recomienda <b style="color:#EEF2FF">convocarlos a una reunión grupal</b>
                              en lugar de reuniones individuales — más eficiente y genera aprendizaje colectivo.
                            </div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                    acciones = propuesta(row)
                    bc = banda_color(row["pct_alto"])
                    items_html = ""
                    for j, (titulo, texto) in enumerate(acciones):
                        sep = "border-bottom:1px solid #0f1429;" if j < len(acciones) - 1 else ""
                        items_html += (
                            f'<div style="display:flex;gap:12px;padding:10px 0;{sep}">'
                            f'<div style="min-width:180px;font-size:11px;font-weight:700;color:#EEF2FF;'
                            f'padding-top:2px;line-height:1.5;flex-shrink:0">{titulo}</div>'
                            f'<div style="font-size:12px;color:#8896B3;line-height:1.6">{texto}</div>'
                            f'</div>'
                        )
                    st.markdown(f"""
                    <div style="background:#0c1022;border:1px solid #151c35;
                         border-left:4px solid {bc};border-radius:14px;
                         padding:16px 20px;margin-bottom:10px">
                      <div style="display:flex;justify-content:space-between;align-items:center;
                           margin-bottom:6px;flex-wrap:wrap;gap:8px">
                        <div style="display:flex;align-items:center;gap:10px">
                          <span style="font-size:11px;font-weight:700;color:{bc};
                               letter-spacing:.06em;text-transform:uppercase;min-width:24px">#{i+1}</span>
                          <span style="font-size:15px;font-weight:700;color:#EEF2FF">{row[grp]}</span>
                        </div>
                        <div style="display:flex;gap:6px;flex-wrap:wrap">
                          <span style="background:rgba(245,56,90,.12);color:#F5385A;
                               padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700">
                            {int(row['alto'])} ALTO · {row['pct_alto']:.0f}%</span>
                          <span style="background:rgba(67,97,238,.10);color:#7B8FF5;
                               padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600">
                            {int(row['donantes'])} altas activas</span>
                          <span style="background:#080b18;color:#5E6A8A;
                               padding:3px 10px;border-radius:20px;font-size:11px">
                            score {row['score_prom']}%</span>
                        </div>
                      </div>
                      {items_html}
                    </div>""", unsafe_allow_html=True)

                # ── TABLA DE GESTIÓN EDITABLE ─────────────────────────────────
                st.markdown('<div class="section-header" style="margin-top:36px"><div class="section-title">Gestión de acciones</div><div class="section-badge">Editable · persiste en la sesión</div></div>', unsafe_allow_html=True)
                st.caption("Completá responsable, fecha y seguimiento para cada facer. Los cambios se mantienen mientras la sesión esté activa.")

                import datetime
                # Inicializar almacén en sesión
                if "gestion_data" not in st.session_state:
                    st.session_state["gestion_data"] = {}

                gestion_rows = []
                for _, row in top20.iterrows():
                    facer = row[grp]
                    stored = st.session_state["gestion_data"].get(facer, {})
                    gestion_rows.append({
                        "Facer":         facer,
                        "Franja":        row["banda"],
                        "% Riesgo":      row["pct_alto"],
                        "ALTO":          int(row["alto"]),
                        "Altas activas": int(row["donantes"]),
                        "Responsable":   stored.get("Responsable", ""),
                        "Fecha acción":  stored.get("Fecha acción", None),
                        "Estado":        stored.get("Estado", "⏳ Pendiente"),
                        "Seguimiento":   stored.get("Seguimiento", ""),
                    })

                gestion_df = pd.DataFrame(gestion_rows)

                edited = st.data_editor(
                    gestion_df,
                    use_container_width=True,
                    hide_index=True,
                    num_rows="fixed",
                    key="gestion_editor",
                    column_config={
                        "Facer":         st.column_config.TextColumn("Facer",         disabled=True, width="medium"),
                        "Franja":        st.column_config.TextColumn("Franja",        disabled=True, width="medium"),
                        "% Riesgo":      st.column_config.NumberColumn("% Riesgo",    disabled=True, format="%.1f%%"),
                        "ALTO":          st.column_config.NumberColumn("ALTO",        disabled=True, format="%d"),
                        "Altas activas": st.column_config.NumberColumn("Altas activas",disabled=True, format="%d"),
                        "Responsable":   st.column_config.TextColumn("Responsable",   width="medium",
                                            help="Persona que lleva las acciones con este facer"),
                        "Fecha acción":  st.column_config.DateColumn("Fecha acción",
                                            min_value=datetime.date(2026, 1, 1),
                                            max_value=datetime.date(2026, 12, 31),
                                            format="DD/MM/YYYY",
                                            help="Fecha en que se realizará la acción"),
                        "Estado":        st.column_config.SelectboxColumn("Estado",
                                            options=["⏳ Pendiente","🔄 En curso","✅ Completado","⛔ Sin acción"],
                                            help="Estado actual de la gestión"),
                        "Seguimiento":   st.column_config.TextColumn("Seguimiento",   width="large",
                                            help="Notas de seguimiento: qué se hizo, próximos pasos"),
                    },
                )

                # Persistir cambios en sesión
                for _, erow in edited.iterrows():
                    st.session_state["gestion_data"][erow["Facer"]] = {
                        "Responsable": erow["Responsable"],
                        "Fecha acción": erow["Fecha acción"],
                        "Estado":       erow["Estado"],
                        "Seguimiento":  erow["Seguimiento"],
                    }

                # Exportar gestión completa
                st.download_button(
                    "↓ Exportar gestión de acciones",
                    edited.to_csv(index=False).encode("utf-8-sig"),
                    "gestion_acciones.csv", "text/csv",
                )

    # ─────────────────────── TAB ANÁLISIS ────────────────────────────────────
    with tab_a:
        RAMP = ["#0DD8A0","#4361EE","#F5A623","#F5385A"]

        def pct_chart(col, title):
            if col not in df.columns: return
            g = df.groupby(col).agg(
                Donantes=("score","count"),
                Alto=("nivel", lambda x:(x=="ALTO").sum()),
            ).reset_index()
            g["pct"] = (g["Alto"]/g["Donantes"]*100).round(1)
            fig = px.bar(g.sort_values("pct", ascending=False), x=col, y="pct",
                         color="pct", color_continuous_scale=["#0DD8A0","#4361EE","#F5385A"],
                         text="pct")
            fig.update_traces(texttemplate="%{text}%", textposition="outside",
                              textfont=dict(size=10, color="#8896B3"), marker_line_width=0)
            fig.update_layout(**PLOT_LAYOUT,
                title=dict(text=title, font=dict(color="#EEF2FF", size=13)),
                coloraxis_showscale=False, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header"><div class="section-title">Perfil del donante</div><div class="section-badge">% alto riesgo por segmento</div></div>', unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1: pct_chart("Tipo de Tarjeta", "% Alto riesgo · Tipo de tarjeta")
        with r1c2: pct_chart("Marca de Tarjeta", "% Alto riesgo · Marca de tarjeta")

        r2c1, r2c2 = st.columns(2)
        with r2c1: pct_chart("Rango etario", "% Alto riesgo · Rango etario")
        with r2c2: pct_chart("IMPUTACION REGION", "% Alto riesgo · Región")

        st.markdown('<div class="section-header" style="margin-top:24px"><div class="section-title">Contactabilidad & monto</div></div>', unsafe_allow_html=True)

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1: pct_chart("Email",   "% Alto riesgo · Tiene email")
        with r3c2: pct_chart("Celular", "% Alto riesgo · Tiene celular")
        with r3c3:
            if "monto_num" in df.columns:
                mdf = df.groupby("nivel")["monto_num"].mean().reset_index()
                mdf.columns = ["Nivel","Monto promedio"]
                mdf["Monto promedio"] = mdf["Monto promedio"].round(0)
                fig_m = px.bar(mdf, x="Nivel", y="Monto promedio", color="Nivel",
                               color_discrete_map={"ALTO":"#F5385A","MEDIO":"#F5A623","BAJO":"#0DD8A0"},
                               text="Monto promedio")
                fig_m.update_traces(texttemplate="$%{text:,.0f}", textposition="outside",
                                    textfont=dict(size=10, color="#8896B3"), marker_line_width=0)
                fig_m.update_layout(**PLOT_LAYOUT,
                    title=dict(text="Monto promedio por nivel", font=dict(color="#EEF2FF", size=13)),
                    showlegend=False)
                st.plotly_chart(fig_m, use_container_width=True)


if __name__ == "__main__":
    main()
