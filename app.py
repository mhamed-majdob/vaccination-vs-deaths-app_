import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import os

# ══════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="VaxInsight — COVID Vaccination Analysis",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0D1117 !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
.stDeployButton, #MainMenu, footer { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #38BDF844; border-radius: 10px; }

.main .block-container {
    max-width: 100% !important;
    padding: 2.5rem 3rem 4rem !important;
}

/* All text base */
body, p, div, label, span {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #E6EDF3 !important;
}

/* ── NAVBAR ── */
.navbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding-bottom: 1.75rem;
    border-bottom: 1px solid #21262D;
    margin-bottom: 2.25rem;
}
.brand { font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.1rem; color:#E6EDF3 !important; }
.brand span { color:#38BDF8; }
.nav-sub { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#8B949E; }
.badge {
    display:inline-flex; align-items:center; gap:0.4rem;
    background:#38BDF810; border:1px solid #38BDF830;
    border-radius:100px; padding:0.28rem 0.85rem;
    font-family:'JetBrains Mono',monospace; font-size:0.67rem; color:#38BDF8;
}
.dot { width:6px;height:6px;border-radius:50%;background:#38BDF8;animation:blink 2s infinite; }
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}

/* ── KPI CARDS ── */
.kpi-row { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:2.25rem; }
.kpi { background:#161B22; border:1px solid #21262D; border-radius:12px; padding:1.1rem 1.3rem; position:relative; overflow:hidden; }
.kpi::before { content:''; position:absolute; top:0;left:0;right:0;height:2px; }
.kpi.sky::before  { background:linear-gradient(90deg,#38BDF8,transparent); }
.kpi.rose::before { background:linear-gradient(90deg,#F472B6,transparent); }
.kpi.grn::before  { background:linear-gradient(90deg,#34D399,transparent); }
.kpi.amb::before  { background:linear-gradient(90deg,#FBBF24,transparent); }
.kpi-v { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.8rem; font-weight:800; color:#E6EDF3; line-height:1.1; margin-bottom:0.25rem; letter-spacing:-0.03em; }
.kpi-u { font-size:0.85rem; font-weight:500; color:#8B949E; }
.kpi-l { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#8B949E; text-transform:uppercase; letter-spacing:0.1em; }

/* ── SECTION HEADERS ── */
.sec-tag { font-family:'JetBrains Mono',monospace; font-size:0.62rem; letter-spacing:0.2em; color:#38BDF8; text-transform:uppercase; margin-bottom:0.4rem; }
.sec-h   { font-family:'Plus Jakarta Sans',sans-serif; font-size:1.25rem; font-weight:700; color:#E6EDF3; margin-bottom:0.25rem; letter-spacing:-0.02em; }
.sec-p   { font-size:0.84rem; color:#8B949E; margin-bottom:1.4rem; line-height:1.6; max-width:600px; }

/* ── CARD ── */
.card { background:#161B22; border:1px solid #21262D; border-radius:14px; padding:1.3rem 1.3rem 0.4rem; margin-bottom:1.25rem; }
.card-h { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.88rem; font-weight:600; color:#E6EDF3; margin-bottom:0.15rem; }
.card-p { font-size:0.74rem; color:#8B949E; margin-bottom:0.5rem; line-height:1.5; }

/* ── CONCLUSION BOX ── */
.conclusion {
    background:#161B22; border:1px solid #21262D;
    border-left: 3px solid #38BDF8;
    border-radius:0 12px 12px 0;
    padding:1.2rem 1.4rem; margin-bottom:1rem;
}
.conclusion.positive { border-left-color:#F472B6; }
.conclusion.neutral  { border-left-color:#FBBF24; }
.conc-country { font-family:'Plus Jakarta Sans',sans-serif; font-size:0.78rem; font-weight:700; color:#8B949E; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem; }
.conc-verdict { font-family:'Plus Jakarta Sans',sans-serif; font-size:1rem; font-weight:700; color:#E6EDF3; margin-bottom:0.5rem; }
.conc-explain { font-size:0.83rem; color:#C9D1D9; line-height:1.65; }
.conc-stats   { display:flex; gap:1.5rem; margin-top:0.75rem; flex-wrap:wrap; }
.cs { font-family:'JetBrains Mono',monospace; font-size:0.72rem; }
.cs-label { color:#8B949E; }
.cs-val   { font-weight:600; }
.cs-val.grn { color:#34D399; }
.cs-val.amb { color:#FBBF24; }
.cs-val.red { color:#F472B6; }

/* ── TABLE ── */
.data-table {
    width:100%; border-collapse:collapse;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem;
    margin-bottom:1rem;
}
.data-table th {
    text-align:left; padding:0.6rem 0.75rem;
    color:#8B949E; font-weight:500; letter-spacing:0.08em;
    border-bottom:1px solid #30363D; text-transform:uppercase; font-size:0.65rem;
}
.data-table td {
    padding:0.6rem 0.75rem;
    border-bottom:1px solid #21262D;
    color:#E6EDF3;
    line-height:1.8;
}
.data-table tr:last-child td { border-bottom:none; }

/* ── MULTISELECT FIX (white dropdown bg) ── */
[data-testid="stMultiSelect"] label {
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.67rem !important; letter-spacing:0.1em !important;
    color:#8B949E !important; text-transform:uppercase !important;
}
/* Dropdown popup */
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[data-baseweb="select"] ul {
    background:#161B22 !important;
    border:1px solid #30363D !important;
}
[data-baseweb="option"] {
    background:#161B22 !important;
    color:#E6EDF3 !important;
}
[data-baseweb="option"]:hover {
    background:#21262D !important;
}
/* The input/tag area */
[data-testid="stMultiSelect"] > div > div {
    background:#161B22 !important;
    border:1px solid #30363D !important;
    border-radius:8px !important;
}
/* Selected tags */
[data-baseweb="tag"] {
    background:#38BDF820 !important;
    border:1px solid #38BDF840 !important;
}
[data-baseweb="tag"] span { color:#38BDF8 !important; }

/* ── BUTTON ── */
[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#38BDF8,#818CF8) !important;
    color:#0D1117 !important;
    font-family:'Plus Jakarta Sans',sans-serif !important;
    font-weight:700 !important; font-size:0.92rem !important;
    border:none !important; border-radius:10px !important;
    padding:0.7rem 1.5rem !important;
    box-shadow:0 4px 18px #38BDF81E !important;
    transition:opacity 0.2s,transform 0.2s !important;
}
[data-testid="stButton"] > button:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }

/* ── DOWNLOAD ── */
[data-testid="stDownloadButton"] > button {
    background:#21262D !important; color:#C9D1D9 !important;
    font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important;
    border:1px solid #30363D !important; border-radius:8px !important;
    padding:0.55rem 1.1rem !important;
}
[data-testid="stDownloadButton"] > button:hover { border-color:#38BDF844 !important; color:#38BDF8 !important; }

/* ── DIVIDER ── */
.div { height:1px; background:#21262D; margin:2.25rem 0; }

/* ── COLUMN GAP ── */
[data-testid="stHorizontalBlock"] { gap:1.25rem !important; align-items:stretch !important; }

/* ── FOOTER ── */
.footer {
    margin-top:3rem; padding-top:1.5rem;
    border-top:1px solid #21262D;
    display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.75rem;
}
.footer span { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#484F58; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  LOAD + MERGE DATA
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    vax = pd.read_csv(os.path.join(base, 'data', 'daily-covid-19-vaccine-doses-administered-per-million-people.csv'))
    deaths = pd.read_csv(
        os.path.join(base, 'data', 'estimated-cumulative-excess-deaths-per-100000-people-during-covid-19.csv'))

    vax.rename(columns={'COVID-19 doses (daily, 7-day average, per million people)': 'vax_per_million'}, inplace=True)
    deaths.rename(columns={
        'Cumulative excess deaths per 100,000 people (central estimate)': 'excess_deaths_per_100k',
    }, inplace=True)

    vax['Day'] = pd.to_datetime(vax['Day'])
    deaths['Day'] = pd.to_datetime(deaths['Day'])

    merged = pd.merge(vax, deaths, on=['Entity', 'Day'], how='inner')
    merged.dropna(subset=['vax_per_million', 'excess_deaths_per_100k'], inplace=True)
    return merged


merged_data = load_data()

# Country accent colors
COLORS = {
    'India': '#38BDF8', 'United States': '#F472B6', 'Brazil': '#34D399',
    'Germany': '#FBBF24', 'Bangladesh': '#A78BFA', 'France': '#FB923C',
    'Japan': '#6EE7B7', 'United Kingdom': '#FCA5A5', 'Italy': '#67E8F9',
    'Canada': '#86EFAC',
}
DEFAULTS = [c for c in ['India', 'United States', 'Brazil', 'Germany', 'Bangladesh']
            if c in merged_data['Entity'].unique()]

# Shared Plotly base
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='JetBrains Mono, monospace', color='#8B949E', size=11),
    margin=dict(l=4, r=4, t=12, b=4),
    xaxis=dict(gridcolor='#21262D', linecolor='#30363D', zerolinecolor='#21262D',
               tickfont=dict(color='#8B949E', size=10)),
    yaxis=dict(gridcolor='#21262D', linecolor='#30363D', zerolinecolor='rgba(0,0,0,0)',
               tickfont=dict(color='#8B949E', size=10)),
)

# ══════════════════════════════════════════════════════
#  ① NAVBAR
# ══════════════════════════════════════════════════════
d_min = merged_data['Day'].min().strftime('%b %Y')
d_max = merged_data['Day'].max().strftime('%b %Y')

st.markdown(f"""
<div class="navbar">
  <div class="brand">💉 Vax<span>Insight</span></div>
  <div class="nav-sub">COVID-19 Vaccination vs Excess Deaths · Linear Regression</div>
  <div class="badge"><span class="dot"></span> {d_min} — {d_max}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  ② KPI ROW
# ══════════════════════════════════════════════════════
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi sky">
    <div class="kpi-v">{merged_data['Entity'].nunique()}</div>
    <div class="kpi-l">Countries Available</div>
  </div>
  <div class="kpi rose">
    <div class="kpi-v">{len(merged_data):,}</div>
    <div class="kpi-l">Data Points</div>
  </div>
  <div class="kpi grn">
    <div class="kpi-v">{merged_data['vax_per_million'].max():,.0f}</div>
    <div class="kpi-l">Peak Doses / Million</div>
  </div>
  <div class="kpi amb">
    <div class="kpi-v">{(merged_data['Day'].max() - merged_data['Day'].min()).days}</div>
    <div class="kpi-l">Days of Coverage</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  ③ COUNTRY PICKER  (simple, visible)
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="sec-tag">Step 1</div>
<div class="sec-h">Pick Countries to Compare</div>
<div class="sec-p">Select the countries you want to analyze. The app will automatically generate charts and plain-English conclusions for each one.</div>
""", unsafe_allow_html=True)

col_sel, _ = st.columns([2, 1])
with col_sel:
    selected = st.multiselect(
        "Countries",
        options=sorted(merged_data['Entity'].unique()),
        default=DEFAULTS,
    )

if not selected:
    st.warning("👆 Select at least one country above to start.")
    st.stop()

# ══════════════════════════════════════════════════════
#  ④ OVERVIEW CHARTS
# ══════════════════════════════════════════════════════
st.markdown('<div class="div"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-tag">Step 2</div>
<div class="sec-h">Overview — All Selected Countries</div>
<div class="sec-p">See how vaccination rollout and excess deaths evolved over time across your chosen countries.</div>
""", unsafe_allow_html=True)

ov1, ov2 = st.columns(2)

with ov1:
    st.markdown(
        '<div class="card"><div class="card-h">💉 Vaccination Rollout</div><div class="card-p">Daily doses per million people (7-day average). Higher = faster rollout.</div>',
        unsafe_allow_html=True)
    fig_v = go.Figure()
    for c in selected:
        cd = merged_data[merged_data['Entity'] == c]
        col = COLORS.get(c, '#8B949E')
        fig_v.add_trace(go.Scatter(x=cd['Day'], y=cd['vax_per_million'], name=c,
                                   mode='lines', line=dict(color=col, width=2),
                                   hovertemplate=f"<b>{c}</b><br>%{{x|%d %b %Y}}<br>%{{y:,.0f}} doses/M<extra></extra>"))
    fig_v.update_layout(**PL, height=300,
                        legend=dict(orientation='h', y=1.08, font=dict(size=10, color='#8B949E'),
                                    bgcolor='rgba(0,0,0,0)'))
    fig_v.update_yaxes(title_text="Doses per million")
    st.plotly_chart(fig_v, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with ov2:
    st.markdown(
        '<div class="card"><div class="card-h">💀 Excess Deaths</div><div class="card-p">Cumulative excess deaths per 100,000 people. Higher = more deaths above normal levels.</div>',
        unsafe_allow_html=True)
    fig_d = go.Figure()
    for c in selected:
        cd = merged_data[merged_data['Entity'] == c]
        col = COLORS.get(c, '#8B949E')
        fig_d.add_trace(go.Scatter(x=cd['Day'], y=cd['excess_deaths_per_100k'], name=c,
                                   mode='lines', line=dict(color=col, width=2, dash='dot'),
                                   hovertemplate=f"<b>{c}</b><br>%{{x|%d %b %Y}}<br>%{{y:.1f}} per 100k<extra></extra>"))
    fig_d.update_layout(**PL, height=300,
                        legend=dict(orientation='h', y=1.08, font=dict(size=10, color='#8B949E'),
                                    bgcolor='rgba(0,0,0,0)'))
    fig_d.update_yaxes(title_text="Excess deaths per 100k")
    st.plotly_chart(fig_d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  ⑤ PER-COUNTRY: CHART + AUTO CONCLUSION
# ══════════════════════════════════════════════════════
st.markdown('<div class="div"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="sec-tag">Step 3</div>
<div class="sec-h">Country-by-Country Analysis</div>
<div class="sec-p">For each country: a regression chart showing the relationship between vaccination rate and excess deaths, plus a plain-English conclusion.</div>
""", unsafe_allow_html=True)

regression_results = []


def make_conclusion(country, slope, r2, color):
    """Generate a plain-English interpretation of the regression."""

    # Relationship direction
    if slope < -0.001:
        direction = "negative"
        direction_txt = "As vaccinations increased, excess deaths went DOWN."
        direction_emoji = "✅"
        box_class = "conclusion"
    elif slope > 0.001:
        direction = "positive"
        direction_txt = "As vaccinations increased, excess deaths also went UP."
        direction_emoji = "⚠️"
        box_class = "conclusion positive"
    else:
        direction = "flat"
        direction_txt = "Vaccinations had almost no linear effect on excess deaths."
        direction_emoji = "➡️"
        box_class = "conclusion neutral"

    # R² quality
    if r2 >= 0.65:
        r2_txt = f"The relationship is <strong>strong</strong> (R²={r2:.2f}) — vaccination rate explains {r2 * 100:.0f}% of the variation in excess deaths."
        r2_class = "grn"
    elif r2 >= 0.35:
        r2_txt = f"The relationship is <strong>moderate</strong> (R²={r2:.2f}) — vaccination rate explains about {r2 * 100:.0f}% of the variation. Other factors also play a role."
        r2_class = "amb"
    else:
        r2_txt = f"The relationship is <strong>weak</strong> (R²={r2:.2f}) — only {r2 * 100:.0f}% of the variation is explained by vaccination rate alone. Many other factors are likely at play."
        r2_class = "red"

    # Overall verdict
    if direction == "negative" and r2 >= 0.35:
        verdict = f"{direction_emoji} Vaccination appears to have helped reduce deaths in {country}."
    elif direction == "positive":
        verdict = f"{direction_emoji} Counterintuitive result in {country} — likely due to vaccines arriving AFTER the death peaks."
    else:
        verdict = f"{direction_emoji} No clear linear signal found in {country}. The data is complex."

    return f"""
    <div class="{box_class}">
      <div class="conc-country" style="color:{color};">🌍 {country}</div>
      <div class="conc-verdict">{verdict}</div>
      <div class="conc-explain">
        {direction_txt}<br><br>
        {r2_txt}<br><br>
        <strong>Important note:</strong> Vaccines were often deployed <em>after</em> early death waves,
        so a positive slope doesn't mean vaccines caused deaths — it reflects timing.
        Always consider the full timeline, not just the correlation number.
      </div>
      <div class="conc-stats">
        <div class="cs"><span class="cs-label">Slope: </span><span class="cs-val {'grn' if slope < 0 else 'red'}">{slope:+.4f}</span></div>
        <div class="cs"><span class="cs-label">R²: </span><span class="cs-val {r2_class}">{r2:.4f}</span></div>
        <div class="cs"><span class="cs-label">Relationship: </span><span class="cs-val">{direction.capitalize()}</span></div>
      </div>
    </div>
    """


for country in selected:
    cd = merged_data[merged_data['Entity'] == country].copy()
    color = COLORS.get(country, '#8B949E')

    if len(cd) < 10:
        st.warning(f"Not enough data for **{country}** — skipping.")
        continue

    X = cd[['vax_per_million']].values
    y = cd['excess_deaths_per_100k'].values
    mdl = LinearRegression()
    mdl.fit(X, y)
    slope = mdl.coef_[0]
    intercept = mdl.intercept_
    r2 = mdl.score(X, y)
    regression_results.append({'Country': country, 'Slope': round(slope, 4),
                               'Intercept': round(intercept, 4), 'R_squared': round(r2, 4)})

    ch1, ch2 = st.columns([1.1, 0.9])

    with ch1:
        # Regression scatter + fit line
        x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        st.markdown(
            f'<div class="card"><div class="card-h">{country} — Vaccination vs Excess Deaths</div><div class="card-p">Each dot = one date. The line shows the overall trend.</div>',
            unsafe_allow_html=True)
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=cd['vax_per_million'], y=cd['excess_deaths_per_100k'],
            mode='markers', name='Data points',
            marker=dict(color=color, size=5, opacity=0.5, line=dict(width=0)),
            hovertemplate="Doses/M: %{x:,.0f}<br>Deaths/100k: %{y:.1f}<extra></extra>",
        ))
        fig_s.add_trace(go.Scatter(
            x=x_range.flatten(), y=mdl.predict(x_range).flatten(),
            mode='lines', name=f'Trend line',
            line=dict(color='#FBBF24', width=2.5),
        ))
        fig_s.update_layout(**PL, height=300,
                            legend=dict(orientation='h', y=1.08, font=dict(size=10, color='#8B949E'),
                                        bgcolor='rgba(0,0,0,0)'))
        fig_s.update_xaxes(title_text="Doses per million")
        fig_s.update_yaxes(title_text="Excess deaths per 100k")
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        # Auto-generated conclusion
        st.markdown(make_conclusion(country, slope, r2, color), unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  ⑥ SUMMARY TABLE + DOWNLOAD
# ══════════════════════════════════════════════════════
if regression_results:
    st.markdown('<div class="div"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-tag">Step 4</div>
    <div class="sec-h">Summary & Download</div>
    <div class="sec-p">All regression results in one table. Download as CSV to use in your report or presentation.</div>
    """, unsafe_allow_html=True)

    rr = pd.DataFrame(regression_results)

    # Build clean HTML table
    rows_html = ""
    for _, row in rr.iterrows():
        sc = '#34D399' if row['Slope'] < 0 else '#F472B6'
        r2c = '#34D399' if row['R_squared'] >= 0.65 else '#FBBF24' if row['R_squared'] >= 0.35 else '#F472B6'
        r2_label = "Strong" if row['R_squared'] >= 0.65 else "Moderate" if row['R_squared'] >= 0.35 else "Weak"
        dot_col = COLORS.get(row['Country'], '#8B949E')
        rows_html += f"""
        <tr>
          <td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{dot_col};margin-right:0.5rem;"></span>{row['Country']}</td>
          <td style="color:{sc};">{row['Slope']:+.4f}</td>
          <td>{row['Intercept']:.2f}</td>
          <td style="color:{r2c};">{row['R_squared']:.4f} &nbsp;<span style="font-size:0.65rem;opacity:0.7;">({r2_label})</span></td>
        </tr>"""

    st.markdown(f"""
    <div class="card" style="padding-bottom:1.2rem;">
      <div class="card-h">Results Table</div>
      <div class="card-p">Green slope = vaccinations associated with fewer deaths. R² strength: ≥0.65 strong, 0.35–0.65 moderate, below 0.35 weak.</div>
      <table class="data-table">
        <thead>
          <tr>
            <th>Country</th><th>Slope</th><th>Intercept</th><th>R² Score</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="⬇  Download Results as CSV",
            data=rr.to_csv(index=False),
            file_name="vaccination_regression_results.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════
#  ⑦ FOOTER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <span>VaxInsight · scikit-learn · Plotly · Streamlit</span>
  <span>Our World in Data · COVID-19 Dataset · Educational use only — not medical advice</span>
</div>
""", unsafe_allow_html=True)