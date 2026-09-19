"""
Design system for the Airline Satisfaction Simulator.

Theme: flight-deck / avionics HUD. Dark-mode OLED base with cyan-amber accent
pairing, monospace instrument labels, and glass panels.

Accessibility notes:
  * Body text uses #E6EDF7 on #05070F -> contrast ratio ~15:1 (WCAG AAA).
  * Muted text uses #94A3B8 on #05070F -> ~7.8:1 (AAA for normal text).
  * Colour is never the sole signal: state is also carried by icon, label or shape.
  * All motion is wrapped in a prefers-reduced-motion guard.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
TOKENS = {
    # surfaces
    "bg":            "#05070F",
    "bg_elev":       "#0A0F1E",
    "panel":         "rgba(17, 25, 45, 0.72)",
    "panel_solid":   "#111A2D",
    "border":        "rgba(94, 234, 212, 0.16)",
    "border_strong": "rgba(94, 234, 212, 0.38)",
    "grid":          "rgba(56, 189, 248, 0.07)",
    # ink
    "text":          "#E6EDF7",
    "muted":         "#94A3B8",
    "dim":           "#64748B",
    # accents -- cyan (instrument) + amber (caution) is the avionics convention
    "cyan":          "#5EEAD4",
    "cyan_dim":      "#2DD4BF",
    "blue":          "#38BDF8",
    "amber":         "#F59E0B",
    "green":         "#34D399",
    "red":           "#FB7185",
    "violet":        "#A78BFA",
}

FAMILY_COLORS = {
    "baseline":  "#64748B",
    "classical": "#38BDF8",
    "ensemble":  "#34D399",
    "deep":      "#A78BFA",
}

FONT_DISPLAY = "'Chakra Petch', 'Fira Sans', system-ui, sans-serif"
FONT_BODY = "'Fira Sans', system-ui, -apple-system, sans-serif"
FONT_MONO = "'Fira Code', 'SF Mono', Menlo, monospace"

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Chakra+Petch:wght@500;600;700&"
    "family=Fira+Code:wght@400;500;600&"
    "family=Fira+Sans:wght@300;400;500;600&display=swap');"
)


TITLE_FONT = dict(family="Chakra Petch, sans-serif", size=15, color=TOKENS["text"])


def merge_layout(**overrides) -> dict:
    """
    Theme layout merged with per-chart overrides.

    Nested dicts (xaxis, yaxis, legend, ...) are merged key-by-key so a caller
    can set e.g. xaxis=dict(range=...) without discarding the themed gridlines
    or triggering a duplicate-keyword error.
    """
    base = plotly_layout()
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged = dict(base[key])
            merged.update(value)
            base[key] = merged
        else:
            base[key] = value
    base.setdefault("title", {})
    if isinstance(base["title"], dict):
        base["title"].setdefault("font", TITLE_FONT)
    return base


def plotly_layout(height: int | None = None) -> dict:
    """
    Shared Plotly layout so every chart matches the flight-deck theme.

    Deliberately excludes `title` so callers can splat this dict alongside their
    own title without a keyword collision.
    """
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Fira Sans, sans-serif", color=TOKENS["text"], size=12),
        margin=dict(l=12, r=12, t=52, b=12),
        xaxis=dict(gridcolor=TOKENS["grid"], zerolinecolor=TOKENS["grid"],
                   linecolor=TOKENS["border"], tickfont=dict(size=11)),
        yaxis=dict(gridcolor=TOKENS["grid"], zerolinecolor=TOKENS["grid"],
                   linecolor=TOKENS["border"], tickfont=dict(size=11)),
        hoverlabel=dict(bgcolor=TOKENS["panel_solid"], bordercolor=TOKENS["cyan"],
                        font=dict(family="Fira Code, monospace", size=12,
                                  color=TOKENS["text"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        colorway=[TOKENS["cyan"], TOKENS["blue"], TOKENS["amber"], TOKENS["violet"],
                  TOKENS["green"], TOKENS["red"]],
    )
    if height:
        layout["height"] = height
    return layout


def css() -> str:
    """Global stylesheet injected once per session."""
    t = TOKENS
    return f"""
<style>
{FONT_IMPORT}

/* ---------- chrome: remove the default light header band ---------- */
header[data-testid="stHeader"] {{
    background: transparent !important;
    backdrop-filter: none;
}}
div[data-testid="stToolbar"] {{ right: .6rem; }}
div[data-testid="stDecoration"] {{ display: none; }}
#MainMenu {{ color: {t['muted']}; }}

/* ---------- base ---------- */
.stApp {{
    background:
        radial-gradient(1100px 620px at 12% -10%, rgba(56,189,248,0.10), transparent 60%),
        radial-gradient(900px 560px at 88% 4%, rgba(167,139,250,0.09), transparent 62%),
        linear-gradient(180deg, {t['bg']} 0%, #060B18 55%, {t['bg']} 100%);
    color: {t['text']};
    font-family: {FONT_BODY};
}}
/* faint instrument grid */
.stApp::before {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient({t['grid']} 1px, transparent 1px),
        linear-gradient(90deg, {t['grid']} 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(ellipse 80% 55% at 50% 0%, #000 35%, transparent 78%);
}}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1320px;
                    position: relative; z-index: 1; }}

h1, h2, h3, h4 {{ font-family: {FONT_DISPLAY}; letter-spacing: .01em; color: {t['text']}; }}
h1 {{ font-weight: 700; }}
p, li, span, label {{ color: {t['text']}; }}
a {{ color: {t['cyan']}; }}
code, pre, .stCode {{ font-family: {FONT_MONO} !important; }}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #070C18 0%, #060A14 100%);
    border-right: 1px solid {t['border']};
}}
section[data-testid="stSidebar"] * {{ color: {t['text']}; }}

/* ---------- reusable panels ---------- */
.hud-panel {{
    background: {t['panel']};
    border: 1px solid {t['border']};
    border-radius: 14px;
    padding: 1.05rem 1.2rem;
    backdrop-filter: blur(11px);
    -webkit-backdrop-filter: blur(11px);
    position: relative;
    transition: border-color .22s ease, transform .22s ease, box-shadow .22s ease;
}}
.hud-panel:hover {{
    border-color: {t['border_strong']};
    box-shadow: 0 0 0 1px rgba(94,234,212,.08), 0 14px 38px -18px rgba(56,189,248,.55);
}}
/* corner ticks -- the avionics bracket motif */
.hud-panel::before, .hud-panel::after {{
    content: ""; position: absolute; width: 11px; height: 11px;
    border-color: {t['cyan']}; opacity: .55;
}}
.hud-panel::before {{ top: 7px; left: 7px; border-top: 2px solid; border-left: 2px solid;
                      border-top-left-radius: 4px; }}
.hud-panel::after  {{ bottom: 7px; right: 7px; border-bottom: 2px solid; border-right: 2px solid;
                      border-bottom-right-radius: 4px; }}

.hud-label {{
    font-family: {FONT_MONO}; font-size: .68rem; letter-spacing: .16em;
    text-transform: uppercase; color: {t['cyan']}; opacity: .92; margin-bottom: .38rem;
}}
.hud-value {{ font-family: {FONT_DISPLAY}; font-size: 1.95rem; font-weight: 700;
              line-height: 1.08; color: {t['text']}; }}
.hud-sub {{ font-size: .76rem; color: {t['muted']}; margin-top: .22rem; }}

/* metric accent variants -- paired with a text label, never colour alone */
.accent-amber .hud-value {{ color: {t['amber']}; }}
.accent-amber .hud-label {{ color: {t['amber']}; }}
.accent-green .hud-value {{ color: {t['green']}; }}
.accent-violet .hud-value {{ color: {t['violet']}; }}

/* ---------- section heading ---------- */
.section-head {{ display: flex; align-items: center; gap: .8rem; margin: 2rem 0 1rem; }}
.section-head .idx {{
    font-family: {FONT_MONO}; font-size: .72rem; color: {t['bg']};
    background: {t['cyan']}; padding: .2rem .5rem; border-radius: 5px; font-weight: 600;
}}
.section-head .txt {{ font-family: {FONT_DISPLAY}; font-size: 1.22rem; font-weight: 600; }}
.section-head .rule {{ flex: 1; height: 1px;
    background: linear-gradient(90deg, {t['border_strong']}, transparent); }}

/* ---------- verdict ---------- */
.verdict {{
    border-radius: 16px; padding: 1.5rem 1.7rem; margin: .5rem 0 1rem;
    border: 1px solid; position: relative; overflow: hidden;
}}
.verdict.pos {{ background: linear-gradient(135deg, rgba(52,211,153,.17), rgba(52,211,153,.05));
                border-color: rgba(52,211,153,.5); }}
.verdict.neg {{ background: linear-gradient(135deg, rgba(251,113,133,.17), rgba(251,113,133,.05));
                border-color: rgba(251,113,133,.5); }}
.verdict h2 {{ margin: 0; font-size: 1.75rem; letter-spacing: .01em; }}
.verdict .conf {{ font-family: {FONT_MONO}; font-size: .88rem; color: {t['muted']};
                  margin-top: .35rem; }}

/* ---------- streamlit widget restyling ---------- */
/* every interactive control meets the 44px minimum touch target */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
div[data-testid="stTabs"] button,
.stNumberInput button,
div[data-testid="stExpander"] summary {{
    min-height: 44px;
}}
.stNumberInput button {{ min-width: 38px; }}

.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
    font-family: {FONT_DISPLAY}; font-weight: 600; letter-spacing: .03em;
    border-radius: 10px; border: 1px solid {t['border_strong']};
    background: linear-gradient(135deg, rgba(94,234,212,.15), rgba(56,189,248,.09));
    color: {t['text']}; transition: all .2s ease; cursor: pointer;
    min-height: 44px;   /* touch target */
}}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {{
    border-color: {t['cyan']};
    box-shadow: 0 0 22px -6px rgba(94,234,212,.6);
    color: {t['cyan']};
}}
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible,
.stDownloadButton > button:focus-visible {{
    outline: 2px solid {t['cyan']}; outline-offset: 2px;
}}

div[data-testid="stTabs"] button {{
    font-family: {FONT_MONO}; font-size: .8rem; letter-spacing: .07em;
    text-transform: uppercase; color: {t['muted']};
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {t['cyan']}; }}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: {t['cyan']}; }}

div[data-testid="stExpander"] {{
    border: 1px solid {t['border']}; border-radius: 12px;
    background: rgba(17,25,45,.45);
}}
div[data-testid="stExpander"] summary:focus-visible {{
    outline: 2px solid {t['cyan']}; outline-offset: 2px;
}}

div[data-testid="stDataFrame"] {{ border: 1px solid {t['border']}; border-radius: 12px; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {t['cyan']}; }}
.stProgress > div > div > div {{ background: linear-gradient(90deg, {t['cyan']}, {t['blue']}); }}

div[data-testid="stMetricValue"] {{ font-family: {FONT_DISPLAY}; color: {t['cyan']}; }}

/* ---------- form controls: dark surfaces with readable text ----------
   Streamlit renders select/input surfaces via hashed emotion classes with no
   stable hook, so these rules target the widget wrapper's descendants and rely
   on the secondaryBackgroundColor theme var being overridden below. */
/* base="dark" in .streamlit/config.toml does the heavy lifting; these rules
   only restyle the select surface itself, scoped to baseweb roles so they
   cannot leak into layout containers. */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
    background-color: {t['panel_solid']} !important;
    border-color: {t['border']} !important;
}}
.stSelectbox div[data-baseweb="select"] svg,
.stMultiSelect div[data-baseweb="select"] svg {{ fill: {t['cyan']} !important; }}

.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stNumberInput div[data-baseweb="input"],
.stTextInput div[data-baseweb="input"],
.stNumberInput input, .stTextInput input {{
    background-color: {t['panel_solid']} !important;
    border-color: {t['border']} !important;
    color: {t['text']} !important;
}}
.stSelectbox div[data-baseweb="select"] *,
.stMultiSelect div[data-baseweb="select"] * {{ color: {t['text']} !important; }}
.stNumberInput input, .stTextInput input {{ caret-color: {t['cyan']}; }}
.stNumberInput button {{
    background-color: {t['panel_solid']} !important;
    border-color: {t['border']} !important; color: {t['cyan']} !important;
}}
.stSelectbox div[data-baseweb="select"] > div:focus-within,
.stNumberInput div[data-baseweb="input"]:focus-within {{
    border-color: {t['cyan']} !important;
    box-shadow: 0 0 0 1px {t['cyan']};
}}
/* dropdown menus render in a portal at body level */
div[data-baseweb="popover"] div[role="listbox"],
div[data-baseweb="popover"] ul {{
    background-color: {t['panel_solid']} !important;
    border: 1px solid {t['border']};
}}
div[data-baseweb="popover"] li {{ color: {t['text']} !important; }}
div[data-baseweb="popover"] li:hover {{ background-color: rgba(94,234,212,.12) !important; }}
/* multiselect chips */
span[data-baseweb="tag"] {{
    background-color: rgba(94,234,212,.18) !important;
    border: 1px solid {t['border_strong']};
}}
span[data-baseweb="tag"] span {{ color: {t['text']} !important; }}
/* checkbox + radio labels */
.stCheckbox label, .stRadio label {{ color: {t['text']} !important; }}

/* plotly chart container */
.js-plotly-plot {{
    border: 1px solid {t['border']}; border-radius: 14px;
    background: rgba(10,15,30,.5); padding: .35rem;
}}

/* ---------- scroll reveal ---------- */
.reveal {{ opacity: 0; transform: translateY(22px); transition: opacity .7s ease, transform .7s ease; }}
.reveal.in {{ opacity: 1; transform: none; }}

/* ---------- motion safety ---------- */
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: .001ms !important; animation-iteration-count: 1 !important;
        transition-duration: .001ms !important; scroll-behavior: auto !important;
    }}
    .reveal {{ opacity: 1 !important; transform: none !important; }}
}}

/* ---------- responsive ---------- */
@media (max-width: 768px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    .hud-value {{ font-size: 1.55rem; }}
    h1 {{ font-size: 1.6rem; }}
}}
</style>
"""
