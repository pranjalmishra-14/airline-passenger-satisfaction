"""Presentational helpers for the flight-deck themed Streamlit app."""
from __future__ import annotations

import streamlit as st

from app.components.theme import TOKENS as T
from app.components.theme import css as _css


def inject_css() -> None:
    """Inject the design system once per render."""
    st.markdown(_css(), unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = "", accent: str = "") -> None:
    """HUD-style metric panel. `accent` is one of '', 'amber', 'green', 'violet'."""
    cls = f"hud-panel {'accent-' + accent if accent else ''}"
    st.markdown(
        f"""<div class="{cls}">
              <div class="hud-label">{label}</div>
              <div class="hud-value">{value}</div>
              <div class="hud-sub">{help_text}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section(index: str, title: str) -> None:
    """Numbered section heading with a trailing rule."""
    st.markdown(
        f"""<div class="section-head">
              <span class="idx">{index}</span>
              <span class="txt">{title}</span>
              <span class="rule"></span>
            </div>""",
        unsafe_allow_html=True,
    )


def verdict_banner(label: str, probability: float, satisfied: bool) -> None:
    """Large prediction verdict. Carries an icon and text, not colour alone."""
    cls = "pos" if satisfied else "neg"
    icon = (
        '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
        'stroke-linejoin="round" aria-hidden="true">'
        + ('<path d="M20 6 9 17l-5-5"/>' if satisfied
           else '<circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/>')
        + "</svg>"
    )
    st.markdown(
        f"""<div class="verdict {cls}" role="status">
              <div style="display:flex;align-items:center;gap:.7rem;">
                {icon}<h2>{label}</h2>
              </div>
              <div class="conf">MODEL CONFIDENCE &nbsp;{probability:.1%}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def panel_open(extra_class: str = "") -> None:
    st.markdown(f'<div class="hud-panel {extra_class}">', unsafe_allow_html=True)


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def caveat(text: str) -> None:
    st.markdown(
        f'<div style="font-size:.82rem;color:{T["muted"]};border-left:2px solid '
        f'{T["border_strong"]};padding-left:.8rem;margin-top:.8rem;line-height:1.6;">'
        f"{text}</div>",
        unsafe_allow_html=True,
    )
