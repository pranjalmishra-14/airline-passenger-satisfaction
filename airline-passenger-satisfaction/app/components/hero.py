"""
Parallax hero and scroll-reactive components.

Streamlit renders a fixed component tree with no scroll hooks, so true parallax
is implemented inside a sandboxed `components.html` iframe that owns its own
scroll container and listens to scroll events directly.

All motion respects prefers-reduced-motion.
"""
from __future__ import annotations

import warnings

import streamlit.components.v1 as components

from app.components.theme import FONT_IMPORT, TOKENS as T


def _embed(html: str, height: int) -> None:
    """
    Render sandboxed HTML+JS.

    `components.html` is used deliberately: it iframes the content, which is what
    gives the hero its own scroll container and lets JS listen to scroll events.
    `st.html` is not iframed and `st.iframe` takes a URL, so neither is a
    substitute here. Streamlit deprecates this API after 2026-06-01; the call is
    centralised in this helper so the migration is a one-line change.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        components.html(html, height=height, scrolling=False)


def parallax_hero(total_records: int, n_models: int, best_f1: float,
                  best_model: str, n_tied: int, height: int = 620) -> None:
    """
    Scroll-driven parallax hero: layered starfield, horizon, altitude rings and
    a flight-path curve that draws itself as the user scrolls.
    """
    html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{FONT_IMPORT}
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ height:100%; background:{T['bg']}; }}
#scroll {{
  height:{height}px; overflow-y:auto; overflow-x:hidden; position:relative;
  scrollbar-width:thin; scrollbar-color:{T['cyan_dim']} transparent;
}}
#scroll::-webkit-scrollbar {{ width:7px; }}
#scroll::-webkit-scrollbar-thumb {{ background:{T['cyan_dim']}; border-radius:4px; }}
#track {{ height:{int(height*2.15)}px; position:relative; }}

.layer {{ position:absolute; inset:0; will-change:transform; }}

/* --- depth 1: starfield --- */
#stars span {{ position:absolute; background:#cfe8ff; border-radius:50%; opacity:.55; }}

/* --- depth 2: altitude rings --- */
#rings {{ display:flex; align-items:center; justify-content:center; }}
#rings div {{
  position:absolute; border:1px solid rgba(94,234,212,.13); border-radius:50%;
}}

/* --- depth 3: horizon glow --- */
#horizon {{
  position:absolute; left:-10%; right:-10%; top:52%; height:320px;
  background:radial-gradient(ellipse at 50% 0%, rgba(56,189,248,.20), transparent 68%);
  filter:blur(6px);
}}
#horizon-line {{
  position:absolute; left:0; right:0; top:52%; height:1px;
  background:linear-gradient(90deg, transparent, {T['cyan']}, transparent); opacity:.5;
}}

/* --- depth 4: flight path --- */
#path {{ position:absolute; inset:0; }}

/* --- content --- */
#content {{
  position:absolute; top:0; left:0; right:0; height:{height}px;
  display:flex; flex-direction:column;
  align-items:center; justify-content:center; text-align:center;
  padding:0 24px; z-index:5;
}}
.tag {{
  font-family:'Fira Code',monospace; font-size:.68rem; letter-spacing:.24em;
  text-transform:uppercase; color:{T['cyan']};
  border:1px solid rgba(94,234,212,.35); border-radius:999px;
  padding:.4rem 1rem; margin-bottom:1.3rem;
  background:rgba(94,234,212,.06); backdrop-filter:blur(6px);
}}
h1 {{
  font-family:'Chakra Petch',sans-serif; font-weight:700;
  font-size:clamp(1.9rem,5.2vw,3.5rem); line-height:1.08; color:#fff;
  letter-spacing:-.01em; max-width:17ch;
}}
h1 .hl {{
  background:linear-gradient(120deg,{T['cyan']},{T['blue']} 55%,{T['violet']});
  -webkit-background-clip:text; background-clip:text; color:transparent;
}}
.sub {{
  font-family:'Fira Sans',sans-serif; font-size:clamp(.88rem,1.7vw,1.06rem);
  color:{T['muted']}; margin-top:1rem; max-width:56ch; line-height:1.65;
}}
.stats {{ display:flex; gap:12px; margin-top:2.1rem; flex-wrap:wrap; justify-content:center; }}
.stat {{
  min-width:128px; padding:.8rem 1.1rem; border-radius:12px;
  background:rgba(17,25,45,.62); border:1px solid rgba(94,234,212,.18);
  backdrop-filter:blur(10px); text-align:left; position:relative;
}}
.stat .k {{ font-family:'Fira Code',monospace; font-size:.6rem; letter-spacing:.15em;
            text-transform:uppercase; color:{T['cyan']}; }}
.stat .v {{ font-family:'Chakra Petch',sans-serif; font-size:1.42rem; font-weight:700;
            color:#fff; margin-top:.2rem; }}
.stat .s {{ font-size:.66rem; color:{T['muted']}; margin-top:.1rem; }}
.stat.warn .v {{ color:{T['amber']}; }}
.stat.warn .k {{ color:{T['amber']}; }}

#cue {{
  position:absolute; bottom:26px; left:50%; transform:translateX(-50%);
  font-family:'Fira Code',monospace; font-size:.62rem; letter-spacing:.2em;
  color:{T['muted']}; text-transform:uppercase; z-index:6; text-align:center;
}}
#cue .arrow {{ display:block; margin:.45rem auto 0; width:16px; height:16px;
  border-right:1.5px solid {T['cyan']}; border-bottom:1.5px solid {T['cyan']};
  transform:rotate(45deg); animation:bob 1.9s ease-in-out infinite; }}
@keyframes bob {{ 0%,100%{{transform:rotate(45deg) translate(0,0);opacity:.45}}
                  50%{{transform:rotate(45deg) translate(4px,4px);opacity:1}} }}

/* second panel revealed on scroll */
#panel2 {{
  position:absolute; top:{height}px; left:0; right:0; height:{height}px;
  display:flex; align-items:center; justify-content:center; padding:0 26px; z-index:5;
}}
.card {{
  max-width:720px; padding:1.7rem 2rem 1.9rem; border-radius:18px;
  background:rgba(17,25,45,.74); border:1px solid rgba(94,234,212,.22);
  backdrop-filter:blur(14px); opacity:0; transform:translateY(34px);
  transition:opacity .75s ease, transform .75s ease;
}}
.card.in {{ opacity:1; transform:none; }}
.card h2 {{ font-family:'Chakra Petch',sans-serif; font-size:1.42rem; color:#fff;
            margin-bottom:.9rem; }}
.card p {{ font-family:'Fira Sans',sans-serif; color:{T['muted']}; line-height:1.72;
           font-size:.93rem; }}
.card .em {{ color:{T['amber']}; font-weight:600; }}
.card .em2 {{ color:{T['cyan']}; font-weight:600; }}
.bars {{ display:flex; gap:6px; margin-top:1.4rem; align-items:flex-end;
         height:56px; margin-bottom:.2rem; }}
.bar {{ flex:1; border-radius:4px 4px 0 0; background:linear-gradient(180deg,{T['cyan']},rgba(94,234,212,.15));
        height:8%; transition:height .9s cubic-bezier(.22,1,.36,1); }}
.bar.tied {{ background:linear-gradient(180deg,{T['amber']},rgba(245,158,11,.15)); }}

@media (prefers-reduced-motion: reduce) {{
  .layer {{ transform:none !important; }}
  .card {{ opacity:1 !important; transform:none !important; }}
  #cue .arrow {{ animation:none; }}
  .bar {{ transition:none; }}
}}
</style></head><body>
<div id="scroll">
  <div id="track">
    <div class="layer" id="stars"></div>
    <div class="layer" id="rings">
      <div style="width:340px;height:340px"></div>
      <div style="width:560px;height:560px"></div>
      <div style="width:820px;height:820px"></div>
    </div>
    <div class="layer"><div id="horizon"></div><div id="horizon-line"></div></div>
    <svg class="layer" id="path" viewBox="0 0 1000 600" preserveAspectRatio="none">
      <path id="fp" d="M -40 470 Q 250 430 500 330 T 1040 140" fill="none"
            stroke="{T['cyan']}" stroke-width="1.6" opacity=".55"
            stroke-dasharray="1400" stroke-dashoffset="1400"/>
      <circle id="craft" r="4.5" fill="{T['amber']}"/>
    </svg>

    <div id="content">
      <div class="tag">Machine Learning &middot; Deep Learning &middot; Comparative Evaluation</div>
      <h1>Airline Passenger <span class="hl">Satisfaction</span> Simulator</h1>
      <p class="sub">
        {total_records:,} passenger records flown through {n_models} models —
        from a majority-class baseline to deep neural networks — under a strict
        protocol where the test set is opened exactly once.
      </p>
      <div class="stats">
        <div class="stat"><div class="k">Records</div><div class="v">{total_records:,}</div>
             <div class="s">70 / 15 / 15 split</div></div>
        <div class="stat"><div class="k">Models</div><div class="v">{n_models}</div>
             <div class="s">4 families</div></div>
        <div class="stat"><div class="k">Best F1</div><div class="v">{best_f1:.4f}</div>
             <div class="s">{best_model[:26]}</div></div>
        <div class="stat warn"><div class="k">Tied</div><div class="v">{n_tied}</div>
             <div class="s">overlapping 95% CIs</div></div>
      </div>
    </div>

    <div id="panel2">
      <div class="card" id="c2">
        <h2>The result that matters</h2>
        <p>
          The leaderboard says one model won. The statistics say otherwise:
          the top five models differ by <span class="em2">0.00077 F1</span>,
          while the mean 95&#37; bootstrap confidence interval is
          <span class="em">7.6&times; wider</span>.
          <span class="em">{n_tied} models</span> are statistically
          indistinguishable — so the choice should rest on training cost and
          interpretability, not on a difference that sits inside the noise.
        </p>
        <div class="bars" id="bars"></div>
      </div>
    </div>

    <div id="cue">Scroll<span class="arrow"></span></div>
  </div>
</div>

<script>
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// starfield -- three depths
const stars = document.getElementById('stars');
for (let i=0;i<110;i++) {{
  const s=document.createElement('span');
  const d=Math.random();
  const size=d<.6?1:(d<.9?1.7:2.4);
  s.style.width=size+'px'; s.style.height=size+'px';
  s.style.left=(Math.random()*100)+'%';
  s.style.top=(Math.random()*100)+'%';
  s.style.opacity=(.2+Math.random()*.55).toFixed(2);
  s.dataset.d=d.toFixed(2);
  stars.appendChild(s);
}}

// flight path + aircraft marker
const fp=document.getElementById('fp');
const craft=document.getElementById('craft');
const len=fp.getTotalLength();
fp.style.strokeDasharray=len; fp.style.strokeDashoffset=reduce?0:len;

// tied-model bars
const heights=[96,95,92,90,88,72,58,44];
const bars=document.getElementById('bars');
heights.forEach((h,i)=>{{
  const b=document.createElement('div');
  b.className='bar'+(i<5?' tied':'');
  b.dataset.h=h; bars.appendChild(b);
}});

const scroller=document.getElementById('scroll');
const card=document.getElementById('c2');
let ticking=false;

function frame() {{
  const y=scroller.scrollTop;
  const max=scroller.scrollHeight-scroller.clientHeight;
  const p=max>0?y/max:0;

  if (!reduce) {{
    stars.style.transform=`translateY(${{y*0.18}}px)`;
    document.getElementById('rings').style.transform=
        `translateY(${{y*0.34}}px) scale(${{1+p*0.14}})`;
    document.getElementById('horizon').style.transform=`translateY(${{-y*0.10}}px)`;
    document.getElementById('content').style.transform=`translateY(${{y*0.52}}px)`;
    document.getElementById('content').style.opacity=Math.max(0,1-p*2.3);
    document.getElementById('cue').style.opacity=Math.max(0,1-p*4);

    const draw=Math.min(1,p*2.0);
    fp.style.strokeDashoffset=len*(1-draw);
    const pt=fp.getPointAtLength(len*draw);
    craft.setAttribute('cx',pt.x); craft.setAttribute('cy',pt.y);
    craft.style.opacity=draw>0.02?1:0;
  }}

  if (p>0.30 && !card.classList.contains('in')) {{
    card.classList.add('in');
    setTimeout(()=>document.querySelectorAll('.bar').forEach((b,i)=>{{
      setTimeout(()=>b.style.height=b.dataset.h+'%', i*70);
    }}), 260);
  }}
  ticking=false;
}}
scroller.addEventListener('scroll',()=>{{
  if(!ticking){{ requestAnimationFrame(frame); ticking=true; }}
}}, {{passive:true}});
frame();
</script>
</body></html>
"""
    _embed(html, height)


def scroll_reveal() -> None:
    """Fade-and-rise reveal for `.hud-panel` elements as they enter the viewport."""
    _embed("""
<script>
(function(){
  const doc = window.parent.document;
  if (doc.getElementById('__reveal_obs')) return;
  const flag = doc.createElement('meta'); flag.id='__reveal_obs'; doc.head.appendChild(flag);
  if (window.parent.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const io = new IntersectionObserver((entries)=>{
    entries.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target); }});
  }, {threshold:0.08, rootMargin:'0px 0px -40px 0px'});

  function scan(){
    doc.querySelectorAll('.hud-panel:not(.reveal)').forEach(el=>{
      el.classList.add('reveal'); io.observe(el);
    });
  }
  scan();
  new MutationObserver(scan).observe(doc.body, {childList:true, subtree:true});
})();
</script>
""", 0)
