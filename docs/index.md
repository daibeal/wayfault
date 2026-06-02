# wayfault

**Wrong-Way Risk (WWR) estimation for counterparty credit risk.**

[![PyPI](https://img.shields.io/pypi/v/wayfault.svg)](https://pypi.org/project/wayfault/)
[![Python](https://img.shields.io/pypi/pyversions/wayfault.svg)](https://pypi.org/project/wayfault/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/daibeal/wayfault/blob/main/LICENSE)

`wayfault` quantifies the adverse dependence between exposure and counterparty
credit quality — the risk that exposure rises precisely when the counterparty
deteriorates (**Wrong-Way Risk**), and its favourable mirror, **Right-Way
Risk**.

It takes a Monte-Carlo **exposure cube** and a **credit curve** as inputs and
produces:

- baseline (independence-assumption) exposure metrics and CVA,
- a **conditional expected exposure given default** under a pluggable
  dependence model,
- a **WWR-adjusted CVA** and the empirical **alpha multiplier**
  $\alpha = \text{WWR-CVA} / \text{independent-CVA}$,
- ML-based **calibration** of the dependence parameter,
- WWR/RWR **classification and diagnostics**.

The library does **not** generate exposures or bootstrap curves — those are
inputs.

!!! tip "Try it live"
    The **[Playground](playground.md)** runs the real `wayfault` wheel in your
    browser (via WebAssembly/Pyodide) with an interactive dashboard — adjust the
    model and watch the CVA, alpha, and exposure charts update live.

## Why wayfault

!!! tip "Design principles"
    - **Minimal core.** The only hard runtime dependency is `numpy`. Everything
      else (`pandas`, `scikit-learn`, `matplotlib`) is an *optional extra*,
      imported lazily.
    - **Hexagonal architecture.** A pure domain, surrounded by ports
      (Protocols) and swappable adapters. The dependency rule points inward.
    - **Deterministic.** Same seed ⇒ identical results.
    - **Typed & tested.** `mypy --strict` clean, `ruff` clean, ≥ 90 % coverage
      on the core.

## Install

```bash
pip install wayfault                 # core (numpy only)
pip install 'wayfault[io,ml,viz]'    # with optional extras
```

## 30-second example

```python
import numpy as np
from wayfault import estimate_wwr
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel

cube = np.random.default_rng(0).normal(size=(10_000, 12)) + 1.0
tenors = [i / 4 for i in range(1, 13)]   # quarterly to 3y

result = estimate_wwr(
    exposure=InMemoryExposureSource(cube, tenors),
    credit=FlatHazardCreditCurveSource(hazard=0.02, recovery=0.4),
    model=HullWhiteHazardModel(b=0.5),    # b > 0  ->  wrong-way
)

print(result.baseline_cva, result.wwr_cva, result.alpha, result.classification)
```

---

## Worked example: wrong-way risk on a 5-year swap

A realistic end-to-end study, **computed by the real library** (see
[`examples/case_study.py`](https://github.com/daibeal/wayfault/blob/main/examples/case_study.py))
and rendered below with interactive [Plotly](https://plotly.com/javascript/)
charts. The setup: a **5-year receiver interest-rate swap** (humped exposure of
20 000 Monte-Carlo paths over 20 quarterly tenors) facing a **BB-rated
counterparty** with an upward-sloping hazard curve, discounted at a flat 3 %.

<div id="cs-kpis" class="cs-kpis"></div>

### 1. Baseline exposure metrics

From the exposure cube $V(t)$ we form the independence-assumption profiles. The
**expected positive / negative exposure** and **potential future exposure** are

$$
\mathrm{EPE}(t) = \mathbb{E}\!\big[V(t)^{+}\big],\qquad
\mathrm{ENE}(t) = \mathbb{E}\!\big[(-V(t))^{+}\big],\qquad
\mathrm{PFE}_q(t) = \inf\{x : \Pr(V(t)^{+}\le x)\ge q\},
$$

and the **effective EPE** is the time-weighted running maximum
$\mathrm{EEPE} = \tfrac{1}{T}\sum_i \max_{j\le i}\mathrm{EPE}(t_j)\,\Delta t_i$.

<div id="cs-envelope" class="cs-chart"></div>

### 2. Baseline CVA

Unilateral CVA integrates discounted expected exposure against the marginal
default probability, with loss-given-default $1-R$:

$$
\mathrm{CVA} = (1-R)\sum_{i} DF(t_i)\,\mathrm{EE}(t_i)\,
\mathrm{PD}(t_{i-1}, t_i),\qquad
\mathrm{PD}(t_{i-1}, t_i) = S(t_{i-1}) - S(t_i),\quad
S(t) = e^{-\int_0^t \lambda(u)\,du}.
$$

The counterparty's hazard $\lambda(t)$ and survival $S(t)$:

<div id="cs-credit" class="cs-chart"></div>

### 3. The wrong-way adjustment (Hull–White)

Under wrong-way risk, default intensity rises with exposure. The canonical
Hull–White stochastic-hazard model couples them through

$$
\lambda(t) = \exp\!\big(a(t) + b\,V(t)\big),
$$

where $a(t)$ is solved per tenor to reproduce the curve's marginal PDs and $b$
is the wrong-way knob. The conditional expected exposure *given default*
re-weights each scenario by its model-implied default likelihood,
$w_s \propto e^{\,b\,V_s(t)}$, lifting the exposure profile into the
high-exposure tail:

<div id="cs-wwr" class="cs-chart"></div>

### 4. Alpha multiplier and EAD

The empirical **alpha multiplier** and the regulatory **exposure-at-default**
view are

$$
\alpha = \frac{\mathrm{CVA}_{\text{WWR}}}{\mathrm{CVA}_{\text{indep}}},
\qquad
\mathrm{EAD} = \alpha \cdot \mathrm{EEPE}.
$$

Sweeping $b$ traces how the adjustment turns on — monotone in $b$, with
$\alpha \ge 1$ for wrong-way ($b>0$) and $\alpha \le 1$ for right-way ($b<0$):

<div id="cs-sweep" class="cs-chart"></div>

### 5. Where the risk concentrates

The per-tenor ratio $\mathrm{EE}_{\text{cond}}(t)/\mathrm{EPE}(t)$ across the
coupling $b$ shows the adjustment is strongest where exposure peaks — the belly
of the swap:

<div id="cs-heatmap" class="cs-chart"></div>

<div id="cs-surface" class="cs-chart"></div>

### 6. Model comparison

Different dependence families (Hull–White hazard, Gaussian copula, and the
tail-dependent Clayton and Frank copulas) imply different uplifts for the same
book:

<div id="cs-compare" class="cs-chart"></div>

### 7. Arbitrage consistency

A calibrated model must, integrated over the exposure distribution, **reproduce
the curve's marginal PDs** — otherwise it mis-prices the unconditional default.
The Hull–White model's model-implied PDs sit exactly on the target:

$$
\mathbb{E}_{V}\!\big[\mathrm{PD}_i(V)\big] = \mathrm{PD}(t_{i-1}, t_i)
\quad\text{for every tenor } i.
$$

<div id="cs-marginal" class="cs-chart"></div>

<style>
.cs-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(8rem,1fr));gap:.8rem;margin:1.2rem 0}
.cs-kpi{padding:.7rem .9rem;border-radius:.5rem;background:var(--md-code-bg-color)}
.cs-kpi span{display:block;font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;color:var(--md-default-fg-color--light)}
.cs-kpi b{font-size:1.3rem;font-variant-numeric:tabular-nums}
.cs-chart{width:100%;min-height:380px;margin:1rem 0}
</style>

<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.30.0/plotly.min.js"></script>
<script>
(async function () {
  const C = {ink:'#1b1f3b', base:'#4c72b0', wwr:'#c44e52', rwr:'#55a868', pfe:'#8172b3', mut:'#9aa0b5'};
  let data;
  try {
    const here = new URL('.', location.href);
    data = await fetch(new URL('assets/data/case_study.json', here).href).then(r => r.json());
  } catch (e) { console.error('case study data missing', e); return; }

  const layout = (extra={}) => Object.assign({
    margin:{l:55,r:25,t:30,b:45}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color:C.mut, size:12}, legend:{orientation:'h', y:1.12, x:0},
    xaxis:{gridcolor:'rgba(150,150,150,.15)', zeroline:false},
    yaxis:{gridcolor:'rgba(150,150,150,.15)', zeroline:false},
  }, extra);
  const cfg = {displayModeBar:false, responsive:true};
  const plot = (id, traces, lo) => Plotly.newPlot(id, traces, layout(lo), cfg);
  const t = data.tenors;

  // KPI cards
  const cls = data.classification, clsColor = cls==='WRONG_WAY'?C.wwr:cls==='RIGHT_WAY'?C.rwr:C.base;
  document.getElementById('cs-kpis').innerHTML = [
    ['Baseline CVA', data.baseline_cva.toFixed(5)],
    ['WWR CVA', data.wwr_cva.toFixed(5)],
    ['Alpha α', data.alpha.toFixed(3)],
    ['Uplift', (data.uplift_pct>=0?'+':'')+data.uplift_pct.toFixed(1)+'%'],
    ['EEPE', data.eepe.toFixed(4)],
    ['EAD', data.ead.toFixed(4)],
  ].map(([k,v]) => `<div class="cs-kpi"><span>${k}</span><b>${v}</b></div>`).join('') +
    `<div class="cs-kpi"><span>Class</span><b style="color:${clsColor};font-size:1rem">${cls}</b></div>`;

  // 1. exposure envelope
  plot('cs-envelope', [
    {x:t, y:data.pfe99, name:'PFE 99%', mode:'lines', line:{color:C.pfe, dash:'dot', width:1.5}},
    {x:t, y:data.pfe95, name:'PFE 95%', mode:'lines', line:{color:C.pfe, width:1.5}},
    {x:t, y:data.epe, name:'EPE', mode:'lines', line:{color:C.base, width:3}, fill:'tozeroy', fillcolor:'rgba(76,114,176,.12)'},
    {x:t, y:data.ene.map(v=>-v), name:'ENE', mode:'lines', line:{color:C.mut, width:1.5}},
  ], {yaxis:{title:'exposure', gridcolor:'rgba(150,150,150,.15)'}, xaxis:{title:'tenor (years)'}});

  // 2. credit curve (hazard + survival, dual axis)
  plot('cs-credit', [
    {x:t, y:data.hazard, name:'hazard λ(t)', mode:'lines', line:{color:C.wwr, width:2.5, shape:'hv'}},
    {x:t, y:data.survival, name:'survival S(t)', mode:'lines', yaxis:'y2', line:{color:C.base, width:2.5}},
  ], {xaxis:{title:'tenor (years)'}, yaxis:{title:'hazard', gridcolor:'rgba(150,150,150,.15)'},
      yaxis2:{title:'survival', overlaying:'y', side:'right', range:[0,1], showgrid:false}});

  // 3. wrong-way conditional EE vs EPE
  plot('cs-wwr', [
    {x:t, y:data.epe, name:'EPE (independent)', mode:'lines', line:{color:C.base, width:2.5}},
    {x:t, y:data.conditional_ee, name:'Conditional EE | default', mode:'lines',
     line:{color:clsColor, width:2.5}, fill:'tonexty', fillcolor:'rgba(196,78,82,.15)'},
  ], {xaxis:{title:'tenor (years)'}, yaxis:{title:'exposure', gridcolor:'rgba(150,150,150,.15)'}});

  // 4. alpha + cva sweep
  const bs = data.sweep.map(p=>p.b);
  plot('cs-sweep', [
    {x:bs, y:data.sweep.map(p=>p.alpha), name:'alpha', mode:'lines+markers', line:{color:C.ink, width:2.5}},
    {x:bs, y:data.sweep.map(p=>p.wwr_cva), name:'WWR CVA', mode:'lines', yaxis:'y2',
     line:{color:C.pfe, width:2, dash:'dash'}},
  ], {xaxis:{title:'wrong-way coupling b'}, yaxis:{title:'alpha', gridcolor:'rgba(150,150,150,.15)'},
      yaxis2:{title:'CVA', overlaying:'y', side:'right', showgrid:false},
      shapes:[{type:'line', x0:0, x1:0, y0:0, y1:1, yref:'paper', line:{color:C.mut, width:1, dash:'dot'}}]});

  // 5. ratio heatmap
  plot('cs-heatmap', [{
    type:'heatmap', x:data.surface.tenors, y:data.surface.b, z:data.surface.ratio,
    colorscale:'RdBu', reversescale:true, zmid:1, colorbar:{title:'cond/EPE'},
  }], {xaxis:{title:'tenor (years)'}, yaxis:{title:'coupling b'}, margin:{l:55,r:25,t:30,b:45}});

  // 5b. 3D surface
  Plotly.newPlot('cs-surface', [{
    type:'surface', x:data.surface.tenors, y:data.surface.b, z:data.surface.z,
    colorscale:'Viridis', colorbar:{title:'cond EE'},
  }], Object.assign(layout(), {
    title:{text:'Conditional EE(t, b)', font:{size:13}},
    scene:{xaxis:{title:'tenor'}, yaxis:{title:'b'}, zaxis:{title:'cond EE'}},
    margin:{l:0,r:0,t:30,b:0},
  }), cfg);

  // 6. model comparison
  plot('cs-compare', [{
    type:'bar', x:data.comparison.map(m=>m.name), y:data.comparison.map(m=>m.alpha),
    marker:{color:[C.base, C.pfe, C.wwr, C.rwr]},
    text:data.comparison.map(m=>'α='+m.alpha.toFixed(2)+'  ('+(m.uplift>=0?'+':'')+m.uplift.toFixed(0)+'%)'),
    textposition:'outside',
  }], {yaxis:{title:'alpha', gridcolor:'rgba(150,150,150,.15)'}, showlegend:false,
      shapes:[{type:'line', x0:-0.5, x1:3.5, y0:1, y1:1, line:{color:C.mut, width:1, dash:'dot'}}]});

  // 7. marginal PD consistency
  plot('cs-marginal', [
    {x:data.marginal.tenors, y:data.marginal.target, name:'curve target', mode:'lines',
     line:{color:C.base, width:3}},
    {x:data.marginal.tenors, y:data.marginal.implied, name:'model implied', mode:'markers',
     marker:{color:C.wwr, size:7, symbol:'circle-open', line:{width:2}}},
  ], {xaxis:{title:'tenor (years)'}, yaxis:{title:'marginal PD', gridcolor:'rgba(150,150,150,.15)'}});
})();
</script>

---

Continue with the [Getting Started](getting-started.md) guide, or jump to the
[API Reference](api/index.md).
