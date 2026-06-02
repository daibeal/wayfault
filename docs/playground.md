# Playground

This page runs the **real `wayfault` library** — the exact same Python wheel you
`pip install` — directly in your browser. No rewrite, no server, no API.

??? question "How can Python run in the browser? (the porting question)"
    The library is **not** re-implemented in JavaScript. Instead it runs on
    [**Pyodide**](https://pyodide.org), which is CPython (plus numpy, scipy, …)
    compiled to **WebAssembly**. The page:

    1. loads the Pyodide runtime from a CDN,
    2. `micropip`-installs the published `wayfault` wheel
       (`wayfault-*-py3-none-any.whl`) — a pure-Python wheel, so it just
       works,
    3. calls `estimate_wwr(...)` and hands the resulting numbers to a JS chart
       library ([Chart.js](https://www.chartjs.org/)) for rendering.

    Because `wayfault`'s only hard dependency is **numpy** (which Pyodide ships
    natively) and the optional extras are lazily imported, the whole library
    drops into the browser unchanged. This is the recommended way to "port"
    a numpy-based Python library to the web: **don't port it — run it on
    WebAssembly.**

<div id="wf-status" class="wf-status">Initializing…</div>

<div id="wf-playground" hidden>
  <div class="wf-controls">
    <label>Model
      <select id="wf-model">
        <option value="hullwhite">Hull–White hazard</option>
        <option value="copula">Gaussian copula</option>
        <option value="clayton">Clayton copula (lower-tail)</option>
        <option value="frank">Frank copula</option>
        <option value="independent">Independent</option>
      </select>
    </label>
    <label id="wf-param-wrap"><span id="wf-param-label">b</span>: <b id="wf-param-val">0.50</b>
      <input type="range" id="wf-param" min="-1.5" max="1.5" step="0.1" value="0.5">
    </label>
    <label>Hazard λ: <b id="wf-haz-val">0.020</b>
      <input type="range" id="wf-haz" min="0.005" max="0.10" step="0.005" value="0.02">
    </label>
    <label>Recovery R: <b id="wf-rec-val">0.40</b>
      <input type="range" id="wf-rec" min="0" max="0.8" step="0.05" value="0.4">
    </label>
    <label>Seed
      <input type="number" id="wf-seed" min="0" max="9999" step="1" value="0" style="width:5rem">
    </label>
  </div>

  <div class="wf-kpis">
    <div class="wf-kpi"><span>Baseline CVA</span><b id="kpi-base">–</b></div>
    <div class="wf-kpi"><span>WWR CVA</span><b id="kpi-wwr">–</b></div>
    <div class="wf-kpi"><span>Alpha α</span><b id="kpi-alpha">–</b></div>
    <div class="wf-kpi"><span>Uplift</span><b id="kpi-uplift">–</b></div>
    <div class="wf-kpi"><span>EEPE</span><b id="kpi-eepe">–</b></div>
    <div class="wf-kpi wf-kpi-class"><span>Class</span><b id="kpi-class">–</b></div>
  </div>

  <div class="wf-charts">
    <div class="wf-chart"><h4>Exposure profiles</h4><canvas id="wf-exposure"></canvas></div>
    <div class="wf-chart"><h4>Alpha vs dependence knob</h4><canvas id="wf-sweep"></canvas></div>
  </div>
  <p class="wf-foot">Computed live by <code>wayfault.estimate_wwr</code> running on Pyodide
  (4&nbsp;000 Monte-Carlo scenarios × 12 quarterly tenors).</p>
</div>

<style>
.wf-status{padding:.7rem 1rem;border-radius:.4rem;background:var(--md-code-bg-color);
  font-family:var(--md-code-font);font-size:.8rem;margin:1rem 0}
.wf-status.err{color:#c44e52}
#wf-playground .wf-controls{display:flex;flex-wrap:wrap;gap:1rem 1.4rem;align-items:end;
  padding:1rem;border:1px solid var(--md-default-fg-color--lightest);border-radius:.5rem;margin-bottom:1rem}
#wf-playground .wf-controls label{display:flex;flex-direction:column;gap:.25rem;font-size:.72rem;
  text-transform:uppercase;letter-spacing:.04em;color:var(--md-default-fg-color--light)}
#wf-playground .wf-controls input[type=range]{width:11rem}
#wf-playground select,#wf-playground input{font-size:.85rem}
.wf-kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(8.5rem,1fr));gap:.8rem;margin-bottom:1.2rem}
.wf-kpi{padding:.7rem .9rem;border-radius:.5rem;background:var(--md-code-bg-color)}
.wf-kpi span{display:block;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;
  color:var(--md-default-fg-color--light)}
.wf-kpi b{font-size:1.25rem;font-variant-numeric:tabular-nums}
.wf-kpi-class b{font-size:.95rem}
.wf-charts{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media (max-width:60rem){.wf-charts{grid-template-columns:1fr}}
.wf-chart{border:1px solid var(--md-default-fg-color--lightest);border-radius:.5rem;padding:.8rem}
.wf-chart h4{margin:.1rem 0 .6rem}
.wf-foot{font-size:.72rem;color:var(--md-default-fg-color--light);margin-top:1rem}
</style>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js"></script>
<script>
const WF_PY = `
import json
import numpy as np
from wayfault import estimate_wwr
from wayfault.adapters.outbound.exposure_inmemory import InMemoryExposureSource
from wayfault.adapters.outbound.credit_flat import FlatHazardCreditCurveSource
from wayfault.adapters.outbound.dependence_independent import IndependentModel
from wayfault.adapters.outbound.dependence_hullwhite import HullWhiteHazardModel
from wayfault.adapters.outbound.dependence_copula import GaussianCopulaModel
from wayfault.adapters.outbound.dependence_archimedean import ClaytonCopulaModel, FrankCopulaModel

_TENORS = [i / 4 for i in range(1, 13)]

def _make_cube(seed):
    rng = np.random.default_rng(int(seed))
    return np.asarray(_TENORS) + rng.normal(scale=0.6, size=(4000, len(_TENORS)))

def _model(name, p):
    p = float(p)
    if name == 'hullwhite':
        return HullWhiteHazardModel(b=p)
    if name == 'copula':
        return GaussianCopulaModel(rho=float(np.clip(p, -0.98, 0.98)))
    if name == 'clayton':
        return ClaytonCopulaModel(theta=max(p, 1e-3))
    if name == 'frank':
        return FrankCopulaModel(theta=p if abs(p) > 1e-3 else 1e-3)
    return IndependentModel()

def _grid(name):
    if name == 'clayton':
        return list(np.linspace(0.05, 5.0, 13))
    if name == 'copula':
        return list(np.linspace(-0.95, 0.95, 13))
    if name == 'frank':
        return list(np.linspace(-8.0, 8.0, 13))
    if name == 'hullwhite':
        return list(np.linspace(-1.5, 1.5, 13))
    return [0.0]

def compute(name, param, hazard, recovery, seed):
    cube = _make_cube(seed)
    ex = InMemoryExposureSource(cube, _TENORS)
    cr = FlatHazardCreditCurveSource(hazard=float(hazard), recovery=float(recovery))
    r = estimate_wwr(ex, cr, _model(name, param))
    d = r.to_dict()
    d['sweep'] = [
        {'x': float(g), 'alpha': float(estimate_wwr(ex, cr, _model(name, g)).alpha)}
        for g in _grid(name)
    ]
    return json.dumps(d)
`;

const CFG = {
  hullwhite:   {min:-1.5, max:1.5, step:0.1,  val:0.5, label:'b (Hull–White coupling)'},
  copula:      {min:-0.95,max:0.95,step:0.05, val:0.5, label:'ρ (Gaussian correlation)'},
  clayton:     {min:0.05, max:5.0, step:0.05, val:2.0, label:'θ (Clayton, lower-tail)'},
  frank:       {min:-8.0, max:8.0, step:0.5,  val:4.0, label:'θ (Frank)'},
  independent: {min:0,    max:0,   step:1,    val:0,   label:'(no parameter)'},
};
const CLASS_COLOR = {WRONG_WAY:'#c44e52', RIGHT_WAY:'#55a868', NEUTRAL:'#4c72b0'};

(async function () {
  const status = document.getElementById('wf-status');
  const set = (t, err=false) => { status.textContent = t; status.classList.toggle('err', err); };
  try {
    set('Loading Python runtime (Pyodide ~10 MB, first load only)…');
    const pyodide = await loadPyodide();
    set('Loading numpy…');
    await pyodide.loadPackage(['numpy', 'micropip']);
    set('Installing the wayfault wheel…');
    const here = new URL('.', location.href);
    // The wheel filename (and version) is resolved from a generated manifest so
    // the playground always installs whatever version the docs were built with.
    const manifest = await fetch(new URL('../assets/wheels/manifest.json', here).href)
      .then(r => r.json());
    const wheelUrl = new URL('../assets/wheels/' + manifest.wheel, here).href;
    const micropip = pyodide.pyimport('micropip');
    await micropip.install(wheelUrl);
    set('Initializing models…');
    await pyodide.runPythonAsync(WF_PY);
    const compute = pyodide.globals.get('compute');

    document.getElementById('wf-playground').hidden = false;
    status.style.display = 'none';

    const $ = (id) => document.getElementById(id);
    const mk = (canvas, cfg) => new Chart($(canvas).getContext('2d'), cfg);
    const line = (label, data, color, dash=false, axis='y') => ({
      label, data, borderColor:color, backgroundColor:color+'22', tension:.25,
      pointRadius:0, borderWidth:2, borderDash:dash?[6,4]:[], yAxisID:axis, fill:false,
    });

    const expChart = mk('wf-exposure', {type:'line', data:{labels:[],datasets:[]},
      options:{responsive:true, interaction:{mode:'index',intersect:false},
        scales:{x:{title:{display:true,text:'Tenor (years)'}}, y:{title:{display:true,text:'Exposure'}}}}});
    const sweepChart = mk('wf-sweep', {type:'line', data:{labels:[],datasets:[]},
      options:{responsive:true,
        scales:{x:{title:{display:true,text:'dependence parameter'}},
                y:{title:{display:true,text:'alpha'}}}}});

    function update(d) {
      const c = CLASS_COLOR[d.classification] || '#4c72b0';
      $('kpi-base').textContent = d.baseline_cva.toFixed(5);
      $('kpi-wwr').textContent  = d.wwr_cva.toFixed(5);
      $('kpi-alpha').textContent= d.alpha.toFixed(4);
      $('kpi-uplift').textContent = (d.uplift_pct>=0?'+':'') + d.uplift_pct.toFixed(1) + '%';
      $('kpi-eepe').textContent = d.eepe.toFixed(4);
      const cl = $('kpi-class'); cl.textContent = d.classification; cl.style.color = c;

      expChart.data.labels = d.tenors.map(t => t.toFixed(2));
      expChart.data.datasets = [
        line('PFE (95%)', d.pfe, '#8172b3', true),
        line('EPE (independent)', d.epe, '#4c72b0'),
        line('Conditional EE | default', d.conditional_ee, c),
      ];
      expChart.update();

      sweepChart.data.labels = d.sweep.map(p => p.x.toFixed(2));
      sweepChart.data.datasets = [line('alpha', d.sweep.map(p=>p.alpha), '#1b1f3b')];
      sweepChart.update();
    }

    function run() {
      const name = $('wf-model').value;
      const json = compute(name, parseFloat($('wf-param').value),
        parseFloat($('wf-haz').value), parseFloat($('wf-rec').value),
        parseInt($('wf-seed').value || '0', 10));
      update(JSON.parse(json));
    }

    function syncModel() {
      const cfg = CFG[$('wf-model').value];
      const p = $('wf-param');
      p.min=cfg.min; p.max=cfg.max; p.step=cfg.step; p.value=cfg.val;
      $('wf-param-label').textContent = cfg.label;
      $('wf-param-wrap').style.visibility = ($('wf-model').value==='independent')?'hidden':'visible';
      $('wf-param-val').textContent = (+cfg.val).toFixed(2);
      run();
    }

    $('wf-model').addEventListener('change', syncModel);
    $('wf-param').addEventListener('input', () => { $('wf-param-val').textContent=(+$('wf-param').value).toFixed(2); run(); });
    $('wf-haz').addEventListener('input', () => { $('wf-haz-val').textContent=(+$('wf-haz').value).toFixed(3); run(); });
    $('wf-rec').addEventListener('input', () => { $('wf-rec-val').textContent=(+$('wf-rec').value).toFixed(2); run(); });
    $('wf-seed').addEventListener('change', run);
    syncModel();
  } catch (e) {
    set('Failed to start the playground: ' + e, true);
    console.error(e);
  }
})();
</script>
