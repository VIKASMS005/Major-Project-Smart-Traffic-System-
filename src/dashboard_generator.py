"""
Standalone junction simulator (output layer).

Writes one self-contained HTML file (no server, no internet, no external JS)
that animates a 4-way junction using the signal plan computed by main.py:
lanes take turns in order, each gets its allocated green time, then 3 s
yellow; lanes flagged with an incident are skipped and shown as BLOCKED.
The junction drawing code is the same `static/junction.js` used by the web
app, inlined into the page.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic Light Simulator</title>
<style>
:root{--bg:#0b1016;--panel:#121a23;--border:#223041;--text:#e6edf5;--muted:#8a9bb0;--green:#2fe36b;--yellow:#ffc61a;--red:#ff4d4d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Roboto,Arial,sans-serif}
header{padding:18px 26px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
h1{margin:0;font-size:20px}.muted{color:var(--muted)}
main{display:grid;grid-template-columns:minmax(320px,520px) 1fr;gap:20px;padding:22px 26px}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:16px}
canvas{width:100%;aspect-ratio:1;display:block;border-radius:10px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-content:start}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px;transition:.3s}
.card.GREEN{border-color:var(--green);box-shadow:0 0 16px rgba(47,227,107,.2)}.card.YELLOW{border-color:var(--yellow)}
.card.BLOCKED{border-color:var(--red)}
.top{display:flex;justify-content:space-between;align-items:center}.name{font-weight:700;font-size:16px}
.lamps{display:flex;gap:6px;background:#05080b;padding:5px 7px;border-radius:14px}
.lamps i{width:14px;height:14px;border-radius:50%;background:#2a2f36}
.lamps i.r.on{background:var(--red);box-shadow:0 0 8px var(--red)}.lamps i.y.on{background:var(--yellow);box-shadow:0 0 8px var(--yellow)}
.lamps i.g.on{background:var(--green);box-shadow:0 0 8px var(--green)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);margin-top:12px;gap:6px}.stats div{font-size:11px;color:var(--muted)}
.stats b{display:block;color:var(--text);font-size:18px}.mix{margin-top:10px;font-size:12px;color:var(--muted)}
.badge{font-size:11px;font-weight:700;padding:3px 8px;border-radius:10px}
.Low{color:var(--green);background:rgba(47,227,107,.12)}.Medium{color:var(--yellow);background:rgba(255,198,26,.12)}
.High{color:var(--red);background:rgba(255,77,77,.13)}.blocked{color:#fff;background:var(--red)}
.status{margin-top:10px;font-weight:700;font-size:13px}
@media(max-width:900px){main{grid-template-columns:1fr}.cards{grid-template-columns:1fr}}
</style></head><body>
<header><div><div class="muted" style="font-size:12px;letter-spacing:.8px">ADAPTIVE SIGNAL PLAN FROM VIDEO ANALYSIS</div>
<h1>Traffic Light Simulator</h1></div><div class="muted" id="cycle">Cycle 1</div></header>
<main><div class="panel"><canvas id="junction"></canvas></div><div class="cards" id="cards"></div></main>
<script>__JUNCTION_JS__</script>
<script>
const PLAN = __PLAN__;
const YELLOW = 3;
const view = new JunctionView(document.getElementById("junction"));
const cards = document.getElementById("cards");
cards.innerHTML = PLAN.map(l => `
 <div class="card" id="card-${l.id}">
  <div class="top"><div class="name">${l.name} <span class="muted" style="font-weight:500">${l.direction}</span></div>
   <div class="lamps"><i class="r"></i><i class="y"></i><i class="g"></i></div></div>
  <div class="stats"><div>Vehicles<b>${l.count}</b></div><div>Density<b><span class="badge ${l.density}">${l.density}</span></b></div>
   <div>Green time<b>${l.incident ? "skipped" : l.green_time + " s"}</b></div></div>
  <div class="mix">${Object.entries(l.by_type).filter(([,v]) => v).map(([k,v]) => v + " " + k).join(" · ") || "no vehicles"}</div>
  <div class="status" id="st-${l.id}"></div></div>`).join("");

const order = PLAN.filter(l => !l.incident);
let idx = 0, phase = "GREEN", end = 0, cycle = 1;
function start() {
  if (!order.length) return;
  phase = "GREEN"; end = Date.now() + order[idx].green_time * 1000;
}
function lights() {
  const act = order[idx];
  return Object.fromEntries(PLAN.map(l => [l.id, l.incident ? "BLOCKED" : (act && l.id === act.id ? phase : "RED")]));
}
function tick() {
  if (order.length && Date.now() >= end) {
    if (phase === "GREEN") { phase = "YELLOW"; end = Date.now() + YELLOW * 1000; }
    else { idx = (idx + 1) % order.length; if (idx === 0) cycle++; start(); }
  }
  const L = lights(), rem = Math.max(0, (end - Date.now()) / 1000);
  view.update({ lanes: PLAN.map(l => ({ ...l, light: L[l.id] })), active_lane: order[idx] && order[idx].id, phase, remaining: rem });
  PLAN.forEach(l => {
    const s = L[l.id];
    document.getElementById("card-" + l.id).className = "card " + s;
    const [r, y, g] = document.querySelector(`#card-${l.id} .lamps`).children;
    r.className = "r" + (s === "RED" || s === "BLOCKED" ? " on" : ""); y.className = "y" + (s === "YELLOW" ? " on" : "");
    g.className = "g" + (s === "GREEN" ? " on" : "");
    document.getElementById("st-" + l.id).innerHTML = s === "BLOCKED" ? '<span class="badge blocked">INCIDENT: LANE BLOCKED</span>'
      : s === "GREEN" ? `<span style="color:var(--green)">GREEN · ${Math.ceil(rem)} s left</span>`
      : s === "YELLOW" ? `<span style="color:var(--yellow)">YELLOW · ${Math.ceil(rem)} s</span>` : '<span class="muted">RED · waiting</span>';
  });
  document.getElementById("cycle").textContent = "Cycle " + cycle;
}
start(); setInterval(tick, 200); tick();
</script></body></html>
"""


def generate_dashboard(lane_results: list[dict], out_path: str) -> str:
    """Write the simulator HTML.

    lane_results: one dict per lane with keys
        id, name, direction, count, density, by_type, green_time, incident
    """
    with open(os.path.join(ROOT, "static", "junction.js"), encoding="utf-8") as f:
        junction_js = f.read()
    html = TEMPLATE.replace("__JUNCTION_JS__", junction_js).replace("__PLAN__", json.dumps(lane_results))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
