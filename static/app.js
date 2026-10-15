// Front-end for the Smart Traffic Management web app.
const $ = (id) => document.getElementById(id);
const COLORS = { "Fixed-time": "#9aa0a6", "Adaptive (density)": "#1a73e8", "RL agent (Q-learning)": "#e8710a" };
const LANE_COLORS = ["#1a73e8", "#e8710a", "#188038", "#a142f4"];
const CLASS_COLORS = ["#34a853", "#fbbc04", "#1a73e8", "#a142f4"];

// ------------------------------------------------------------------ tabs
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  $(t.dataset.tab).classList.add("active");
  if (t.dataset.tab === "results") loadResults();
}));

// slider value labels
[["imgsz", "imgszOut"], ["conf", "confOut"], ["stall", "stallOut"], ["fixedGreen", "fixedOut"]]
  .forEach(([i, o]) => $(i).addEventListener("input", () => { $(o).textContent = $(i).value; }));

const metric = (label, value, sub = "", cls = "") =>
  `<div class="metric"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="sub">${sub}</div></div>`;

// --------------------------------------------------------- live detection
async function loadVideos(select) {
  const vids = await (await fetch("/api/videos")).json();
  $("video").innerHTML = vids.map((v) => `<option ${v === select ? "selected" : ""}>${v}</option>`).join("");
}
loadVideos();

$("upload").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("video", f);
  $("progressText").textContent = "Uploading…";
  const r = await (await fetch("/api/upload", { method: "POST", body: fd })).json();
  $("progressText").textContent = r.name ? `Uploaded ${r.name}` : "Upload failed";
  if (r.name) loadVideos(r.name);
});

const laneChart = new Chart($("laneChart"), {
  type: "line", data: { labels: [], datasets: [] },
  options: { animation: false, elements: { point: { radius: 0 } },
    scales: { x: { title: { display: true, text: "Video time (s)" } }, y: { beginAtZero: true, title: { display: true, text: "Vehicles" } } } },
});
const classChart = new Chart($("classChart"), {
  type: "bar", data: { labels: ["car", "motorcycle", "bus", "truck"], datasets: [{ data: [0, 0, 0, 0], backgroundColor: CLASS_COLORS }] },
  options: { animation: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
});

let poll = null;
$("runDetect").addEventListener("click", () => {
  const sid = Math.random().toString(36).slice(2);
  const p = new URLSearchParams({
    sid, video: $("video").value, model: $("model").value, imgsz: $("imgsz").value,
    conf: $("conf").value, lanes: $("lanes").value, stall: $("stall").value, max_frames: $("maxFrames").value,
  });
  $("placeholder").style.display = "none";
  $("stream").style.display = "block";
  $("stream").src = `/video_feed?${p}`;
  $("alertBox").classList.add("hidden");
  $("runDetect").disabled = true;
  $("runDetect").textContent = "Running…";
  clearInterval(poll);
  poll = setInterval(() => updateLive(sid), 700);
});

async function updateLive(sid) {
  const d = await (await fetch(`/api/live_stats?sid=${sid}`)).json();
  if (!d.ready || !d.stats) return;
  const s = d.stats;
  $("progressBar").style.width = `${(100 * d.frame) / Math.max(d.total, 1)}%`;
  $("progressText").textContent = `Frame ${d.frame} / ${d.total} · ${s.infer_ms} ms per frame`;

  let html = metric("Vehicles in frame", s.vehicles) + metric("Unique vehicles tracked", s.unique_vehicles);
  for (let k = 1; k <= d.lanes; k++) {
    const dens = s[`lane${k}_density`];
    html += metric(`Lane ${k} · <span class="${dens}">${dens}</span>`, `${s[`lane${k}_green_s`]} s green`, `${s[`lane${k}_count`]} vehicles`);
  }
  html += metric("Incident alerts", d.alerts.length, "", d.alerts.length ? "HIGH" : "");
  $("liveMetrics").innerHTML = html;

  if (d.alerts.length) {
    const a = d.alerts[d.alerts.length - 1];
    $("alertBox").textContent = `⚠ t=${a.time_s}s · ${a.type} · ${a.vehicle} #${a.track_id}`;
    $("alertBox").classList.remove("hidden");
  }
  $("alertTable").querySelector("tbody").innerHTML = d.alerts.map((a) =>
    `<tr><td>${a.time_s}</td><td>${a.vehicle}</td><td>${a.track_id}</td><td>${a.type}</td></tr>`).join("")
    || `<tr><td colspan="4" class="muted">No incidents detected</td></tr>`;

  laneChart.data.labels = d.history.map((h) => h.time_s);
  laneChart.data.datasets = Array.from({ length: d.lanes }, (_, i) => ({
    label: `Lane ${i + 1}`, data: d.history.map((h) => h[`lane${i + 1}_count`]),
    borderColor: LANE_COLORS[i], borderWidth: 1.5, tension: 0.2,
  }));
  laneChart.update();
  if (d.summary) {
    classChart.data.datasets[0].data = ["car", "motorcycle", "bus", "truck"].map((c) => d.summary.class_totals[c]);
    classChart.update();
  }
  if (d.done) {
    clearInterval(poll);
    $("runDetect").disabled = false;
    $("runDetect").textContent = "▶ Run detection";
    $("progressText").textContent = `Finished: ${d.summary.frames} frames · ${d.summary.unique_vehicles} vehicles tracked · avg ${d.summary.avg_infer_ms} ms per frame`;
  }
}

// -------------------------------------------------------------- simulation
const queueChart = new Chart($("queueChart"), {
  type: "line", data: { labels: [], datasets: [] },
  options: { animation: false, elements: { point: { radius: 0 } },
    scales: { x: { title: { display: true, text: "Time (minutes)" } }, y: { beginAtZero: true, title: { display: true, text: "Vehicles waiting" } } } },
});

$("runSim").addEventListener("click", async () => {
  $("runSim").disabled = true;
  $("runSim").textContent = "Simulating…";
  const r = await (await fetch("/api/simulate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario: $("scenario").value, fixed_green: +$("fixedGreen").value, seed: +$("seed").value }),
  })).json();
  $("runSim").disabled = false;
  $("runSim").textContent = "▶ Run simulation";

  const base = r.results[0].avg_wait_s;
  $("simMetrics").innerHTML = r.results.map((x, i) => metric(`${x.controller}: average wait`, `${x.avg_wait_s.toFixed(1)} s`,
    i === 0 ? "baseline" : `<span class="good">${(100 * (base - x.avg_wait_s) / base).toFixed(0)}% less waiting than fixed-time</span>`)).join("");

  queueChart.data.labels = r.minutes;
  queueChart.data.datasets = Object.entries(r.queues).map(([name, q]) => ({
    label: name, data: q, borderColor: COLORS[name], borderWidth: 2, tension: 0.2,
  }));
  queueChart.update();

  $("simTable").innerHTML = `<thead><tr><th>Controller</th><th class="num">Avg wait (s)</th><th class="num">Avg queue</th>
    <th class="num">Max queue</th><th class="num">Vehicles served</th><th class="num">Vehicles arrived</th></tr></thead><tbody>` +
    r.results.map((x) => `<tr><td>${x.controller}</td><td class="num">${x.avg_wait_s.toFixed(1)}</td><td class="num">${x.avg_queue.toFixed(1)}</td>
    <td class="num">${x.max_queue}</td><td class="num">${x.throughput}</td><td class="num">${x.arrived}</td></tr>`).join("") + "</tbody>";
});

// ----------------------------------------------------------------- results
let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return;
  const r = await (await fetch("/api/results")).json();
  resultsLoaded = true;
  if (!r.signal.length) {
    $("resSignal").innerHTML = "<tr><td>Run <code>python run_experiments.py</code> to generate results.</td></tr>";
    return;
  }
  $("resSignal").innerHTML = `<thead><tr><th>Scenario</th><th>Controller</th><th class="num">Avg wait (s)</th><th class="num">Avg queue</th>
    <th class="num">Max queue</th><th class="num">Vehicles served %</th></tr></thead><tbody>` +
    r.signal.map((x) => `<tr><td>${x.scenario}</td><td>${x.controller}</td><td class="num">${x.avg_wait_s.toFixed(1)}</td>
    <td class="num">${x.avg_queue.toFixed(1)}</td><td class="num">${x.max_queue.toFixed(0)}</td><td class="num">${x.served_pct.toFixed(1)}</td></tr>`).join("") + "</tbody>";
  $("resSignalCharts").innerHTML = r.signal_charts.map((c) => `<img src="${c}" alt="">`).join("");
  $("resDetection").innerHTML = r.detection.map((d) => `
    <div class="card">
      <h2>${d.video}</h2>
      <div class="metrics">
        ${metric("Frames processed", d.frames)}
        ${metric("Processing speed", `${d.fps.toFixed(1)} FPS`, `${d.avg_infer_ms.toFixed(0)} ms per frame (CPU)`)}
        ${metric("Time at density", `${d.density_share_pct.LOW}% / ${d.density_share_pct.MEDIUM}% / ${d.density_share_pct.HIGH}%`, "low / medium / high")}
        ${metric("Incident alerts", d.alerts.length)}
      </div>
      <div class="det">
        <video src="${d.video_url}" controls muted loop></video>
        <div>${d.charts.map((c) => `<img src="${c}" alt="">`).join("")}</div>
      </div>
    </div>`).join("");
}
