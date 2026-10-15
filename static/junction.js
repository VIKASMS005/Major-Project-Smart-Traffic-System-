/*
 * Animated top-down view of a 4-way junction (left-hand traffic, as in India).
 * Used by the live Control Center and by the standalone HTML simulator.
 *
 *   const view = new JunctionView(canvasElement);
 *   view.update({ lanes: [{ id, direction, count, light, by_type }], active_lane, phase, remaining });
 *
 * light is one of GREEN | YELLOW | RED | BLOCKED.
 */
(function () {
  // 2W two-wheelers, 3W auto-rickshaws, 4W cars, HV heavy vehicles (buses, trucks)
  const CAR_COLORS = { "4W": ["#e8eaed", "#4dabf7", "#adb5bd", "#f06595", "#ffd43b"], "2W": ["#ffa94d"],
    "3W": ["#c8e639"], HV: ["#fab005", "#9775fa"] };
  const SIZE = { "2W": [7, 12], "3W": [11, 16], "4W": [14, 22], HV: [16, 34] };   // width, length
  const LAMP = { GREEN: "#2fe36b", YELLOW: "#ffc61a", RED: "#ff3b3b", BLOCKED: "#ff3b3b" };

  // geometry per approach: where the queue sits and which way traffic moves
  // (left-hand traffic: each approach uses the lane on its own left)
  function approach(dir, c, r) {
    const half = r / 2;
    switch (dir) {
      case "North": return { x: c + half, y: c - r, dx: 0, dy: 1, rot: Math.PI };       // heading south
      case "South": return { x: c - half, y: c + r, dx: 0, dy: -1, rot: 0 };            // heading north
      case "East":  return { x: c + r, y: c + half, dx: -1, dy: 0, rot: -Math.PI / 2 }; // heading west
      default:      return { x: c - r, y: c - half, dx: 1, dy: 0, rot: Math.PI / 2 };   // West, heading east
    }
  }

  function vehiclePalette(byType, n) {
    const list = [];
    const t = byType || { "4W": n };
    ["HV", "3W", "2W", "4W"].forEach((k) => { for (let i = 0; i < (t[k] || 0); i++) list.push(k); });
    while (list.length < n) list.push("4W");
    // interleave so big vehicles are not all at the front
    return list.sort((a, b) => ((a.charCodeAt(0) * 7 + b.length) % 3) - 1).slice(0, n);
  }

  class JunctionView {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.state = { lanes: [] };
      this.moving = [];
      this.lastSpawn = {};
      this.t0 = performance.now();
      const loop = (t) => { this.draw(t); requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }

    update(state) { this.state = state; }

    draw(now) {
      const cv = this.canvas, ctx = this.ctx;
      const dpr = window.devicePixelRatio || 1;
      const size = cv.clientWidth;
      if (cv.width !== size * dpr) { cv.width = size * dpr; cv.height = size * dpr; }
      ctx.setTransform((dpr * size) / 520, 0, 0, (dpr * size) / 520, 0, 0);
      const S = 520, c = S / 2, r = 62;

      // ground
      ctx.fillStyle = "#1c2a22"; ctx.fillRect(0, 0, S, S);
      ctx.fillStyle = "#24362b";
      [[0, 0], [c + r, 0], [0, c + r], [c + r, c + r]].forEach(([x, y]) => ctx.fillRect(x + 10, y + 10, c - r - 20, c - r - 20));
      // footpaths
      ctx.fillStyle = "#4a4f55";
      ctx.fillRect(c - r - 8, 0, 8, S); ctx.fillRect(c + r, 0, 8, S);
      ctx.fillRect(0, c - r - 8, S, 8); ctx.fillRect(0, c + r, S, 8);
      // roads
      ctx.fillStyle = "#30353b";
      ctx.fillRect(c - r, 0, 2 * r, S); ctx.fillRect(0, c - r, S, 2 * r);
      // centre lines (yellow, dashed) outside the box
      ctx.strokeStyle = "#e6b800"; ctx.lineWidth = 2; ctx.setLineDash([14, 10]);
      ctx.beginPath();
      ctx.moveTo(c, 0); ctx.lineTo(c, c - r - 18); ctx.moveTo(c, c + r + 18); ctx.lineTo(c, S);
      ctx.moveTo(0, c); ctx.lineTo(c - r - 18, c); ctx.moveTo(c + r + 18, c); ctx.lineTo(S, c);
      ctx.stroke(); ctx.setLineDash([]);
      // zebra crossings
      ctx.fillStyle = "#d9dde1";
      for (let i = -r + 6; i < r - 4; i += 12) {
        ctx.fillRect(c + i, c - r - 16, 7, 12); ctx.fillRect(c + i, c + r + 4, 7, 12);
        ctx.fillRect(c - r - 16, c + i, 12, 7); ctx.fillRect(c + r + 4, c + i, 12, 7);
      }
      // stop lines
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(c, c - r - 20, r, 3); ctx.fillRect(c - r, c + r + 17, r, 3);
      ctx.fillRect(c + r + 17, c, 3, r); ctx.fillRect(c - r - 20, c - r, 3, r);

      const lanes = this.state.lanes || [];
      const dt = (now - (this._last || now)) / 1000; this._last = now;

      lanes.forEach((ln) => {
        const a = approach(ln.direction, c, r);
        const queue = Math.min(ln.count || 0, 9);
        const kinds = vehiclePalette(ln.by_type, queue);
        // queued vehicles behind the stop line
        let offset = 30;
        kinds.forEach((k, i) => {
          const len = SIZE[k][1];
          const x = a.x - a.dx * (offset + len / 2), y = a.y - a.dy * (offset + len / 2);
          this.drawVehicle(x, y, a.rot, k, i);
          offset += len + 6;
        });
        // spawn moving vehicles while green
        if (ln.light === "GREEN" && queue > 0 && now - (this.lastSpawn[ln.id] || 0) > 900) {
          this.lastSpawn[ln.id] = now;
          this.moving.push({ x: a.x - a.dx * 20, y: a.y - a.dy * 20, dx: a.dx, dy: a.dy, rot: a.rot,
            kind: kinds[0] || "4W", seed: Math.floor(now) % 5 });
        }
        this.drawSignal(ln, c, r);
      });

      // moving vehicles
      this.moving.forEach((m) => { m.x += m.dx * 130 * dt; m.y += m.dy * 130 * dt; this.drawVehicle(m.x, m.y, m.rot, m.kind, m.seed); });
      this.moving = this.moving.filter((m) => m.x > -40 && m.x < S + 40 && m.y > -40 && m.y < S + 40);

      // centre readout
      const act = lanes.find((l) => l.id === this.state.active_lane);
      ctx.fillStyle = "rgba(10,14,18,0.78)";
      ctx.beginPath(); ctx.arc(c, c, 36, 0, Math.PI * 2); ctx.fill();
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      if (act && (this.state.phase === "GREEN" || this.state.phase === "YELLOW")) {
        ctx.fillStyle = LAMP[this.state.phase];
        ctx.font = "bold 22px Inter, Segoe UI, sans-serif";
        ctx.fillText(Math.ceil(this.state.remaining || 0) + "s", c, c - 6);
        ctx.font = "600 10px Inter, Segoe UI, sans-serif";
        ctx.fillStyle = "#cfd8e3";
        ctx.fillText((act.label || act.direction).toUpperCase(), c, c + 14);
      } else {
        ctx.fillStyle = "#ff6b6b"; ctx.font = "bold 12px Inter, Segoe UI, sans-serif";
        ctx.fillText("ALL RED", c, c);
      }
      // compass labels at the road ends
      ctx.font = "700 12px Inter, Segoe UI, sans-serif";
      [["N", c, 12], ["S", c, S - 12], ["W", 12, c], ["E", S - 12, c]].forEach(([t, x, y]) => {
        ctx.fillStyle = "rgba(10,14,18,0.8)"; ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#cfd8e3"; ctx.fillText(t, x, y + 1);
      });
    }

    drawVehicle(x, y, rot, kind, seed) {
      const ctx = this.ctx;
      const [w, l] = SIZE[kind] || SIZE["4W"];
      const pal = CAR_COLORS[kind] || CAR_COLORS["4W"];
      ctx.save(); ctx.translate(x, y); ctx.rotate(rot);
      ctx.fillStyle = "rgba(0,0,0,0.35)"; ctx.fillRect(-w / 2 + 2, -l / 2 + 2, w, l);
      ctx.fillStyle = pal[seed % pal.length];
      ctx.beginPath(); ctx.roundRect(-w / 2, -l / 2, w, l, 3); ctx.fill();
      if (kind !== "2W") {
        ctx.fillStyle = "rgba(20,30,45,0.85)";
        ctx.fillRect(-w / 2 + 2, -l / 2 + 4, w - 4, kind === "4W" ? 5 : 4);
        if (kind === "4W") ctx.fillRect(-w / 2 + 2, l / 2 - 7, w - 4, 4);
      }
      ctx.restore();
    }

    drawSignal(ln, c, r) {
      const ctx = this.ctx;
      // signal head sits on the corner to the driver's left of the stop line
      const pos = { North: [c + r + 14, c - r - 44, 0], South: [c - r - 22, c + r + 14, 0],
        East: [c + r + 14, c + r + 14, 1], West: [c - r - 52, c - r - 22, 1] }[ln.direction];
      const [x, y, horiz] = pos;
      const w = horiz ? 38 : 14, h = horiz ? 14 : 38;
      ctx.fillStyle = "#111"; ctx.beginPath(); ctx.roundRect(x, y, w, h, 3); ctx.fill();
      ["RED", "YELLOW", "GREEN"].forEach((col, i) => {
        const on = ln.light === col || (ln.light === "BLOCKED" && col === "RED");
        const cx = horiz ? x + 7 + i * 12 : x + 7, cy = horiz ? y + 7 : y + 7 + i * 12;
        ctx.fillStyle = on ? LAMP[col] : "#2a2a2a";
        if (on) { ctx.shadowColor = LAMP[col]; ctx.shadowBlur = 12; }
        ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      });
      if (ln.light === "BLOCKED") {
        const a = approach(ln.direction, c, r);
        const bx = a.x - a.dx * 26, by = a.y - a.dy * 26;
        ctx.fillStyle = "#ff3b3b";
        ctx.beginPath(); ctx.moveTo(bx, by - 13); ctx.lineTo(bx + 13, by + 10); ctx.lineTo(bx - 13, by + 10); ctx.closePath(); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.font = "bold 14px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText("!", bx, by + 2);
      }
    }
  }
  window.JunctionView = JunctionView;
})();
