(function () {
  "use strict";

  const el = (id) => document.getElementById(id);
  const colours = { blue: "#1d4ed8", orange: "#c75b12", green: "#16805d", red: "#dc2626", grey: "#8993a4", dark: "#1f2430" };
  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));

  function drawAxes(ctx, width, height, padding, xLabel, yLabel) {
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#d9dde5";
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, padding, width - 2 * padding, height - 2 * padding);
    ctx.fillStyle = "#5b6475";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(xLabel, width / 2, height - 5);
    ctx.save();
    ctx.translate(12, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }

  function supervisedWidget() {
    const root = el("supervised-widget");
    if (!root) return;
    const points = [
      [0.10, 0.20, 0], [0.20, 0.45, 0], [0.28, 0.18, 0], [0.36, 0.62, 0],
      [0.45, 0.35, 0], [0.52, 0.72, 1], [0.62, 0.42, 1], [0.70, 0.76, 1],
      [0.82, 0.30, 1], [0.90, 0.65, 1]
    ];
    const threshold = el("supervised-threshold");
    const labels = el("supervised-labels");

    function observedLabel(point) {
      if (labels.value === "one-flipped" && point === points[5]) return 0;
      if (labels.value === "three-flipped" && (point === points[5] || point === points[7] || point === points[8])) return 0;
      return point[2];
    }

    function draw() {
      const canvas = el("supervised-canvas");
      const ctx = canvas.getContext("2d");
      const padding = 42;
      const px = (x) => padding + x * (canvas.width - 2 * padding);
      const py = (y) => canvas.height - padding - y * (canvas.height - 2 * padding);
      drawAxes(ctx, canvas.width, canvas.height, padding, "object width", "object height");
      const cut = Number(threshold.value);
      ctx.strokeStyle = colours.dark;
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(px(cut), padding); ctx.lineTo(px(cut), canvas.height - padding); ctx.stroke();
      ctx.setLineDash([]);
      let correct = 0, noisy = 0;
      points.forEach((point) => {
        const predicted = point[0] >= cut ? 1 : 0;
        const target = point[2];
        const observed = observedLabel(point);
        if (predicted === target) correct++;
        if (observed !== target) noisy++;
        ctx.fillStyle = observed ? colours.orange : colours.blue;
        ctx.strokeStyle = observed !== target ? colours.red : "#fff";
        ctx.lineWidth = observed !== target ? 3 : 1;
        ctx.beginPath(); ctx.arc(px(point[0]), py(point[1]), 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      });
      el("supervised-threshold-value").textContent = cut.toFixed(2);
      el("supervised-stats").textContent = `Threshold = ${cut.toFixed(2)} | Accuracy against true labels = ${(correct / points.length * 100).toFixed(0)}% | Incorrect training labels = ${noisy}/${points.length}.`;
    }

    threshold.addEventListener("input", draw);
    labels.addEventListener("change", draw);
    el("supervised-reset").addEventListener("click", () => { threshold.value = "0.5"; labels.value = "clean"; draw(); });
    draw();
  }

  function selfSupervisedWidget() {
    const root = el("self-supervised-widget");
    if (!root) return;
    const task = el("self-task");
    const strength = el("self-strength");
    let round = 0;

    function draw() {
      const canvas = el("self-supervised-canvas");
      const ctx = canvas.getContext("2d");
      const padding = 24;
      const amount = Number(strength.value);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const panels = [
        { x: padding, title: "Input A", mask: task.value === "mask" ? Math.floor(amount * 5) : 0 },
        { x: canvas.width / 2 + 8, title: task.value === "mask" ? "Recovery target" : "Augmented view B", mask: 0 }
      ];
      panels.forEach((panel, panelIndex) => {
        ctx.fillStyle = "#f8fafc";
        ctx.fillRect(panel.x, padding, canvas.width / 2 - padding - 8, canvas.height - 2 * padding);
        ctx.strokeStyle = "#d9dde5";
        ctx.strokeRect(panel.x, padding, canvas.width / 2 - padding - 8, canvas.height - 2 * padding);
        ctx.fillStyle = "#34475a";
        ctx.font = "bold 13px system-ui, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(panel.title, panel.x + 12, padding + 20);
        const cell = 30;
        for (let i = 0; i < 6; i++) {
          for (let j = 0; j < 3; j++) {
            const bx = panel.x + 18 + i * (cell + 2);
            const by = padding + 40 + j * (cell + 2);
            const road = j === 1 && i >= 1 && i <= 4;
            const object = (i === 1 && j === 1) || (i === 4 && j === 1);
            let fill = road ? "#5b6475" : object ? colours.orange : "#dbe7d3";
            if (panelIndex === 1 && task.value === "contrast" && ((i + round) % 3 === 0)) fill = "#b8d4ef";
            if (panel.mask > 0 && i < panel.mask) fill = "#20242c";
            ctx.fillStyle = fill;
            ctx.fillRect(bx, by, cell, cell);
          }
        }
      });
      el("self-strength-value").textContent = `${Math.round(amount * 100)}%`;
      const sameSceneSimilarity = 1 - 0.45 * amount + 0.015 * round;
      if (task.value === "mask") {
        const masked = Math.max(1, Math.round(amount * 5));
        el("self-task-output").textContent = `Recover ${masked} hidden image column${masked === 1 ? "" : "s"}`;
        el("self-stats").textContent = `Round ${round}: the target is generated from the same unlabelled image. Masked input coverage = ${Math.round(amount * 100)}%.`;
      } else {
        el("self-task-output").textContent = `Pull two views together; keep unrelated scenes apart`;
        el("self-stats").textContent = `Round ${round}: positive-pair similarity = ${sameSceneSimilarity.toFixed(2)}; negative-pair similarity = ${(0.28 - 0.01 * round).toFixed(2)}. Lower augmentation strength makes matching easier.`;
      }
    }

    el("self-train").addEventListener("click", () => { round++; draw(); });
    el("self-reset").addEventListener("click", () => { round = 0; draw(); });
    task.addEventListener("change", () => { round = 0; draw(); });
    strength.addEventListener("input", draw);
    draw();
  }

  function unsupervisedWidget() {
    const root = el("unsupervised-widget");
    if (!root) return;
    const points = [
      [0.12, 0.18], [0.18, 0.30], [0.25, 0.20], [0.30, 0.36], [0.20, 0.48],
      [0.50, 0.66], [0.58, 0.78], [0.66, 0.62], [0.72, 0.82], [0.58, 0.52],
      [0.80, 0.20], [0.88, 0.34], [0.72, 0.28], [0.90, 0.46], [0.76, 0.44]
    ];
    const count = el("cluster-count");
    let centres = [], iteration = 0, assignments = [];

    function reset() {
      const k = Number(count.value);
      centres = Array.from({ length: k }, (_, i) => [0.12 + i * 0.76 / Math.max(1, k - 1), 0.2 + (i % 2) * 0.55]);
      iteration = 0; assignments = [];
      draw();
    }

    function assign() {
      assignments = points.map(([x, y]) => centres.reduce((best, centre, i) => {
        const distance = (x - centre[0]) ** 2 + (y - centre[1]) ** 2;
        return distance < best.distance ? { index: i, distance } : best;
      }, { index: 0, distance: Infinity }).index);
    }

    function advance() {
      assign();
      centres = centres.map((centre, index) => {
        const members = points.filter((_, i) => assignments[i] === index);
        if (!members.length) return centre;
        return [
          members.reduce((sum, point) => sum + point[0], 0) / members.length,
          members.reduce((sum, point) => sum + point[1], 0) / members.length
        ];
      });
      iteration++;
      assign();
      draw();
    }

    function draw() {
      const canvas = el("unsupervised-canvas");
      const ctx = canvas.getContext("2d");
      const padding = 34;
      const px = (x) => padding + x * (canvas.width - 2 * padding);
      const py = (y) => canvas.height - padding - y * (canvas.height - 2 * padding);
      drawAxes(ctx, canvas.width, canvas.height, padding, "scenario feature 1", "scenario feature 2");
      const palette = [colours.blue, colours.orange, colours.green, "#7c4d9e"];
      points.forEach((point, i) => {
        ctx.fillStyle = assignments.length ? palette[assignments[i] % palette.length] : colours.grey;
        ctx.beginPath(); ctx.arc(px(point[0]), py(point[1]), 5, 0, 2 * Math.PI); ctx.fill();
      });
      centres.forEach((centre, i) => {
        ctx.strokeStyle = palette[i % palette.length];
        ctx.lineWidth = 3;
        const x = px(centre[0]), y = py(centre[1]);
        ctx.beginPath(); ctx.moveTo(x - 7, y - 7); ctx.lineTo(x + 7, y + 7); ctx.moveTo(x + 7, y - 7); ctx.lineTo(x - 7, y + 7); ctx.stroke();
      });
      const inertia = assignments.length ? points.reduce((sum, point, i) => sum + (point[0] - centres[assignments[i]][0]) ** 2 + (point[1] - centres[assignments[i]][1]) ** 2, 0) : NaN;
      el("cluster-count-value").textContent = count.value;
      el("cluster-stats").textContent = `k = ${count.value} | Iteration = ${iteration} | ${Number.isNaN(inertia) ? "Run an iteration to calculate clusters." : `Within-cluster squared distance = ${inertia.toFixed(3)}.`}`;
    }

    el("cluster-step").addEventListener("click", advance);
    el("cluster-reset").addEventListener("click", reset);
    count.addEventListener("input", reset);
    reset();
  }

  function semiSupervisedWidget() {
    const root = el("semi-supervised-widget");
    if (!root) return;
    const threshold = el("pseudo-threshold");
    const seeds = el("pseudo-seeds");
    const labelled = [
      { x: 0.08, truth: 0 }, { x: 0.18, truth: 0 }, { x: 0.30, truth: 0 },
      { x: 0.70, truth: 1 }, { x: 0.84, truth: 1 }, { x: 0.93, truth: 1 }
    ];
    const unlabelled = [0.38, 0.44, 0.49, 0.55, 0.60, 0.66];
    let round = 0;

    function data() {
      const training = labelled.map((item, index) => ({ ...item, label: item.truth }));
      if (seeds.value === "noisy") training[2].label = 1;
      return training;
    }

    function result() {
      const training = data();
      const selected = [];
      let accepted = 0;
      let correct = 0;
      for (let r = 0; r < round; r++) {
        const left = training.filter((item) => item.label === 0).reduce((sum, item) => sum + item.x, 0) / training.filter((item) => item.label === 0).length;
        const right = training.filter((item) => item.label === 1).reduce((sum, item) => sum + item.x, 0) / training.filter((item) => item.label === 1).length;
        const candidates = unlabelled.filter((x) => !selected.some((item) => item.x === x));
        let candidate = null;
        candidates.forEach((x) => {
          const d0 = Math.abs(x - left), d1 = Math.abs(x - right);
          const label = d1 < d0 ? 1 : 0;
          const confidence = 0.5 + 0.5 * Math.abs(d1 - d0) / Math.max(d0 + d1, 1e-9);
          if (confidence >= Number(threshold.value) && (!candidate || confidence > candidate.confidence)) candidate = { x, label, confidence, truth: x >= 0.5 ? 1 : 0 };
        });
        if (!candidate) break;
        selected.push(candidate);
        training.push(candidate);
        accepted++;
        if (candidate.label === candidate.truth) correct++;
      }
      return { selected, accepted, correct };
    }

    function draw() {
      const canvas = el("semi-supervised-canvas");
      const ctx = canvas.getContext("2d");
      const padding = 34;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const px = (x) => padding + x * (canvas.width - 2 * padding);
      const y = canvas.height / 2;
      ctx.strokeStyle = "#d9dde5"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(padding, y); ctx.lineTo(canvas.width - padding, y); ctx.stroke();
      ctx.strokeStyle = colours.dark; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(px(0.5), 30); ctx.lineTo(px(0.5), canvas.height - 30); ctx.stroke(); ctx.setLineDash([]);
      const { selected, accepted, correct } = result();
      data().forEach((item) => {
        ctx.fillStyle = item.label ? colours.orange : colours.blue;
        ctx.strokeStyle = item.label !== item.truth ? colours.red : "#fff";
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(px(item.x), y, 7, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      });
      unlabelled.forEach((x) => {
        const pseudo = selected.find((item) => item.x === x);
        ctx.fillStyle = pseudo ? (pseudo.label ? colours.orange : colours.blue) : "#fff";
        ctx.strokeStyle = pseudo && pseudo.label !== pseudo.truth ? colours.red : colours.grey;
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(px(x), y, 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      });
      el("pseudo-threshold-value").textContent = Number(threshold.value).toFixed(2);
      el("pseudo-stats").textContent = `Round ${round} | Pseudo-labels accepted = ${accepted} | Correct = ${correct} | Seed-label error = ${seeds.value === "noisy" ? "1 of 6" : "0 of 6"}. The threshold trades coverage for label confidence.`;
    }

    function reset() { round = 0; draw(); }
    el("pseudo-round").addEventListener("click", () => { round++; draw(); });
    el("pseudo-reset").addEventListener("click", reset);
    threshold.addEventListener("input", reset);
    seeds.addEventListener("change", reset);
    reset();
  }

  function reinforcementWidget() {
    const root = el("reinforcement-widget");
    if (!root) return;
    const penalty = el("rl-penalty");
    const goals = [[0, 4]];
    const hazards = [[2, 2], [3, 2]];
    const actions = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    const q = new Map();
    let episodes = 0, route = [], lastReturn = 0;
    let seed = 17;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    const key = (r, c, action) => `${r},${c},${action}`;
    const value = (r, c, action) => q.get(key(r, c, action)) || 0;

    function validPosition(r, c) { return r >= 0 && r < 5 && c >= 0 && c < 5; }
    function isAt(position, locations) { return locations.some(([r, c]) => r === position[0] && c === position[1]); }
    function choose(r, c, explore) {
      if (explore && random() < 0.25) return Math.floor(random() * 4);
      const values = actions.map((_, a) => value(r, c, a));
      const max = Math.max(...values);
      const best = values.map((entry, index) => entry === max ? index : -1).filter((index) => index >= 0);
      return best[Math.floor(random() * best.length)];
    }

    function runEpisode(count = 1) {
      for (let episode = 0; episode < count; episode++) {
        let state = [4, 0], total = 0, path = [state], steps = 0;
        while (steps < 30 && !isAt(state, goals)) {
          const action = choose(state[0], state[1], true);
          const [dr, dc] = actions[action];
          const next = [state[0] + dr, state[1] + dc];
          const destination = validPosition(next[0], next[1]) ? next : state;
          const reward = isAt(destination, goals) ? 10 : isAt(destination, hazards) ? -Number(penalty.value) : -0.1;
          const nextBest = Math.max(...actions.map((_, a) => value(destination[0], destination[1], a)));
          const old = value(state[0], state[1], action);
          q.set(key(state[0], state[1], action), old + 0.2 * (reward + 0.9 * nextBest - old));
          total += reward;
          state = destination;
          path.push(state);
          steps++;
          if (isAt(state, hazards)) break;
        }
        route = path;
        lastReturn = total;
        episodes++;
      }
      draw();
    }

    function draw() {
      const canvas = el("rl-canvas");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const margin = 20, size = (Math.min(canvas.width, canvas.height) - 2 * margin) / 5;
      const ox = (canvas.width - size * 5) / 2, oy = (canvas.height - size * 5) / 2;
      for (let r = 0; r < 5; r++) {
        for (let c = 0; c < 5; c++) {
          ctx.fillStyle = isAt([r, c], goals) ? "#dff1e7" : isAt([r, c], hazards) ? "#fde2e2" : "#f8fafc";
          ctx.fillRect(ox + c * size, oy + r * size, size - 2, size - 2);
          ctx.strokeStyle = "#d9dde5"; ctx.strokeRect(ox + c * size, oy + r * size, size - 2, size - 2);
        }
      }
      route.forEach(([r, c], i) => {
        ctx.fillStyle = i === route.length - 1 ? colours.blue : "rgba(29,78,216,0.45)";
        ctx.beginPath(); ctx.arc(ox + (c + 0.5) * size, oy + (r + 0.5) * size, i === route.length - 1 ? 8 : 4, 0, 2 * Math.PI); ctx.fill();
      });
      const start = [4, 0];
      if (!route.length) {
        ctx.fillStyle = colours.blue; ctx.beginPath(); ctx.arc(ox + (start[1] + 0.5) * size, oy + (start[0] + 0.5) * size, 8, 0, 2 * Math.PI); ctx.fill();
      }
      el("rl-stats").textContent = `Episodes trained = ${episodes} | Last return = ${lastReturn.toFixed(1)} | Hazard penalty = ${Number(penalty.value).toFixed(1)}. The learned Q-values favour actions with higher estimated cumulative reward.`;
    }

    function reset() { q.clear(); episodes = 0; route = []; lastReturn = 0; seed = 17; draw(); }
    el("rl-step").addEventListener("click", () => runEpisode(1));
    el("rl-train").addEventListener("click", () => runEpisode(50));
    el("rl-reset").addEventListener("click", reset);
    penalty.addEventListener("change", reset);
    reset();
  }

  function imitationWidget() {
    const root = el("imitation-widget");
    if (!root) return;
    const method = el("imitation-method");
    const error = el("imitation-error");
    let step = 0;

    function rollout() {
      const correction = method.value === "dagger" ? 0.72 : 0.46;
      const states = [0];
      const actions = [];
      for (let i = 0; i < 16; i++) {
        const state = states[states.length - 1];
        const action = -correction * state;
        actions.push(action);
        const next = state + action + Number(error.value) * (1 - correction);
        states.push(clamp(next, -5, 5));
      }
      return { states, actions };
    }

    function draw() {
      const { states } = rollout();
      const shown = states.slice(0, step + 1);
      const canvas = el("imitation-canvas");
      const ctx = canvas.getContext("2d");
      const padding = 42;
      drawAxes(ctx, canvas.width, canvas.height, padding, "time step", "lateral offset");
      const px = (index) => padding + index / 16 * (canvas.width - 2 * padding);
      const py = (offset) => canvas.height / 2 - offset / 5 * (canvas.height / 2 - padding);
      ctx.strokeStyle = "#d9dde5";
      ctx.beginPath(); ctx.moveTo(padding, py(0)); ctx.lineTo(canvas.width - padding, py(0)); ctx.stroke();
      ctx.strokeStyle = method.value === "dagger" ? colours.green : colours.red;
      ctx.lineWidth = 2.5;
      ctx.beginPath(); shown.forEach((state, index) => { if (index === 0) ctx.moveTo(px(index), py(state)); else ctx.lineTo(px(index), py(state)); }); ctx.stroke();
      shown.forEach((state, index) => { ctx.fillStyle = method.value === "dagger" ? colours.green : colours.red; ctx.beginPath(); ctx.arc(px(index), py(state), 4, 0, 2 * Math.PI); ctx.fill(); });
      const maxOffset = Math.max(...states.map(Math.abs));
      const finalOffset = states[states.length - 1];
      el("imitation-error-value").textContent = Number(error.value).toFixed(2);
      el("imitation-stats").textContent = `Step ${step}/16 | Initial position error = ${states[0].toFixed(2)} | Final offset = ${finalOffset.toFixed(2)} | Maximum offset = ${maxOffset.toFixed(2)}. ${method.value === "dagger" ? "DAgger-style corrections teach the policy on states its own rollout visits." : "Behaviour cloning compounds small policy errors when the rollout leaves the expert's state distribution."}`;
      el("imitation-step").disabled = step === 16;
    }

    function reset() { step = 0; draw(); }
    el("imitation-step").addEventListener("click", () => { step = Math.min(16, step + 1); draw(); });
    el("imitation-reset").addEventListener("click", reset);
    method.addEventListener("change", reset);
    error.addEventListener("input", reset);
    reset();
  }

  supervisedWidget();
  selfSupervisedWidget();
  unsupervisedWidget();
  semiSupervisedWidget();
  reinforcementWidget();
  imitationWidget();
})();
