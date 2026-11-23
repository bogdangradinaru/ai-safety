(function () {
  "use strict";

  const el = (id) => document.getElementById(id);
  const colours = { blue: "#1d4ed8", orange: "#c75b12", grey: "#8993a4", red: "#dc2626" };

  function plot(canvas, xRange, yRange, lines, points, xLabel, yLabel) {
    const ctx = canvas.getContext("2d");
    const { width: w, height: h } = canvas;
    const left = 54, right = 18, top = 18, bottom = 40;
    const px = (x) => left + (x - xRange[0]) / (xRange[1] - xRange[0]) * (w - left - right);
    const py = (y) => h - bottom - (y - yRange[0]) / (yRange[1] - yRange[0]) * (h - top - bottom);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = "#445";
    ctx.textAlign = "center";
    ctx.fillText(xLabel, (left + w - right) / 2, h - 5);
    ctx.textAlign = "left";
    ctx.fillText(yLabel, left, 13);
    ctx.strokeStyle = "#d9dde5";
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, w - left - right, h - top - bottom);
    ctx.fillStyle = "#5b6475";
    ctx.textAlign = "center";
    ctx.fillText(String(xRange[0]), left, h - bottom + 16);
    ctx.fillText(String(xRange[1]), w - right, h - bottom + 16);
    ctx.textAlign = "right";
    ctx.fillText(String(yRange[0]), left - 6, h - bottom);
    ctx.fillText(String(Number(yRange[1].toFixed(3))), left - 6, top + 5);
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top, w - left - right, h - top - bottom);
    ctx.clip();
    lines.forEach(({ data, colour, dashed }) => {
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.setLineDash(dashed ? [5, 4] : []);
      ctx.beginPath();
      data.forEach(([x, y], i) => {
        if (i === 0) ctx.moveTo(px(x), py(y)); else ctx.lineTo(px(x), py(y));
      });
      ctx.stroke();
    });
    ctx.setLineDash([]);
    points.forEach(({ x, y, colour, radius = 4 }) => {
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(px(x), py(y), radius, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.restore();
  }

  function batchWidget() {
    if (!el("batch-widget")) return;
    const targets = [0, 4, 1, 5, 2, 6, 3, 7];
    const order = [0, 5, 2, 7, 4, 1, 6, 3];
    const size = el("batch-size");
    let theta, index, epoch, updates, last;

    function draw() {
      const selected = last ? last.indices : [];
      plot(el("batch-canvas"), [1, 8], [-1, 8], [
        { data: [[1, theta], [8, theta]], colour: colours.red }
      ], targets.map((target, i) => ({
        x: i + 1, y: target, colour: selected.includes(i) ? colours.blue : colours.grey
      })), "training example", "target / prediction");
      const full = 2 * (theta - 3.5);
      el("batch-stats").textContent = `Prediction θ = ${theta.toFixed(3)} | Updates = ${updates} | Completed epochs = ${epoch} | Full gradient at current θ = ${full.toFixed(3)}. ` +
        (last ? `Last batch: examples ${last.indices.map((i) => i + 1).join(", ")}; batch gradient before update = ${last.gradient.toFixed(3)}; full gradient before update = ${last.full.toFixed(3)}.` : "Press Next update to select a batch.");
    }

    function reset() {
      theta = 0; index = 0; epoch = 0; updates = 0; last = null;
      draw();
    }
    el("batch-step").addEventListener("click", () => {
      const indices = order.slice(index, index + Number(size.value));
      const mean = indices.reduce((sum, i) => sum + targets[i], 0) / indices.length;
      const gradient = 2 * (theta - mean);
      last = { indices, gradient, full: 2 * (theta - 3.5) };
      theta -= 0.1 * gradient;
      updates += 1;
      index += indices.length;
      if (index === targets.length) { index = 0; epoch += 1; }
      draw();
    });
    el("batch-reset").addEventListener("click", reset);
    size.addEventListener("change", reset);
    reset();
  }

  function optimiserWidget() {
    if (!el("optim-widget")) return;
    const method = el("optim-method"), rate = el("optim-rate");
    let position, velocity, second, step, path, stopped;

    function draw() {
      el("optim-rate-value").textContent = Number(rate.value).toFixed(2);
      const loss = (position[0] ** 2 + 10 * position[1] ** 2) / 2;
      const limit = Math.max(5, ...path.flat().map(Math.abs)) * 1.05;
      plot(el("optim-canvas"), [-limit, limit], [-limit, limit], [
        { data: path, colour: colours.blue }
      ], [
        { x: 0, y: 0, colour: colours.grey, radius: 5 },
        { x: position[0], y: position[1], colour: colours.red, radius: 5 }
      ], "x", "y");
      el("optim-stats").textContent = `Step ${step} | (x, y) = (${position[0].toFixed(3)}, ${position[1].toFixed(3)}) | Loss = ${loss.toFixed(3)} | Gradient = (${position[0].toFixed(3)}, ${(10 * position[1]).toFixed(3)}).` +
        (stopped ? " Path stopped because the parameters grew beyond 1,000; reset or lower the learning rate." : "");
      el("optim-step").disabled = stopped;
    }

    function reset() {
      position = [4, 4]; velocity = [0, 0]; second = [0, 0];
      step = 0; path = [[...position]]; stopped = false;
      draw();
    }
    el("optim-step").addEventListener("click", () => {
      const eta = Number(rate.value), gradient = [position[0], 10 * position[1]];
      step += 1;
      position = position.map((value, i) => {
        const g = gradient[i];
        if (method.value === "sgd") return value - eta * g;
        if (method.value === "momentum") {
          velocity[i] = 0.9 * velocity[i] + g;
          return value - eta * velocity[i];
        }
        if (method.value === "rmsprop") {
          second[i] = 0.9 * second[i] + 0.1 * g * g;
          return value - eta * g / (Math.sqrt(second[i]) + 1e-8);
        }
        velocity[i] = 0.9 * velocity[i] + 0.1 * g;
        second[i] = 0.999 * second[i] + 0.001 * g * g;
        const mHat = velocity[i] / (1 - 0.9 ** step);
        const vHat = second[i] / (1 - 0.999 ** step);
        return value - eta * mHat / (Math.sqrt(vHat) + 1e-8);
      });
      path.push([...position]);
      if (position.some((value) => Math.abs(value) > 1000)) stopped = true;
      draw();
    });
    el("optim-reset").addEventListener("click", reset);
    method.addEventListener("change", reset);
    rate.addEventListener("input", reset);
    reset();
  }

  function scheduleWidget() {
    if (!el("schedule-widget")) return;
    const kind = el("schedule-kind"), iteration = el("schedule-iteration");
    function value(t) {
      if (kind.value === "step") return 0.1 * 0.5 ** Math.floor(t / 10);
      if (kind.value === "cosine") return 0.05 * (1 + Math.cos(Math.PI * t / 40));
      if (kind.value === "warmup") return 0.1 * Math.min(1, t / 8);
      return t <= 8 ? 0.1 * t / 8 : 0.05 * (1 + Math.cos(Math.PI * (t - 8) / 32));
    }
    function draw() {
      const t = Number(iteration.value);
      el("schedule-iteration-value").textContent = String(t);
      const data = [];
      for (let i = 0; i <= 40; i++) {
        if (kind.value === "step" && i > 0) data.push([i, value(i - 1)]);
        data.push([i, value(i)]);
      }
      plot(el("schedule-canvas"), [0, 40], [0, 0.11], [
        { data, colour: colours.blue }
      ], [{ x: t, y: value(t), colour: colours.red, radius: 5 }], "iteration", "learning rate η");
      el("schedule-stats").textContent = `Iteration ${t} of 40 | Learning rate η = ${value(t).toFixed(5)}.` +
        (kind.value === "step" ? " The rate halves every 10 iterations." : "");
    }
    kind.addEventListener("change", draw);
    iteration.addEventListener("input", draw);
    draw();
  }

  const trainX = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1];
  const trainY = [1.05, 0.43, 0.35, -0.02, 0.16, 0.18, 0.68, 0.94];
  const validX = [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875];
  const validY = validX.map((x) => x * x);
  const mse = (xs, ys, predict) => xs.reduce((sum, x, i) => sum + (predict(x) - ys[i]) ** 2, 0) / xs.length;

  function polynomialFit(degree) {
    const n = degree + 1;
    const matrix = Array.from({ length: n }, (_, row) => [
      ...Array.from({ length: n }, (_, col) => trainX.reduce((sum, x) => sum + x ** (row + col), 0)),
      trainX.reduce((sum, x, i) => sum + trainY[i] * x ** row, 0)
    ]);
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(matrix[row][col]) > Math.abs(matrix[pivot][col])) pivot = row;
      }
      [matrix[col], matrix[pivot]] = [matrix[pivot], matrix[col]];
      const divisor = matrix[col][col];
      for (let j = col; j <= n; j++) matrix[col][j] /= divisor;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = matrix[row][col];
        for (let j = col; j <= n; j++) matrix[row][j] -= factor * matrix[col][j];
      }
    }
    const coefficients = matrix.map((row) => row[n]);
    return (x) => coefficients.reduce((sum, coefficient, i) => sum + coefficient * x ** i, 0);
  }

  function fitWidget() {
    if (!el("fit-widget")) return;
    const select = el("fit-model");
    const models = { linear: polynomialFit(1), quadratic: polynomialFit(2), interpolating: polynomialFit(7) };
    function draw() {
      const predict = models[select.value];
      const data = Array.from({ length: 201 }, (_, i) => {
        const x = -1 + i / 100;
        return [x, predict(x)];
      });
      plot(el("fit-canvas"), [-1, 1], [-0.5, 1.5], [
        { data, colour: colours.blue }
      ], [
        ...trainX.map((x, i) => ({ x, y: trainY[i], colour: colours.blue })),
        ...validX.map((x, i) => ({ x, y: validY[i], colour: colours.orange }))
      ], "input x", "target / prediction");
      el("fit-stats").textContent = `Training MSE = ${mse(trainX, trainY, predict).toFixed(4)} | Validation MSE = ${mse(validX, validY, predict).toFixed(4)}. Values outside the plot are clipped.`;
    }
    select.addEventListener("change", draw);
    draw();
  }

  function curvesWidget() {
    if (!el("curves-widget")) return;
    const scenario = el("curves-scenario"), epoch = el("curves-epoch");
    function losses(t) {
      if (scenario.value === "under") return [0.85 + 0.3 * Math.exp(-t / 7), 0.92 + 0.3 * Math.exp(-t / 7)];
      if (scenario.value === "good") return [0.12 + Math.exp(-t / 8), 0.18 + Math.exp(-t / 8)];
      return [0.08 + Math.exp(-t / 8), 0.18 + Math.exp(-t / 8) + 0.0015 * Math.max(0, t - 12) ** 2];
    }
    function draw() {
      const t = Number(epoch.value);
      const training = [], validation = [];
      for (let i = 1; i <= 40; i++) {
        const [tr, val] = losses(i);
        training.push([i, tr]); validation.push([i, val]);
      }
      const best = validation.reduce((a, b) => b[1] < a[1] ? b : a);
      const [tr, val] = losses(t);
      el("curves-epoch-value").textContent = String(t);
      plot(el("curves-canvas"), [1, 40], [0, 1.6], [
        { data: training, colour: colours.blue }, { data: validation, colour: colours.orange }
      ], [
        { x: t, y: tr, colour: colours.blue, radius: 5 },
        { x: t, y: val, colour: colours.orange, radius: 5 }
      ], "epoch", "loss");
      el("curves-stats").textContent = `Epoch ${t} | Training loss = ${tr.toFixed(3)} | Validation loss = ${val.toFixed(3)} | Lowest displayed validation loss = ${best[1].toFixed(3)} at epoch ${best[0]}.`;
    }
    scenario.addEventListener("change", draw);
    epoch.addEventListener("input", draw);
    draw();
  }

  function hyperWidget() {
    if (!el("hyper-widget")) return;
    const rate = el("hyper-rate"), epochs = el("hyper-epochs"), lambda = el("hyper-lambda");
    const batch = el("hyper-batch"), depth = el("hyper-depth"), width = el("hyper-width");
    const optimiser = el("hyper-optimiser");

    function randomGenerator(seed) {
      let state = seed;
      return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
      };
    }

    function draw() {
      const eta = Number(rate.value), count = Number(epochs.value), regularisation = Number(lambda.value);
      const batchSize = Number(batch.value), hiddenLayers = Number(depth.value), neurons = Number(width.value);
      const initialise = randomGenerator(42), shuffle = randomGenerator(123);
      const sizes = [1, ...Array(hiddenLayers).fill(neurons), 1];
      const parameter = (value) => ({ value, gradient: 0, first: 0, second: 0 });
      const layers = sizes.slice(1).map((outputSize, layerIndex) => {
        const inputSize = sizes[layerIndex];
        const limit = Math.sqrt(6 / (inputSize + outputSize));
        return {
          weights: Array.from({ length: outputSize }, () =>
            Array.from({ length: inputSize }, () => parameter((2 * initialise() - 1) * limit))),
          biases: Array.from({ length: outputSize }, () => parameter(0))
        };
      });
      const parameters = layers.flatMap((layer) => [
        ...layer.weights.flat().map((p) => ({ p, isWeight: true })),
        ...layer.biases.map((p) => ({ p, isWeight: false }))
      ]);

      function forward(x) {
        const activations = [[x]];
        layers.forEach((layer, index) => {
          const previous = activations[index];
          const output = layer.weights.map((row, i) => {
            const sum = row.reduce((total, p, j) => total + p.value * previous[j], layer.biases[i].value);
            return index === layers.length - 1 ? sum : Math.tanh(sum);
          });
          activations.push(output);
        });
        return activations;
      }

      let updates = 0, diverged = false;
      const training = [], validation = [];
      for (let epoch = 1; epoch <= count; epoch++) {
        const order = trainX.map((_, i) => i);
        for (let i = order.length - 1; i > 0; i--) {
          const j = Math.floor(shuffle() * (i + 1));
          [order[i], order[j]] = [order[j], order[i]];
        }
        for (let start = 0; start < order.length; start += batchSize) {
          const indices = order.slice(start, start + batchSize);
          parameters.forEach(({ p }) => { p.gradient = 0; });
          indices.forEach((example) => {
            const activations = forward(trainX[example]);
            let delta = [2 * (activations[layers.length][0] - trainY[example]) / indices.length];
            for (let index = layers.length - 1; index >= 0; index--) {
              const layer = layers[index], previous = activations[index];
              const nextDelta = previous.map(() => 0);
              layer.weights.forEach((row, i) => {
                layer.biases[i].gradient += delta[i];
                row.forEach((p, j) => {
                  p.gradient += delta[i] * previous[j];
                  nextDelta[j] += p.value * delta[i];
                });
              });
              delta = nextDelta.map((value, j) => value * (1 - previous[j] ** 2));
            }
          });
          updates++;
          parameters.forEach(({ p, isWeight }) => {
            const gradient = p.gradient + (isWeight ? 2 * regularisation * p.value : 0);
            if (optimiser.value === "momentum") {
              p.first = 0.9 * p.first + gradient;
              p.value -= eta * p.first;
            } else if (optimiser.value === "adam") {
              p.first = 0.9 * p.first + 0.1 * gradient;
              p.second = 0.999 * p.second + 0.001 * gradient ** 2;
              const mean = p.first / (1 - 0.9 ** updates);
              const variance = p.second / (1 - 0.999 ** updates);
              p.value -= eta * mean / (Math.sqrt(variance) + 1e-8);
            } else {
              p.value -= eta * gradient;
            }
          });
        }
        const predict = (x) => forward(x)[layers.length][0];
        const tr = mse(trainX, trainY, predict), val = mse(validX, validY, predict);
        if (!Number.isFinite(tr) || !Number.isFinite(val) || tr > 1e6 || val > 1e6 ||
            parameters.some(({ p }) => !Number.isFinite(p.value))) {
          diverged = true;
          break;
        }
        training.push([epoch, tr]); validation.push([epoch, val]);
      }
      el("hyper-epochs-value").textContent = String(count);
      const last = training.length - 1;
      plot(el("hyper-canvas"), [0, count], [0, 1], [
        { data: training, colour: colours.blue }, { data: validation, colour: colours.orange }
      ], training.length === 1 ? [
        { x: training[0][0], y: training[0][1], colour: colours.blue },
        { x: validation[0][0], y: validation[0][1], colour: colours.orange }
      ] : [], "epoch", "MSE (clipped at 1)");
      el("hyper-stats").textContent = last >= 0
        ? `Epoch ${training[last][0]} | Training MSE = ${training[last][1].toFixed(4)} | Validation MSE = ${validation[last][1].toFixed(4)} | Parameters = ${parameters.length} | ${diverged ? "Attempted updates" : "Updates"} = ${updates}.` + (diverged ? " Training diverged; later epochs are not shown." : "")
        : `Training diverged before the first epoch completed. Parameters = ${parameters.length} | Attempted updates = ${updates}.`;
    }
    rate.addEventListener("change", draw);
    epochs.addEventListener("input", draw);
    lambda.addEventListener("change", draw);
    batch.addEventListener("change", draw);
    depth.addEventListener("change", draw);
    width.addEventListener("change", draw);
    optimiser.addEventListener("change", draw);
    draw();
  }

  batchWidget();
  optimiserWidget();
  scheduleWidget();
  fitWidget();
  curvesWidget();
  hyperWidget();
})();
