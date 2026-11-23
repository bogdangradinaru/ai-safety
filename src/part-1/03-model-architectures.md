
# 3. AI Model Architectures and Technologies

## 3.1 Introduction

In Chapter 2 we saw *how* a model can be taught: with labels, without labels, through rewards, or by imitating an expert. In this chapter we look at *what* is being taught — the model itself.

A [**neural network**](#glossary-neural-network) is a computer program that learns a task from examples instead of following hand-written rules. Its [**architecture**](#glossary-model-architecture) is its shape and wiring: how many parts it has, what each part does, and how the parts are connected. Different tasks need different architectures. A network that reads camera images is wired differently from one that reads [LiDAR](#glossary-point-cloud) points or predicts where a pedestrian will walk next.

Key technical terms are linked to the **Glossary** at the end of this chapter. The separate **Mathematical notation** table summarises the main symbols used in equations and examples.

Recall from Section 2.2 that training means finding the [parameters](#glossary-model-parameters) [$\theta$](#notation-theta) that make a [loss function](#glossary-loss-function) small:
In the objective below, [$f_\theta$](#notation-f-theta) is the model function, [$x_i$](#notation-data) and [$y_i$](#notation-data) are the input and target for example $i$, [$N$](#notation-dataset-size) is the number of examples, and [$\mathcal{L}$](#notation-loss) is the loss function.

$$
\theta^{*} = \arg\min_{\theta} \; \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}\big(f_\theta(x_i),\, y_i\big)
$$

#### Try it: the simplest possible $f_\theta$ — [linear regression](#glossary-linear-regression)

Here, [$\hat y$](#notation-data) denotes the predicted value and $y$ the target.

Before opening up a full neural network, it helps to see this formula at work in its simplest case: $f_\theta(x) = wx + b$ (a straight line) with $\mathcal{L} = (\hat y - y)^2$. This is ordinary **linear regression** — one [weight](#glossary-model-parameters) [$w$](#notation-weight-bias), one [bias](#glossary-model-parameters) [$b$](#notation-weight-bias), no [activation function](#glossary-activation-function).

Edit the points below and watch $w$, $b$ and the [mean squared error](#glossary-loss-function) update live.

<div id="lr-widget" class="aisafety-widget">
<div class="aisw-controls">
<button type="button" data-lr-preset="clean">Clean linear trend</button>
<button type="button" data-lr-preset="outlier">With an outlier</button>
<button type="button" data-lr-preset="curved">Curved data</button>
<button type="button" data-lr-preset="clear">Clear points</button>
<label><input type="checkbox" id="lr-residuals"> show residuals</label>
</div>
<canvas id="lr-canvas" width="680" height="380"></canvas>
<p class="aisw-hint">Click empty space to add a point, drag a point to move it, double-click a point to delete it.</p>
<div id="lr-stats" class="aisw-stats"></div>
</div>

The widget also reports [$R^2$](#notation-regression-metrics), the coefficient of determination, as a measure of how well the fitted line explains the variation in the points.

<script>
(function () {
  const canvas = document.getElementById("lr-canvas");
  const ctx = canvas.getContext("2d");
  const statsEl = document.getElementById("lr-stats");
  const residualsBox = document.getElementById("lr-residuals");

  const xMin = 0, xMax = 10, yMin = 0, yMax = 10;
  const pad = 36;

  const presets = {
    clean:   [[1,1.8],[2,2.6],[3,4.1],[4,4.7],[5,5.9],[6,6.6],[7,7.4],[8,8.3]],
    outlier: [[1,1.8],[2,2.6],[3,4.1],[4,4.7],[5,5.9],[6,6.6],[7,7.4],[8,0.5]],
    curved:  [[1,1.0],[2,1.3],[3,2.0],[4,3.3],[5,5.2],[6,7.6],[7,9.3],[8,9.8]],
  };

  let points = presets.clean.map(([x, y]) => ({ x, y }));
  let dragIndex = -1;

  function toPx(x, y) {
    const w = canvas.width, h = canvas.height;
    return [
      pad + (x - xMin) / (xMax - xMin) * (w - 2 * pad),
      h - pad - (y - yMin) / (yMax - yMin) * (h - 2 * pad),
    ];
  }
  function toData(px, py) {
    const w = canvas.width, h = canvas.height;
    return [
      xMin + (px - pad) / (w - 2 * pad) * (xMax - xMin),
      yMin + (h - pad - py) / (h - 2 * pad) * (yMax - yMin),
    ];
  }

  function fitLine() {
    const n = points.length;
    if (n < 2) return { w: 0, b: n ? points[0].y : 0, mse: NaN, r2: NaN };
    const xbar = points.reduce((s, p) => s + p.x, 0) / n;
    const ybar = points.reduce((s, p) => s + p.y, 0) / n;
    let num = 0, den = 0;
    for (const p of points) { num += (p.x - xbar) * (p.y - ybar); den += (p.x - xbar) ** 2; }
    const w = den === 0 ? 0 : num / den;
    const b = ybar - w * xbar;
    let ssRes = 0, ssTot = 0;
    for (const p of points) {
      const pred = w * p.x + b;
      ssRes += (p.y - pred) ** 2;
      ssTot += (p.y - ybar) ** 2;
    }
    const mse = ssRes / n;
    const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
    return { w, b, mse, r2 };
  }

  function draw() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // axes
    ctx.strokeStyle = "#d9dde5";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);

    const { w: slope, b: intercept, mse, r2 } = fitLine();

    // residuals
    if (residualsBox.checked && points.length >= 2) {
      ctx.strokeStyle = "#f2b8b5";
      for (const p of points) {
        const [px, py1] = toPx(p.x, p.y);
        const [, py2] = toPx(p.x, slope * p.x + intercept);
        ctx.beginPath(); ctx.moveTo(px, py1); ctx.lineTo(px, py2); ctx.stroke();
      }
    }

    // fitted line
    if (points.length >= 2) {
      const [x1, y1] = toPx(xMin, slope * xMin + intercept);
      const [x2, y2] = toPx(xMax, slope * xMax + intercept);
      ctx.strokeStyle = "#1d4ed8";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }

    // points
    ctx.fillStyle = "#1f2430";
    for (const p of points) {
      const [px, py] = toPx(p.x, p.y);
      ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill();
    }

    statsEl.innerHTML = points.length < 2
      ? "Add at least two points to fit a line."
      : `w (slope) = ${slope.toFixed(3)} &nbsp;|&nbsp; b (intercept) = ${intercept.toFixed(3)} &nbsp;|&nbsp; MSE = ${mse.toFixed(3)} &nbsp;|&nbsp; R² = ${r2.toFixed(3)}`;
  }

  function nearestIndex(px, py) {
    let best = -1, bestDist = 12; // px hit radius
    points.forEach((p, i) => {
      const [qx, qy] = toPx(p.x, p.y);
      const d = Math.hypot(qx - px, qy - py);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  }

  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const idx = nearestIndex(px, py);
    if (idx >= 0) {
      dragIndex = idx;
    } else {
      const [x, y] = toData(px, py);
      if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
        points.push({ x, y });
        dragIndex = points.length - 1;
      }
    }
    draw();
  });

  canvas.addEventListener("mousemove", (e) => {
    if (dragIndex < 0) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const [x, y] = toData(px, py);
    points[dragIndex] = {
      x: Math.min(xMax, Math.max(xMin, x)),
      y: Math.min(yMax, Math.max(yMin, y)),
    };
    draw();
  });

  window.addEventListener("mouseup", () => { dragIndex = -1; });

  canvas.addEventListener("dblclick", (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const idx = nearestIndex(px, py);
    if (idx >= 0) { points.splice(idx, 1); draw(); }
  });

  residualsBox.addEventListener("change", draw);

  document.querySelectorAll("#lr-widget [data-lr-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-lr-preset");
      points = key === "clear" ? [] : presets[key].map(([x, y]) => ({ x, y }));
      draw();
    });
  });

  draw();
})();
</script>

> **Note:** $w$ and $b$ above are computed with the closed-form least-squares solution, $w = \dfrac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2}$, $b = \bar y - w\bar x$ — the direct solution for this simplest case of $f_\theta$. Section 3.3 introduces the general method ([gradient descent](#glossary-gradient-descent)) used when $f_\theta$ is a full neural network and no closed-form solution exists.

In this chapter, $f_\theta$ — the "model" — is opened up and explained. The chapter is organised in the order a beginner needs it:

1. **Building blocks** (Section 3.2): the small parts every neural network is made of.
2. **Training mechanics** (Section 3.3): how a network adjusts itself to make fewer mistakes.
3. [**Generalisation**](#glossary-generalisation) (Section 3.4): how to make sure the network also works on data it has never seen, including the settings ([hyperparameters](#glossary-hyperparameter)) that control training.
4. **Architecture families** (Section 3.5): [MLPs](#glossary-mlp), [CNNs](#glossary-cnn), [LSTMs](#glossary-rnn), [encoder–decoders](#glossary-encoder-decoder), [transformers](#glossary-transformer), [point-cloud networks](#glossary-point-cloud) and [bird's-eye-view (BEV) networks](#glossary-bev).
5. **Putting it together** (Section 3.6): how the pieces combine into a complete perception network.

> **Note:** This chapter explains the ideas. Where each model is *used* in a vehicle is the subject of Chapter 4, how the models fit into a complete driving system is covered in Chapter 5, and how they are made small and fast enough to run on vehicle hardware is covered in Chapter 6.

## 3.2 Neural Network Building Blocks

### 3.2.1 The [neuron](#glossary-neuron)

The smallest unit of a neural network is the **neuron**. It receives some numbers as input, multiplies each one by a **weight** (how important that input is), adds the results together with a **bias** (a constant shift), and then passes the total through an **activation function** [$\phi$](#notation-activation):

$$
y = \phi\big(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b\big) = \phi\big(\mathbf{w}^\top \mathbf{x} + b\big)
$$

You can think of a neuron as a tiny calculator that makes a small decision, such as "does this part of the image look like an edge?".

<div id="neuron-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>w1 <input type="range" id="neuron-w1" min="-3" max="3" step="0.1" value="1"> <span id="neuron-w1-val">1.0</span></label>
<label>w2 <input type="range" id="neuron-w2" min="-3" max="3" step="0.1" value="-1"> <span id="neuron-w2-val">-1.0</span></label>
<label>b <input type="range" id="neuron-b" min="-5" max="5" step="0.1" value="0"> <span id="neuron-b-val">0.0</span></label>
<label>activation <select id="neuron-phi"><option value="none">none</option><option value="relu">ReLU</option><option value="sigmoid" selected>Sigmoid</option><option value="tanh">Tanh</option></select></label>
</div>
<canvas id="neuron-canvas" width="420" height="420"></canvas>
<p class="aisw-hint">Click anywhere on the plot to probe a point (x1, x2) and read z and y. The dark line is the decision boundary z = 0, where the sign of the pre-activation flips.</p>
<div id="neuron-stats" class="aisw-stats"></div>
</div>

<script>
(function () {
  const canvas = document.getElementById("neuron-canvas");
  const ctx = canvas.getContext("2d");
  const statsEl = document.getElementById("neuron-stats");
  const w1Input = document.getElementById("neuron-w1");
  const w2Input = document.getElementById("neuron-w2");
  const bInput = document.getElementById("neuron-b");
  const phiSelect = document.getElementById("neuron-phi");
  const w1Val = document.getElementById("neuron-w1-val");
  const w2Val = document.getElementById("neuron-w2-val");
  const bVal = document.getElementById("neuron-b-val");

  const xMin = -5, xMax = 5, yMin = -5, yMax = 5;
  const pad = 30;
  const cells = 50;
  let probe = null;

  function phi(name, z) {
    switch (name) {
      case "relu": return Math.max(0, z);
      case "sigmoid": return 1 / (1 + Math.exp(-z));
      case "tanh": return Math.tanh(z);
      default: return z;
    }
  }

  function rangeFor(name) {
    switch (name) {
      case "relu": return [0, 5];
      case "sigmoid": return [0, 1];
      case "tanh": return [-1, 1];
      default: return [-5, 5];
    }
  }

  function toPx(x, y) {
    const w = canvas.width, h = canvas.height;
    return [
      pad + (x - xMin) / (xMax - xMin) * (w - 2 * pad),
      h - pad - (y - yMin) / (yMax - yMin) * (h - 2 * pad),
    ];
  }
  function toData(px, py) {
    const w = canvas.width, h = canvas.height;
    return [
      xMin + (px - pad) / (w - 2 * pad) * (xMax - xMin),
      yMin + (h - pad - py) / (h - 2 * pad) * (yMax - yMin),
    ];
  }

  function colorFor(t) {
    t = Math.min(1, Math.max(0, t));
    let r, g, b;
    if (t < 0.5) {
      const k = t / 0.5;
      r = Math.round(59 + k * (255 - 59));
      g = Math.round(130 + k * (255 - 130));
      b = Math.round(246 + k * (255 - 246));
    } else {
      const k = (t - 0.5) / 0.5;
      r = 255;
      g = Math.round(255 - k * (255 - 68));
      b = Math.round(255 - k * (255 - 68));
    }
    return `rgb(${r},${g},${b})`;
  }

  function draw() {
    const w1 = parseFloat(w1Input.value);
    const w2 = parseFloat(w2Input.value);
    const b = parseFloat(bInput.value);
    const phiName = phiSelect.value;
    w1Val.textContent = w1.toFixed(1);
    w2Val.textContent = w2.toFixed(1);
    bVal.textContent = b.toFixed(1);

    const [lo, hi] = rangeFor(phiName);
    const w = canvas.width, h = canvas.height;
    const cellW = (w - 2 * pad) / cells;
    const cellH = (h - 2 * pad) / cells;

    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const x1 = xMin + (i + 0.5) / cells * (xMax - xMin);
        const x2 = yMin + (j + 0.5) / cells * (yMax - yMin);
        const z = w1 * x1 + w2 * x2 + b;
        const y = phi(phiName, z);
        const t = (y - lo) / (hi - lo);
        ctx.fillStyle = colorFor(t);
        const px = pad + i * cellW;
        const py = h - pad - (j + 1) * cellH;
        ctx.fillRect(px, py, cellW + 1, cellH + 1);
      }
    }

    ctx.strokeStyle = "#1f2430";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);

    ctx.strokeStyle = "#1f2430";
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (Math.abs(w2) > 1e-6) {
      const [x1a, y1a] = toPx(xMin, -(w1 * xMin + b) / w2);
      const [x2a, y2a] = toPx(xMax, -(w1 * xMax + b) / w2);
      ctx.moveTo(x1a, y1a); ctx.lineTo(x2a, y2a);
    } else if (Math.abs(w1) > 1e-6) {
      const xb = -b / w1;
      const [x1a, y1a] = toPx(xb, yMin);
      const [x2a, y2a] = toPx(xb, yMax);
      ctx.moveTo(x1a, y1a); ctx.lineTo(x2a, y2a);
    }
    ctx.stroke();

    let statsHtml = `w1 = ${w1.toFixed(1)}, w2 = ${w2.toFixed(1)}, b = ${b.toFixed(1)}, φ = ${phiName}`;
    if (probe) {
      const z = w1 * probe.x1 + w2 * probe.x2 + b;
      const y = phi(phiName, z);
      const [px, py] = toPx(probe.x1, probe.x2);
      ctx.fillStyle = "#1f2430";
      ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
      statsHtml += `<br>probe: x1 = ${probe.x1.toFixed(2)}, x2 = ${probe.x2.toFixed(2)} &rarr; z = ${z.toFixed(2)}, y = φ(z) = ${y.toFixed(3)}`;
    }
    statsEl.innerHTML = statsHtml;
  }

  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    const [x1, x2] = toData(px, py);
    probe = { x1, x2 };
    draw();
  });

  [w1Input, w2Input, bInput, phiSelect].forEach((el) => el.addEventListener("input", draw));
  phiSelect.addEventListener("change", draw);

  draw();
})();
</script>

### 3.2.2 Weights and biases

**Weights** and **biases** are the "knobs" of the network. At the start of training they have arbitrary values and the network's answers are poor. Training is the process of turning all these knobs, little by little, until the answers become good. A modern network can have millions or billions of such knobs. Together they are called the model **parameters**, written $\theta$.

### 3.2.3 [Layers](#glossary-layer)

Neurons are arranged in rows called **layers**:

- The **input layer** receives the data (for example, the pixel values of an image).
- One or more **hidden layers** process the data step by step.
- The **output layer** produces the answer (for example, "pedestrian" or "car").

If a layer has [$n_{in}$](#notation-layer-size) inputs and [$n_{out}$](#notation-layer-size) neurons, all of its weights can be written as a matrix [$W$](#notation-weight-matrix) and a bias vector [$\mathbf b$](#notation-weight-matrix); the layer computes:

$$
\mathbf{h} = \phi\big(W\mathbf{x} + \mathbf{b}\big)
$$

Such a layer has $n_{in} \times n_{out}$ weights and $n_{out}$ biases. For example, a layer taking 100 numbers and producing 50 has $100 \times 50 + 50 = 5\,050$ parameters.

A network with many hidden layers is called a **deep** network, which is where the term [*deep learning*](#glossary-deep-learning) comes from. Early layers tend to learn simple patterns, and later layers combine them into more complex ones.

<div id="layer-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>n_in <input type="range" id="layer-nin" min="1" max="6" step="1" value="3"> <span id="layer-nin-val">3</span></label>
<label>hidden layers <select id="layer-hidden-count"><option value="0" selected>0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select></label>
<label>hidden size <input type="range" id="layer-hidden-size" min="1" max="6" step="1" value="4"> <span id="layer-hidden-size-val">4</span></label>
<label>n_out <input type="range" id="layer-nout" min="1" max="6" step="1" value="4"> <span id="layer-nout-val">4</span></label>
</div>
<canvas id="layer-canvas" width="640" height="360"></canvas>
<div class="aisw-controls">
<label id="layer-selected-label">Click an edge to edit its weight.</label>
<label>weight <input type="number" id="layer-weight-input" step="0.1" style="width:5rem;" disabled></label>
</div>
<p class="aisw-hint">Every node in one column connects to every node in the next (fully connected). Edge colour and thickness reflect each weight's value. Changing the architecture resets all weights to 1.</p>
<div id="layer-stats" class="aisw-stats"></div>
</div>

<script>
(function () {
  const canvas = document.getElementById("layer-canvas");
  const ctx = canvas.getContext("2d");
  const statsEl = document.getElementById("layer-stats");
  const ninInput = document.getElementById("layer-nin");
  const noutInput = document.getElementById("layer-nout");
  const hiddenCountSelect = document.getElementById("layer-hidden-count");
  const hiddenSizeInput = document.getElementById("layer-hidden-size");
  const ninVal = document.getElementById("layer-nin-val");
  const noutVal = document.getElementById("layer-nout-val");
  const hiddenSizeVal = document.getElementById("layer-hidden-size-val");
  const selectedLabel = document.getElementById("layer-selected-label");
  const weightInput = document.getElementById("layer-weight-input");

  const pad = 30, leftMargin = 70;
  let sizes = [];
  let weights = [];
  let selected = null;

  function nodePositions(n, x, h) {
    const positions = [];
    for (let i = 0; i < n; i++) {
      const y = n === 1 ? h / 2 : pad + (i / (n - 1)) * (h - 2 * pad);
      positions.push([x, y]);
    }
    return positions;
  }

  function colFor(index, total, w) {
    if (total === 1) return leftMargin;
    return leftMargin + index * ((w - 2 * leftMargin) / (total - 1));
  }

  function columnsFor(w, h) {
    return sizes.map((size, l) => ({
      size,
      nodes: nodePositions(size, colFor(l, sizes.length, w), h),
    }));
  }

  function layerLabel(l) {
    if (l === 0) return "x";
    if (l === sizes.length - 1) return "o";
    return "h";
  }

  function rebuild() {
    const nIn = parseInt(ninInput.value, 10);
    const nOut = parseInt(noutInput.value, 10);
    const hiddenCount = parseInt(hiddenCountSelect.value, 10);
    const hiddenSize = parseInt(hiddenSizeInput.value, 10);
    ninVal.textContent = nIn;
    noutVal.textContent = nOut;
    hiddenSizeVal.textContent = hiddenSize;

    sizes = [nIn];
    for (let i = 0; i < hiddenCount; i++) sizes.push(hiddenSize);
    sizes.push(nOut);

    weights = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const mat = [];
      for (let i = 0; i < sizes[l]; i++) {
        const row = [];
        for (let j = 0; j < sizes[l + 1]; j++) row.push(1);
        mat.push(row);
      }
      weights.push(mat);
    }
    selected = null;
    updateSelectedUi();
  }

  function colorForWeight(v) {
    const t = Math.min(1, Math.max(0, v / 3 * 0.5 + 0.5));
    let r, g, b;
    if (t < 0.5) {
      const k = t / 0.5;
      r = Math.round(59 + k * (255 - 59)); g = Math.round(130 + k * (255 - 130)); b = Math.round(246 + k * (255 - 246));
    } else {
      const k = (t - 0.5) / 0.5;
      r = 255; g = Math.round(255 - k * (255 - 68)); b = Math.round(255 - k * (255 - 68));
    }
    return `rgb(${r},${g},${b})`;
  }

  function draw() {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const columns = columnsFor(w, h);

    for (let l = 0; l < columns.length - 1; l++) {
      for (let i = 0; i < columns[l].nodes.length; i++) {
        for (let j = 0; j < columns[l + 1].nodes.length; j++) {
          const [x1, y1] = columns[l].nodes[i];
          const [x2, y2] = columns[l + 1].nodes[j];
          const v = weights[l][i][j];
          const isSel = selected && selected.layer === l && selected.from === i && selected.to === j;
          ctx.strokeStyle = isSel ? "#1f2430" : colorForWeight(v);
          ctx.lineWidth = isSel ? 3 : Math.min(4, 0.6 + Math.abs(v));
          ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        }
      }
    }

    columns.forEach((col, l) => {
      col.nodes.forEach(([x, y], i) => {
        ctx.fillStyle = "#1f2430";
        ctx.beginPath(); ctx.arc(x, y, 10, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(layerLabel(l) + (i + 1), x, y);
      });
    });

    let totalParams = 0;
    const breakdown = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const wCount = sizes[l] * sizes[l + 1];
      const bCount = sizes[l + 1];
      totalParams += wCount + bCount;
      breakdown.push(`${sizes[l]}&times;${sizes[l + 1]}+${sizes[l + 1]}`);
    }
    statsEl.innerHTML = `layer sizes: ${sizes.join(" &rarr; ")} &nbsp;|&nbsp; parameters per layer: ${breakdown.join(" + ")} &nbsp;|&nbsp; total = ${totalParams}`;
  }

  function updateSelectedUi() {
    if (selected) {
      const v = weights[selected.layer][selected.from][selected.to];
      selectedLabel.textContent = `Selected: ${layerLabel(selected.layer)}${selected.from + 1} -> ${layerLabel(selected.layer + 1)}${selected.to + 1}`;
      weightInput.disabled = false;
      weightInput.value = v;
    } else {
      selectedLabel.textContent = "Click an edge to edit its weight.";
      weightInput.disabled = true;
      weightInput.value = "";
    }
  }

  function distToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const len2 = dx * dx + dy * dy;
    let t = len2 === 0 ? 0 : ((px - x1) * dx + (py - y1) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    const projX = x1 + t * dx, projY = y1 + t * dy;
    return Math.hypot(px - projX, py - projY);
  }

  function nearestEdge(px, py) {
    const columns = columnsFor(canvas.width, canvas.height);
    let best = null, bestDist = 10;
    for (let l = 0; l < columns.length - 1; l++) {
      for (let i = 0; i < columns[l].nodes.length; i++) {
        for (let j = 0; j < columns[l + 1].nodes.length; j++) {
          const [x1, y1] = columns[l].nodes[i];
          const [x2, y2] = columns[l + 1].nodes[j];
          const d = distToSegment(px, py, x1, y1, x2, y2);
          if (d < bestDist) { bestDist = d; best = { layer: l, from: i, to: j }; }
        }
      }
    }
    return best;
  }

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    selected = nearestEdge(px, py);
    updateSelectedUi();
    draw();
  });

  weightInput.addEventListener("input", () => {
    if (!selected) return;
    const v = parseFloat(weightInput.value);
    weights[selected.layer][selected.from][selected.to] = isNaN(v) ? 0 : v;
    draw();
  });

  [ninInput, noutInput, hiddenSizeInput].forEach((el) => el.addEventListener("input", () => { rebuild(); draw(); }));
  hiddenCountSelect.addEventListener("change", () => { rebuild(); draw(); });

  rebuild();
  draw();
})();
</script>

### 3.2.4 Activation functions

An **activation function** is a small mathematical function applied to the output of each neuron.

**Why do we need it?** Without activation functions, a neuron only does multiplication and addition. Stacking many such layers would still be equivalent to a *single* multiplication and addition, so the network could only learn straight-line relationships. Activation functions bend the straight lines, which allows the network to learn complex, curved patterns.

| Function | Formula | Plain-language description | Typical use |
|---|---|---|---|
| **ReLU** | $\max(0, x)$ | Negative values become zero; positive values pass unchanged. | Default choice in hidden layers of CNNs |
| **Leaky ReLU** | $\max(0.01x, x)$ | Like ReLU, but lets a tiny amount of negative values through. | Detection networks (e.g. [YOLO](#glossary-yolo) variants) |
| **Sigmoid** | $\dfrac{1}{1+e^{-x}}$ | Squashes any number into the range 0 to 1. | Output of a yes/no decision |
| **Tanh** | $\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | Squashes any number into the range −1 to 1. | Inside LSTM cells |
| **Softmax** | $\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | Turns a list of numbers into probabilities that add up to 100%. | Output of a multi-class classifier |
| **GELU / SiLU** | $x\,\Phi(x)$ / $x\,\sigma(x)$ | Smooth versions of ReLU. | Transformers and modern CNNs |

> [**The vanishing-gradient problem**](#glossary-gradient). Sigmoid and tanh become almost flat for very large or very small inputs. In a flat region, the signal used to teach the network (the gradient, Section 3.3) is close to zero, so early layers of a deep network barely learn. ReLU-type functions do not flatten for positive values, which is one reason they dominate modern networks.

<div id="act-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>function <select id="act-fn"><option value="relu">ReLU</option><option value="lrelu">Leaky ReLU</option><option value="sigmoid" selected>Sigmoid</option><option value="tanh">Tanh</option><option value="gelu">GELU</option></select></label>
</div>
<canvas id="act-canvas" width="560" height="340"></canvas>
<p class="aisw-hint">Drag along the plot to probe a point x and read φ(x) (solid) and the numerically estimated slope φ'(x) (dashed). Watch the dashed line flatten near the tails for Sigmoid/Tanh but not for ReLU.</p>
<div id="act-stats" class="aisw-stats"></div>
</div>

<script>
(function () {
  const canvas = document.getElementById("act-canvas");
  const ctx = canvas.getContext("2d");
  const statsEl = document.getElementById("act-stats");
  const fnSelect = document.getElementById("act-fn");

  const xMin = -5, xMax = 5;
  const pad = 36;
  let probeX = 1.5;
  let dragging = false;

  function phi(name, x) {
    switch (name) {
      case "relu": return Math.max(0, x);
      case "lrelu": return x >= 0 ? x : 0.01 * x;
      case "sigmoid": return 1 / (1 + Math.exp(-x));
      case "tanh": return Math.tanh(x);
      case "gelu": {
        const c = Math.sqrt(2 / Math.PI);
        return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x ** 3)));
      }
      default: return x;
    }
  }

  function derivative(name, x) {
    const h = 1e-3;
    return (phi(name, x + h) - phi(name, x - h)) / (2 * h);
  }

  function toPx(x, y, yMin, yMax, w, h) {
    return [
      pad + (x - xMin) / (xMax - xMin) * (w - 2 * pad),
      h - pad - (y - yMin) / (yMax - yMin) * (h - 2 * pad),
    ];
  }

  function draw() {
    const name = fnSelect.value;
    const w = canvas.width, hgt = canvas.height;
    ctx.clearRect(0, 0, w, hgt);

    const samples = [];
    const derivs = [];
    const n = 200;
    for (let i = 0; i <= n; i++) {
      const x = xMin + (i / n) * (xMax - xMin);
      samples.push([x, phi(name, x)]);
      derivs.push([x, derivative(name, x)]);
    }
    let yMin = Math.min(...samples.map((p) => p[1]), ...derivs.map((p) => p[1]));
    let yMax = Math.max(...samples.map((p) => p[1]), ...derivs.map((p) => p[1]));
    const margin = (yMax - yMin) * 0.1 || 0.5;
    yMin -= margin; yMax += margin;

    ctx.strokeStyle = "#d9dde5";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2 * pad, hgt - 2 * pad);
    const [zx1, zy1] = toPx(xMin, 0, yMin, yMax, w, hgt);
    const [zx2, zy2] = toPx(xMax, 0, yMin, yMax, w, hgt);
    ctx.beginPath(); ctx.moveTo(zx1, zy1); ctx.lineTo(zx2, zy2); ctx.stroke();

    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    samples.forEach(([x, y], i) => {
      const [px, py] = toPx(x, y, yMin, yMax, w, hgt);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    ctx.strokeStyle = "#ef4444";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    derivs.forEach(([x, y], i) => {
      const [px, py] = toPx(x, y, yMin, yMax, w, hgt);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    const [px, pyTop] = toPx(probeX, yMax, yMin, yMax, w, hgt);
    const [, pyBot] = toPx(probeX, yMin, yMin, yMax, w, hgt);
    ctx.strokeStyle = "#5b6475";
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, pyTop); ctx.lineTo(px, pyBot); ctx.stroke();
    ctx.setLineDash([]);

    const pVal = phi(name, probeX);
    const dVal = derivative(name, probeX);
    const [dotX, dotY] = toPx(probeX, pVal, yMin, yMax, w, hgt);
    ctx.fillStyle = "#1f2430";
    ctx.beginPath(); ctx.arc(dotX, dotY, 5, 0, 2 * Math.PI); ctx.fill();

    statsEl.innerHTML = `x = ${probeX.toFixed(2)} &nbsp;|&nbsp; φ(x) = ${pVal.toFixed(3)} &nbsp;|&nbsp; φ'(x) &asymp; ${dVal.toFixed(3)}`;
  }

  function setProbeFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const x = xMin + (px - pad) / (canvas.width - 2 * pad) * (xMax - xMin);
    probeX = Math.min(xMax, Math.max(xMin, x));
    draw();
  }

  canvas.addEventListener("mousedown", (e) => { dragging = true; setProbeFromEvent(e); });
  canvas.addEventListener("mousemove", (e) => { if (dragging) setProbeFromEvent(e); });
  window.addEventListener("mouseup", () => { dragging = false; });
  fnSelect.addEventListener("change", draw);

  draw();
})();
</script>

### 3.2.5 [Convolution](#glossary-convolution) and [pooling](#glossary-pooling)

Images contain a lot of numbers (a $640 \times 640$ colour image has over a million). Connecting every pixel to every neuron would need far too many weights. **Convolution** solves this with a clever shortcut.

A **convolution** slides a small window — called a [**filter**](#glossary-convolution) or [**kernel**](#glossary-convolution), for example $3\times3$ pixels — across the image. At each position it multiplies the pixels under the window by the filter's weights and adds them up. The *same* filter is reused at every position, so the number of weights stays small, and a pattern (such as a vertical edge) is detected wherever it appears in the image. The result is called a [**feature map**](#glossary-convolution).

Three settings control a convolution:

- [**Kernel size**](#glossary-convolution) [$k$](#notation-convolution): the size of the window.
- [**Stride**](#glossary-convolution) [$s$](#notation-convolution): how many pixels the window moves each time. A stride of 2 halves the width and height.
- [**Padding**](#glossary-convolution) [$p$](#notation-convolution): zeros added around the border so that edges are not lost.

The output width (or height) for an input of size $n$ is:

$$
n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1
$$

**Pooling** shrinks a feature map by keeping a summary of each small region — for example, **max pooling** keeps only the largest value in each $2\times2$ block. This reduces the amount of computation and makes the network less sensitive to tiny shifts of an object in the image.

<div id="conv-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>kernel <select id="conv-preset"><option value="vert">Vertical edge</option><option value="horiz">Horizontal edge</option><option value="blur">Blur</option><option value="sharpen">Sharpen</option><option value="identity">Identity</option></select></label>
<label>stride <select id="conv-stride"><option value="1" selected>1</option><option value="2">2</option></select></label>
<label>padding <select id="conv-pad"><option value="0" selected>0</option><option value="1">1</option></select></label>
<button type="button" id="conv-step">Step</button>
<button type="button" id="conv-auto">Auto-run</button>
<button type="button" id="conv-reset">Reset</button>
</div>
<div style="display:flex;gap:1.25rem;flex-wrap:wrap;align-items:flex-start;">
<div id="conv-kernel-grid" style="display:inline-grid;grid-template-columns:repeat(3,2.6rem);gap:2px;"></div>
<canvas id="conv-input-canvas" width="280" height="280" style="max-width:280px;"></canvas>
<canvas id="conv-output-canvas" width="200" height="200" style="max-width:200px;"></canvas>
</div>
<p class="aisw-hint">Click a cell in the input grid to cycle its intensity (0-3). Edit the kernel numbers directly, then use Step/Auto-run to slide the kernel across the input and build the output feature map.</p>
<div id="conv-stats" class="aisw-stats"></div>
</div>

<script>
(function () {
  const n = 8;
  const kernelGridEl = document.getElementById("conv-kernel-grid");
  const inputCanvas = document.getElementById("conv-input-canvas");
  const outputCanvas = document.getElementById("conv-output-canvas");
  const inCtx = inputCanvas.getContext("2d");
  const outCtx = outputCanvas.getContext("2d");
  const statsEl = document.getElementById("conv-stats");
  const presetSelect = document.getElementById("conv-preset");
  const strideSelect = document.getElementById("conv-stride");
  const padSelect = document.getElementById("conv-pad");
  const stepBtn = document.getElementById("conv-step");
  const autoBtn = document.getElementById("conv-auto");
  const resetBtn = document.getElementById("conv-reset");

  const presets = {
    vert: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    horiz: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    blur: [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
    sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    identity: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
  };

  let kernel = presets.vert.map((row) => row.slice());
  let grid = [];
  for (let r = 0; r < n; r++) {
    const row = [];
    for (let c = 0; c < n; c++) row.push(c < n / 2 ? 0 : 3);
    grid.push(row);
  }

  let stepIndex = -1;
  let autoTimer = null;
  let outputCache = [];

  function buildKernelInputs() {
    kernelGridEl.innerHTML = "";
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const inp = document.createElement("input");
        inp.type = "number";
        inp.step = "0.1";
        inp.value = Math.round(kernel[i][j] * 100) / 100;
        inp.style.width = "100%";
        inp.addEventListener("input", () => {
          kernel[i][j] = parseFloat(inp.value) || 0;
          resetProgress();
          draw();
        });
        kernelGridEl.appendChild(inp);
      }
    }
  }

  function dims() {
    const s = parseInt(strideSelect.value, 10);
    const p = parseInt(padSelect.value, 10);
    const k = 3;
    const nOut = Math.floor((n + 2 * p - k) / s) + 1;
    return { s, p, k, nOut };
  }

  function valueAt(r, c) {
    if (r < 0 || r >= n || c < 0 || c >= n) return 0;
    return grid[r][c];
  }

  function windowSum(topRow, topCol) {
    let sum = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        sum += valueAt(topRow + i, topCol + j) * kernel[i][j];
      }
    }
    return sum;
  }

  function resetProgress() {
    stepIndex = -1;
    outputCache = [];
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = "Auto-run"; }
  }

  function cellColorInput(v) {
    const t = v / 3;
    const g = Math.round(255 - t * 255);
    return `rgb(${g},${g},${g})`;
  }

  function diverging(v, maxAbs) {
    const t = Math.min(1, Math.max(0, (v / (maxAbs || 1)) * 0.5 + 0.5));
    let r, g, b;
    if (t < 0.5) {
      const k = t / 0.5;
      r = Math.round(59 + k * (255 - 59)); g = Math.round(130 + k * (255 - 130)); b = Math.round(246 + k * (255 - 246));
    } else {
      const k = (t - 0.5) / 0.5;
      r = 255; g = Math.round(255 - k * (255 - 68)); b = Math.round(255 - k * (255 - 68));
    }
    return `rgb(${r},${g},${b})`;
  }

  function draw() {
    const { s, p, k, nOut } = dims();
    const cell = inputCanvas.width / n;
    inCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        inCtx.fillStyle = cellColorInput(grid[r][c]);
        inCtx.fillRect(c * cell, r * cell, cell - 1, cell - 1);
      }
    }

    if (stepIndex >= 0 && stepIndex < nOut * nOut) {
      const row = Math.floor(stepIndex / nOut);
      const col = stepIndex % nOut;
      const topRow = row * s - p;
      const topCol = col * s - p;
      inCtx.strokeStyle = "#1d4ed8";
      inCtx.lineWidth = 3;
      inCtx.strokeRect(topCol * cell, topRow * cell, k * cell, k * cell);
    }

    const outCell = outputCanvas.width / nOut;
    outCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    const maxAbs = Math.max(1, ...outputCache.map((v) => Math.abs(v)));
    for (let idx = 0; idx < outputCache.length; idx++) {
      const row = Math.floor(idx / nOut);
      const col = idx % nOut;
      outCtx.fillStyle = diverging(outputCache[idx], maxAbs);
      outCtx.fillRect(col * outCell, row * outCell, outCell - 1, outCell - 1);
    }
    if (stepIndex >= 0 && stepIndex < nOut * nOut) {
      const row = Math.floor(stepIndex / nOut);
      const col = stepIndex % nOut;
      outCtx.strokeStyle = "#1d4ed8";
      outCtx.lineWidth = 2;
      outCtx.strokeRect(col * outCell, row * outCell, outCell, outCell);
    }

    let statsHtml = `n = ${n}, k = ${k}, stride = ${s}, padding = ${p} &rarr; n_out = floor((${n} + 2&times;${p} - ${k}) / ${s}) + 1 = ${nOut}`;
    if (stepIndex >= 0 && stepIndex < nOut * nOut) {
      statsHtml += `<br>position ${stepIndex + 1} / ${nOut * nOut}: &sum; window &middot; kernel = ${outputCache[stepIndex].toFixed(2)}`;
    } else if (stepIndex >= nOut * nOut) {
      statsHtml += `<br>done — ${nOut * nOut} positions computed.`;
    }
    statsEl.innerHTML = statsHtml;
  }

  function step() {
    const { s, p, nOut } = dims();
    const total = nOut * nOut;
    if (stepIndex >= total - 1) { stepIndex = total; draw(); return; }
    stepIndex++;
    const row = Math.floor(stepIndex / nOut);
    const col = stepIndex % nOut;
    outputCache[stepIndex] = windowSum(row * s - p, col * s - p);
    draw();
  }

  stepBtn.addEventListener("click", () => { if (!autoTimer) step(); });
  autoBtn.addEventListener("click", () => {
    if (autoTimer) {
      clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = "Auto-run";
    } else {
      autoBtn.textContent = "Pause";
      autoTimer = setInterval(() => {
        const { nOut } = dims();
        if (stepIndex >= nOut * nOut - 1) { clearInterval(autoTimer); autoTimer = null; autoBtn.textContent = "Auto-run"; return; }
        step();
      }, 400);
    }
  });
  resetBtn.addEventListener("click", () => { resetProgress(); draw(); });

  inputCanvas.addEventListener("click", (e) => {
    const rect = inputCanvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (inputCanvas.width / rect.width);
    const py = (e.clientY - rect.top) * (inputCanvas.height / rect.height);
    const cell = inputCanvas.width / n;
    const c = Math.floor(px / cell), r = Math.floor(py / cell);
    if (r >= 0 && r < n && c >= 0 && c < n) {
      grid[r][c] = (grid[r][c] + 1) % 4;
      resetProgress();
      draw();
    }
  });

  presetSelect.addEventListener("change", () => {
    kernel = presets[presetSelect.value].map((row) => row.slice());
    buildKernelInputs();
    resetProgress();
    draw();
  });
  strideSelect.addEventListener("change", () => { resetProgress(); draw(); });
  padSelect.addEventListener("change", () => { resetProgress(); draw(); });

  buildKernelInputs();
  draw();
})();
</script>

### 3.2.6 [Normalisation layers](#glossary-normalisation)

As numbers pass through many layers they can become very large or very small, which makes learning slow and unstable. A **normalisation layer** rescales the numbers to a standard range:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^{2} + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta
$$

where [$\mu$](#notation-normalisation) and [$\sigma^2$](#notation-normalisation) are the mean and variance of the values, and [$\gamma$](#notation-normalisation) and [$\beta$](#notation-normalisation) are two learnable parameters that let the network undo the rescaling if that turns out to be better.

- [**Batch Normalisation (BatchNorm)**](#glossary-normalisation) computes the mean and variance across a batch of examples. Typical in CNNs.
- [**Layer Normalisation (LayerNorm)**](#glossary-normalisation) computes them across the features of a single example. Typical in transformers and recurrent networks.

### 3.2.7 [Weight initialisation](#glossary-weight-initialisation)

Before training, the weights need starting values. If they are all zero, every neuron learns the same thing; if they are too large or too small, learning can stall. Standard recipes choose small random values whose size depends on the layer width:

- **Xavier (Glorot) initialisation:** variance $\dfrac{2}{n_{in}+n_{out}}$ — used with tanh and sigmoid.
- **He initialisation:** variance $\dfrac{2}{n_{in}}$ — used with ReLU.

### 3.2.8 [Residual (skip) connections](#glossary-residual-connection)

In a **residual connection**, the input of a block is added to its output:

$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x}
$$

The block $F$ now only needs to learn the *change* to apply to its input, and information (and the learning signal) can travel along the shortcut across many layers. This makes very deep networks much easier to train and is the key idea of **ResNet**. Residual connections are used in almost every modern architecture, including transformers.

### 3.2.9 [Attention](#glossary-attention) (a first look)

**Attention** lets a network decide which parts of its input to focus on when producing each part of its output — like highlighting the important words in a sentence. For every element, the network builds a [**query**](#glossary-attention) [$Q$](#notation-attention) ("what am I looking for?"), and every element offers a [**key**](#glossary-attention) [$K$](#notation-attention) ("what do I contain?") and a [**value**](#glossary-attention) [$V$](#notation-attention) ("what information do I give?"). Keys that match the query receive a high weight, and the output is the weighted mix of the values:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Attention is explained fully in Section 3.5.5 as the core of the transformer.

### 3.2.10 Parameters versus hyperparameters

| | Parameters | Hyperparameters |
|---|---|---|
| **Examples** | Weights and biases | [Learning rate](#glossary-learning-rate), batch size, number of layers, [dropout rate](#glossary-regularisation) |
| **Who sets them?** | The network, by learning | The engineer, before training |
| **When?** | During training | Before (and between) training runs |

Hyperparameters are discussed in Section 3.4.

## 3.3 How a Network Learns: Training Mechanics

Training is a repeated four-step loop:

1. [**Forward pass**](#glossary-forward-pass): feed examples in and compute predictions.
2. **Loss:** measure how wrong the predictions are.
3. **Backward pass:** work out how each weight contributed to the error.
4. **Update:** nudge each weight in the direction that reduces the error.

This loop runs thousands or millions of times until the error stops improving.

> **Running example.** To make the loop concrete, Sections 3.3.1–3.3.4 follow one tiny neuron through every stage with real numbers. The neuron has no activation function, to keep the arithmetic simple: $\hat{y} = wx + b$. It sees a single training example with input $x=2$ and true answer $y=3$. Its weights start at $w=0.5$ and $b=0$, the loss is $\mathcal{L} = (\hat{y}-y)^2$, and the learning rate is [$\eta$](#notation-learning-rate)$=0.1$. The stepper in Section 3.3.3 lets you replay the whole loop on these numbers.

### 3.3.1 Forward propagation

In **forward propagation**, data enters the input layer and flows through the layers one by one, each applying its weights and activation function, until the output layer produces a prediction. This is the network "making a guess". Forward propagation is also exactly what happens when a trained network is used (this is called [**inference**](#glossary-inference)).

> **Running example — stage 1.** Our neuron has a single layer and no activation, so the forward pass is one line:
> $\hat{y} = w x + b = 0.5 \times 2 + 0 = 1$. The network guesses 1, but the true answer is 3.

### 3.3.2 Loss functions

A **loss function** $\mathcal{L}$ is a score that tells us how wrong a prediction $\hat{y}$ is compared with the true answer $y$. Lower is better. The choice of loss depends on the task:

In the focal-loss expression, [$p_t$](#notation-focal-probability) is the probability assigned to the target class. In the Dice-loss expression, [$A$](#notation-dice-regions) and [$B$](#notation-dice-regions) denote the predicted and target regions.


| Task | Loss | Formula / idea |
|---|---|---|
| Classification | [**Cross-entropy**](#glossary-loss-function) | $-\sum_c y_c \log \hat{y}_c$ — penalises being confident and wrong |
| Predicting a number | [**MSE**](#glossary-loss-function) (mean squared error) | $(\hat{y}-y)^2$ — penalises large errors strongly |
| Predicting a number | [**L1**](#glossary-loss-function) (mean absolute error) | $\lvert\hat{y}-y\rvert$ — less sensitive to outliers |
| Predicting a number | [**Huber / Smooth-L1**](#glossary-loss-function) | MSE for small errors, L1 for large ones |
| Bounding boxes | [**IoU / GIoU loss**](#glossary-loss-function) | $1 - \text{IoU}$, where IoU is the overlap between predicted and true box divided by their union |
| Rare classes | [**Focal loss**](#glossary-loss-function) | $-(1-p_t)^{\gamma}\log p_t$ — down-weights easy examples so rare, hard ones count more |
| Segmentation | [**Dice loss**](#glossary-loss-function) | $1 - \dfrac{2\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert}$ — based on the overlap of predicted and true regions |

Focal loss is particularly useful in driving data, where "background" vastly outnumbers objects of interest.

> **Running example — stage 2.** Our neuron predicts a number, so we use MSE: $\mathcal{L} = (\hat{y}-y)^2 = (1-3)^2 = 4$. The loss is the single number that the rest of the loop tries to push towards zero.

The widget below compares three of the "predicting a number" losses for a true answer of $y=3$. Drag the prediction $\hat{y}$ (or click on the plot) and watch how differently each loss reacts as the prediction moves further from the target.

<div id="loss-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>ŷ <input type="range" id="loss-yhat" min="-2" max="8" step="0.05" value="1"> <span id="loss-yhat-val">1.00</span></label>
<label>Highlight
<select id="loss-sel">
<option value="all">all three</option>
<option value="mse">MSE</option>
<option value="l1">L1</option>
<option value="huber">Huber</option>
</select>
</label>
<button type="button" id="loss-example">Running example (ŷ = 1)</button>
</div>
<canvas id="loss-canvas" width="560" height="320"></canvas>
<p class="aisw-hint">True answer y = 3 (dashed line). MSE = e², L1 = |e|, Huber (δ = 1) = ½e² for |e| ≤ 1 and |e| − ½ beyond, where e = ŷ − y. Near the target all three are small; far from it MSE explodes while L1 and Huber grow only linearly — which is why MSE "penalises large errors strongly" and L1/Huber are less sensitive to outliers.</p>
<div id="loss-stats" class="aisw-stats"></div>
</div>

<script>
(function () {
  const canvas = document.getElementById("loss-canvas");
  const ctx = canvas.getContext("2d");
  const statsEl = document.getElementById("loss-stats");
  const yhatInput = document.getElementById("loss-yhat");
  const yhatVal = document.getElementById("loss-yhat-val");
  const selInput = document.getElementById("loss-sel");
  const exampleBtn = document.getElementById("loss-example");

  const y = 3, delta = 1;
  const xMin = -2, xMax = 8, lMax = 26;
  const pad = 36;
  const fns = {
    mse: (e) => e * e,
    l1: (e) => Math.abs(e),
    huber: (e) => (Math.abs(e) <= delta ? 0.5 * e * e : delta * (Math.abs(e) - 0.5 * delta)),
  };
  const names = { mse: "MSE", l1: "L1", huber: "Huber" };
  const colors = { mse: "#1d4ed8", l1: "#ef4444", huber: "#16a34a" };

  function toPx(v, l) {
    const W = canvas.width, H = canvas.height;
    return [
      pad + (v - xMin) / (xMax - xMin) * (W - 2 * pad),
      H - pad - Math.min(l, lMax) / lMax * (H - 2 * pad),
    ];
  }

  function draw() {
    const yhat = parseFloat(yhatInput.value);
    const sel = selInput.value;
    yhatVal.textContent = yhat.toFixed(2);
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = "#d9dde5";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);
    ctx.fillStyle = "#5b6475";
    ctx.font = "12px system-ui, sans-serif";
    ctx.textAlign = "center";
    for (let v = xMin; v <= xMax; v += 2) {
      const [px] = toPx(v, 0);
      ctx.fillText(String(v), px, H - pad + 16);
    }
    ctx.textAlign = "right";
    for (let l = 0; l <= 25; l += 5) {
      const [, py] = toPx(xMin, l);
      ctx.fillText(String(l), pad - 6, py + 4);
    }
    ctx.textAlign = "center";
    ctx.fillText("prediction ŷ", W / 2, H - 6);
    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("loss", 0, 0);
    ctx.restore();

    const [tx] = toPx(y, 0);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "#5b6475";
    ctx.beginPath(); ctx.moveTo(tx, pad); ctx.lineTo(tx, H - pad); ctx.stroke();
    ctx.setLineDash([]);

    Object.keys(fns).forEach((key) => {
      const on = sel === "all" || sel === key;
      ctx.globalAlpha = on ? 1 : 0.25;
      ctx.strokeStyle = colors[key];
      ctx.lineWidth = on ? 3 : 1.5;
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const v = xMin + (i / 200) * (xMax - xMin);
        const [px, py] = toPx(v, fns[key](v - y));
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      if (on) {
        const [px, py] = toPx(yhat, fns[key](yhat - y));
        ctx.fillStyle = colors[key];
        ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
      }
      ctx.globalAlpha = 1;
    });

    const e = yhat - y;
    statsEl.innerHTML = `error e = ŷ − y = ${e.toFixed(2).replace("-", "−")} &nbsp;|&nbsp; ` +
      Object.keys(fns).map((k) => `<span style="color:${colors[k]}">${names[k]} = ${fns[k](e).toFixed(2)}</span>`).join(" &nbsp;|&nbsp; ");
  }

  function setFromPointer(ev) {
    const rect = canvas.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * (canvas.width / rect.width);
    const v = xMin + (px - pad) / (canvas.width - 2 * pad) * (xMax - xMin);
    yhatInput.value = Math.min(xMax, Math.max(xMin, v));
    draw();
  }
  let dragging = false;
  canvas.addEventListener("pointerdown", (ev) => { dragging = true; canvas.setPointerCapture(ev.pointerId); setFromPointer(ev); });
  canvas.addEventListener("pointermove", (ev) => { if (dragging) setFromPointer(ev); });
  canvas.addEventListener("pointerup", () => { dragging = false; });

  yhatInput.addEventListener("input", draw);
  selInput.addEventListener("change", draw);
  exampleBtn.addEventListener("click", () => { yhatInput.value = 1; draw(); });

  draw();
})();
</script>

### 3.3.3 [Gradients](#glossary-gradient) and [backpropagation](#glossary-backpropagation)

The **gradient** of the loss with respect to a weight, $\dfrac{\partial \mathcal{L}}{\partial w}$, tells us how the loss changes if that weight is increased slightly. Imagine standing on a hill in thick fog: you cannot see the valley, but you can feel which way the ground slopes. The gradient is that slope, and moving *against* it takes you downhill, towards a lower loss.

**Backpropagation** is the efficient method for computing the gradient of *every* weight in the network. Starting at the loss, it moves backwards layer by layer, using the [**chain rule**](#glossary-chain-rule) from calculus: the effect of a weight on the loss is the product of the effects along the path from that weight to the loss. For a weight $w$ feeding a neuron with pre-activation $z$ and output $\hat{y}$:

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

Deep-learning frameworks such as PyTorch compute this automatically ([**automatic differentiation**](#glossary-automatic-differentiation)), so engineers rarely write it by hand.

> **Running example — stage 3.** Our neuron has no activation function, so $\hat{y} = z$ and the middle factor is simply 1. Working backwards from the loss $\mathcal{L}=(\hat{y}-y)^2$:
>
> - How does the loss change with the prediction? $\dfrac{\partial \mathcal{L}}{\partial \hat{y}} = 2(\hat{y}-y) = 2(1-3) = -4$.
> - How does the prediction change with each weight? $\dfrac{\partial \hat{y}}{\partial w} = x = 2$ and $\dfrac{\partial \hat{y}}{\partial b} = 1$.
> - Multiplying along each path (chain rule): $\dfrac{\partial \mathcal{L}}{\partial w} = -4 \times 2 = -8$ and $\dfrac{\partial \mathcal{L}}{\partial b} = -4 \times 1 = -4$.
>
> Both gradients are negative: increasing $w$ or $b$ would *lower* the loss, so the update will push them up.

This is exactly what `loss.backward()` does automatically in PyTorch for networks with millions of weights: apply the chain rule layer by layer, rather than working it out by hand as above.

#### Interactive: the training loop on the running example

The stepper below runs the four stages of the loop on the running example. Click **Next stage** to walk through forward pass, loss, backward pass and update; the picture shows the loss for every possible pair $(w, b)$ — blue is low, red is high. The white dot is the neuron's current weights, the arrow shows the downhill direction found by the backward pass, and the dashed line shows where the update will move it. Click **Apply update** to start the next round.

<div id="bp-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>w <input type="range" id="bp-w" min="-2" max="3" step="0.05" value="0.5"> <span id="bp-w-val">0.50</span></label>
<label>b <input type="range" id="bp-b" min="-3" max="3" step="0.05" value="0"> <span id="bp-b-val">0.00</span></label>
<label>η <input type="range" id="bp-eta" min="0.01" max="0.3" step="0.01" value="0.1"> <span id="bp-eta-val">0.10</span></label>
<button type="button" id="bp-next">Next stage ▸</button>
<button type="button" id="bp-reset">Reset</button>
</div>
<div id="bp-stages" class="aisw-stats"></div>
<canvas id="bp-canvas" width="420" height="380"></canvas>
<p class="aisw-hint">x = 2, y = 3 — the running example. With the default settings, round 1 reproduces the numbers in the text exactly and lands on the target (loss 0). Now try changing η: at 0.2 each update overshoots to the other side of the valley and bounces back and forth; above 0.2 it diverges (the dot pins to the edge of the plot). Drag w or b to restart from anywhere.</p>
<div id="bp-stats" class="aisw-stats"></div>
</div>

<script>
(function () {
  const canvas = document.getElementById("bp-canvas");
  const ctx = canvas.getContext("2d");
  const stagesEl = document.getElementById("bp-stages");
  const statsEl = document.getElementById("bp-stats");
  const wInput = document.getElementById("bp-w");
  const bInput = document.getElementById("bp-b");
  const etaInput = document.getElementById("bp-eta");
  const wVal = document.getElementById("bp-w-val");
  const bVal = document.getElementById("bp-b-val");
  const etaVal = document.getElementById("bp-eta-val");
  const nextBtn = document.getElementById("bp-next");
  const resetBtn = document.getElementById("bp-reset");

  const x = 2, y = 3;
  const wMin = -2, wMax = 3, bMin = -3, bMax = 3;
  const pad = 36;
  let trail = [];
  let stage = 0;
  let round = 0;

  const f = (v) => v.toFixed(2).replace("-", "−");
  const clampW = (v) => Math.min(wMax, Math.max(wMin, v));
  const clampB = (v) => Math.min(bMax, Math.max(bMin, v));
  function loss(w, b) { const yhat = w * x + b; return (yhat - y) ** 2; }
  function grad(w, b) { const d = 2 * (w * x + b - y); return [d * x, d]; }

  function toPx(w, b) {
    const W = canvas.width, H = canvas.height;
    return [
      pad + (w - wMin) / (wMax - wMin) * (W - 2 * pad),
      H - pad - (b - bMin) / (bMax - bMin) * (H - 2 * pad),
    ];
  }

  function colorFor(t) {
    t = Math.min(1, Math.max(0, t));
    const r = Math.round(255 * t);
    const g = Math.round(255 * (1 - t) * 0.9);
    const bch = Math.round(255 * (1 - t));
    return `rgb(${r},${g},${bch})`;
  }

  const surface = document.createElement("canvas");
  surface.width = canvas.width;
  surface.height = canvas.height;
  (function paintSurface() {
    const sctx = surface.getContext("2d");
    const W = surface.width, H = surface.height;
    const cells = 40;
    let maxL = 0;
    const losses = [];
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const ww = wMin + (i + 0.5) / cells * (wMax - wMin);
        const bb = bMin + (j + 0.5) / cells * (bMax - bMin);
        const l = loss(ww, bb);
        losses.push(l);
        if (l > maxL) maxL = l;
      }
    }
    const cellW = (W - 2 * pad) / cells;
    const cellH = (H - 2 * pad) / cells;
    let idx = 0;
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        const t = Math.sqrt(Math.min(1, losses[idx++] / (maxL || 1)));
        sctx.fillStyle = colorFor(t);
        sctx.fillRect(pad + i * cellW, H - pad - (j + 1) * cellH, cellW + 1, cellH + 1);
      }
    }
  })();

  const titles = ["1 · Forward pass", "2 · Loss", "3 · Backward pass", "4 · Update"];

  function stageText(i, w, b, eta) {
    const yhat = w * x + b;
    const d = 2 * (yhat - y);
    const [gw, gb] = grad(w, b);
    if (i === 0) return `ŷ = w·x + b = ${f(w)} × ${x} + ${f(b)} = <b>${f(yhat)}</b> &nbsp;(true answer y = ${y})`;
    if (i === 1) return `L = (ŷ − y)² = (${f(yhat)} − ${y})² = <b>${f(loss(w, b))}</b>`;
    if (i === 2) return `∂L/∂ŷ = 2(ŷ − y) = ${f(d)} &nbsp;→&nbsp; ∂L/∂w = ${f(d)} × ${x} = <b>${f(gw)}</b> &nbsp;,&nbsp; ∂L/∂b = ${f(d)} × 1 = <b>${f(gb)}</b>`;
    return `w ← ${f(w)} − ${f(eta)} × (${f(gw)}) = <b>${f(w - eta * gw)}</b> &nbsp;,&nbsp; b ← ${f(b)} − ${f(eta)} × (${f(gb)}) = <b>${f(b - eta * gb)}</b>`;
  }

  function draw() {
    const w = parseFloat(wInput.value);
    const b = parseFloat(bInput.value);
    const eta = parseFloat(etaInput.value);
    wVal.textContent = f(w);
    bVal.textContent = f(b);
    etaVal.textContent = f(eta);

    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(surface, 0, 0);
    ctx.strokeStyle = "#1f2430";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);

    if (trail.length > 0) {
      ctx.strokeStyle = "#1f2430";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      trail.concat([[w, b]]).forEach(([tw, tb], i) => {
        const [px, py] = toPx(tw, tb);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    const [px, py] = toPx(w, b);
    const [gw, gb] = grad(w, b);

    if (stage >= 2) {
      const gnorm = Math.hypot(gw, gb) || 1;
      const arrowLen = 35;
      ctx.strokeStyle = "#1d4ed8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px - (gw / gnorm) * arrowLen, py + (gb / gnorm) * arrowLen);
      ctx.stroke();
    }
    if (stage >= 3) {
      const [nx, ny] = toPx(clampW(w - eta * gw), clampB(b - eta * gb));
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#1f2430";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(nx, ny); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.beginPath(); ctx.arc(nx, ny, 6, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = "#1f2430"; ctx.lineWidth = 2; ctx.stroke();
    }

    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "#1f2430"; ctx.lineWidth = 2; ctx.stroke();

    stagesEl.innerHTML = titles.map((t, i) => {
      const shown = i <= stage;
      const style = `padding:.2rem .5rem;border-left:3px solid ${i === stage ? "#1d4ed8" : "transparent"};background:${i === stage ? "#eef4ff" : "transparent"};opacity:${shown ? 1 : 0.4}`;
      return `<div style="${style}"><b>${t}</b>${shown ? " — " + stageText(i, w, b, eta) : ""}</div>`;
    }).join("");
    nextBtn.textContent = stage < 3 ? "Next stage ▸" : "Apply update ▸ new round";
    statsEl.innerHTML = `round ${round} &nbsp;|&nbsp; current loss L = ${f(loss(w, b))}`;
  }

  function next() {
    if (stage < 3) { stage++; draw(); return; }
    const w = parseFloat(wInput.value);
    const b = parseFloat(bInput.value);
    const eta = parseFloat(etaInput.value);
    const [gw, gb] = grad(w, b);
    trail.push([w, b]);
    wInput.value = clampW(w - eta * gw);
    bInput.value = clampB(b - eta * gb);
    stage = 0;
    round++;
    draw();
  }

  function restart() { trail = []; stage = 0; round = 0; draw(); }

  nextBtn.addEventListener("click", next);
  resetBtn.addEventListener("click", () => {
    wInput.value = 0.5; bInput.value = 0; etaInput.value = 0.1; restart();
  });
  [wInput, bInput].forEach((el) => el.addEventListener("input", restart));
  etaInput.addEventListener("input", draw);

  draw();
})();
</script>

### 3.3.4 [Gradient descent](#glossary-gradient-descent) and the [learning rate](#glossary-learning-rate)

**Gradient descent** repeats the update step: move each parameter a small step *against* its gradient.

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}
$$

The **learning rate** $\eta$ is the size of the step:

- **Too large:** the steps overshoot the valley and the loss bounces around or even grows.
- **Too small:** learning is very slow and may get stuck.
- **Just right:** the loss decreases steadily.

> **Running example — stage 4.** Using the gradients from stage 3 and $\eta = 0.1$:
> $w \leftarrow 0.5 - 0.1 \times (-8) = 1.3$ and $b \leftarrow 0 - 0.1 \times (-4) = 0.4$.
> Running the forward pass again with the new weights gives $\hat{y} = 1.3 \times 2 + 0.4 = 3.0$, which is exactly the target: the loss has dropped from 4 to 0 in a single step. The loop then starts over from stage 1 with the new weights.
>
> This one-step success is special to $\eta = 0.1$ and to this tiny neuron; in real networks many small steps are needed. For this neuron, one update multiplies the error $\hat{y}-y$ by $(1 - 10\eta)$ (the 10 comes from $2(x^2+1)$). At $\eta=0.1$ that factor is 0, at $\eta=0.2$ it is $-1$ (the error flips sign every step and never shrinks), and above $0.2$ it is larger than 1 in size, so the error grows. You can see all three behaviours in the stepper above; the widget below shows the same effect on an even simpler curve.

<div id="eta-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>η <input type="range" id="eta-eta" min="0.01" max="1.3" step="0.01" value="0.2"> <span id="eta-eta-val">0.20</span></label>
<button type="button" id="eta-step">Step</button>
<button type="button" id="eta-play">Play</button>
<button type="button" id="eta-reset">Reset</button>
</div>
<canvas id="eta-canvas" width="560" height="320"></canvas>
<p class="aisw-hint">A toy loss L(θ) = (θ-3)² + 0.5, starting far from the minimum. Try η around 0.1-0.3 (steady), then push past η = 1.0 to watch it diverge, and down near η = 0.02 to watch it crawl.</p>
<div id="eta-stats" class="aisw-stats"></div>
</div>

<script>
(function () {
  const canvas = document.getElementById("eta-canvas");
  const ctx = canvas.getContext("2d");
  const statsEl = document.getElementById("eta-stats");
  const etaInput = document.getElementById("eta-eta");
  const etaVal = document.getElementById("eta-eta-val");
  const stepBtn = document.getElementById("eta-step");
  const playBtn = document.getElementById("eta-play");
  const resetBtn = document.getElementById("eta-reset");

  const thetaMin = -2, thetaMax = 8;
  const pad = 36;
  const minimum = 3;
  let theta = 8;
  let history = [theta];
  let timer = null;

  function L(t) { return (t - minimum) ** 2 + 0.5; }
  function gradL(t) { return 2 * (t - minimum); }

  function yRange() {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i <= 100; i++) {
      const t = thetaMin + (i / 100) * (thetaMax - thetaMin);
      const v = L(t);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    history.forEach((t) => { const v = L(t); if (v > hi) hi = v; });
    return [lo, hi * 1.05];
  }

  function toPx(t, l, yMin, yMax, w, h) {
    return [
      pad + (t - thetaMin) / (thetaMax - thetaMin) * (w - 2 * pad),
      h - pad - (l - yMin) / (yMax - yMin) * (h - 2 * pad),
    ];
  }

  function classify(eta) {
    const r = Math.abs(1 - 2 * eta);
    if (r >= 1) return "diverging";
    if (r >= 0.9) return "oscillating, barely bounded";
    if (r <= 0.3) return "converging quickly";
    return "converging steadily";
  }

  function draw() {
    const eta = parseFloat(etaInput.value);
    etaVal.textContent = eta.toFixed(2);

    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    const [yMin, yMax] = yRange();

    ctx.strokeStyle = "#d9dde5";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2 * pad, h - 2 * pad);

    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const t = thetaMin + (i / 100) * (thetaMax - thetaMin);
      const [px, py] = toPx(t, L(t), yMin, yMax, w, h);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    if (history.length > 1) {
      ctx.strokeStyle = "rgba(31,36,48,0.4)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      history.forEach((t, i) => {
        const [px, py] = toPx(t, L(t), yMin, yMax, w, h);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    const [px, py] = toPx(theta, L(theta), yMin, yMax, w, h);
    ctx.fillStyle = "#ef4444";
    ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI); ctx.fill();

    statsEl.innerHTML = `θ = ${theta.toFixed(3)} &nbsp;|&nbsp; L(θ) = ${L(theta).toFixed(3)} &nbsp;|&nbsp; step ${history.length - 1} &nbsp;|&nbsp; ${classify(eta)}`;
  }

  function step() {
    const eta = parseFloat(etaInput.value);
    theta = theta - eta * gradL(theta);
    theta = Math.min(1e6, Math.max(-1e6, theta));
    history.push(theta);
    if (history.length > 60) history.shift();
    draw();
  }

  stepBtn.addEventListener("click", step);
  playBtn.addEventListener("click", () => {
    if (timer) {
      clearInterval(timer); timer = null; playBtn.textContent = "Play";
    } else {
      playBtn.textContent = "Pause";
      timer = setInterval(() => {
        step();
        if (Math.abs(theta - minimum) > 1000) { clearInterval(timer); timer = null; playBtn.textContent = "Play"; }
      }, 350);
    }
  });
  resetBtn.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "Play"; }
    theta = 8; history = [theta]; draw();
  });
  etaInput.addEventListener("input", () => {
    if (timer) { clearInterval(timer); timer = null; playBtn.textContent = "Play"; }
    theta = 8; history = [theta]; draw();
  });

  draw();
})();
</script>

#### Interactive: comparing four CNN architectures

Explore how different CNN architectures are organised, from a classic image classifier to modern networks used for object detection.



### 3.3.5 [Mini-batches](#glossary-sgd) and [Stochastic Gradient Descent (SGD)](#glossary-sgd)

Computing the gradient over the *entire* dataset for every single step is too slow for large datasets. Instead, we use a small random subset — a **mini-batch**, typically 16 to 512 examples — to estimate the gradient. This is **Stochastic Gradient Descent (SGD)**. Each step is a little noisy, but steps are much cheaper, and the noise even helps the network avoid poor solutions.

Some vocabulary:

- **Batch size:** the number of examples in each mini-batch.
- **Iteration:** one update using one mini-batch.
- **Epoch:** one full pass through the entire training set.

#### Try it: mini-batch updates

<div id="batch-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Batch size <select id="batch-size"><option>1</option><option>2</option><option>4</option><option selected>8</option></select></label>
<button type="button" id="batch-step">Next update</button>
<button type="button" id="batch-reset">Reset</button>
</div>
<canvas id="batch-canvas" width="560" height="250" role="img" aria-label="Training targets and current prediction"></canvas>
<p class="aisw-hint">Fit a constant prediction to eight fixed targets with squared-error loss. Blue dots are the current mini-batch; grey dots are the remaining examples. Compare the mini-batch gradient with the full-dataset gradient. A fixed shuffled order is repeated each epoch so the experiment is reproducible. Changing batch size resets it.</p>
<div id="batch-stats" class="aisw-stats" aria-live="polite"></div>
</div>

### 3.3.6 Optimisation algorithms

An [**optimiser**](#glossary-optimiser) is the rule that turns gradients into parameter updates. Plain SGD is the starting point, and the others improve on it.

In the update rules below, [$g$](#notation-gradient) denotes the current gradient used to calculate a parameter update.

| Optimiser | Idea in plain words | Update rule (simplified) |
|---|---|---|
| **SGD** | Step downhill using the current mini-batch gradient $g$. | $\theta \leftarrow \theta - \eta\, g$ |
| **SGD with momentum** | Like a rolling ball that keeps some speed from previous steps; it smooths out noise and rolls through small bumps. | $v \leftarrow \beta v + g$; $\theta \leftarrow \theta - \eta\, v$ |
| **RMSprop** | Gives each parameter its own step size: parameters with consistently large gradients take smaller steps. | $s \leftarrow \rho s + (1-\rho)g^2$; $\theta \leftarrow \theta - \eta\,\dfrac{g}{\sqrt{s}+\epsilon}$ |
| **Adam** | Combines momentum (average direction) with RMSprop (per-parameter step size). The most common default. | $m \leftarrow \beta_1 m + (1-\beta_1)g$; $s \leftarrow \beta_2 s + (1-\beta_2)g^2$; $\theta \leftarrow \theta - \eta\,\dfrac{\hat{m}}{\sqrt{\hat{s}}+\epsilon}$ |
| **AdamW** | Adam with [weight decay](#glossary-regularisation) (Section 3.4.4) applied correctly, as a separate step. Widely used for transformers. | Adam update plus $-\eta\lambda\theta$ |

> **Rule of thumb:** Adam/AdamW usually works well with little tuning, which makes it a good first choice. SGD with momentum often needs more tuning but is still popular for training CNNs.

#### Try it: compare optimisers

<div id="optim-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Optimiser <select id="optim-method"><option value="sgd">SGD</option><option value="momentum">SGD with momentum</option><option value="rmsprop">RMSprop</option><option value="adam">Adam</option></select></label>
<label>Learning rate <input type="range" id="optim-rate" min="0.01" max="0.25" step="0.01" value="0.10"> <span id="optim-rate-value"></span></label>
<button type="button" id="optim-step">Step</button>
<button type="button" id="optim-reset">Reset</button>
</div>
<canvas id="optim-canvas" width="560" height="300" role="img" aria-label="Optimiser path on a two-dimensional quadratic loss"></canvas>
<p class="aisw-hint">Minimise L(x, y) = (x² + 10y²)/2 from (4, 4). Each point is an update; the minimum is at (0, 0). All methods use the exact same gradient, with no mini-batch noise. Changing the method or learning rate restarts the path.</p>
<div id="optim-stats" class="aisw-stats" aria-live="polite"></div>
</div>

### 3.3.7 [Learning-rate schedules](#glossary-learning-rate-schedule)

The best learning rate is not constant during training: large steps are useful at the start, and small steps are better at the end for settling precisely into a good solution. A **schedule** changes $\eta$ over time:

- **Step decay:** multiply $\eta$ by a factor (e.g. 0.1) every few epochs.
- **Cosine annealing:** decrease $\eta$ smoothly following a cosine curve: $\eta_t = \eta_{min} + \tfrac{1}{2}(\eta_{max}-\eta_{min})\big(1+\cos(\pi t/T)\big)$.
- **Warm-up:** start with a very small $\eta$ and increase it over the first few iterations, which avoids unstable early steps.
- **One-cycle:** warm up to a maximum and then anneal down to near zero.

#### Try it: inspect schedules

<div id="schedule-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Schedule <select id="schedule-kind"><option value="step">Step decay</option><option value="cosine">Cosine annealing</option><option value="warmup">Warm-up then constant</option><option value="cycle">One-cycle</option></select></label>
<label>Iteration <input type="range" id="schedule-iteration" min="0" max="40" step="1" value="0"> <span id="schedule-iteration-value"></span></label>
</div>
<canvas id="schedule-canvas" width="560" height="250" role="img" aria-label="Learning rate against iteration"></canvas>
<p class="aisw-hint">All examples use 40 iterations and a maximum learning rate of 0.1. Move the slider to inspect the rate at each iteration; these are illustrative schedules, not a training run.</p>
<div id="schedule-stats" class="aisw-stats" aria-live="polite"></div>
</div>

### 3.3.8 [Gradient clipping](#glossary-gradient-clipping)

Occasionally a gradient becomes unusually large and would throw the weights far off course. **Gradient clipping** caps its size: if the length of the gradient $\lVert g \rVert$ exceeds a threshold $c$, the gradient is scaled down:

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right)
$$

It is especially common when training recurrent networks and transformers.

### 3.3.9 [Mixed-precision training](#glossary-mixed-precision)

Numbers in a computer can be stored with different precision: 32 bits (FP32) or 16 bits (FP16 or BF16). **Mixed-precision training** uses the smaller 16-bit numbers for most calculations and keeps 32-bit numbers where precision matters. This makes training faster and uses about half the memory, with little or no loss in quality. (Using lower precision *after* training, to make models smaller and faster on vehicle hardware, is called quantisation and is covered in Chapter 6.)

## 3.4 Making the Network Work on New Data: Generalisation

The goal of training is not to do well on the training examples, but to do well on **new data** the network has never seen. This ability is called **generalisation**.

### 3.4.1 Training, validation and test sets

The available data is split into three separate piles:

| Set | Used for | Analogy |
|---|---|---|
| **Training set** | Adjusting the weights | The exercises a student practises on |
| **Validation set** | Choosing hyperparameters and deciding when to stop | A practice exam |
| **Test set** | The final, one-time evaluation | The real exam |

The test set must stay untouched during development; otherwise the final score is no longer an honest estimate of performance on new data.

### 3.4.2 [Overfitting](#glossary-overfitting) and [underfitting](#glossary-overfitting)

- **Underfitting:** the model is too simple (or trained too little) to learn the pattern. It performs poorly on both training and validation data.
- **Overfitting:** the model learns the training examples — including their noise and accidents — by heart, instead of the general idea. It performs very well on training data but poorly on validation data. It is like a student who memorises past exam answers without understanding the subject.

Overfitting is more likely when the model has many parameters compared with the amount (or variety) of data.

#### Try it: fit and generalisation

<div id="fit-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Model <select id="fit-model"><option value="linear">Linear (degree 1)</option><option value="quadratic" selected>Quadratic (degree 2)</option><option value="interpolating">Interpolating (degree 7)</option></select></label>
</div>
<canvas id="fit-canvas" width="560" height="300" role="img" aria-label="Fitted curve compared with training and validation points"></canvas>
<p class="aisw-hint">The same eight noisy training points are fitted in each case. Blue points are used to fit the curve; orange points are held out for validation. The degree-7 curve passes through every training point but can oscillate between them.</p>
<div id="fit-stats" class="aisw-stats" aria-live="polite"></div>
</div>

### 3.4.3 Learning curves

A **learning curve** plots the training loss and the validation loss against the number of epochs. It is the main diagnostic tool:

| What you see | What it means | What to try |
|---|---|---|
| Both losses high | Underfitting | Larger model, train longer, higher learning rate |
| Training loss low, validation loss high (a gap) | Overfitting | More data, augmentation, [regularisation](#glossary-regularisation), smaller model |
| Both losses low and close together | Good fit | Done (or fine-tune) |
| Validation loss starts to rise while training loss keeps falling | Overfitting has begun | [Early stopping](#glossary-regularisation) |

#### Try it: read learning curves

<div id="curves-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Scenario <select id="curves-scenario"><option value="under">Underfitting</option><option value="good">Good fit</option><option value="over" selected>Overfitting</option></select></label>
<label>Epoch <input type="range" id="curves-epoch" min="1" max="40" step="1" value="20"> <span id="curves-epoch-value"></span></label>
</div>
<canvas id="curves-canvas" width="560" height="250" role="img" aria-label="Illustrative training and validation loss curves by epoch"></canvas>
<p class="aisw-hint">These curves are synthetic examples, not measured training runs. Blue is training loss; orange is validation loss. The reported best epoch is where the validation loss is lowest over the displayed 40 epochs.</p>
<div id="curves-stats" class="aisw-stats" aria-live="polite"></div>
</div>

### 3.4.4 Regularisation

**Regularisation** is a family of techniques that deliberately make it harder for the network to memorise, so that it learns the general pattern.

- [**L2 regularisation**](#glossary-regularisation) / [**weight decay**](#glossary-regularisation). Add a penalty for large weights to the loss: $\mathcal{L}_{total} = \mathcal{L} + \lambda \lVert \mathbf{w}\rVert_2^2$. Large weights are discouraged, which keeps the model simpler and smoother. [$\lambda$](#notation-weight-decay) controls the strength.
- **L1 regularisation.** Penalty $\lambda \lVert \mathbf{w}\rVert_1$ (sum of absolute values). It pushes many weights to exactly zero, which gives sparser models.
- [**Dropout**](#glossary-regularisation). During training, each neuron is randomly switched off with probability $p$ (e.g. 0.5) in every step. The network cannot rely on any single neuron, so it learns more robust, redundant features. At inference time, all neurons are used.
- [**Data augmentation**](#glossary-data-augmentation). Create extra, slightly different training examples from the existing ones — flipping, cropping, changing brightness, adding noise. The network sees more variety without more data collection. Covered in detail in Chapter 11.
- [**Early stopping**](#glossary-regularisation). Monitor the validation loss and stop training when it stops improving, keeping the best version of the model.
- **Label smoothing.** Instead of telling the network "this is 100% a car", train with "this is 90% a car". This stops the network from becoming over-confident.
- **Stochastic depth.** Randomly skip entire residual blocks during training (a "dropout for layers"), used in very deep networks.

### 3.4.5 Hyperparameters

**Hyperparameters** are the settings chosen by the engineer *before* training. The most important are:

| Hyperparameter | What it controls | Typical effect |
|---|---|---|
| **Learning rate** $\eta$ | Step size of each update | Usually the single most important setting |
| **Batch size** | Examples per update | Larger: smoother but needs more memory; smaller: noisier |
| **Number of epochs** | How long to train | Too few: underfit; too many: overfit |
| **Depth and width** | Number of layers / neurons per layer | More capacity, but a higher risk of overfitting |
| **Optimiser and its settings** | e.g. Adam's $\beta_1$, $\beta_2$, momentum | Affects speed and stability |
| **Regularisation strength** | Weight-decay $\lambda$, dropout rate $p$ | Trade-off between underfitting and overfitting |

#### Try it: change training hyperparameters

<div id="hyper-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Learning rate <select id="hyper-rate"><option value="0.01">0.01</option><option value="0.05" selected>0.05</option><option value="0.1">0.1</option><option value="0.5">0.5</option><option value="1.5">1.5</option></select></label>
<label>Batch size <select id="hyper-batch"><option value="1">1</option><option value="2">2</option><option value="4" selected>4</option><option value="8">8 (full batch)</option></select></label>
<label>Epochs <input type="range" id="hyper-epochs" min="1" max="200" step="1" value="80"> <span id="hyper-epochs-value"></span></label>
<label>Hidden layers <select id="hyper-depth"><option value="1" selected>1</option><option value="2">2</option><option value="3">3</option></select></label>
<label>Neurons per layer <select id="hyper-width"><option value="2">2</option><option value="4" selected>4</option><option value="8">8</option></select></label>
<label>Optimiser <select id="hyper-optimiser"><option value="sgd" selected>SGD</option><option value="momentum">SGD + momentum (0.9)</option><option value="adam">Adam (β₁=0.9, β₂=0.999)</option></select></label>
<label>L2 strength λ <select id="hyper-lambda"><option value="0" selected>0</option><option value="0.01">0.01</option><option value="0.1">0.1</option><option value="1">1</option></select></label>
</div>
<canvas id="hyper-canvas" width="560" height="250" role="img" aria-label="Training and validation mean squared error for selected hyperparameters"></canvas>
<p class="aisw-hint">Train a small tanh neural network from scratch on eight noisy synthetic training points; eight separate points are used for validation. Blue is training MSE and orange is validation MSE, clipped at 1 on the plot. The L2 penalty applies to weights, not biases, and is excluded from both displayed errors. Each change restarts with a fixed seed and reproducible shuffled batches. An epoch visits all eight training examples, so smaller batches make more updates per epoch. No test set is used.</p>
<div id="hyper-stats" class="aisw-stats" aria-live="polite"></div>
</div>

<script src="training-widgets.js"></script>

### 3.4.6 Tuning hyperparameters

Finding good hyperparameters is called **tuning**. Each candidate is evaluated by training a model and measuring its validation performance.

| Method | How it works | Remarks |
|---|---|---|
| **Manual tuning** | Adjust settings by hand, guided by learning curves and experience | Good for first experiments |
| **Grid search** | Try every combination from a fixed list of values | Simple, but the number of runs explodes as settings are added |
| **Random search** | Try randomly chosen combinations | Often finds good settings faster than grid search |
| **Bayesian optimisation** | Use the results of earlier trials to decide which settings to try next | Efficient when each training run is expensive |

> **Tip:** Change one thing at a time when learning, and always record what was tried. **Experiment-tracking tools** such as Weights & Biases, MLflow, or Optuna (a tuning library) store each run's settings and results, so experiments can be compared and repeated. Fixing the random seed makes runs repeatable.

## 3.5 Architecture Families

The building blocks in Section 3.2 combine into architecture families suited to different kinds of data. Here we focus on the core idea and automotive use; the comparison in Section 3.7 summarises their trade-offs.

### 3.5.1 Multi-Layer Perceptron (MLP)

An **MLP** (fully connected or dense network) passes a vector through layers where every neuron connects to every neuron in the next layer (Section 3.2.3). MLPs work well with tabular sensor data and often form prediction heads, but are inefficient on raw images because they do not exploit pixel neighbourhoods.

#### Interactive: inputs through a small MLP

```{=html}
<div id="mlp-explorer" class="aisafety-widget">
<div class="aisw-controls">
<label>Relative speed <input type="range" id="mlp-speed" min="0" max="1" step="0.1" value="0.4"> <span id="mlp-speed-value"></span></label>
<label>Proximity <input type="range" id="mlp-proximity" min="0" max="1" step="0.1" value="0.6"> <span id="mlp-proximity-value"></span></label>
<button type="button" id="mlp-step">Next layer</button>
<button type="button" id="mlp-reset">Reset</button>
</div>
<div class="architecture-graph-scroll">
<svg class="architecture-graph" viewBox="0 0 600 270" aria-hidden="true">
<text class="graph-heading" x="85" y="24">Inputs</text><text class="graph-heading" x="300" y="24">Hidden · ReLU</text><text class="graph-heading" x="515" y="24">Output</text>
<g class="graph-edges graph-stage-1">
<line x1="120" y1="85" x2="265" y2="85"/><line x1="120" y1="85" x2="265" y2="195"/>
<line x1="120" y1="195" x2="265" y2="85"/><line x1="120" y1="195" x2="265" y2="195"/>
</g>
<g class="graph-edges graph-stage-2"><line x1="335" y1="85" x2="480" y2="140"/><line x1="335" y1="195" x2="480" y2="140"/></g>
<g class="graph-node graph-input"><circle cx="85" cy="85" r="34"/><text id="mlp-graph-input-1" x="85" y="90"></text></g>
<g class="graph-node graph-input"><circle cx="85" cy="195" r="34"/><text id="mlp-graph-input-2" x="85" y="200"></text></g>
<g class="graph-node graph-stage-1"><circle cx="300" cy="85" r="34"/><text id="mlp-graph-hidden-1" x="300" y="90"></text></g>
<g class="graph-node graph-stage-1"><circle cx="300" cy="195" r="34"/><text id="mlp-graph-hidden-2" x="300" y="200"></text></g>
<g class="graph-node graph-stage-2"><circle cx="515" cy="140" r="34"/><text id="mlp-graph-output" x="515" y="145"></text></g>
<text class="graph-caption" x="85" y="135">speed</text><text class="graph-caption" x="85" y="245">proximity</text>
<text class="graph-caption" x="300" y="135">neuron 1</text><text class="graph-caption" x="300" y="245">neuron 2</text>
<text class="graph-caption" x="515" y="195">weighted sum</text>
</svg>
</div>
<div class="architecture-flow" aria-label="MLP processing stages">
<div><strong>1. Inputs</strong><output id="mlp-input"></output></div>
<div><strong>2. Hidden layer</strong><output id="mlp-hidden"></output></div>
<div><strong>3. Output</strong><output id="mlp-output"></output></div>
</div>
<p class="aisw-hint">Move the sliders to recalculate every node immediately, or press Reset and step through the fully connected 2 → 2 → 1 network one layer at a time. Hidden neurons use ReLU; the output is a weighted sum. Values are dimensionless and weights are hand-set: the output is not a driving decision or safety score.</p>
<div id="mlp-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

### 3.5.2 Convolutional Neural Networks (CNN)

**CNNs** reuse the same convolution filters across an image (Section 3.2.5), building feature maps that later layers combine into more complex patterns.

```
Image -> [Conv+ReLU] -> [Pool] -> [Conv+ReLU] -> [Pool] -> ... -> [MLP head] -> "pedestrian"
         edges            ↓         shapes          ↓                 decision
```

**Well-known CNNs.**

| Network | Key idea |
|---|---|
| **ResNet / EfficientNet** | Image feature extractors used as network backbones |
| [**YOLO**](#glossary-yolo) | Predicts object boxes and classes in one network evaluation |
| **U-Net** | Produces a label for each image pixel |

CNNs are used for camera-based object detection, sign and lane recognition, and image segmentation.

#### Interactive: from a camera image to CNN feature maps

This visual explorer uses synthetic scenes and hand-set filters to illustrate convolution, feature maps, ReLU, stride, padding and max pooling. It is a teaching model, not a trained classifier.

```{=html}
<iframe
  src="cnn-feature-explorer.html"
  title="Interactive CNN feature-map explorer"
  width="100%"
  height="820"
  loading="lazy"
  style="border:0; border-radius:12px;">
</iframe>
```

For an optional coding exercise, [open the CNN architecture notebook in Colab](https://colab.research.google.com/github/bogdangradinaru/ai-safety/blob/main/notebooks/part-1/07-model-architectures.ipynb) or [view it on GitHub](https://github.com/bogdangradinaru/ai-safety/blob/main/notebooks/part-1/07-model-architectures.ipynb).

The [CNN architecture explorer](cnn-architecture-explorer.html) is available as an optional interactive resource.

### 3.5.3 Recurrent Networks: [RNN](#glossary-rnn), [LSTM](#glossary-rnn) and [GRU](#glossary-rnn)

**RNNs** process sequences one step at a time, carrying a hidden state forward as memory. **LSTMs** and **GRUs** use gates to control what information is retained; they are used for tasks such as trajectory prediction and tracking. Their sequential computation can be slower to train than transformers (Section 3.5.5).

#### Interactive: a recurrent state over time

```{=html}
<div id="rnn-explorer" class="aisafety-widget">
<div class="aisw-controls">
<label>Motion sequence <select id="rnn-sequence"><option value="approach">Approaching</option><option value="recede">Moving away</option><option value="alternating">Alternating</option></select></label>
<label>Memory weight <input type="range" id="rnn-memory" min="0" max="1" step="0.1" value="0.5"> <span id="rnn-memory-value"></span></label>
<button type="button" id="rnn-step">Next time step</button>
<button type="button" id="rnn-reset">Reset</button>
</div>
<div class="architecture-graph-scroll">
<svg class="architecture-graph" viewBox="0 0 600 230" aria-hidden="true">
<defs><marker id="rnn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8091a5"/></marker></defs>
<text class="graph-heading" x="300" y="24">The same recurrent cell at successive time steps →</text>
<g class="graph-edges" marker-end="url(#rnn-arrow)"><line x1="113" y1="126" x2="207" y2="126"/><line x1="253" y1="126" x2="347" y2="126"/><line x1="393" y1="126" x2="487" y2="126"/></g>
<text class="graph-caption" x="25" y="132">h₀ = 0</text><line class="graph-initial-edge" x1="48" y1="126" x2="67" y2="126" marker-end="url(#rnn-arrow)"/>
<g id="rnn-graph-cell-1" class="graph-cell"><text class="graph-caption" x="90" y="51">t = 1</text><text id="rnn-graph-input-1" class="graph-caption" x="90" y="74"></text><line class="graph-input-edge" x1="90" y1="81" x2="90" y2="99"/><rect x="67" y="100" width="46" height="52" rx="8"/><text id="rnn-graph-state-1" class="graph-value" x="90" y="132"></text><text class="graph-caption" x="90" y="180">h₁</text></g>
<g id="rnn-graph-cell-2" class="graph-cell"><text class="graph-caption" x="230" y="51">t = 2</text><text id="rnn-graph-input-2" class="graph-caption" x="230" y="74"></text><line class="graph-input-edge" x1="230" y1="81" x2="230" y2="99"/><rect x="207" y="100" width="46" height="52" rx="8"/><text id="rnn-graph-state-2" class="graph-value" x="230" y="132"></text><text class="graph-caption" x="230" y="180">h₂</text></g>
<g id="rnn-graph-cell-3" class="graph-cell"><text class="graph-caption" x="370" y="51">t = 3</text><text id="rnn-graph-input-3" class="graph-caption" x="370" y="74"></text><line class="graph-input-edge" x1="370" y1="81" x2="370" y2="99"/><rect x="347" y="100" width="46" height="52" rx="8"/><text id="rnn-graph-state-3" class="graph-value" x="370" y="132"></text><text class="graph-caption" x="370" y="180">h₃</text></g>
<g id="rnn-graph-cell-4" class="graph-cell"><text class="graph-caption" x="510" y="51">t = 4</text><text id="rnn-graph-input-4" class="graph-caption" x="510" y="74"></text><line class="graph-input-edge" x1="510" y1="81" x2="510" y2="99"/><rect x="487" y="100" width="46" height="52" rx="8"/><text id="rnn-graph-state-4" class="graph-value" x="510" y="132"></text><text class="graph-caption" x="510" y="180">h₄</text></g>
<text class="graph-caption" x="300" y="211">Each state passes to the next cell</text>
</svg>
</div>
<div class="architecture-flow" aria-label="Recurrent processing stages">
<div><strong>1. Input sequence</strong><output id="rnn-input"></output></div>
<div><strong>2. Previous state</strong><output id="rnn-previous"></output></div>
<div><strong>3. New state</strong><output id="rnn-state"></output></div>
</div>
<p class="aisw-hint">Step through four dimensionless motion measurements, starting with state 0. Each box shows the state produced at that time; a line carries it to the next copy of the same cell. The cell computes h = tanh(input + memory weight × previous state). Moving the memory-weight slider recalculates all four states; changing the sequence resets them. Press Reset to step through again. This is a simplified RNN, not an LSTM or GRU and not a trained tracker.</p>
<div id="rnn-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

### 3.5.4 [Encoder–Decoder Networks](#glossary-encoder-decoder) and [Autoencoders](#glossary-encoder-decoder)

An **encoder** maps input to a compact [latent representation](#glossary-encoder-decoder); a **decoder** turns it into the desired output.

```
Input -> [ Encoder ] -> compact summary (latent) -> [ Decoder ] -> Output
```

In an **autoencoder**, the decoder reconstructs the input, enabling unsupervised representation learning and anomaly detection. Encoder–decoder designs also support pixel-wise segmentation (for example, U-Net) and sequence prediction.

#### Interactive: a bottleneck and reconstruction

```{=html}
<div id="encoder-explorer" class="aisafety-widget">
<div class="aisw-controls">
<label>Sensor 1 <input type="range" id="encoder-input-1" min="0" max="1" step="0.1" value="0.2"> <span id="encoder-value-1"></span></label>
<label>Sensor 2 <input type="range" id="encoder-input-2" min="0" max="1" step="0.1" value="0.8"> <span id="encoder-value-2"></span></label>
<label>Sensor 3 <input type="range" id="encoder-input-3" min="0" max="1" step="0.1" value="0.4"> <span id="encoder-value-3"></span></label>
<label>Sensor 4 <input type="range" id="encoder-input-4" min="0" max="1" step="0.1" value="0.6"> <span id="encoder-value-4"></span></label>
<button type="button" id="encoder-step">Next stage</button>
<button type="button" id="encoder-reset">Reset</button>
</div>
<div class="architecture-graph-scroll">
<svg class="architecture-graph" viewBox="0 0 600 320" aria-hidden="true">
<text class="graph-heading" x="85" y="24">4 inputs</text><text class="graph-heading" x="300" y="24">2 latent values</text><text class="graph-heading" x="515" y="24">4 reconstructed</text>
<g class="graph-edges graph-stage-1">
<line x1="113" y1="70" x2="272" y2="110"/><line x1="113" y1="130" x2="272" y2="110"/>
<line x1="113" y1="190" x2="272" y2="230"/><line x1="113" y1="250" x2="272" y2="230"/>
</g>
<g class="graph-edges graph-stage-2">
<line x1="328" y1="110" x2="487" y2="70"/><line x1="328" y1="110" x2="487" y2="130"/>
<line x1="328" y1="230" x2="487" y2="190"/><line x1="328" y1="230" x2="487" y2="250"/>
</g>
<g class="graph-node graph-input"><circle cx="85" cy="70" r="27"/><text id="encoder-graph-input-1" x="85" y="75"></text></g>
<g class="graph-node graph-input"><circle cx="85" cy="130" r="27"/><text id="encoder-graph-input-2" x="85" y="135"></text></g>
<g class="graph-node graph-input"><circle cx="85" cy="190" r="27"/><text id="encoder-graph-input-3" x="85" y="195"></text></g>
<g class="graph-node graph-input"><circle cx="85" cy="250" r="27"/><text id="encoder-graph-input-4" x="85" y="255"></text></g>
<g class="graph-node graph-stage-1"><circle cx="300" cy="110" r="27"/><text id="encoder-graph-latent-1" x="300" y="115"></text></g>
<g class="graph-node graph-stage-1"><circle cx="300" cy="230" r="27"/><text id="encoder-graph-latent-2" x="300" y="235"></text></g>
<g class="graph-node graph-stage-2"><circle cx="515" cy="70" r="27"/><text id="encoder-graph-output-1" x="515" y="75"></text></g>
<g class="graph-node graph-stage-2"><circle cx="515" cy="130" r="27"/><text id="encoder-graph-output-2" x="515" y="135"></text></g>
<g class="graph-node graph-stage-2"><circle cx="515" cy="190" r="27"/><text id="encoder-graph-output-3" x="515" y="195"></text></g>
<g class="graph-node graph-stage-2"><circle cx="515" cy="250" r="27"/><text id="encoder-graph-output-4" x="515" y="255"></text></g>
<text class="graph-caption" x="190" y="300">Encoder: pair averages →</text><text class="graph-caption" x="410" y="300">Decoder: repeat →</text>
</svg>
</div>
<div class="architecture-flow" aria-label="Encoder-decoder processing stages">
<div><strong>1. Inputs</strong><output id="encoder-input"></output></div>
<div><strong>2. Latent representation</strong><output id="encoder-latent"></output></div>
<div><strong>3. Reconstruction</strong><output id="encoder-output"></output></div>
</div>
<p class="aisw-hint">The fixed encoder averages adjacent pairs of four dimensionless sensor values; the decoder repeats each average. Moving a slider immediately recalculates the entire 4 → 2 → 4 transformation. Press Reset to step through it instead. This is a hand-set linear autoencoder, not a trained model; a bottleneck can lose information.</p>
<div id="encoder-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

<script src="architecture-widgets.js"></script>

### 3.5.5 Transformers and Attention

**Transformers** use attention to relate tokens—such as image patches, objects or map elements—and can process them in parallel. Each block combines attention with an MLP, residual connections and normalisation (Section 3.2). Self-attention relates tokens within one input; cross-attention combines information from different inputs, as in camera-to-BEV models.

**Well-known examples.**

| Model | Key idea |
|---|---|
| **Transformer** | Original attention-based architecture for language |
| **ViT** | Treats an image as a sequence of patches |
| **DETR** | Uses learned queries to detect objects from image features |

Transformers capture long-range relationships and are used in vision, detection, trajectory prediction and sensor fusion. Their attention cost grows quadratically with token count, and they are often data- and compute-intensive.

### 3.5.6 Point-Cloud Networks: PointNet and PointPillars

LiDAR produces an unordered, sparse [point cloud](#glossary-point-cloud), unlike an image's regular grid. **PointNet** applies the same small network to each point and combines the results without depending on point order. **PointPillars** bins points into vertical columns, converts those columns to a pseudo-image, and uses a 2D CNN to detect 3D objects.

```
Point cloud -> Pillars -> PointNet per pillar -> Pseudo-image -> 2D CNN -> 3D boxes
```

Other LiDAR detectors include **SECOND**, which uses sparse 3D convolutions, and **CenterPoint**, which predicts object centres on a BEV grid. These models support 3D detection of vehicles, pedestrians and cyclists.

### 3.5.7 Bird's-Eye-View (BEV) Networks

Bird's-eye view (**BEV**) represents the surroundings in a shared top-down coordinate frame, making it useful for combining sensors and supporting detection, mapping and planning. Camera-based systems must estimate depth to transform perspective images into BEV features.

**Main approaches.**

| Model | How it builds the BEV |
|---|---|
| **Lift-Splat-Shoot (LSS)** | Lifts image features into 3D using estimated depth, then aggregates them into BEV cells |
| **BEVFormer** | Uses BEV queries to gather spatial and temporal information with attention |
| **BEVFusion** | Combines camera and LiDAR features in BEV |

```
Camera images (multiple views)  ->  image backbone  ->  view transform  ->  BEV features -> heads
LiDAR point cloud               ->  PointPillars-style encoder ------------>  BEV features ↗
                                                                    (fuse)       (detection, segmentation, ...)
```

BEV is used for multi-camera or camera–LiDAR fusion, 3D detection, lane mapping and as an input to motion prediction.

### 3.5.8 Other Architectures (Brief Overview)

Other families you may encounter include:

- **Graph neural networks:** model relationships between entities, such as road users and map elements.
- **Occupancy networks:** predict which cells in the surrounding 3D space are occupied.
- **End-to-end driving models:** map sensor inputs directly to plans or controls (Chapter 5).
- **Generative and vision-language models:** generate data or connect visual input with text.

## 3.6 Putting It Together: A Modern Perception Network

Most modern perception networks, whatever the task, are built from the same three parts:

| Part | Role | Typical building blocks |
|---|---|---|
| **Backbone** | Reads the raw input and extracts features | ResNet, EfficientNet, ViT; PointPillars encoder for LiDAR |
| [**Neck**](#glossary-backbone-neck-head) | Combines features from different depths or scales, so that both small (far-away) and large (nearby) objects are handled | [Feature Pyramid Network (FPN)](#glossary-backbone-neck-head), [PAN](#glossary-backbone-neck-head), BEV view transform |
| [**Head**](#glossary-backbone-neck-head) | Produces the final task output | Detection head (boxes + classes), segmentation head (per-pixel labels), regression head |

### Worked example: a YOLO-style camera detector

The detector's final post-processing step can use [**Non-Maximum Suppression (NMS)**](#glossary-nms) to remove duplicate overlapping detections.

```
Input image            3 × 640 × 640
   │
   ▼  BACKBONE (CNN: conv + normalisation + activation + residual blocks)
Feature maps           at strides 8, 16, 32  →  80×80, 40×40, 20×20 grids
   │
   ▼  NECK (FPN/PAN: mixes fine detail from large maps with meaning from small maps)
Fused feature maps     at the same three scales
   │
   ▼  HEAD (small convolutions per scale)
Predictions            for each grid cell: box (x, y, w, h), "is there an object?", class probabilities
   │
   ▼  POST-PROCESSING (remove duplicate boxes with Non-Maximum Suppression)
Final detections       e.g. "car, 92%, box (…)"
```

Every item from this chapter appears:

- **Building blocks** (Section 3.2): convolutions, normalisation, activations, residual connections.
- **Training mechanics** (Section 3.3): a loss that combines box-overlap (IoU), [objectness](#glossary-objectness) and classification terms, minimised with an optimiser and a learning-rate schedule.
- **Generalisation** (Section 3.4): data augmentation, weight decay and early stopping; hyperparameters chosen using a validation set.

The same backbone–neck–head pattern also describes a PointPillars detector (the pillar encoder and 2D CNN form the backbone, and an SSD-style head produces 3D boxes) and a BEV network (image backbone and view transform, then BEV heads).

## 3.7 Comparison of Architectures

| Architecture | Typical input | Main strength | Main limitation | Typical automotive use |
|---|---|---|---|---|
| **MLP** | Vector of numbers | Simple and flexible | Inefficient for images; ignores structure | Output heads, small regressors |
| **CNN** | Images (regular grid) | Efficient detection of local patterns; fast | Limited global context | Object detection, segmentation, sign classification |
| **RNN / LSTM / GRU** | Sequences | Built-in memory of the past | Slow to train; weak on very long sequences | Trajectory prediction, tracking |
| **Encoder–decoder / autoencoder** | Image, sequence | Compact representations; same-size outputs | Needs a well-designed bottleneck | Segmentation, anomaly detection, feature learning |
| **Transformer** | Tokens (patches, objects, queries) | Captures long-range relations; scales well | Compute grows with square of tokens; data-hungry | Backbones, BEV, prediction, fusion |
| **PointNet / PointPillars** | LiDAR point cloud | Works directly on 3D data; PointPillars is fast | Sparse data; resolution limited by grid | 3D object detection from LiDAR |
| **BEV networks** | Multi-camera and/or LiDAR | Common top-down frame for fusion and planning | Complex; depth estimation from cameras is hard | 3D detection, mapping, fusion |

## 3.8 Summary

A neural network is built from simple parts — neurons with weights and biases, arranged in layers and made expressive by activation functions — plus specialised pieces such as convolutions, normalisation, residual connections and attention. It learns by repeating a loop: make a prediction (forward propagation), measure the error (loss), find how each weight contributed (backpropagation), and adjust the weights with an optimiser such as SGD or Adam.

Good training means more than a low training loss: regularisation techniques, a careful train/validation/test split, and well-chosen hyperparameters are what allow a network to generalise to new data.

Different data types call for different architectures: MLPs for vectors, CNNs for images, LSTMs for sequences, encoder–decoders for compression and segmentation, transformers for flexible relationships across tokens, PointPillars for LiDAR point clouds, and BEV networks for a unified top-down view that fuses multiple sensors. In practice they are combined in a backbone–neck–head design. The following chapters show where these models are used in a vehicle (Chapter 4), how they fit into complete driving systems (Chapter 5), and how they are deployed on vehicle hardware (Chapter 6).

## Further Reading

**Textbooks and courses**
- Goodfellow, I., Bengio, Y., Courville, A. — *Deep Learning* (MIT Press) — a comprehensive introductory textbook.
- Zhang, A. et al. — *Dive into Deep Learning* (d2l.ai) — free, interactive textbook with code.
- Stanford CS231n — *Convolutional Neural Networks for Visual Recognition* (course notes).

**Building blocks and training**
- Glorot, X., Bengio, Y. — *Understanding the Difficulty of Training Deep Feedforward Neural Networks* (Xavier initialisation).
- He, K. et al. — *Delving Deep into Rectifiers* (He initialisation).
- Ioffe, S., Szegedy, C. — *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.
- Ba, J. L., Kiros, J. R., Hinton, G. E. — *Layer Normalization*.
- Kingma, D., Ba, J. — *Adam: A Method for Stochastic Optimization*.
- Loshchilov, I., Hutter, F. — *Decoupled Weight Decay Regularization* (AdamW).
- Srivastava, N. et al. — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
- Lin, T.-Y. et al. — *Focal Loss for Dense Object Detection*.
- Bergstra, J., Bengio, Y. — *Random Search for Hyper-Parameter Optimization*.

**Architectures**
- He, K. et al. — *Deep Residual Learning for Image Recognition* (ResNet).
- Tan, M., Le, Q. — *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
- Redmon, J. et al. — *You Only Look Once: Unified, Real-Time Object Detection* (YOLO).
- Ronneberger, O. et al. — *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Lin, T.-Y. et al. — *Feature Pyramid Networks for Object Detection*.
- Hochreiter, S., Schmidhuber, J. — *Long Short-Term Memory*.
- Cho, K. et al. — *Learning Phrase Representations using RNN Encoder–Decoder* (GRU).
- Vaswani, A. et al. — *Attention Is All You Need* (Transformer).
- Dosovitskiy, A. et al. — *An Image Is Worth 16×16 Words* (ViT).
- Carion, N. et al. — *End-to-End Object Detection with Transformers* (DETR).
- Qi, C. R. et al. — *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*.
- Lang, A. H. et al. — *PointPillars: Fast Encoders for Object Detection from Point Clouds*.
- Yan, Y. et al. — *SECOND: Sparsely Embedded Convolutional Detection*.
- Yin, T. et al. — *Center-based 3D Object Detection and Tracking* (CenterPoint).
- Philion, J., Fidler, S. — *Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D*.
- Li, Z. et al. — *BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers*.
- Liu, Z. et al. — *BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation*.
- Gao, J. et al. — *VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation*.

## Glossary and notation

The glossary definitions below are **paraphrases**, not verbatim quotations. The source column distinguishes three cases:

- **Standard definition (paraphrased):** the cited clause contains a definition of the term or a directly equivalent term.
- **Standard context only:** the standard discusses or gives an example of the concept, but does not define the glossary term itself.
- **No formal definition identified:** the selected standards do not define the term. The short explanatory description is retained for teaching purposes and must not be presented as an ISO/IEC definition.

Clause references identify the relevant entry or technical discussion; they are not requirement numbers unless explicitly stated. The mathematical notation section is retained as chapter-specific notation and is not attributed to a standard.

### Neural-network fundamentals

| Term | Definition (paraphrased) | Standard source / status |
|---|---|---|
| <span id="glossary-neural-network"><strong>Neural network</strong></span> | A network of one or more layers of neurons connected through weighted links, with adjustable weights, that receives input data and produces an output. | **Standard definition:** [ISO/IEC 22989:2022, 3.4.8](https://www.iso.org/standard/74296.html). Also defined for the robustness context in [ISO/IEC TR 24029-1:2021, 3.4](https://www.iso.org/obp/ui?_escaped_fragment_=iso%3Astd%3Aiso-iec%3Atr%3A24029%3A-1%3Aed-1%3Av1%3Aen). |
| <span id="glossary-model-architecture"><strong>Model architecture</strong></span> | In this chapter, the architecture is the arrangement of a model's computational elements and their connections, which determines how information is transformed from input to output. | **Standard context only:** [ISO/PAS 8800:2024, 6.3–6.4](https://www.iso.org/standard/83303.html) discusses abstraction layers and an example AI-system architecture. The selected standards do not provide a standalone Clause 3 definition of *model architecture*. |
| <span id="glossary-linear-regression"><strong>Linear regression</strong></span> | A machine-learning method that determines the intercept and weights of a linear function relating an input to a numerical output. | **Standard context only:** [ISO/PAS 8800:2024, 3.1.30](https://www.iso.org/standard/83303.html) uses a trained univariate linear function as an example when describing an ML algorithm; it does not provide a separate glossary definition of *linear regression*, but specifies that the process of determining the intercept and weights for a linear function is known as *linear regression*. |
| <span id="glossary-model-parameters"><strong>Model parameters, weights and biases</strong></span> | A model parameter is an internal variable that affects how the model computes its outputs. Neural-network weights are an explicit example; this chapter also treats the bias values in its equations as model parameters. | **Standard definition:** [ISO/IEC 22989:2022, 3.3.8](https://www.iso.org/standard/74296.html). The standard explicitly gives neural-network weights as an example; the grouping with biases follows the chapter's model equations. |
| <span id="glossary-neuron"><strong>Neuron</strong></span> | A primitive processing element that combines one or more input values and applies an activation function to produce an output value. | **Standard definition:** [ISO/IEC 22989:2022, 3.4.9](https://www.iso.org/standard/74296.html). |
| <span id="glossary-activation-function"><strong>Activation function</strong></span> | A function applied to the weighted combination of a neuron's inputs. Activation functions are typically nonlinear and allow a neural network to learn more complex features. | **Standard definition:** [ISO/IEC 22989:2022, 3.4.1](https://www.iso.org/standard/74296.html). The standard defines the general term, not each named function (ReLU, sigmoid, tanh, GELU or SiLU). |
| <span id="glossary-layer"><strong>Layer</strong></span> | A stage in a neural network. In the chapter's examples, layers receive inputs, apply a transformation and pass their outputs to the next stage. | **Standard context only:** [ISO/IEC 22989:2022, 3.4.8](https://www.iso.org/standard/74296.html) defines the *artificial neural network* as comprising one or more *layers* of neurons, but the selected standards do not give a specific  definition of *layer*. |
| <span id="glossary-deep-learning"><strong>Deep learning</strong></span> | An approach that learns hierarchical representations by training neural networks with many hidden layers; it is a subset of machine learning. | **Standard definition:** [ISO/IEC 22989:2022, 3.4.4](https://www.iso.org/standard/74296.html). |

### Network operations and components

| Term | Definition (paraphrased) | Standard source / status |
|---|---|---|
| <span id="glossary-convolution"><strong>Convolution, filter, kernel, feature map, stride and padding</strong></span> | Convolution is a mathematical operation that applies a sliding dot product or cross-correlation to input data. In this chapter, *filter/kernel* refers to the values applied in the sliding operation; the resulting responses are shown as a *feature map*. *Stride* is the step between positions, and *padding* adds values around the input boundary. | **Standard definition for convolution only:** [ISO/IEC 22989:2022, 3.4.3](https://www.iso.org/standard/74296.html). **Not separately defined in the selected standards:** filter/kernel, feature map, stride and padding; their explanations are retained as educational descriptions. |
| <span id="glossary-pooling"><strong>Pooling</strong></span> | An operation that summarises values in a local region of a feature map, often reducing its spatial dimensions. Max pooling retains the largest value in each region. | **No formal definition identified** in the selected standards. This is an educational description of a common CNN operation. |
| <span id="glossary-normalisation"><strong>Normalisation layers (BatchNorm and LayerNorm)</strong></span> | Layers that transform activation values using statistics of a batch or of an individual example. Batch Normalisation uses batch-level statistics; Layer Normalisation uses statistics across the features of an example. | **No formal definition identified** in the selected standards. The description is an educational summary; BatchNorm and LayerNorm are not separately defined in the cited standards. |
| <span id="glossary-weight-initialisation"><strong>Weight initialisation</strong></span> | The procedure for assigning initial values to a model's learnable weights before training. Xavier/Glorot and He are named initialisation methods discussed in this chapter. | **No formal definition identified** in the selected standards. The selected standards' model-parameter and training terminology does not define initialisation methods. |
| <span id="glossary-residual-connection"><strong>Residual or skip connection</strong></span> | A network connection that adds a block's input to the block's transformed output, creating a shorter information path through the network. | **No formal definition identified** in the selected standards. This is an architectural concept explained in the chapter; see the ResNet paper in Further Reading. |
| <span id="glossary-attention"><strong>Attention, query, key and value</strong></span> | Attention assigns different importance to parts of an input when computing an output. In the query–key–value formulation, a query is compared with keys to calculate weights, and those weights are used to combine values. | **No published definition identified in the selected edition.** A definition of *attention* appears in proposed [ISO/IEC 22989:2022/FDAmd 1, 3.3.18](https://www.iso.org/standard/88145.html), which is still under approval as of 20 November 2026 and is therefore not cited here as a published standard. Query, key and value are not separately defined in the selected standards. |

### Training and optimisation

| Term | Definition (paraphrased) | Standard source / status |
|---|---|---|
| <span id="glossary-hyperparameter"><strong>Hyperparameter</strong></span> | A parameter of the AI technology that affects model performance and the learning process. Hyperparameters are selected before training and can be used to help determine model parameters. | **Standard definition:** [ISO/PAS 8800:2024, 3.1.27](https://www.iso.org/standard/83303.html). |
| <span id="glossary-forward-pass"><strong>Forward pass or forward propagation</strong></span> | The calculation that carries input values through successive network operations to produce an output or prediction. | **No separate definition identified** for *forward pass/forward propagation*. **Standard context only:** [ISO/IEC 22989:2022, 3.4.6](https://www.iso.org/standard/74296.html) defines a feed-forward neural network as one in which information flows from input to output in one direction. |
| <span id="glossary-inference"><strong>Inference</strong></span> | Reasoning by which conclusions are derived from known premises. In AI, the premises can include facts, rules, models, features or raw data; in this chapter, the term is used for obtaining an output from a trained model. | **Standard definition:** [ISO/IEC 22989:2022, 3.1.17](https://www.iso.org/standard/74296.html). For the output of an AI system, see also *prediction* in ISO/IEC 22989:2022, 3.1.27. |
| <span id="glossary-loss-function"><strong>Loss function and common losses</strong></span> | In this chapter, a loss function is the numerical measure used during training to express the discrepancy between a model's output and the target. The chapter introduces cross-entropy, MSE, L1, Huber/Smooth-L1, IoU/GIoU, focal and Dice losses. | **No formal definition identified** for *loss function* or the listed loss variants in the selected standards. **Standard context only:** [ISO/PAS 8800:2024, 3.1.32](https://www.iso.org/standard/83303.html) refers to a cost function as part of the definition of *ML model training* which   can adapt during model training. |
| <span id="glossary-gradient"><strong>Gradient</strong></span> | The collection of partial derivatives of a quantity, such as the loss, with respect to model parameters. It indicates how that quantity changes locally as the parameters change. | **No general definition identified** in the selected standards. [ISO/IEC 22989:2022, 3.4.5](https://www.iso.org/standard/74296.html) defines *exploding gradient*, a related training phenomenon, but not the general term *gradient*. |
| <span id="glossary-backpropagation"><strong>Backpropagation</strong></span> | A neural-network training method that uses the error at the output layer to adjust the weights of connections in preceding layers. | **Standard definition:** [ISO/IEC 23053:2022, 3.2.1](https://www.iso.org/standard/74438.html). |
| <span id="glossary-chain-rule"><strong>Chain rule</strong></span> | A calculus rule for differentiating a composition of functions. In neural-network training, it allows derivatives to be combined across successive operations. | **No formal definition identified** in the selected standards. This is a mathematical concept used to explain backpropagation. |
| <span id="glossary-automatic-differentiation"><strong>Automatic differentiation</strong></span> | A computational technique that evaluates derivatives by applying differentiation rules to the operations in a program. | **No formal definition identified** in the selected standards. This is a supporting computational technique described in the chapter. |
| <span id="glossary-gradient-descent"><strong>Gradient descent</strong></span> | An iterative optimisation method that updates model parameters using the loss gradient, generally moving in the direction that reduces the loss. | **No standalone definition identified** for *gradient descent*. **Standard context only:** [ISO/IEC 23053:2022, 3.2.2](https://www.iso.org/standard/74438.html) defines learning rate as the step size for a gradient method. |
| <span id="glossary-learning-rate"><strong>Learning rate</strong></span> | The step size used by a gradient-based method. For neural networks, it is an important hyperparameter influencing whether and how quickly training converges. | **Standard definition:** [ISO/IEC 23053:2022, 3.2.2](https://www.iso.org/standard/74438.html). |
| <span id="glossary-sgd"><strong>Mini-batch and stochastic gradient descent (SGD)</strong></span> | A mini-batch is a subset of training examples used for one update. SGD estimates an update from one example or a subset of examples rather than using the complete training dataset for every update. | **No formal definition identified** for *mini-batch* or *SGD* in the selected standards. ISO/IEC 23053:2022, Clause 7 describes ML learning approaches, but does not provide a definition for these two terms. |
| <span id="glossary-optimiser"><strong>Optimiser</strong></span> | An algorithm that uses gradients to determine updates to model parameters. Examples discussed in this chapter include SGD with momentum, Adam and AdamW. | **No formal definition identified** in the selected standards. The entry is a chapter-level explanation of a common training component. |
| <span id="glossary-learning-rate-schedule"><strong>Learning-rate schedule</strong></span> | A rule that changes the learning rate during training, for example by reducing it as training progresses. | **No formal definition identified** in the selected standards. |
| <span id="glossary-gradient-clipping"><strong>Gradient clipping</strong></span> | A training technique that limits gradient magnitude or norm before an optimisation update, reducing the effect of unusually large gradients. | **No formal definition identified** in the selected standards. |
| <span id="glossary-mixed-precision"><strong>Mixed-precision training</strong></span> | Training that uses more than one numerical precision, commonly using lower precision for selected computations while retaining higher precision where needed. | **No formal definition identified** in the selected standards. |

### Generalisation and regularisation

| Term | Definition (paraphrased) | Standard source / status |
|---|---|---|
| <span id="glossary-generalisation"><strong>Generalisation</strong></span> | The ability of a trained model to make correct predictions on input data that it has not previously seen. | **Standard definition:** [ISO/IEC 23053:2022, 3.1.3](https://www.iso.org/standard/74438.html). Automotive AI-specific wording is also given in [ISO/PAS 8800:2024, 3.1.5](https://www.iso.org/standard/83303.html). |
| <span id="glossary-overfitting"><strong>Overfitting and underfitting</strong></span> | Overfitting describes a model that fits its training data too closely and consequently fails to generalise well to new data. Underfitting describes a model that does not fit the training data sufficiently and therefore also produces incorrect predictions on new data. | **Standard definitions:** [ISO/IEC 23053:2022, 3.1.4 and 3.1.5](https://www.iso.org/standard/74438.html). |
| <span id="glossary-regularisation"><strong>Regularisation, dropout, weight decay and early stopping</strong></span> | In this chapter, regularisation refers to techniques used to reduce overfitting and improve performance on unseen data. Dropout, weight decay and early stopping are examples of such techniques. | **No formal definitions identified** for these terms in the selected standards. The descriptions are educational summaries, not ISO/IEC definitions. |
| <span id="glossary-data-augmentation"><strong>Data augmentation</strong></span> | Creating modified versions of training examples, for example through geometric or appearance transformations, to increase the variety of examples used during training. | **No standalone definition identified** in the selected standards. The selected documents address dataset development and lifecycle activities (e.g. [ISO/PAS 8800:2024, Clause 11](https://www.iso.org/standard/83303.html)), but this entry is an educational explanation rather than a paraphrase of a defined term. |

### Architecture families and perception

| Term | Definition (paraphrased) | Standard source / status |
|---|---|---|
| <span id="glossary-mlp"><strong>Multi-layer perceptron (MLP)</strong></span> | A feed-forward neural-network architecture arranged in an input layer, one or more hidden layers and an output layer. | **No separate MLP definition identified** in the selected standards. [ISO/IEC 22989:2022, 3.4.6 and 3.4.8](https://www.iso.org/standard/74296.html) provide related definitions for feed-forward neural networks and neural networks, but do not define *MLP* in the cited entries. |
| <span id="glossary-cnn"><strong>Convolutional neural network (CNN)</strong></span> | A feed-forward neural network that uses convolution in at least one of its layers. | **Standard definition:** [ISO/IEC 22989:2022, 3.4.2](https://www.iso.org/standard/74296.html). |
| <span id="glossary-rnn"><strong>Recurrent neural network (RNN), LSTM and GRU</strong></span> | An RNN feeds information from a previous processing step into the current step. LSTM is a type of RNN intended to handle both short- and long-span dependencies. GRU is another gated RNN variant discussed in this chapter. | **Standard definitions:** [ISO/IEC 22989:2022, 3.4.10 (RNN) and 3.4.7 (LSTM)](https://www.iso.org/standard/74296.html). **No separate GRU definition identified** in the selected standards; the GRU description is educational. |
| <span id="glossary-encoder-decoder"><strong>Encoder–decoder, autoencoder and latent representation</strong></span> | An encoder maps input data to an internal representation; a decoder maps that representation to an output. An autoencoder is trained to reconstruct its input, using the intermediate representation as a compact encoding. | **No formal definitions identified** in the selected standards. The entry is an educational explanation of common architecture patterns. |
| <span id="glossary-transformer"><strong>Transformer and tokens</strong></span> | A Transformer is a neural-network architecture built around attention mechanisms. It operates on a sequence of tokens, which can represent words, image patches or other input units depending on the application. | **No published Transformer or token definition identified** in the selected editions. A token definition is proposed in [ISO/IEC 22989:2022/FDAmd 1, 3.1.41](https://www.iso.org/standard/88145.html), which remains under approval as of 20 November 2026 and is not treated here as a published definition. |
| <span id="glossary-attention-patterns"><strong>Positional encoding, self-attention, cross-attention and multi-head attention</strong></span> | Positional encoding adds information about sequence order or position. Self-attention relates elements within one input sequence; cross-attention relates elements from different inputs; multi-head attention computes several attention relationships in parallel. | **No formal definitions identified** for these variants in the selected published standards. The proposed [ISO/IEC 22989:2022/FDAmd 1, 3.3.18](https://www.iso.org/standard/88145.html) addresses *attention* generally but remains under approval, not a published standard definition. |
| <span id="glossary-point-cloud"><strong>Point cloud, LiDAR and voxels</strong></span> | A point cloud is a set of points represented in three-dimensional space. LiDAR is a sensing technology that uses laser light to measure distances; a voxel is a volumetric cell used to discretise 3D space. | **No formal definitions identified** in the selected standards for these three terms in the context used here. The descriptions are explanatory; the selected standards do not define this architecture-specific vocabulary. |
| <span id="glossary-bev"><strong>Bird's-eye view (BEV)</strong></span> | A top-down spatial representation around a vehicle. Information from camera, LiDAR or other sensors can be transformed into this representation and combined. | **No formal definition identified** in the selected standards. This is an explanatory term used in automotive perception literature. |
| <span id="glossary-backbone-neck-head"><strong>Backbone, neck and head</strong></span> | Common functional stages in a perception network: the backbone extracts features, the neck combines or redistributes features, and the head produces task-specific outputs. | **No formal definitions identified** in the selected standards. These are architecture-convention terms used in computer-vision literature. |
| <span id="glossary-yolo"><strong>You Only Look Once (YOLO)</strong></span> | A family of object-detection architectures that predict object locations and classes in a single network evaluation. | **No formal definition identified** in the selected standards. This is the name and concept of a research architecture; the original YOLO paper is listed in Further Reading. |
| <span id="glossary-nms"><strong>Non-Maximum Suppression (NMS)</strong></span> | A post-processing operation that suppresses low-relevance or duplicate bounding-box candidates, typically retaining the higher-scoring detection. | **Standard context only:** [ISO/PAS 8800:2024, 3.1.9](https://www.iso.org/standard/83303.html) gives non-maximum suppression as an example of AI post-processing and describes its use to remove low-relevance and duplicate bounding boxes. It does not provide a standalone definition of *NMS*. |
| <span id="glossary-objectness"><strong>Objectness</strong></span> | A detector output or score estimating whether a candidate region contains an object, distinct from the predicted object class. | **No formal definition identified** in the selected standards. This is a detector-specific explanatory term. |

### Standards reviewed and scope notes

The references below are standards-level sources for the entries above. The glossary uses only clauses that directly define a term or explicitly discuss the concept; it does not infer a definition merely because a standard is relevant to AI safety in general.

- [ISO/IEC 22989:2022 — Artificial intelligence concepts and terminology](https://www.iso.org/standard/74296.html): the primary terminology source for neural networks, neurons, activation functions, convolution, deep learning, parameters, RNNs and LSTMs.
- [ISO/IEC 23053:2022 — Framework for AI systems using machine learning](https://www.iso.org/standard/74438.html): the source for backpropagation, learning rate, generalisation, overfitting and underfitting. It was included as an additional relevant standard because it was identified during the preceding discussion.
- [ISO/PAS 8800:2024 — Road vehicles — Safety and artificial intelligence](https://www.iso.org/standard/83303.html): the source for hyperparameters and AI-specific generalisation; it also provides context for model architecture, linear regression, cost functions, and NMS as an AI post-processing example.
- [ISO/IEC TR 24029-1:2021 — Assessment of the robustness of neural networks — Part 1: Overview](https://www.iso.org/obp/ui?_escaped_fragment_=iso%3Astd%3Aiso-iec%3Atr%3A24029%3A-1%3Aed-1%3Av1%3Aen): relevant to neural networks and robustness terminology. Part 2, [ISO/IEC 24029-2:2023](https://www.iso.org/standard/79804.html), is a methodology for applying formal methods to assess neural-network robustness; it is not used as a source for the architecture definitions in this chapter.
- [ISO/IEC TR 5469:2024 — Functional safety and AI systems](https://www.iso.org/standard/81283.html): its Clause 3 uses the terminology in ISO/IEC 22989 and adds functional-safety concepts. Those safety terms are more appropriate to the later safety chapters than to this neural-network architecture glossary.
- [ISO/TR 4804:2020 — Safety and cybersecurity for automated driving systems](https://www.iso.org/standard/80363.html) is withdrawn and has been superseded by [ISO/TS 5083:2025](https://committee.iso.org/cms/live/live/en/sites/isoorg/contents/data/standard/08/19/81920.html). These documents address automated-driving-system safety, verification and validation, rather than defining most of the neural-network architecture terms in this chapter.

### Mathematical notation

The same symbol can have different meanings in different contexts. The definitions below follow the usage in this chapter; in particular, \(p\) is padding in the convolution equation and the dropout probability in the regularisation discussion, while \(s\) is stride in the convolution equation and a moving estimate in the Adam update.

| Symbol | Meaning in this chapter |
|---|---|
| <span id="notation-f-theta">\(f_\theta\)</span> | The model function, parameterised by \(\theta\). |
| <span id="notation-theta">\(\theta\), \(\theta^*\)</span> | The set of learnable model parameters and, with the asterisk, an optimal parameter setting for the stated objective. |
| <span id="notation-data">\(x_i\), \(y_i\), \(\hat y\)</span> | An input example, its target value, and a model prediction, respectively. |
| <span id="notation-weight-bias">\(w\), \(b\)</span> | A scalar weight and bias in the simple linear model or neuron example. |
| <span id="notation-weight-matrix">\(W\), \(\mathbf b\)</span> | A layer's weight matrix and bias vector. |
| <span id="notation-activation">\(\phi\)</span> | The activation function applied to a neuron's pre-activation value. |
| <span id="notation-loss">\(\mathcal{L}\)</span> | The loss function. |
| <span id="notation-dataset-size">\(N\)</span> | The number of examples in the dataset or objective being averaged. |
| <span id="notation-attention">\(Q\), \(K\), \(V\), \(d_k\)</span> | Query, key and value representations, and the dimension of the key used in the attention scaling term. |
| <span id="notation-layer-size">\(n_{in}\), \(n_{out}\)</span> | The number of inputs and the number of output neurons in a layer. |
| <span id="notation-normalisation">\(\mu\), \(\sigma^2\), \(\epsilon\), \(\gamma\), \(\beta\)</span> | Mean, variance, a small numerical-stability constant, and learnable scale and shift parameters in the normalisation equation. The optimiser sections also use \(\beta\)-subscripts for Adam's moving averages. |
| <span id="notation-learning-rate">\(\eta\)</span> | Learning rate, the step size used in parameter updates. |
| <span id="notation-gradient">\(g\)</span> | A gradient used in the optimiser update equations. |
| <span id="notation-weight-decay">\(\lambda\)</span> | The regularisation or weight-decay coefficient in the equations where it appears. |
| <span id="notation-convolution">\(k\), \(s\), \(p\)</span> | Kernel size, stride and padding in the convolution output-size equation. |
| <span id="notation-focal-probability">\(p_t\)</span> | The model probability assigned to the target class in the focal-loss expression. |
| <span id="notation-dice-regions">\(A\), \(B\)</span> | The predicted and target regions used in the Dice-loss expression. |
| <span id="notation-regression-metrics">\(R^2\), \(\mathrm{IoU}\)</span> | Coefficient of determination in the linear-regression widget, and intersection over union for comparing regions or boxes. |