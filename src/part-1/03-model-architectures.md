
# 3. AI Model Architectures and Technologies

## 3.1 Introduction

In Chapter 2 we saw *how* a model can be taught: with labels, without labels, through rewards, or by imitating an expert. In this chapter we look at *what* is being taught — the model itself.

A **neural network** is a computer program that learns a task from examples instead of following hand-written rules. Its **architecture** is its shape and wiring: how many parts it has, what each part does, and how the parts are connected. Different tasks need different architectures. A network that reads camera images is wired differently from one that reads LiDAR points or predicts where a pedestrian will walk next.

Recall from Section 2.2 that training means finding the parameters $\theta$ that make a loss function small:

$$
\theta^{*} = \arg\min_{\theta} \; \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}\big(f_\theta(x_i),\, y_i\big)
$$

#### Try it: the simplest possible $f_\theta$ — linear regression

Before opening up a full neural network, it helps to see this formula at work in its simplest case: $f_\theta(x) = wx + b$ (a straight line) with $\mathcal{L} = (\hat y - y)^2$. This is ordinary **linear regression** — one weight $w$, one bias $b$, no activation function.

Edit the points below and watch $w$, $b$ and the mean squared error update live.

<div id="lr-widget" style="border:1px solid #d9dde5;border-radius:8px;padding:1rem;margin:1.25rem 0;max-width:720px;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;">
<div style="display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:.75rem;">
<button type="button" data-lr-preset="clean">Clean linear trend</button>
<button type="button" data-lr-preset="outlier">With an outlier</button>
<button type="button" data-lr-preset="curved">Curved data</button>
<button type="button" data-lr-preset="clear">Clear points</button>
<label style="margin-left:.5rem;display:flex;align-items:center;gap:.3rem;">
<input type="checkbox" id="lr-residuals"> show residuals
</label>
</div>
<canvas id="lr-canvas" width="680" height="380" style="width:100%;height:auto;border:1px solid #d9dde5;border-radius:6px;cursor:crosshair;touch-action:none;"></canvas>
<p style="font-size:.9rem;color:#5b6475;margin:.5rem 0 0;">Click empty space to add a point, drag a point to move it, double-click a point to delete it.</p>
<div id="lr-stats" style="margin-top:.5rem;font-size:.95rem;line-height:1.6;"></div>
</div>

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

> **Note:** $w$ and $b$ above are computed with the closed-form least-squares solution, $w = \dfrac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2}$, $b = \bar y - w\bar x$ — the direct solution for this simplest case of $f_\theta$. Section 3.3 introduces the general method (gradient descent) used when $f_\theta$ is a full neural network and no closed-form solution exists.

In this chapter, $f_\theta$ — the "model" — is opened up and explained. The chapter is organised in the order a beginner needs it:

1. **Building blocks** (Section 3.2): the small parts every neural network is made of.
2. **Training mechanics** (Section 3.3): how a network adjusts itself to make fewer mistakes.
3. **Generalisation** (Section 3.4): how to make sure the network also works on data it has never seen, including the settings (hyperparameters) that control training.
4. **Architecture families** (Section 3.5): MLPs, CNNs, LSTMs, encoder–decoders, transformers, point-cloud networks and bird's-eye-view (BEV) networks.
5. **Putting it together** (Section 3.6): how the pieces combine into a complete perception network.

> **Note:** This chapter explains the ideas. Where each model is *used* in a vehicle is the subject of Chapter 4, how the models fit into a complete driving system is covered in Chapter 5, and how they are made small and fast enough to run on vehicle hardware is covered in Chapter 6.

## 3.2 Neural Network Building Blocks

### 3.2.1 The neuron

The smallest unit of a neural network is the **neuron**. It receives some numbers as input, multiplies each one by a **weight** (how important that input is), adds the results together with a **bias** (a constant shift), and then passes the total through an **activation function** $\phi$:

$$
y = \phi\big(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b\big) = \phi\big(\mathbf{w}^\top \mathbf{x} + b\big)
$$

You can think of a neuron as a tiny calculator that makes a small decision, such as "does this part of the image look like an edge?".

### 3.2.2 Weights and biases

**Weights** and **biases** are the "knobs" of the network. At the start of training they have arbitrary values and the network's answers are poor. Training is the process of turning all these knobs, little by little, until the answers become good. A modern network can have millions or billions of such knobs. Together they are called the model **parameters**, written $\theta$.

### 3.2.3 Layers

Neurons are arranged in rows called **layers**:

- The **input layer** receives the data (for example, the pixel values of an image).
- One or more **hidden layers** process the data step by step.
- The **output layer** produces the answer (for example, "pedestrian" or "car").

If a layer has $n_{in}$ inputs and $n_{out}$ neurons, all of its weights can be written as a matrix $W$ and the layer computes:

$$
\mathbf{h} = \phi\big(W\mathbf{x} + \mathbf{b}\big)
$$

Such a layer has $n_{in} \times n_{out}$ weights and $n_{out}$ biases. For example, a layer taking 100 numbers and producing 50 has $100 \times 50 + 50 = 5\,050$ parameters.

A network with many hidden layers is called a **deep** network, which is where the term *deep learning* comes from. Early layers tend to learn simple patterns, and later layers combine them into more complex ones.

### 3.2.4 Activation functions

An **activation function** is a small mathematical function applied to the output of each neuron.

**Why do we need it?** Without activation functions, a neuron only does multiplication and addition. Stacking many such layers would still be equivalent to a *single* multiplication and addition, so the network could only learn straight-line relationships. Activation functions bend the straight lines, which allows the network to learn complex, curved patterns.

| Function | Formula | Plain-language description | Typical use |
|---|---|---|---|
| **ReLU** | $\max(0, x)$ | Negative values become zero; positive values pass unchanged. | Default choice in hidden layers of CNNs |
| **Leaky ReLU** | $\max(0.01x, x)$ | Like ReLU, but lets a tiny amount of negative values through. | Detection networks (e.g. YOLO variants) |
| **Sigmoid** | $\dfrac{1}{1+e^{-x}}$ | Squashes any number into the range 0 to 1. | Output of a yes/no decision |
| **Tanh** | $\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$ | Squashes any number into the range −1 to 1. | Inside LSTM cells |
| **Softmax** | $\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | Turns a list of numbers into probabilities that add up to 100%. | Output of a multi-class classifier |
| **GELU / SiLU** | $x\,\Phi(x)$ / $x\,\sigma(x)$ | Smooth versions of ReLU. | Transformers and modern CNNs |

> **The vanishing-gradient problem.** Sigmoid and tanh become almost flat for very large or very small inputs. In a flat region, the signal used to teach the network (the gradient, Section 3.3) is close to zero, so early layers of a deep network barely learn. ReLU-type functions do not flatten for positive values, which is one reason they dominate modern networks.

### 3.2.5 Convolution and pooling

Images contain a lot of numbers (a $640 \times 640$ colour image has over a million). Connecting every pixel to every neuron would need far too many weights. **Convolution** solves this with a clever shortcut.

A **convolution** slides a small window — called a **filter** or **kernel**, for example $3\times3$ pixels — across the image. At each position it multiplies the pixels under the window by the filter's weights and adds them up. The *same* filter is reused at every position, so the number of weights stays small, and a pattern (such as a vertical edge) is detected wherever it appears in the image. The result is called a **feature map**.

Three settings control a convolution:

- **Kernel size $k$:** the size of the window.
- **Stride $s$:** how many pixels the window moves each time. A stride of 2 halves the width and height.
- **Padding $p$:** zeros added around the border so that edges are not lost.

The output width (or height) for an input of size $n$ is:

$$
n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1
$$

**Pooling** shrinks a feature map by keeping a summary of each small region — for example, **max pooling** keeps only the largest value in each $2\times2$ block. This reduces the amount of computation and makes the network less sensitive to tiny shifts of an object in the image.

### 3.2.6 Normalisation layers

As numbers pass through many layers they can become very large or very small, which makes learning slow and unstable. A **normalisation layer** rescales the numbers to a standard range:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^{2} + \epsilon}}, \qquad y = \gamma\,\hat{x} + \beta
$$

where $\mu$ and $\sigma^2$ are the mean and variance of the values, and $\gamma$ and $\beta$ are two learnable parameters that let the network undo the rescaling if that turns out to be better.

- **Batch Normalisation (BatchNorm)** computes the mean and variance across a batch of examples. Typical in CNNs.
- **Layer Normalisation (LayerNorm)** computes them across the features of a single example. Typical in transformers and recurrent networks.

### 3.2.7 Weight initialisation

Before training, the weights need starting values. If they are all zero, every neuron learns the same thing; if they are too large or too small, learning can stall. Standard recipes choose small random values whose size depends on the layer width:

- **Xavier (Glorot) initialisation:** variance $\dfrac{2}{n_{in}+n_{out}}$ — used with tanh and sigmoid.
- **He initialisation:** variance $\dfrac{2}{n_{in}}$ — used with ReLU.

### 3.2.8 Residual (skip) connections

In a **residual connection**, the input of a block is added to its output:

$$
\mathbf{y} = F(\mathbf{x}) + \mathbf{x}
$$

The block $F$ now only needs to learn the *change* to apply to its input, and information (and the learning signal) can travel along the shortcut across many layers. This makes very deep networks much easier to train and is the key idea of **ResNet**. Residual connections are used in almost every modern architecture, including transformers.

### 3.2.9 Attention (a first look)

**Attention** lets a network decide which parts of its input to focus on when producing each part of its output — like highlighting the important words in a sentence. For every element, the network builds a **query** $Q$ ("what am I looking for?"), and every element offers a **key** $K$ ("what do I contain?") and a **value** $V$ ("what information do I give?"). Keys that match the query receive a high weight, and the output is the weighted mix of the values:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Attention is explained fully in Section 3.5.5 as the core of the transformer.

### 3.2.10 Parameters versus hyperparameters

| | Parameters | Hyperparameters |
|---|---|---|
| **Examples** | Weights and biases | Learning rate, batch size, number of layers, dropout rate |
| **Who sets them?** | The network, by learning | The engineer, before training |
| **When?** | During training | Before (and between) training runs |

Hyperparameters are discussed in Section 3.4.

## 3.3 How a Network Learns: Training Mechanics

Training is a repeated four-step loop:

1. **Forward pass:** feed examples in and compute predictions.
2. **Loss:** measure how wrong the predictions are.
3. **Backward pass:** work out how each weight contributed to the error.
4. **Update:** nudge each weight in the direction that reduces the error.

This loop runs thousands or millions of times until the error stops improving.

### 3.3.1 Forward propagation

In **forward propagation**, data enters the input layer and flows through the layers one by one, each applying its weights and activation function, until the output layer produces a prediction. This is the network "making a guess". Forward propagation is also exactly what happens when a trained network is used (this is called **inference**).

### 3.3.2 Loss functions

A **loss function** $\mathcal{L}$ is a score that tells us how wrong a prediction $\hat{y}$ is compared with the true answer $y$. Lower is better. The choice of loss depends on the task:

| Task | Loss | Formula / idea |
|---|---|---|
| Classification | **Cross-entropy** | $-\sum_c y_c \log \hat{y}_c$ — penalises being confident and wrong |
| Predicting a number | **MSE** (mean squared error) | $(\hat{y}-y)^2$ — penalises large errors strongly |
| Predicting a number | **L1** (mean absolute error) | $\lvert\hat{y}-y\rvert$ — less sensitive to outliers |
| Predicting a number | **Huber / Smooth-L1** | MSE for small errors, L1 for large ones |
| Bounding boxes | **IoU / GIoU loss** | $1 - \text{IoU}$, where IoU is the overlap between predicted and true box divided by their union |
| Rare classes | **Focal loss** | $-(1-p_t)^{\gamma}\log p_t$ — down-weights easy examples so rare, hard ones count more |
| Segmentation | **Dice loss** | $1 - \dfrac{2\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert}$ — based on the overlap of predicted and true regions |

Focal loss is particularly useful in driving data, where "background" vastly outnumbers objects of interest.

### 3.3.3 Gradients and backpropagation

The **gradient** of the loss with respect to a weight, $\dfrac{\partial \mathcal{L}}{\partial w}$, tells us how the loss changes if that weight is increased slightly. Imagine standing on a hill in thick fog: you cannot see the valley, but you can feel which way the ground slopes. The gradient is that slope, and moving *against* it takes you downhill, towards a lower loss.

**Backpropagation** is the efficient method for computing the gradient of *every* weight in the network. Starting at the loss, it moves backwards layer by layer, using the **chain rule** from calculus: the effect of a weight on the loss is the product of the effects along the path from that weight to the loss. For a weight $w$ feeding a neuron with pre-activation $z$ and output $\hat{y}$:

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

Deep-learning frameworks such as PyTorch compute this automatically (**automatic differentiation**), so engineers rarely write it by hand.

#### A tiny worked example

Take a single neuron without activation, $\hat{y} = wx + b$, with input $x=2$, true answer $y=3$, and starting values $w=0.5$, $b=0$. The loss is $\mathcal{L} = (\hat{y}-y)^2$.

| Step | Calculation | Result |
|---|---|---|
| Forward pass | $\hat{y} = 0.5 \times 2 + 0 $ | $\hat{y}=1$ |
| Loss | $(1-3)^2$ | $\mathcal{L}=4$ |
| Gradient w.r.t. $\hat{y}$ | $2(\hat{y}-y)$ | $-4$ |
| Gradient w.r.t. $w$ | $-4 \times x = -4 \times 2$ | $-8$ |
| Gradient w.r.t. $b$ | $-4 \times 1$ | $-4$ |
| Update (learning rate $\eta=0.1$) | $w \leftarrow 0.5 - 0.1\times(-8)$; $b \leftarrow 0 - 0.1\times(-4)$ | $w=1.3$, $b=0.4$ |
| New forward pass | $1.3 \times 2 + 0.4$ | $\hat{y}=3.0$ |

After a single update the prediction moved from 1 to 3, which is exactly the target. (The learning rate was chosen to keep the numbers clean; in real networks many small steps are needed.)

This is exactly what `loss.backward()` does automatically in PyTorch for networks with millions of weights: apply the chain rule layer by layer, rather than working it out by hand as above.

### 3.3.4 Gradient descent and the learning rate

**Gradient descent** repeats the update step: move each parameter a small step *against* its gradient.

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}
$$

The **learning rate** $\eta$ is the size of the step:

- **Too large:** the steps overshoot the valley and the loss bounces around or even grows.
- **Too small:** learning is very slow and may get stuck.
- **Just right:** the loss decreases steadily.

### 3.3.5 Mini-batches and Stochastic Gradient Descent (SGD)

Computing the gradient over the *entire* dataset for every single step is too slow for large datasets. Instead, we use a small random subset — a **mini-batch**, typically 16 to 512 examples — to estimate the gradient. This is **Stochastic Gradient Descent (SGD)**. Each step is a little noisy, but steps are much cheaper, and the noise even helps the network avoid poor solutions.

Some vocabulary:

- **Batch size:** the number of examples in each mini-batch.
- **Iteration:** one update using one mini-batch.
- **Epoch:** one full pass through the entire training set.

### 3.3.6 Optimisation algorithms

An **optimiser** is the rule that turns gradients into parameter updates. Plain SGD is the starting point, and the others improve on it.

| Optimiser | Idea in plain words | Update rule (simplified) |
|---|---|---|
| **SGD** | Step downhill using the current mini-batch gradient $g$. | $\theta \leftarrow \theta - \eta\, g$ |
| **SGD with momentum** | Like a rolling ball that keeps some speed from previous steps; it smooths out noise and rolls through small bumps. | $v \leftarrow \beta v + g$; $\theta \leftarrow \theta - \eta\, v$ |
| **RMSprop** | Gives each parameter its own step size: parameters with consistently large gradients take smaller steps. | $s \leftarrow \rho s + (1-\rho)g^2$; $\theta \leftarrow \theta - \eta\,\dfrac{g}{\sqrt{s}+\epsilon}$ |
| **Adam** | Combines momentum (average direction) with RMSprop (per-parameter step size). The most common default. | $m \leftarrow \beta_1 m + (1-\beta_1)g$; $s \leftarrow \beta_2 s + (1-\beta_2)g^2$; $\theta \leftarrow \theta - \eta\,\dfrac{\hat{m}}{\sqrt{\hat{s}}+\epsilon}$ |
| **AdamW** | Adam with weight decay (Section 3.4.4) applied correctly, as a separate step. Widely used for transformers. | Adam update plus $-\eta\lambda\theta$ |

> **Rule of thumb:** Adam/AdamW usually works well with little tuning, which makes it a good first choice. SGD with momentum often needs more tuning but is still popular for training CNNs.

### 3.3.7 Learning-rate schedules

The best learning rate is not constant during training: large steps are useful at the start, and small steps are better at the end for settling precisely into a good solution. A **schedule** changes $\eta$ over time:

- **Step decay:** multiply $\eta$ by a factor (e.g. 0.1) every few epochs.
- **Cosine annealing:** decrease $\eta$ smoothly following a cosine curve: $\eta_t = \eta_{min} + \tfrac{1}{2}(\eta_{max}-\eta_{min})\big(1+\cos(\pi t/T)\big)$.
- **Warm-up:** start with a very small $\eta$ and increase it over the first few iterations, which avoids unstable early steps.
- **One-cycle:** warm up to a maximum and then anneal down to near zero.

### 3.3.8 Gradient clipping

Occasionally a gradient becomes unusually large and would throw the weights far off course. **Gradient clipping** caps its size: if the length of the gradient $\lVert g \rVert$ exceeds a threshold $c$, the gradient is scaled down:

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right)
$$

It is especially common when training recurrent networks and transformers.

### 3.3.9 Mixed-precision training

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

### 3.4.2 Overfitting and underfitting

- **Underfitting:** the model is too simple (or trained too little) to learn the pattern. It performs poorly on both training and validation data.
- **Overfitting:** the model learns the training examples — including their noise and accidents — by heart, instead of the general idea. It performs very well on training data but poorly on validation data. It is like a student who memorises past exam answers without understanding the subject.

Overfitting is more likely when the model has many parameters compared with the amount (or variety) of data.

### 3.4.3 Learning curves

A **learning curve** plots the training loss and the validation loss against the number of epochs. It is the main diagnostic tool:

| What you see | What it means | What to try |
|---|---|---|
| Both losses high | Underfitting | Larger model, train longer, higher learning rate |
| Training loss low, validation loss high (a gap) | Overfitting | More data, augmentation, regularisation, smaller model |
| Both losses low and close together | Good fit | Done (or fine-tune) |
| Validation loss starts to rise while training loss keeps falling | Overfitting has begun | Early stopping |

### 3.4.4 Regularisation

**Regularisation** is a family of techniques that deliberately make it harder for the network to memorise, so that it learns the general pattern.

- **L2 regularisation / weight decay.** Add a penalty for large weights to the loss: $\mathcal{L}_{total} = \mathcal{L} + \lambda \lVert \mathbf{w}\rVert_2^2$. Large weights are discouraged, which keeps the model simpler and smoother. $\lambda$ controls the strength.
- **L1 regularisation.** Penalty $\lambda \lVert \mathbf{w}\rVert_1$ (sum of absolute values). It pushes many weights to exactly zero, which gives sparser models.
- **Dropout.** During training, each neuron is randomly switched off with probability $p$ (e.g. 0.5) in every step. The network cannot rely on any single neuron, so it learns more robust, redundant features. At inference time, all neurons are used.
- **Data augmentation.** Create extra, slightly different training examples from the existing ones — flipping, cropping, changing brightness, adding noise. The network sees more variety without more data collection. Covered in detail in Chapter 11.
- **Early stopping.** Monitor the validation loss and stop training when it stops improving, keeping the best version of the model.
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

The building blocks of Section 3.2 can be combined in many ways. This section introduces the main families relevant to automotive AI. Each subsection answers three questions: *What is the idea? How does it work? Where is it used in vehicles?*

### 3.5.1 Multi-Layer Perceptron (MLP)

**Idea.** The simplest neural network: a stack of layers in which every neuron is connected to every neuron of the next layer (also called a *fully connected* or *dense* network).

**How it works.** The input is a list of numbers (a vector). Each layer computes $\mathbf{h} = \phi(W\mathbf{x}+\mathbf{b})$ and passes the result on (Section 3.2.3).

**Strengths and limits.** MLPs are flexible and easy to build, and they work well on tables of numbers. For images they are inefficient: every pixel would need its own weights, and the network has no built-in notion that neighbouring pixels belong together.

**In vehicles.**
- The final "decision" layers of larger networks, for example turning image features into class probabilities or box coordinates.
- Small regressors, such as mapping sensor values to an estimate.
- Components inside transformers (the feed-forward block).

### 3.5.2 Convolutional Neural Networks (CNN)

**Idea.** A network designed for images. It uses convolutions (Section 3.2.5) so that the same pattern detector is reused across the whole image.

**How it works.** A CNN stacks convolution, activation and pooling layers. Early layers detect simple patterns such as edges and colours, middle layers combine them into shapes such as wheels or windows, and later layers recognise whole objects such as cars or pedestrians. As the data moves deeper, the feature maps get smaller in size but richer in meaning.

```
Image -> [Conv+ReLU] -> [Pool] -> [Conv+ReLU] -> [Pool] -> ... -> [MLP head] -> "pedestrian"
         edges            ↓         shapes          ↓                 decision
```

**Well-known CNNs.**

| Network | Key idea |
|---|---|
| **LeNet** | Early CNN for digit recognition; showed the basic recipe |
| **ResNet** | Residual connections (Section 3.2.8) allow networks with 50–150+ layers; very common backbone |
| **EfficientNet** | Scales depth, width and image size together for good accuracy at low cost |
| **YOLO** | "You Only Look Once" — a CNN that predicts boxes and classes for all objects in a single pass; fast (used in Chapter 2's Lab 1) |
| **U-Net / FCN** | CNNs whose output is an image of the same size, used for segmentation (labelling every pixel) |

**In vehicles.** Object detection, traffic-sign classification, lane detection, semantic segmentation, and the image feature extractors inside larger systems.

### 3.5.3 Recurrent Networks: RNN, LSTM and GRU

**Idea.** A network with **memory**, designed for sequences — data where order matters, such as video frames, or the positions of a vehicle over time.

**How it works.** A **Recurrent Neural Network (RNN)** processes the sequence one step at a time and carries a hidden state (its "memory") from one step to the next:

$$
\mathbf{h}_t = \phi\big(W_x \mathbf{x}_t + W_h \mathbf{h}_{t-1} + \mathbf{b}\big)
$$

Plain RNNs struggle to remember things over long sequences (the vanishing-gradient problem of Section 3.2.4 applies across time). Two improved designs solve much of this:

- **LSTM (Long Short-Term Memory):** adds **gates** — small learned switches that decide what to *forget*, what new information to *store*, and what to *output*. This lets it keep important information for a long time.
- **GRU (Gated Recurrent Unit):** a simpler variant of the LSTM with fewer gates and parameters.

**In vehicles.**
- Predicting the future trajectory of surrounding vehicles and pedestrians from their past positions.
- Tracking objects across video frames.
- Filtering noisy sensor sequences.

**Limits.** Because they process one step after another, recurrent networks are slow to train and still find very long sequences hard. For many sequence tasks they have been largely replaced by transformers (Section 3.5.5), but they remain common in smaller, lightweight models.

### 3.5.4 Encoder–Decoder Networks and Autoencoders

**Idea.** Split a network into two halves. The **encoder** compresses the input into a short summary (called the **latent representation** or *bottleneck*); the **decoder** expands that summary into the desired output.

```
Input -> [ Encoder ] -> compact summary (latent) -> [ Decoder ] -> Output
```

**Variants.**

- **Autoencoder.** The output is a reconstruction of the input itself. The network is trained to rebuild the input from the compact summary, so it must learn the most important features. No labels are needed (see Section 2.3 and 2.4).
- **Segmentation networks (e.g. U-Net).** The decoder outputs a label for every pixel. **Skip connections** copy detailed information from encoder to decoder so that fine boundaries are preserved.
- **Sequence-to-sequence models.** An encoder reads an input sequence (for example, past motion and map) and a decoder generates an output sequence (for example, future motion). Transformers use this same encoder–decoder pattern.

**In vehicles.** Pixel-wise scene understanding, learning compact features from unlabelled sensor data, anomaly detection (a model trained to rebuild "normal" data rebuilds unusual data poorly), and trajectory generation.

### 3.5.5 Transformers and Attention

**Idea.** Instead of reading data step by step or through small local windows, a **transformer** looks at *all* parts of the input at once and learns which parts are related to each other. It is the architecture behind modern language models, and it has become central to vision and driving models too.

**How it works.**

1. The input is split into pieces called **tokens** — for a sentence, words; for an image, small patches (for example $16\times16$ pixels); for a driving scene, objects, map elements or BEV cells.
2. Each token is converted into a vector. Because attention itself has no sense of order or position, a **positional encoding** is added so that the model knows where each token is.
3. **Self-attention** (Section 3.2.9) lets every token gather information from every other token:

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

   - $Q$ (queries), $K$ (keys) and $V$ (values) are all computed from the tokens by learned weights.
   - $QK^\top$ measures how relevant each token is to each other token.
   - Softmax turns relevance scores into weights that sum to 1, and the output is the weighted mix of values.
4. **Multi-head attention** runs several attention operations in parallel, so that different "heads" can look for different kinds of relationships.
5. Each attention step is followed by a small MLP, with residual connections and LayerNorm around both. This **transformer block** is stacked many times.

**Self-attention vs. cross-attention.** In *self-attention*, queries, keys and values all come from the same data. In *cross-attention*, queries come from one source and keys and values from another — for example, queries from a BEV grid and keys and values from camera images (Section 3.5.7).

**Well-known examples.**

| Model | Key idea |
|---|---|
| **Transformer** (Vaswani et al., 2017) | Original encoder–decoder transformer for language |
| **ViT** (Vision Transformer) | Treat an image as a sequence of patches and process it with a transformer |
| **DETR** | Detects objects by letting a set of learned queries attend to image features, which removes many hand-designed detection components |

**Strengths and limits.** Transformers capture long-range relationships well and can be trained efficiently in parallel. The cost of attention grows with the *square* of the number of tokens, and transformers usually need a lot of training data.

**In vehicles.** Image backbones, detection heads, camera-to-BEV conversion, trajectory prediction and fusion of multiple sensors.

### 3.5.6 Point-Cloud Networks: PointNet and PointPillars

**The data.** A LiDAR sensor measures distances with laser pulses and produces a **point cloud**: a list of 3D points, each with a position $(x,y,z)$ and usually a reflection intensity $r$. This is very different from an image:

- It is **unordered** (shuffling the list does not change the scene).
- It is **sparse** (most of space is empty).
- The number of points varies from scan to scan.

Standard CNNs expect a regular grid, so point clouds need a different approach.

**PointNet.** Processes each point with the same small MLP and then combines all points with a symmetric operation (max pooling) that does not depend on their order. It was the first network to work directly on raw point clouds.

**PointPillars.** A fast and widely used LiDAR detector that gets the best of both worlds: it turns the point cloud into a pseudo-image, so that fast 2D CNNs can be used. The steps are:

1. **Pillars.** Divide the ground plane (viewed from above) into a regular grid. Each cell, extended over the full height, is a vertical column — a **pillar**. Every point falls into one pillar.
2. **Pillar features.** Each point is described by its own values $(x,y,z,r)$ plus its offsets from the pillar centre and from the average of the points in the pillar. A small PointNet-style network (linear layer, BatchNorm, ReLU, then max pooling over the points in the pillar) turns each pillar into one feature vector.
3. **Pseudo-image.** The pillar feature vectors are placed back on the grid, which creates an image-like tensor of shape $C \times H \times W$ (channels $\times$ height $\times$ width).
4. **2D CNN backbone.** An ordinary CNN processes the pseudo-image at several scales.
5. **Detection head.** A final layer predicts 3D bounding boxes (position, size, orientation) and a class for each object.

```
Point cloud -> Pillars -> PointNet per pillar -> Pseudo-image -> 2D CNN -> 3D boxes
```

**Related networks.** **SECOND** (uses 3D sparse convolutions on voxels) and **CenterPoint** (detects objects as centre points on a BEV grid, then regresses box size and orientation) are other widely used LiDAR detectors.

**In vehicles.** 3D object detection from LiDAR: finding the position, size and heading of cars, pedestrians and cyclists.

### 3.5.7 Bird's-Eye-View (BEV) Networks

**Idea.** A **bird's-eye view (BEV)** is a top-down map of the surroundings, as if seen from a drone hovering above the vehicle. In this view, distances and positions can be read directly, objects do not appear larger or smaller depending on how close they are, and data from several sensors can be placed in one common frame. This makes BEV a natural "common language" for perception, prediction and planning.

**The challenge.** Cameras see the world in a perspective view, not top-down, and a single camera image does not tell how far away things are. A BEV network must therefore convert from camera images to the top-down map.

**Main approaches.**

| Model | How it builds the BEV |
|---|---|
| **Lift-Splat-Shoot (LSS)** | **Lift:** for each pixel, predict a distribution over possible depths and create a "frustum" of 3D features along the viewing ray. **Splat:** add these features into the cells of a BEV grid. **Shoot:** use the BEV for tasks such as planning. |
| **BEVFormer** | Uses a grid of learnable **BEV queries**. **Spatial cross-attention** lets each BEV cell look up the relevant camera images at the right locations; **temporal self-attention** lets it also use the BEV from the previous frame. |
| **BEVFusion** | Builds a BEV feature map from the cameras and another from LiDAR, then fuses the two into a single BEV representation. |

```
Camera images (multiple views)  ->  image backbone  ->  view transform  ->  BEV features -> heads
LiDAR point cloud               ->  PointPillars-style encoder ------------>  BEV features ↗
                                                                    (fuse)       (detection, segmentation, ...)
```

**In vehicles.** 3D detection and tracking from several cameras, map and lane segmentation in the BEV, fusion of camera and LiDAR, and the input to motion prediction and planning.

### 3.5.8 Other Architectures (Brief Overview)

The following are active areas that students will meet in the literature. They are introduced here only by name and one-line idea:

- **Graph Neural Networks (GNN).** Work on data that forms a graph — nodes connected by edges. In driving, lanes, vehicles and pedestrians can be nodes, and their relationships the edges (for example **VectorNet** for map and trajectory prediction).
- **Occupancy networks.** Predict, for each small 3D cell around the vehicle, whether it is occupied, instead of drawing boxes around known object classes.
- **End-to-end driving models.** A single network that maps from sensor input directly to a driving plan or control output, replacing separate perception, prediction and planning modules (see Chapter 5).
- **Diffusion and other generative models.** Learn to generate realistic new data (images, scenes, trajectories); one use is creating synthetic training data (see Chapter 11).
- **Vision-language models.** Combine images and text in a single model, for example to describe a scene or answer questions about it.

## 3.6 Putting It Together: A Modern Perception Network

Most modern perception networks, whatever the task, are built from the same three parts:

| Part | Role | Typical building blocks |
|---|---|---|
| **Backbone** | Reads the raw input and extracts features | ResNet, EfficientNet, ViT; PointPillars encoder for LiDAR |
| **Neck** | Combines features from different depths or scales, so that both small (far-away) and large (nearby) objects are handled | Feature Pyramid Network (FPN), PAN, BEV view transform |
| **Head** | Produces the final task output | Detection head (boxes + classes), segmentation head (per-pixel labels), regression head |

### Worked example: a YOLO-style camera detector

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
- **Training mechanics** (Section 3.3): a loss that combines box-overlap (IoU), objectness and classification terms, minimised with an optimiser and a learning-rate schedule.
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
