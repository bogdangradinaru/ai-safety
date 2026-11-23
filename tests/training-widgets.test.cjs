const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../src/part-1/training-widgets.js"), "utf8");

function setup() {
  const defaults = {
    "batch-size": "8",
    "optim-method": "sgd", "optim-rate": "0.10",
    "schedule-kind": "step", "schedule-iteration": "0",
    "fit-model": "quadratic",
    "curves-scenario": "over", "curves-epoch": "20",
    "hyper-rate": "0.05", "hyper-epochs": "80", "hyper-lambda": "0",
    "hyper-batch": "4", "hyper-depth": "1", "hyper-width": "4", "hyper-optimiser": "sgd"
  };
  const elements = new Map();
  const context = new Proxy({}, {
    get(target, key) {
      if (!(key in target)) target[key] = (...args) => {
        for (const arg of args) {
          if (typeof arg === "number") assert.ok(Number.isFinite(arg), `${String(key)} received ${arg}`);
        }
      };
      return target[key];
    }
  });
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {
          value: defaults[id] || "", textContent: "", disabled: false,
          width: 560, height: 250, listeners: {},
          getContext: () => context,
          addEventListener(type, handler) { this.listeners[type] = handler; }
        });
      }
      return elements.get(id);
    }
  };
  vm.runInNewContext(script, { document });
  function get(id) { return document.getElementById(id); }
  function trigger(id, type, value) {
    const element = get(id);
    if (value !== undefined) element.value = value;
    element.listeners[type]();
  }
  return { get, trigger };
}

test("mini-batches compute gradients and count epochs", () => {
  const { get, trigger } = setup();
  trigger("batch-step", "click");
  assert.match(get("batch-stats").textContent, /Prediction θ = 0\.700.*Completed epochs = 1/);
  assert.match(get("batch-stats").textContent, /batch gradient before update = -7\.000/);
  trigger("batch-size", "change", "1");
  trigger("batch-step", "click");
  assert.match(get("batch-stats").textContent, /Prediction θ = 0\.000.*Completed epochs = 0/);
  assert.match(get("batch-stats").textContent, /batch gradient before update = 0\.000; full gradient before update = -7\.000/);
  for (let i = 0; i < 7; i++) trigger("batch-step", "click");
  assert.match(get("batch-stats").textContent, /Updates = 8 \| Completed epochs = 1/);
});

test("optimisers use their actual updates and reset on changes", () => {
  const { get, trigger } = setup();
  trigger("optim-step", "click");
  assert.match(get("optim-stats").textContent, /Step 1 \| \(x, y\) = \(3\.600, 0\.000\)/);
  trigger("optim-method", "change", "momentum");
  trigger("optim-step", "click");
  trigger("optim-step", "click");
  assert.match(get("optim-stats").textContent, /Step 2 \| \(x, y\) = \(2\.880, -3\.600\)/);
  trigger("optim-method", "change", "adam");
  trigger("optim-step", "click");
  assert.match(get("optim-stats").textContent, /Step 1 \| \(x, y\) = \(3\.900, 3\.900\)/);
  trigger("optim-method", "change", "rmsprop");
  trigger("optim-step", "click");
  assert.match(get("optim-stats").textContent, /Step 1 \| \(x, y\) = \(3\.684, 3\.684\)/);
  trigger("optim-rate", "input", "0.25");
  assert.match(get("optim-stats").textContent, /Step 0/);
});

test("schedules have correct warm-up, step and cosine endpoints", () => {
  const { get, trigger } = setup();
  trigger("schedule-iteration", "input", "10");
  assert.match(get("schedule-stats").textContent, /0\.05000/);
  trigger("schedule-kind", "change", "cosine");
  trigger("schedule-iteration", "input", "20");
  assert.match(get("schedule-stats").textContent, /0\.05000/);
  trigger("schedule-iteration", "input", "40");
  assert.match(get("schedule-stats").textContent, /0\.00000/);
  trigger("schedule-kind", "change", "cycle");
  trigger("schedule-iteration", "input", "8");
  assert.match(get("schedule-stats").textContent, /0\.10000/);
  trigger("schedule-iteration", "input", "40");
  assert.match(get("schedule-stats").textContent, /0\.00000/);
  trigger("schedule-kind", "change", "warmup");
  assert.match(get("schedule-stats").textContent, /0\.10000/);
});

test("model capacity illustrates underfitting and overfitting on held-out points", () => {
  const { get, trigger } = setup();
  function errors() {
    const match = get("fit-stats").textContent.match(/Training MSE = (\d+\.\d+) \| Validation MSE = (\d+\.\d+)/);
    return [Number(match[1]), Number(match[2])];
  }
  const quadratic = errors();
  trigger("fit-model", "change", "linear");
  const linear = errors();
  trigger("fit-model", "change", "interpolating");
  const overfit = errors();
  assert.ok(linear[0] > quadratic[0] && linear[1] > quadratic[1], JSON.stringify({ linear, quadratic }));
  assert.ok(overfit[0] < quadratic[0] && overfit[1] > quadratic[1], JSON.stringify({ overfit, quadratic }));
});

test("learning curves react to selections", () => {
  const { get, trigger } = setup();
  assert.match(get("curves-stats").textContent, /Lowest displayed validation loss = .* at epoch (?!40)\d+/);
  trigger("curves-scenario", "change", "good");
  assert.match(get("curves-stats").textContent, /at epoch 40/);
});

test("hyperparameter runs use minibatches, architecture and optimiser selections", () => {
  const { get, trigger } = setup();
  const baseline = get("hyper-stats").textContent;
  assert.match(baseline, /Epoch 80 .*Parameters = 13 \| Updates = 160/);
  trigger("hyper-batch", "change", "8");
  assert.match(get("hyper-stats").textContent, /Parameters = 13 \| Updates = 80/);
  assert.notEqual(get("hyper-stats").textContent, baseline);
  trigger("hyper-batch", "change", "4");
  assert.equal(get("hyper-stats").textContent, baseline);
  trigger("hyper-depth", "change", "2");
  assert.match(get("hyper-stats").textContent, /Parameters = 33/);
  trigger("hyper-width", "change", "2");
  assert.match(get("hyper-stats").textContent, /Parameters = 13/);
  const sgd = get("hyper-stats").textContent;
  trigger("hyper-optimiser", "change", "momentum");
  const momentum = get("hyper-stats").textContent;
  assert.notEqual(momentum, sgd);
  trigger("hyper-optimiser", "change", "adam");
  assert.notEqual(get("hyper-stats").textContent, momentum);
  assert.notEqual(get("hyper-stats").textContent, sgd);
  const adam = get("hyper-stats").textContent;
  trigger("hyper-lambda", "change", "1");
  assert.notEqual(get("hyper-stats").textContent, adam);
});

test("one full-batch SGD update matches numerical gradients including L2", () => {
  const { get, trigger } = setup();
  trigger("hyper-width", "change", "2");
  trigger("hyper-batch", "change", "8");
  trigger("hyper-lambda", "change", "0.1");
  trigger("hyper-epochs", "input", "1");

  let state = 42;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const bound = Math.sqrt(2);
  const weights = Array.from({ length: 4 }, () => (2 * random() - 1) * bound);
  const values = [...weights, 0, 0, 0];
  const trainX = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1];
  const trainY = [1.05, 0.43, 0.35, -0.02, 0.16, 0.18, 0.68, 0.94];
  const validX = [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875];
  const predict = (p, x) => p[2] * Math.tanh(p[0] * x + p[4]) +
    p[3] * Math.tanh(p[1] * x + p[5]) + p[6];
  const mse = (p, xs, ys) => xs.reduce((sum, x, i) => sum + (predict(p, x) - ys[i]) ** 2, 0) / xs.length;
  const objective = (p) => mse(p, trainX, trainY) + 0.1 * p.slice(0, 4).reduce((sum, w) => sum + w * w, 0);
  const epsilon = 1e-5;
  const updated = values.map((value, i) => {
    const plus = values.slice(), minus = values.slice();
    plus[i] += epsilon;
    minus[i] -= epsilon;
    return value - 0.05 * (objective(plus) - objective(minus)) / (2 * epsilon);
  });
  const match = get("hyper-stats").textContent.match(/Training MSE = (\d+\.\d+) \| Validation MSE = (\d+\.\d+)/);
  assert.ok(match);
  assert.ok(Math.abs(Number(match[1]) - mse(updated, trainX, trainY)) < 0.00006);
  assert.ok(Math.abs(Number(match[2]) - mse(updated, validX, validX.map((x) => x * x))) < 0.00006);
});

test("unstable hyperparameter runs report divergence", () => {
  const { get, trigger } = setup();
  trigger("hyper-lambda", "change", "1");
  trigger("hyper-rate", "change", "1.5");
  assert.match(get("hyper-stats").textContent, /Training diverged/);
});
