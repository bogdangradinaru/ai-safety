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
    "hyper-rate": "0.1", "hyper-epochs": "40", "hyper-lambda": "0"
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

test("learning curves and hyperparameter runs react to selections", () => {
  const { get, trigger } = setup();
  assert.match(get("curves-stats").textContent, /Lowest displayed validation loss = .* at epoch (?!40)\d+/);
  trigger("curves-scenario", "change", "good");
  assert.match(get("curves-stats").textContent, /at epoch 40/);
  const baseline = get("hyper-stats").textContent;
  assert.match(baseline, /Epoch 40/);
  trigger("hyper-lambda", "change", "1");
  assert.notEqual(get("hyper-stats").textContent, baseline);
  trigger("hyper-rate", "change", "1.5");
  trigger("hyper-epochs", "input", "80");
  assert.match(get("hyper-stats").textContent, /Training diverged/);
});
