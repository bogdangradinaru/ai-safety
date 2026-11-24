const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../src/part-1/learning-methods-widgets.js"), "utf8");

function setup() {
  const defaults = {
    "supervised-threshold": "0.5", "supervised-labels": "clean",
    "self-task": "mask", "self-strength": "0.4",
    "cluster-count": "3",
    "pseudo-threshold": "0.65", "pseudo-seeds": "clean",
    "rl-penalty": "5",
    "imitation-method": "cloning", "imitation-error": "0.5"
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
          value: defaults[id] || "", textContent: "", disabled: false, listeners: {},
          width: id.includes("rl-") ? 520 : 620,
          height: 300,
          getContext: () => context,
          addEventListener(type, handler) { this.listeners[type] = handler; }
        });
      }
      return elements.get(id);
    }
  };
  vm.runInNewContext(script, { document, Math, Number, Infinity, NaN });
  const get = (id) => document.getElementById(id);
  const trigger = (id, type, value) => {
    const element = get(id);
    if (value !== undefined) element.value = value;
    element.listeners[type]();
  };
  return { get, trigger };
}

test("supervised example separates label quality from true performance", () => {
  const { get, trigger } = setup();
  assert.match(get("supervised-stats").textContent, /Accuracy against true labels = 100%.*Incorrect training labels = 0\/10/);
  trigger("supervised-threshold", "input", "0.65");
  assert.match(get("supervised-stats").textContent, /Accuracy against true labels = 80%/);
  trigger("supervised-labels", "change", "three-flipped");
  assert.match(get("supervised-stats").textContent, /Incorrect training labels = 3\/10/);
});

test("self-supervised tasks derive masking or contrastive targets without labels", () => {
  const { get, trigger } = setup();
  assert.match(get("self-task-output").textContent, /Recover 2 hidden image columns/);
  trigger("self-train", "click");
  assert.match(get("self-stats").textContent, /Round 1/);
  trigger("self-task", "change", "contrast");
  assert.match(get("self-task-output").textContent, /Pull two views together/);
  assert.match(get("self-stats").textContent, /positive-pair similarity/);
});

test("k-means iteration assigns points and decreases within-cluster distance", () => {
  const { get, trigger } = setup();
  assert.match(get("cluster-stats").textContent, /Run an iteration/);
  trigger("cluster-step", "click");
  const first = Number(get("cluster-stats").textContent.match(/distance = (\d+\.\d+)/)[1]);
  trigger("cluster-step", "click");
  const second = Number(get("cluster-stats").textContent.match(/distance = (\d+\.\d+)/)[1]);
  assert.ok(second <= first, `${second} should not exceed ${first}`);
  trigger("cluster-count", "input", "4");
  assert.match(get("cluster-stats").textContent, /k = 4 \| Iteration = 0/);
});

test("pseudo-labelling rounds expose a bad seed and stricter thresholds", () => {
  const { get, trigger } = setup();
  trigger("pseudo-round", "click");
  const clean = get("pseudo-stats").textContent;
  assert.match(clean, /Pseudo-labels accepted = 1 \| Correct = 1/);
  trigger("pseudo-seeds", "change", "noisy");
  assert.match(get("pseudo-stats").textContent, /Seed-label error = 1 of 6/);
  trigger("pseudo-threshold", "input", "0.95");
  trigger("pseudo-round", "click");
  assert.match(get("pseudo-stats").textContent, /Pseudo-labels accepted = 0/);
});

test("reinforcement training accumulates episodes and responds to reward penalties", () => {
  const { get, trigger } = setup();
  assert.match(get("rl-stats").textContent, /Episodes trained = 0/);
  trigger("rl-train", "click");
  assert.match(get("rl-stats").textContent, /Episodes trained = 50.*Hazard penalty = 5\.0/);
  trigger("rl-penalty", "change", "20");
  assert.match(get("rl-stats").textContent, /Episodes trained = 0.*Hazard penalty = 20\.0/);
});

test("DAgger-style corrections reduce final imitation rollout drift", () => {
  const { get, trigger } = setup();
  for (let step = 0; step < 16; step++) trigger("imitation-step", "click");
  const cloned = Number(get("imitation-stats").textContent.match(/Final offset = (-?\d+\.\d+)/)[1]);
  trigger("imitation-method", "change", "dagger");
  for (let step = 0; step < 16; step++) trigger("imitation-step", "click");
  const corrected = Number(get("imitation-stats").textContent.match(/Final offset = (-?\d+\.\d+)/)[1]);
  assert.ok(Math.abs(corrected) < Math.abs(cloned), `${corrected} should be smaller than ${cloned}`);
  assert.equal(get("imitation-step").disabled, true);
});
