const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../src/part-1/architecture-widgets.js"), "utf8");

function setup() {
  const defaults = {
    "mlp-speed": "0.4", "mlp-proximity": "0.6",
    "rnn-sequence": "approach", "rnn-memory": "0.5",
    "encoder-input-1": "0.2", "encoder-input-2": "0.8",
    "encoder-input-3": "0.4", "encoder-input-4": "0.6"
  };
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {
          value: defaults[id] || "", textContent: "", disabled: false, listeners: {},
          addEventListener(type, handler) { this.listeners[type] = handler; }
        });
      }
      return elements.get(id);
    }
  };
  vm.runInNewContext(script, { document });
  const get = (id) => document.getElementById(id);
  const trigger = (id, type, value) => {
    const element = get(id);
    if (value !== undefined) element.value = value;
    element.listeners[type]();
  };
  return { get, trigger };
}

test("MLP computes each dense layer and resets on input changes", () => {
  const { get, trigger } = setup();
  assert.equal(get("mlp-hidden").textContent, "Not computed");
  trigger("mlp-step", "click");
  assert.equal(get("mlp-hidden").textContent, "[0.50, 0.00]");
  assert.equal(get("mlp-output").textContent, "Not computed");
  trigger("mlp-step", "click");
  assert.equal(get("mlp-output").textContent, "0.45");
  assert.equal(get("mlp-step").disabled, true);
  trigger("mlp-proximity", "input", "0");
  assert.equal(get("mlp-output").textContent, "Not computed");
  assert.equal(get("mlp-step").disabled, false);
  trigger("mlp-step", "click");
  assert.equal(get("mlp-hidden").textContent, "[0.00, 0.60]");
  trigger("mlp-step", "click");
  assert.equal(get("mlp-output").textContent, "-0.14");
  trigger("mlp-reset", "click");
  assert.equal(get("mlp-hidden").textContent, "Not computed");
});

test("RNN carries state across time and resets when its settings change", () => {
  const { get, trigger } = setup();
  trigger("rnn-step", "click");
  assert.equal(get("rnn-previous").textContent, "0.000");
  assert.equal(get("rnn-state").textContent, Math.tanh(0.2).toFixed(3));
  trigger("rnn-step", "click");
  assert.equal(get("rnn-previous").textContent, Math.tanh(0.2).toFixed(3));
  assert.equal(get("rnn-state").textContent, Math.tanh(0.4 + 0.5 * Math.tanh(0.2)).toFixed(3));
  trigger("rnn-step", "click");
  trigger("rnn-step", "click");
  assert.equal(get("rnn-step").disabled, true);
  trigger("rnn-memory", "input", "0");
  assert.equal(get("rnn-step").disabled, false);
  assert.equal(get("rnn-state").textContent, "Not computed");
  trigger("rnn-step", "click");
  trigger("rnn-step", "click");
  assert.equal(get("rnn-state").textContent, Math.tanh(0.4).toFixed(3));
  trigger("rnn-sequence", "change", "alternating");
  assert.equal(get("rnn-state").textContent, "Not computed");
  trigger("rnn-step", "click");
  trigger("rnn-step", "click");
  assert.equal(get("rnn-state").textContent, Math.tanh(-0.8).toFixed(3));
});

test("autoencoder shows information loss and exact reconstructions", () => {
  const { get, trigger } = setup();
  trigger("encoder-step", "click");
  assert.equal(get("encoder-latent").textContent, "[0.50, 0.50]");
  trigger("encoder-step", "click");
  assert.equal(get("encoder-output").textContent, "[0.50, 0.50, 0.50, 0.50]");
  assert.match(get("encoder-stats").textContent, /MSE = 0\.0500/);
  assert.equal(get("encoder-step").disabled, true);
  trigger("encoder-input-2", "input", "0.2");
  trigger("encoder-input-4", "input", "0.4");
  assert.equal(get("encoder-output").textContent, "Not computed");
  trigger("encoder-step", "click");
  trigger("encoder-step", "click");
  assert.equal(get("encoder-output").textContent, "[0.20, 0.20, 0.40, 0.40]");
  assert.match(get("encoder-stats").textContent, /MSE = 0\.0000/);
});
