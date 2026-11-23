(function () {
  "use strict";

  const el = (id) => document.getElementById(id);
  const format = (values) => `[${values.map((value) => value.toFixed(2)).join(", ")}]`;

  function mlpExplorer() {
    if (!el("mlp-explorer")) return;
    const speed = el("mlp-speed"), proximity = el("mlp-proximity");
    let stage = 0;

    function draw() {
      const x = Number(speed.value), y = Number(proximity.value);
      const hidden = [Math.max(0, x + y - 0.5), Math.max(0, x - y + 0.2)];
      const output = 0.7 * hidden[0] - 0.4 * hidden[1] + 0.1;
      el("mlp-speed-value").textContent = x.toFixed(1);
      el("mlp-proximity-value").textContent = y.toFixed(1);
      el("mlp-input").textContent = format([x, y]);
      el("mlp-hidden").textContent = stage >= 1 ? format(hidden) : "Not computed";
      el("mlp-output").textContent = stage >= 2 ? output.toFixed(2) : "Not computed";
      el("mlp-stats").textContent = stage === 0
        ? "Press Next layer to calculate the hidden activations."
        : stage === 1
          ? `Hidden = ReLU([speed + proximity - 0.5, speed - proximity + 0.2]) = ${format(hidden)}.`
          : `Output = 0.7 × ${hidden[0].toFixed(2)} - 0.4 × ${hidden[1].toFixed(2)} + 0.1 = ${output.toFixed(2)}.`;
      el("mlp-step").disabled = stage === 2;
    }

    function reset() { stage = 0; draw(); }
    el("mlp-step").addEventListener("click", () => { if (stage < 2) stage++; draw(); });
    el("mlp-reset").addEventListener("click", reset);
    speed.addEventListener("input", reset);
    proximity.addEventListener("input", reset);
    reset();
  }

  function rnnExplorer() {
    if (!el("rnn-explorer")) return;
    const sequence = el("rnn-sequence"), memory = el("rnn-memory");
    const sequences = {
      approach: [0.2, 0.4, 0.6, 0.8],
      recede: [0.8, 0.6, 0.4, 0.2],
      alternating: [0.8, -0.8, 0.8, -0.8]
    };
    let step = 0, state = 0, previous = 0;

    function draw() {
      const values = sequences[sequence.value];
      el("rnn-memory-value").textContent = Number(memory.value).toFixed(1);
      el("rnn-input").textContent = format(values);
      el("rnn-previous").textContent = step ? previous.toFixed(3) : "0.000 (initial)";
      el("rnn-state").textContent = step ? state.toFixed(3) : "Not computed";
      el("rnn-stats").textContent = step
        ? `Step ${step} of ${values.length}: tanh(${values[step - 1].toFixed(1)} + ${Number(memory.value).toFixed(1)} × ${previous.toFixed(3)}) = ${state.toFixed(3)}.`
        : "Press Next time step to process the first measurement.";
      el("rnn-step").disabled = step === values.length;
    }

    function reset() { step = 0; state = 0; previous = 0; draw(); }
    el("rnn-step").addEventListener("click", () => {
      const values = sequences[sequence.value];
      if (step < values.length) {
        previous = state;
        state = Math.tanh(values[step] + Number(memory.value) * previous);
        step++;
      }
      draw();
    });
    el("rnn-reset").addEventListener("click", reset);
    sequence.addEventListener("change", reset);
    memory.addEventListener("input", reset);
    reset();
  }

  function encoderExplorer() {
    if (!el("encoder-explorer")) return;
    const inputs = [1, 2, 3, 4].map((i) => el(`encoder-input-${i}`));
    let stage = 0;

    function draw() {
      const values = inputs.map((input) => Number(input.value));
      const latent = [(values[0] + values[1]) / 2, (values[2] + values[3]) / 2];
      const reconstructed = [latent[0], latent[0], latent[1], latent[1]];
      const error = values.reduce((sum, value, i) => sum + (value - reconstructed[i]) ** 2, 0) / values.length;
      values.forEach((value, i) => { el(`encoder-value-${i + 1}`).textContent = value.toFixed(1); });
      el("encoder-input").textContent = format(values);
      el("encoder-latent").textContent = stage >= 1 ? format(latent) : "Not computed";
      el("encoder-output").textContent = stage >= 2 ? format(reconstructed) : "Not computed";
      el("encoder-stats").textContent = stage === 0
        ? "Press Next stage to encode the input."
        : stage === 1
          ? `Encoder averages adjacent input pairs: ${format(latent)}.`
          : `Decoder repeats the two averages. Reconstruction MSE = ${error.toFixed(4)}.`;
      el("encoder-step").disabled = stage === 2;
    }

    function reset() { stage = 0; draw(); }
    el("encoder-step").addEventListener("click", () => { if (stage < 2) stage++; draw(); });
    el("encoder-reset").addEventListener("click", reset);
    inputs.forEach((input) => input.addEventListener("input", reset));
    reset();
  }

  mlpExplorer();
  rnnExplorer();
  encoderExplorer();
})();
