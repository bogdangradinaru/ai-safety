
# 2. AI Learning Methods and Training Paradigms

## 2.1 Introduction

Artificial Intelligence systems can be trained using different learning approaches, depending on the available data, the training objectives, and the desired system behaviour.

The main learning paradigms include:

- **Supervised Learning**
- **Self-Supervised Learning**
- **Unsupervised Learning**
- **Semi-Supervised Learning**
- **Reinforcement Learning**
- **Imitation Learning**

The choice of learning paradigm is not only an engineering decision — it is also a **safety-relevant** decision. Each paradigm makes different assumptions about data, and each has characteristic failure modes that a safety case must account for. For example, a model trained by imitation learning can fail in ways a model trained by supervised learning on the same data would not (Section 2.7). This chapter introduces each paradigm, its automotive applications, and — for each — the specific safety implications developed further in Section 2.10 and in Part III and Part IV of this book.

## 2.2 Supervised Learning

Supervised learning is a machine learning approach in which a model is trained using labelled data. Formally, given a dataset of input–label pairs $\{(x_i, y_i)\}_{i=1}^{N}$, training finds model parameters $\theta$ that minimise a loss function:

$$
\theta^{*} = \arg\min_{\theta} \; \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}\big(f_\theta(x_i),\, y_i\big)
$$

Each training example contains:
- **Input:** e.g. an image captured by an automotive camera.
- **Label:** e.g. a bounding box identifying a pedestrian.
- **Output:** the model prediction $f_\theta(x_i)$, which is compared against the ground-truth label $y_i$ via the loss $\mathcal{L}$.

### Automotive applications

- Object detection (vehicles, pedestrians, cyclists).
- Traffic sign classification.
- Lane detection.
- Semantic segmentation.

#### Try it: labels shape a classifier

```{=html}
<div id="supervised-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Width threshold <input type="range" id="supervised-threshold" min="0.1" max="0.9" step="0.01" value="0.5"> <span id="supervised-threshold-value"></span></label>
<label>Training labels <select id="supervised-labels"><option value="clean">Clean</option><option value="one-flipped">One wrong label</option><option value="three-flipped">Three wrong labels</option></select></label>
<label>Point class <select id="supervised-point-class"><option value="0">Blue</option><option value="1">Orange</option></select></label>
<button type="button" id="supervised-reset">Reset</button>
</div>
<canvas id="supervised-canvas" width="620" height="300" role="img" aria-label="Editable labelled examples and a movable classification threshold"></canvas>
<p class="aisw-hint">Choose Blue or Orange, then click empty space to add a point. Click a point to select it and use the class control to relabel it; double-click a point to remove it. Red rings mark incorrect training labels. Accuracy is calculated against the point's true label.</p>
<div id="supervised-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

### Automatic labelling and supervised learning

Labels do not necessarily have to be created manually.

Automated labelling techniques can generate labels using:
- Sensor fusion (e.g. LiDAR and camera data).
- Pre-trained AI models.
- Rule-based algorithms.
- Multi-frame tracking and temporal information.

The generated labels can subsequently be used to train another model through supervised learning.

> **Important:** Automatically generated labels still require quality assessment. Incorrect or incomplete labels can introduce systematic errors into the training dataset. Because supervised learning directly optimises against these labels, any systematic labelling bias (e.g. a detector that under-performs on children or on partially occluded pedestrians) is learned by the model as if it were correct behaviour.

## 2.3 Self-Supervised Learning

Self-supervised learning creates training objectives — known as **pretext tasks** — directly from the data itself, without requiring manually annotated ground-truth labels for the specific pre-training task. A label-free training signal $\tilde{y}$ is derived automatically from the input $x$ itself (e.g. a masked portion of it, or a second augmented view of it), and the model is trained as in supervised learning but against $\tilde{y}$ instead of a human-provided label.

### Examples

- Predicting masked regions of an image (masked-image modelling).
- Predicting future observations from previous frames.
- Learning representations by comparing different augmented views of the same scene (contrastive learning).

### Automotive applications

- Learning visual representations from large collections of driving videos.
- Pre-training perception models before fine-tuning on a smaller labelled dataset.
- Learning useful features from unlabeled sensor data (camera, LiDAR) collected across a vehicle fleet.

#### Try it: create a learning signal from the image

```{=html}
<div id="self-supervised-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Pretext task <select id="self-task"><option value="mask">Masked-image prediction</option><option value="contrast">Contrastive views</option></select></label>
<label>Augmentation <input type="range" id="self-strength" min="0.2" max="0.8" step="0.1" value="0.4"> <span id="self-strength-value"></span></label>
<button type="button" id="self-train">Training round</button>
<button type="button" id="self-reset">Reset</button>
</div>
<canvas id="self-supervised-canvas" width="620" height="260" role="img" aria-label="Masked image recovery or contrastive image views"></canvas>
<div id="self-task-output" class="aisw-stats"></div>
<p class="aisw-hint">The model derives its own target from the image. Masking asks it to recover hidden content; contrastive learning asks it to keep two augmented views of the same scene close in representation space. Neither task needs an object label.</p>
<div id="self-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

> **Note:** Self-supervised pretraining does not remove the need for labelled data — it typically reduces how much labelled data is required for a given level of performance, with the model later fine-tuned via supervised learning (Section 2.2).

## 2.4 Unsupervised Learning

Unsupervised learning identifies patterns, structures, or representations in data without conventional ground-truth labels, and without a pretext task constructed to mimic supervision (contrast with Section 2.3).

### Examples

- Clustering.
- Dimensionality reduction.
- Anomaly detection.

### Automotive applications

- Identifying unusual driving situations.
- Grouping similar driving scenarios (e.g. for test-case selection or dataset curation).
- Detecting anomalies in vehicle sensor data.

#### Try it: discover groups with k-means

```{=html}
<div id="unsupervised-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Number of clusters k <input type="range" id="cluster-count" min="2" max="4" step="1" value="3"> <span id="cluster-count-value"></span></label>
<label>Point colour <select id="cluster-point-color"><option value="0">Blue</option><option value="1">Orange</option><option value="2">Green</option></select></label>
<button type="button" id="cluster-step">Run iteration</button>
<button type="button" id="cluster-reset">Reset centres</button>
<button type="button" id="cluster-restore">Restore points</button>
</div>
<canvas id="unsupervised-canvas" width="620" height="320" role="img" aria-label="Editable scenario points clustered using k-means"></canvas>
<p class="aisw-hint">Choose a colour and click empty space to add a point. Click a point to select and recolour it; double-click to remove it. Run an iteration to discard the seed colours and recolour points by the clusters discovered from their positions.</p>
<div id="cluster-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

## 2.5 Semi-Supervised Learning

Semi-supervised learning combines a small amount of labelled data with a larger pool of unlabelled data. This is particularly relevant in automotive AI, where raw sensor data is abundant but manual annotation (e.g. drawing bounding boxes or segmentation masks) is expensive and slow.

A common technique is **pseudo-labelling**: a model trained on the available labelled data is used to generate labels for the unlabelled data, which are then treated as additional training targets (a form of automatic labelling, see Section 2.2). A related technique, **consistency regularisation**, trains the model to produce the same prediction for different augmented versions of the same unlabelled input.

### Automotive applications

- Extending a small, manually labelled perception dataset with pseudo-labelled data from a large unlabelled fleet dataset.
- Reducing annotation cost for rare object classes (e.g. construction equipment, animals).

#### Try it: build a larger training set with pseudo-labels

```{=html}
<div id="semi-supervised-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Confidence threshold <input type="range" id="pseudo-threshold" min="0.55" max="0.95" step="0.05" value="0.65"> <span id="pseudo-threshold-value"></span></label>
<label>Seed labels <select id="pseudo-seeds"><option value="clean">Clean</option><option value="noisy">One incorrect seed</option></select></label>
<button type="button" id="pseudo-round">Next round</button>
<button type="button" id="pseudo-reset">Reset</button>
</div>
<canvas id="semi-supervised-canvas" width="620" height="180" role="img" aria-label="Labelled seeds and accepted pseudo-labels on a one-dimensional feature axis"></canvas>
<p class="aisw-hint">Filled circles are labelled examples. Each round accepts only the most confident currently unlabelled example, retraining the one-dimensional classifier after every addition. A wrong seed or a low threshold can make the model reinforce its own mistaken boundary.</p>
<div id="pseudo-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

> **Important:** Because pseudo-labels are generated by the model itself, semi-supervised learning carries a risk of **confirmation bias**: if the initial model is systematically wrong on a certain input pattern (e.g. a rare occlusion case), it will generate incorrect pseudo-labels for similar unlabelled examples, and training on those labels reinforces the original error rather than correcting it.

## 2.6 Reinforcement Learning

Reinforcement learning (RL) is an approach in which an agent learns a policy through interactions with an environment, formalised as a Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$: a set of states $\mathcal{S}$, a set of actions $\mathcal{A}$, a transition model $P(s'\mid s,a)$, a reward function $R(s,a)$, and a discount factor $\gamma$. The agent learns a policy $\pi(a\mid s)$ that maximises expected cumulative (discounted) reward:

$$
\pi^{*} = \arg\max_{\pi} \; \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^{t} R(s_t, a_t)\right]
$$

### Main elements

1. **Agent:** the learning system.
2. **Environment:** the system with which the agent interacts.
3. **State:** the information available to the agent.
4. **Action:** the decision made by the agent.
5. **Reward:** feedback received after an action.

### Online vs. offline reinforcement learning

- **Online RL** — the agent learns by acting in the environment and observing the consequences in real time. Exploring freely in a real vehicle is unsafe, so online RL for driving is almost always restricted to simulation.
- **Offline (batch) RL** — the agent learns a policy from a fixed, previously collected dataset of interactions, without further real-time exploration. This is the more common setting when any real driving data is involved, precisely because it avoids unsafe exploratory actions on public roads.

### Automotive applications

- Learning driving policies in simulation.
- Lane-change and manoeuvre selection.
- Motion planning.
- Reward shaping for specific driving behaviours (e.g. comfort, efficiency, rule compliance).

#### Try it: change the reward, change the policy

```{=html}
<div id="reinforcement-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Near-miss penalty <select id="rl-penalty"><option value="1">1</option><option value="5" selected>5</option><option value="20">20</option></select></label>
<button type="button" id="rl-step">Run episode</button>
<button type="button" id="rl-train">Train 50 episodes</button>
<button type="button" id="rl-reset">Reset Q-values</button>
</div>
<canvas id="rl-canvas" width="520" height="360" role="img" aria-label="Small grid environment where an agent learns from rewards"></canvas>
<p class="aisw-hint">The agent starts at bottom left and learns a route to the top-right goal. Red cells are near-miss hazards. Change their penalty and retrain: the destination stays fixed, but the value of each route can change.</p>
<div id="rl-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

> **Important — reward specification.** An RL agent optimises exactly the reward function it is given, not the designer's underlying intent. A poorly specified reward (e.g. one that rewards progress without sufficiently penalising near-misses) can lead to **reward hacking**: a policy that scores well on the specified reward while behaving in an unintended or unsafe way. This is a central concern when applying RL to driving and is revisited in Section 2.10.

## 2.7 Imitation Learning

Imitation learning trains a model to reproduce decisions or behaviours demonstrated by an expert, typically a human driver. In its simplest form, **behaviour cloning**, imitation learning is treated as supervised learning over expert state–action pairs $\{(s_i, a_i)\}$:

$$
\theta^{*} = \arg\min_{\theta} \; \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}\big(\pi_\theta(s_i),\, a_i\big)
$$

### Automotive applications

- Learning driving policies from human driving demonstrations.
- Behaviour cloning.
- Learning vehicle manoeuvres from expert driving data.

> **Important — distributional shift.** Behaviour cloning is trained only on the states an expert actually visited. Once deployed, small prediction errors cause the vehicle to drift into states slightly different from the training distribution; because those states were rarely or never demonstrated, the model's errors there tend to be larger, driving the vehicle further off-distribution. This **compounding error** problem is one of the best-known failure modes in automotive AI and is distinct from the failure modes of supervised learning on i.i.d. data. A standard mitigation is **DAgger (Dataset Aggregation)**: the learned policy is run in the environment (in simulation), the expert labels the *actual* visited states with the correct action, and these corrective examples are added to the training set — iteratively closing the gap between the training distribution and the states the policy actually encounters.

#### Try it: rollout error and corrective demonstrations

```{=html}
<div id="imitation-widget" class="aisafety-widget">
<div class="aisw-controls">
<label>Training data <select id="imitation-method"><option value="cloning">Expert demonstrations only</option><option value="dagger">Expert + rollout corrections</option></select></label>
<label>Control error <input type="range" id="imitation-error" min="0.1" max="1" step="0.1" value="0.5"> <span id="imitation-error-value"></span></label>
<button type="button" id="imitation-step">Next time step</button>
<button type="button" id="imitation-reset">Reset rollout</button>
</div>
<canvas id="imitation-canvas" width="620" height="280" role="img" aria-label="Compounding lateral drift during an imitation-learning rollout"></canvas>
<p class="aisw-hint">Advance the rollout one time step at a time. Plain behaviour cloning repeatedly applies a weaker correction learned only on expert states. The DAgger-style option includes corrective examples from off-centre states and returns more quickly toward the lane centre.</p>
<div id="imitation-stats" class="aisw-stats" aria-live="polite"></div>
</div>
```

<script src="learning-methods-widgets.js"></script>

## 2.8 Comparison of Learning Paradigms

| Learning method | Training data | Learning objective | Typical safety-relevant failure mode |
|---|---|---|---|
| Supervised | Labelled data | Predict known labels | Systematic label bias is learned as correct behaviour |
| Self-supervised | Unlabelled data | Generate learning targets from data | Pretext task may not transfer to the safety-relevant downstream task |
| Unsupervised | Unlabelled data | Discover data structure | Discovered structure may not align with safety-relevant distinctions |
| Semi-supervised | Labelled + unlabelled data | Combine both data sources | Confirmation bias from incorrect pseudo-labels |
| Reinforcement | Interaction and rewards | Learn an effective policy | Reward hacking / specification gaming; unsafe exploration if learning online |
| Imitation | Expert demonstrations | Reproduce demonstrated behaviour | Compounding error under distributional shift |

## 2.9 Combining Learning Paradigms in Practice

### Transfer learning and fine-tuning

**Transfer learning** reuses knowledge acquired on a source task or domain as the starting point for learning a target task or operating in a target domain. It can compensate for limited target-domain data or reduce the training time and data required to reach the target level of performance. In automotive AI, the source may be a general visual-recognition task, a different driving domain, or a simulation environment, while the target may be object detection in a fleet dataset, operation under a different sensor configuration, or deployment in a different geographic region.

Transfer-learning methods can be classified by the information available in the source and target domains:

- **Inductive transfer learning** uses labelled source information to support a related but different target task for which some labelled target data is available.
- **Transductive transfer learning** transfers knowledge between domains while the source and target tasks remain the same; labelled source data may be used when target-domain labels are scarce or unavailable.
- **Unsupervised transfer learning** transfers knowledge for unsupervised tasks when labelled data is unavailable in both the source and target domains.

A second classification concerns the compatibility of the source and target representations. **Homogeneous transfer learning** applies when their feature and label spaces are consistent or sufficiently aligned. **Heterogeneous transfer learning** is needed when the domains use different feature spaces, and may also require bridging different label spaces; for example, transferring knowledge between camera- and lidar-based perception systems can require an explicit alignment between their representations.

Transfer can occur through different forms of **fine-tuning**:

- **Feature extraction** keeps the pretrained model as a fixed representation extractor and trains a new task-specific head.
- **Partial fine-tuning** updates the task-specific components and a subset of the pretrained layers.
- **Full fine-tuning** updates all or most model parameters using target-task data.

It is also useful to distinguish between **task transfer**, where a model learned for one prediction task is adapted to another, and **domain adaptation**, where the task remains similar but the source and target data distributions differ. For example, a vision backbone pretrained by self-supervised learning on unlabelled images can be adapted to supervised pedestrian detection, while a model trained in simulation can be fine-tuned using real driving data to reduce the sim-to-real domain gap.

Transfer learning is therefore not a separate learning paradigm alongside supervised or reinforcement learning. It is a training strategy that can be combined with those paradigms to reuse previously learned representations and reduce the target data or training effort required.

Reusing a model trained on a large and diverse corpus can improve the generalisation and robustness of a model by supplying representations learned from more varied examples than the target dataset alone provides. These benefits are not automatic: the transferred model can also inherit biases or other undesirable emergent properties from the source data, training objective, or model. If the source and target domains, tasks, feature spaces, or label spaces are insufficiently related, adaptation can even reduce performance compared with training for the target domain alone. This is called **negative transfer**.

For safety-related AI, the decision to transfer should therefore begin with an assessment of relatedness between the source and target domains, tasks, features, and labels, including differences in sensors, environments, operating conditions, and safety-relevant classes. That assessment should guide which knowledge is extracted and which components are frozen, adapted, replaced, or trained from scratch. Transfer learning does not remove the inherited risks or replace target-domain evidence: representative verification and validation must demonstrate that the adapted model meets the target domain's performance and safety KPIs, including measures of robustness, bias, and behaviour on rare or out-of-distribution cases.

In practice, automotive AI systems rarely rely on a single learning paradigm. A common pattern is:

1. **Self-supervised pretraining** on a large pool of unlabelled fleet data to learn general-purpose representations that can be reused through transfer learning (Section 2.3).
2. **Supervised fine-tuning** on a smaller, labelled (manually or automatically annotated) dataset for the specific task, such as object detection (Section 2.2).
3. **Semi-supervised extension** of the labelled set using pseudo-labelling to cover rare cases more cheaply (Section 2.5).
4. **Imitation and/or reinforcement learning**, often in simulation, to learn or refine a driving policy on top of the resulting perception and prediction models (Sections 2.6–2.7).

> **Note on industry practice:** Different manufacturers and suppliers combine these paradigms in different, and evolving, ways, and the details of any specific production pipeline are typically only partially disclosed. This book deliberately focuses on the general principles above rather than on claims about a specific company's current system; where a concrete industrial example is needed, it will be explicitly cited to a primary source (e.g. a published paper, engineering blog, or conference talk) rather than asserted as general fact.

## 2.10 Safety Implications by Learning Paradigm

Each learning paradigm shapes *how* a model can fail, which in turn shapes what verification and validation evidence is needed:

- **Supervised learning** failures trace back to the labelled dataset: coverage gaps and annotation errors become model errors. This motivates the data-quality and bias topics in Part II.
- **Self-supervised / unsupervised learning** failures are harder to characterise directly, since there is no task-specific ground truth at pretraining time; safety evidence must instead come from evaluating the downstream, fine-tuned task.
- **Semi-supervised learning** adds the specific risk of confirmation bias compounding across pseudo-labelling rounds.
- **Reinforcement learning** adds the risk of reward misspecification and, where learning happens online, the risk inherent in exploratory actions themselves.
- **Imitation learning** adds the risk of compounding error under distributional shift between training (expert) and deployment (learned-policy) state distributions.

These paradigm-specific risks are one reason ISO 21448 (SOTIF) treats performance limitations of a function — rather than only classical malfunctions — as a source of hazard, and why ISO/PAS 8800 defines lifecycle activities (including data management and model verification) that differ by how the model was trained. Both standards are covered in depth in Part III; the evaluation techniques used to detect these failure modes (e.g. uncertainty estimation, out-of-distribution detection) are covered in Part IV.

## 2.11 Summary

Different learning paradigms address different training challenges, and in automotive applications they are typically combined rather than used in isolation — for example, a perception model pre-trained using self-supervised learning and subsequently fine-tuned using supervised learning with manually or automatically generated labels. Critically for this book, each paradigm also carries a distinct profile of safety-relevant failure modes — label bias, confirmation bias, reward hacking, and compounding error under distributional shift — which must be accounted for in the safety case for any AI-based automotive function.

## Further Reading

- Ross, S., Bagnell, D. — *Efficient Reductions for Imitation Learning* (introduces the compounding-error problem in behaviour cloning).
- Ross, S., Gordon, G., Bagnell, D. — *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (the DAgger algorithm).
- Amodei, D. et al. — *Concrete Problems in AI Safety* (reward specification and related RL safety issues).
- Chen, T. et al. — *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)* (contrastive self-supervised learning).
- He, K. et al. — *Masked Autoencoders Are Scalable Vision Learners* (masked-image-modelling self-supervised learning).
- ISO 21448 — *Road vehicles – Safety of the intended functionality (SOTIF)*.
- ISO/PAS 8800 — *Road vehicles – Safety and artificial intelligence*.

## Practical Lab 1 — Automotive Object Detection

In this exercise, we will train a YOLO object detection model using annotated images from the KITTI dataset.

**Learning objectives:**
- Understand ground-truth annotations.
- Train an object detection model.
- Evaluate detection performance.
- Investigate AI safety implications.

**Tools:** KITTI, YOLO, PyTorch, Google Colab.

[Open notebook in Google Colab](https://colab.research.google.com/github/bogdangradinaru/ai-safety/blob/main/notebooks/part-1/01-supervised-kitti-yolo.ipynb)

[View notebook on GitHub](https://github.com/bogdangradinaru/ai-safety/blob/main/notebooks/part-1/01-supervised-kitti-yolo.ipynb)
