
# 1. Introduction to Artificial Intelligence

## 1.1 What is Artificial Intelligence?

Artificial Intelligence (AI) is the field of computer science concerned with building systems that perform tasks which typically require human intelligence, such as perceiving an environment, recognising patterns, reasoning about uncertain information, and making decisions.

AI is best understood as an umbrella term rather than a single technique. It includes:

- **Symbolic / rule-based approaches**, where behaviour is explicitly programmed by engineers (e.g. if–then logic, expert systems).
- **Statistical and data-driven approaches**, where behaviour is learned from data (machine learning and deep learning).

Modern automotive AI systems — such as perception and prediction modules in advanced driver assistance systems (ADAS) and autonomous driving (AD) stacks — are almost entirely built using the data-driven branch of AI, which is the focus of this book.

> **Note on scope:** All AI systems discussed in this book are examples of **narrow AI** — systems designed to perform a specific task (e.g. detecting pedestrians) well, as opposed to **general AI**, a hypothetical system with human-like, task-independent intelligence. General AI does not currently exist and is outside the scope of this book.

## 1.2 AI, Machine Learning, and Deep Learning

The terms *AI*, *Machine Learning (ML)*, and *Deep Learning (DL)* are often used interchangeably, but they describe nested concepts:

```
┌─────────────────────────────────────────────┐
│  Artificial Intelligence (AI)                │
│  Any technique that enables machines to      │
│  mimic aspects of human intelligence.        │
│                                               │
│   ┌───────────────────────────────────────┐  │
│   │  Machine Learning (ML)                │  │
│   │  Systems that learn patterns from      │  │
│   │  data rather than following explicitly │  │
│   │  programmed rules.                     │  │
│   │                                        │  │
│   │   ┌────────────────────────────────┐   │  │
│   │   │  Deep Learning (DL)            │   │  │
│   │   │  ML using multi-layer neural   │   │  │
│   │   │  networks to learn hierarchical │   │  │
│   │   │  representations directly from  │   │  │
│   │   │  raw data (e.g. pixels).        │   │  │
│   │   └────────────────────────────────┘   │  │
│   └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

| Term | Scope | Typical techniques | Automotive example |
|---|---|---|---|
| **AI** | Any method producing intelligent behaviour | Rule-based systems, ML, DL, search, planning | A finite-state machine deciding when to issue a forward-collision warning |
| **Machine Learning** | Learning patterns from data | Decision trees, SVMs, boosted trees, shallow neural networks | A classifier predicting lane-change intent from vehicle kinematics |
| **Deep Learning** | ML with deep neural networks | CNNs, transformers, graph neural networks | A CNN detecting pedestrians directly from camera images |

Chapter 2 covers *how* these systems learn (training paradigms); Chapter 3 covers *what* architectures they are built from.

## 1.3 A Brief History Relevant to Automotive AI

Understanding how automotive AI arrived at its current form helps explain both its capabilities and its safety challenges.

- **Rule-based era (1950s–1980s).** Early AI research focused on symbolic reasoning and expert systems with explicitly encoded rules. Behaviour was fully specified by engineers and, in principle, exhaustively testable.
- **Classical machine learning era (1980s–2000s).** Statistical methods (e.g. decision trees, support vector machines) and hand-crafted feature extraction (e.g. edge detectors, HOG features) enabled early vision-based driver assistance functions, such as lane-departure warning.
- **Deep learning era (2012–present).** The success of deep convolutional neural networks on large-scale image datasets shifted automotive perception from hand-crafted features to learned representations, substantially improving detection and classification performance.
- **Large-scale, data-driven fleets (2016–present).** Growth in fleet-collected driving data enabled training perception, prediction, and planning models on millions of real-world driving miles, combined with simulation.
- **Foundation models and end-to-end approaches (2020–present).** Large pretrained models (including transformer-based and multimodal architectures) are increasingly used as a starting point for perception and planning, fine-tuned for specific automotive tasks.

> **Important:** Specific industry implementations change rapidly and vary by manufacturer. This book focuses on durable, general principles rather than claims about any single company's current production system; where a specific system is used as an example, it will be clearly cited.

## 1.4 Core Concepts and Terminology

The following terms are used throughout this book and should be understood before proceeding to Chapter 2.

| Term | Definition | Automotive example |
|---|---|---|
| **Model** | A function, with learned parameters, that maps inputs to outputs | A neural network mapping a camera image to bounding boxes |
| **Parameters (weights)** | Numerical values learned during training that define the model's behaviour | The millions of weights in a perception CNN |
| **Training** | The process of adjusting parameters to minimise a loss function on training data | Training a traffic-sign classifier on labelled images |
| **Inference** | Using a trained, fixed model to produce predictions on new inputs | Running a trained model on live camera frames in the vehicle |
| **Dataset (train / validation / test split)** | Data partitioned to fit, tune, and evaluate a model without leaking information between stages | KITTI images split for training a detector and for measuring its accuracy |
| **Loss function** | A measure of the discrepancy between model predictions and desired outputs, minimised during training | Cross-entropy loss for classifying traffic signs |
| **Generalisation** | A model's ability to perform well on inputs not seen during training | A pedestrian detector performing well in a new city not in the training data |
| **Overfitting / underfitting** | Fitting noise in the training data (overfitting) vs. failing to capture real patterns (underfitting) | A model that memorises training-set vehicles but fails on new vehicle types |
| **Inductive bias** | Built-in assumptions that shape what a model can learn | Convolutional layers assuming spatial locality in images |

## 1.5 The AI System Lifecycle

AI systems are not just trained once and deployed — they follow a recurring lifecycle that this book returns to repeatedly, particularly in Part II (data) and Part IV (evaluation):

```
 ┌───────────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌───────────────┐
 │ Data           │→ │ Labelling  │→ │ Training  │→ │ Validation │→ │ Deployment  │→ │ Field          │
 │ collection     │  │ /          │  │           │  │            │  │             │  │ monitoring     │
 │                │  │ annotation │  │           │  │            │  │             │  │                │
 └───────────────┘   └───────────┘   └──────────┘   └────────────┘   └────────────┘   └───────┬───────┘
         ▲                                                                                      │
         └──────────────────────────────────────────────────────────────────────────────────────┘
                                   feedback: new / corrected data from the field
```

- **Data collection** — gathering sensor data (camera, LiDAR, radar) from real driving or simulation.
- **Labelling / annotation** — producing ground truth, manually or automatically (see Section 2.2 of Chapter 2).
- **Training** — fitting model parameters to the data using a chosen learning paradigm (Chapter 2).
- **Validation** — assessing performance before deployment (Part IV).
- **Deployment** — integrating the trained model into the vehicle's software stack (Chapter 6).
- **Field monitoring** — observing real-world performance and collecting new or edge-case data, closing the loop.

> **Important:** Unlike traditional software, an AI model's behaviour can change meaningfully between lifecycle iterations (e.g. after retraining on new data), which has direct consequences for safety assurance — a theme developed throughout Part III.

## 1.6 AI in the Autonomous Driving Stack

A typical autonomous driving software stack can be represented as a pipeline from sensing to actuation:

```
 Sensors  →  Perception  →  Prediction  →  Planning  →  Control  →  Actuation
 (camera,     (detection,     (trajectory    (path/        (steering,
  LiDAR,       segmentation,   forecasting    behaviour     throttle,
  radar)       localisation)   of agents)     planning)     braking)
```

- **Perception** — interpreting raw sensor data (e.g. detecting and classifying vehicles, pedestrians, and lane markings). Predominantly learning-based today.
- **Prediction** — forecasting the future behaviour of other road users. Increasingly learning-based.
- **Planning** — deciding the vehicle's intended trajectory or manoeuvre. Often a mix of learned and rule-based / optimisation-based methods.
- **Control** — converting a planned trajectory into actuator commands. Predominantly classical control theory (e.g. PID, model-predictive control), though learned control policies also exist.

### Automotive examples

- Object detection (vehicles, pedestrians, cyclists).
- Traffic sign and traffic light recognition.
- Lane and drivable-area detection.
- Trajectory prediction for surrounding agents.
- Behaviour and motion planning.
- Low-level vehicle control.

> **Note:** The boundary between "learned" and "rule-based" components varies by system architecture and is discussed in depth in Chapter 5 (Autonomous Driving System Architectures).

## 1.7 Why AI Safety Is Different From Traditional Functional Safety

Traditional safety-critical automotive software is engineered under assumptions that classical functional safety standards (e.g. ISO 26262) rely on:

- Behaviour is **explicitly specified** by requirements.
- Correct behaviour can, in principle, be **exhaustively verified** against that specification.
- The software is **deterministic**: identical inputs always produce identical outputs.

Data-driven AI systems challenge each of these assumptions:

| Assumption (traditional software) | AI reality |
|---|---|
| Fully specified behaviour | Behaviour is learned from data; no complete explicit specification exists |
| Exhaustive verification against spec | Performance is statistical; correctness is evaluated probabilistically on sample data, not proven for all inputs |
| Deterministic, input → output mapping is fixed | Behaviour depends on training data distribution and can degrade on inputs unlike anything seen in training |
| Failures are (mostly) specification or coding errors | Failures can emerge from data gaps, distributional shift, or statistically rare but valid inputs, even with "correct" code |

> **Important:** These differences are precisely why the automotive industry developed additional standards beyond ISO 26262:
> - **ISO 21448 (SOTIF)** addresses hazards arising from functional insufficiencies and performance limitations — including those of AI/ML components — rather than only from malfunctions.
> - **ISO/PAS 8800** addresses AI-specific safety considerations across the AI lifecycle described in Section 1.5, including data management, model training, and verification.
>
> These standards are introduced in depth in Part III of this book.

## 1.8 How This Book Is Organized

This book is structured in five parts:

- **Part I — AI Fundamentals** (this part): AI concepts, learning paradigms, model architectures, automotive applications, system architectures, and deployment.
- **Part II — Autonomous Driving Data**: public datasets, data formats, annotation, data quality and bias, data collection and augmentation.
- **Part III — Safety Principles and Standards**: ISO 26262, ISO 21448 (SOTIF), ISO/PAS 8800, and hazard/risk analysis for driving scenarios.
- **Part IV — AI Safety Evaluation**: performance metrics, dataset shift, uncertainty estimation, and testing/validation of perception systems.
- **Part V — Practical Projects**: applied exercises combining the concepts from Parts I–IV.

Each part builds on the previous one: the AI concepts introduced here in Part I are the vocabulary used to discuss data (Part II), which in turn is the basis for the safety standards and evaluation methods covered in Parts III and IV.

## 1.9 Summary

Artificial Intelligence — and specifically machine learning and deep learning — now forms the core of perception, prediction, and increasingly planning in automotive systems. Unlike traditional, fully specified software, AI systems learn their behaviour from data, following a recurring lifecycle of data collection, training, validation, deployment, and field monitoring. This data-driven nature is the fundamental reason why automotive AI requires safety engineering approaches beyond traditional functional safety — the subject developed throughout the rest of this book, starting with the learning paradigms covered in Chapter 2.

## Exercises

1. **Terminology check.** For each of the following, state whether it is best described as AI, ML, DL, or more than one of these: (a) a hand-written rule that triggers automatic emergency braking below a fixed distance threshold; (b) a decision tree predicting lane-change intent from vehicle speed and indicator state; (c) a convolutional neural network detecting pedestrians from a camera image.
2. **Lifecycle mapping.** A perception model performs well in testing but misclassifies pedestrians in heavy snow after deployment. Using the lifecycle diagram in Section 1.5, identify which stage(s) most plausibly contributed to this failure, and justify your answer.
3. **Specification vs. learning.** Compare a traditional cruise-control controller (implemented from explicit control equations) with a learned lane-keeping model trained on driving data. For each, discuss what engineering evidence would be needed to argue the system is safe, and how that evidence would differ between the two.
4. **Stack boundaries.** For the stack in Section 1.6 (perception, prediction, planning, control), give one example of a component that is typically rule-based/classical rather than learned, and explain why that design choice might be made for a safety-critical function.

## Further Reading

- Goodfellow, I., Bengio, Y., Courville, A. — *Deep Learning*, MIT Press.
- SAE J3016 — *Taxonomy and Definitions for Terms Related to Driving Automation Systems*.
- ISO 26262 — *Road vehicles – Functional safety*.
- ISO 21448 — *Road vehicles – Safety of the intended functionality (SOTIF)*.
- ISO/PAS 8800 — *Road vehicles – Safety and artificial intelligence*.
