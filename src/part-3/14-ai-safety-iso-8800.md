# 14. AI Safety and ISO 8800

## 14.1 Introduction

ISO 26262 and ISO 21448 do not give detailed guidance on how to engineer and assure functions built from machine-learning components. **ISO/PAS 8800 — Road vehicles – Safety and artificial intelligence**, published in December 2024, addresses this gap. It is a Publicly Available Specification (PAS) that provides guidance on the safety of AI within safety-related E/E systems in road vehicles, and is designed to be used together with ISO 26262 and ISO 21448.

> **Note:** This chapter summarises the structure and main ideas of the document at a conceptual level. It is not a substitute for the standard, and clause-level requirements should always be taken from the published text.

**Learning objectives.** After this chapter you should be able to:

- Explain why AI needs dedicated safety guidance beyond ISO 26262 and ISO 21448.
- Describe the main concepts: AI system, AI element, the AI safety lifecycle and its data lifecycle.
- Explain how AI-specific insufficiencies lead to safety-relevant failures.
- Outline the evidence types that make up an AI safety assurance argument.

## 14.2 Why a Dedicated Standard?

Recall the properties of ML components that conflict with classical assumptions (Chapter 12, Section 12.7):

- Behaviour is learned from data rather than specified.
- Failures are performance limitations rather than faults.
- Verification by inspection of code or structural coverage is of little use.
- Performance depends on the data distribution, which may shift after deployment.

ISO/PAS 8800 organises responses to these challenges into a coherent lifecycle with explicit treatment of **data**, **models**, **requirements**, **analysis**, and **assurance**.

## 14.3 Position in the Standards Landscape

| Document | Contribution |
|---|---|
| ISO 26262 | Hardware/software malfunctions, ASIL-based process |
| ISO 21448 | Functional insufficiencies and triggering conditions |
| **ISO/PAS 8800** | AI-specific lifecycle, data, requirements, V&V, and assurance |
| ISO/TR 5469, ISO/IEC TS 22440 (in development) | Functional safety and AI more generally |
| UL 4600 | Safety-case approach for autonomous products |
| EU AI Act | Regulation of AI systems in the EU; high-risk obligations for vehicles are largely linked to vehicle type-approval legislation |

## 14.4 Core Concepts

- **AI system** — a system that includes one or more AI elements together with conventional elements that interact with them (sensors, pre-/post-processing, actuators).
- **AI element** — the part of the system that implements a function using a trained model (e.g. a neural-network pedestrian detector).
- **Safety-related AI** — AI whose malfunction or insufficiency could contribute to violating a safety goal.
- **AI safety requirements** — safety requirements allocated to AI elements, which include performance, robustness, and data-related requirements.
- **Residual risk** — risk that remains after all measures; to be shown to be acceptable.

## 14.5 The AI Safety Lifecycle (Conceptual View)

The lifecycle follows the same logic as the V-model in ISO 26262 but with AI-specific activities:

1. **System-level safety requirements** from ISO 26262 and ISO 21448 analyses are allocated to the AI element.
2. **AI safety requirements** are derived — what the element must do (performance on specified classes and conditions), what it must not do, and under which conditions (ODD).
3. **Architecture** of the AI system — including **safety mechanisms** around the model, such as redundancy, monitors, plausibility checks, and fallback behaviour.
4. **Data lifecycle** — specification, collection, annotation, preparation, verification and maintenance of datasets (Part II). Data-related requirements typically cover aspects such as representativeness, correctness, completeness, independence of training and test data, and traceability.
5. **Model development** — selection, training, tuning, and the documentation of training procedures.
6. **Verification and validation** — evaluation against performance and robustness requirements on independent data, plus testing of edge cases, adversarial and out-of-distribution inputs (Part IV).
7. **Safety analysis** — analyses (such as FTA/FMEA-style methods adapted to AI) of how AI-specific insufficiencies propagate to system-level hazards.
8. **Operation and field monitoring** — detection of drift, collection of field data, incident analysis, and updates.
9. **Supporting processes** — configuration management of data, models and tools; change management; tool confidence.

## 14.6 AI-Specific Insufficiencies and Causal Reasoning

A central idea is to reason explicitly about **how** an AI element's behaviour can become insufficient. Typical causes include:

- **Data insufficiencies** — gaps, bias, label errors, or poor representativeness (Part II).
- **Model insufficiencies** — limited capacity, overfitting, sensitivity to small input changes, poor calibration.
- **Training-process insufficiencies** — inappropriate loss, hyper-parameters, or evaluation.
- **Deployment insufficiencies** — quantisation, compilation or hardware effects that change model behaviour (Chapter 6).
- **Operational change** — dataset shift, new environments, sensor ageing.

Establishing such causal chains makes it possible to **link each measure to the risk it addresses** — for instance, label review addresses annotation errors; uncertainty estimation addresses out-of-distribution inputs; diverse redundancy addresses common-cause model errors.

## 14.7 Assurance Argument

The goal is an **assurance argument**: structured claims, arguments and evidence, showing that the AI element's residual risk is acceptable. Evidence typically spans:

| Claim | Example evidence |
|---|---|
| Requirements are adequate and complete | Hazard analysis, ODD definition, scenario analysis |
| Data are suitable | Data specification, coverage analysis, annotation audit results |
| Model performance is sufficient | Per-slice metrics on independent test data, comparison to acceptance criteria |
| Model is robust | Perturbation, corruption and out-of-distribution test results |
| Safety mechanisms work | Monitor detection rates, fault-injection tests |
| Risk remains acceptable in operation | Field monitoring plan, incident-response process |

Arguments are often expressed in a notation such as the **Goal Structuring Notation (GSN)** and maintained as a living safety case.

## 14.8 Practical Consequences

- **Treat data as a safety artefact.** Datasets are specified, versioned, reviewed, and traced like software.
- **State the ODD explicitly** and test within it.
- **Prefer architecture over model perfection.** Because no learned model is perfect, system-level measures (diverse redundancy, monitors, safe fallback) are generally essential.
- **Plan for change.** Retraining or over-the-air updates require impact analysis and re-verification.
- **Document the limitations** of the model and of the evidence.

## 14.9 Summary

ISO/PAS 8800 adds the AI-specific layer to the automotive safety standards: it frames the AI element within a safety lifecycle, treats data and models as engineering artefacts, links AI-specific insufficiencies to safety measures, and structures the evidence into an assurance argument. It is applied together with ISO 26262 and ISO 21448 rather than instead of them.

## Exercises

1. **Mapping.** For a camera-based pedestrian detector, give one example each of a data insufficiency, a model insufficiency, and a deployment insufficiency, and a measure that addresses each.
2. **Evidence.** Draft a claim–argument–evidence structure (a few lines in GSN-style text) for the claim "the pedestrian detector is sufficiently robust in the specified ODD".
3. **Architecture.** Explain why a diverse redundant architecture can reduce AI risk, and what must be shown for the redundancy to be credited.

## Further Reading

- ISO/PAS 8800:2024 — *Road vehicles – Safety and artificial intelligence*.
- ISO/TR 5469 — *Artificial intelligence – Functional safety and AI systems*.
- Kelly, T. — *Arguing Safety: A Systematic Approach to Managing Safety Cases* (GSN).
- Burton, S. et al. — *Mind the gaps: Assuring the safety of autonomous systems from an engineering, ethical, and legal perspective*.
- Regulation (EU) 2024/1689 (AI Act).
