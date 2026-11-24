# 14. AI Safety and ISO/PAS 8800

## 14.1 Introduction

AI behaviour depends on data, learned parameters, training choices, architecture, and the execution environment. A high test score alone therefore does not show that an AI-based vehicle function is safe. This chapter introduces the engineering activities in **Clauses 10 to 13 of ISO/PAS 8800:2024**:

- **Clause 10:** select suitable AI technologies, architectures, and development measures.
- **Clause 11:** manage the safety of the datasets used to develop and test the AI system.
- **Clause 12:** verify AI components and validate the integrated AI system.
- **Clause 13:** analyse safety-relevant AI errors, identify their causes, and define measures to prevent or control them.

> **Scope:** The explanations below are educational paraphrases of Clauses 10–13. They do not reproduce or replace the requirements in the published specification. Requirements used in a safety project must be checked against the licensed text and its current edition.

**Learning objectives.** After this chapter you should be able to:

- Distinguish architectural measures from development measures.
- Explain the train, validation, and test roles within a controlled dataset lifecycle.
- Derive basic test cases and pass/fail criteria for an AI component.
- Trace a failed pedestrian-detection test to a possible requirement, data, model, or integration cause.
- Identify the main work products produced by Clauses 10–13.

## 14.2 Running Example: A Pedestrian Detector

Consider a camera-based pedestrian detector in an automatic emergency braking function. Its allocated safety requirement is stated in simplified form as follows:

> Detect safety-relevant pedestrians in the specified input space and provide detections early enough for the encompassing function to plan and execute an appropriate response.

The detector might be a convolutional neural network such as those introduced in Chapter 3. Its performance depends on the image data and labels introduced in Chapter 2. It is not evaluated in isolation, however, because preprocessing, the AI model, post-processing, tracking, and the vehicle-level decision all influence the final behaviour.

The four clauses form one evidence chain:

```text
Safety requirement
        |
        v
Clause 10: select technology, architecture, and development measures
        |
        v
Clause 11: specify and control the datasets
        |
        v
Clause 12: test components, integrate the system, and validate it
        |
        v
Clause 13: analyse failed or unsafe behaviour and feed improvements back
```

## 14.3 Clause 10: Selecting AI Technologies and Measures

### 14.3.1 Purpose

Clause 10 asks whether the selected AI technology and development approach can support the allocated safety requirements. A technically impressive model is not automatically a suitable safety-related model. Its behaviour, limitations, and the measures available to control them must be considered together.

The selection process has four main questions:

1. Which AI technology and method are capable of performing the intended function?
2. Which safety requirements are allocated to each AI component?
3. Which measures prevent AI errors or reduce the risk when AI errors occur?
4. Do the development and target execution environments create relevant differences in behaviour?

### 14.3.2 Architectural and development measures

| Measure type | Main role | Pedestrian-detector examples |
|---|---|---|
| **Architectural** | Add structure around or within the AI system to detect, contain, or control errors | Diverse sensors, redundant models, voting or switching, reject behaviour for uncertain or out-of-distribution inputs, a conventional safety monitor, fallback or degraded operation |
| **Development** | Improve the model or its evidence during data preparation, training, and evaluation | Data augmentation, regularisation, hyperparameter tuning, fault-aware or adversarial training, confidence calibration, uncertainty estimation, saliency analysis, targeted retraining |
| **Combined** | Use model development and system structure to control the same risk | Train a reject class and connect its output to a monitor that restricts or stops the function |

> **Important:** These measures are not interchangeable. A measure that improves average model performance may reduce an AI error, while an architectural monitor may limit the consequence of that error. Both can be needed.

### 14.3.3 Justifying the selection

The justification connects each choice to the property or risk it is expected to control. Clause 10 gives examples across safety-related AI properties such as robustness, generalisation, reliability, resilience, controllability, explainability, predictability, alignment, bias, and fairness.

A basic selection rationale can be recorded as follows:

| Safety concern | Proposed measure | Evidence expected later |
|---|---|---|
| Reduced performance after small input changes | Perturbation-oriented data augmentation and robustness tests | Stable pass/fail results across specified perturbations |
| Unusual inputs not represented in training | Out-of-distribution rejection plus a safe fallback | Detection rate on selected unknown-input tests and safe fallback behaviour |
| Correlated failures from similar models | Diversity in architecture, training, or sensing | Independence argument and combined-system performance |
| Poorly calibrated confidence | Calibration and uncertainty evaluation | Agreement between confidence and observed correctness |
| Bias between safety-relevant groups or conditions | Coverage analysis and targeted data collection | Per-condition and per-group results against criteria |

The effectiveness of the combination must be argued; simply listing measures does not demonstrate that the remaining risk is acceptable.

### 14.3.4 Design and refine iteratively

A candidate architecture and development process are trained with the training dataset and evaluated with the validation dataset. If no candidate can meet its allocated requirements, the design or requirements must be reconsidered with the relevant stakeholders. If several candidates are feasible, their performance, cost, complexity, evidence burden, and limitations can be compared.

The selection must also consider differences between the development environment and the vehicle execution environment. For example, a model trained with one camera configuration, numerical precision, compiler, or preprocessing pipeline may behave differently after deployment. Relevant differences should be identified and controlled through development or architecture.

### 14.3.5 Worked selection example

For the pedestrian detector, one possible selection is:

- A CNN for image feature extraction and detection.
- A diverse second perception channel or conservative conventional check.
- Uncertainty estimation and an out-of-distribution monitor.
- A defined degraded response when reliable detection cannot be supported.
- Data augmentation, regularisation, and hyperparameter tuning during training.

This is only a candidate. Clauses 11 and 12 must then establish whether the data and evidence support it, while Clause 13 investigates any resulting AI errors.

## 14.4 Clause 11: Data-Related Considerations

### 14.4.1 Why data need their own lifecycle

For a learned model, data are an engineering input that directly shapes behaviour. Clause 11 therefore treats datasets as controlled lifecycle products rather than as an unstructured collection of files.

The dataset lifecycle includes:

1. Dataset requirements development.
2. Dataset design.
3. Dataset implementation.
4. Dataset safety analysis.
5. Dataset verification.
6. Dataset validation.
7. Dataset maintenance and revision when inputs, requirements, or observed insufficiencies change.

Training and validation are used iteratively during development. The test dataset is reserved for evaluation of the trained component so that it provides a more credible estimate of performance on independent data.

### 14.4.2 Dataset roles

| Dataset | Main purpose | Safety concern |
|---|---|---|
| **Training** | Adjust model parameters and learn the intended mapping | Coverage gaps, label errors, imbalance, and unintended bias can become model behaviour |
| **Validation** | Compare candidates, tune hyperparameters, and decide whether training should continue | Repeated use can overfit decisions to the validation set; it must remain separate from training |
| **Test** | Evaluate the selected model against defined pass/fail criteria | Leakage or reuse weakens the independence of the evidence |

### 14.4.3 Data-related safety properties

Dataset safety analysis is organised around properties relevant to the intended use:

| Property | Question to ask |
|---|---|
| **Accuracy** | Do values and annotations represent the intended meaning at the needed resolution? |
| **Completeness** | Are required input-space regions, metadata, scenarios, and perturbations covered? |
| **Correctness or fidelity** | Does the recorded data faithfully capture the phenomenon and retain the features needed to characterise it? |
| **Independence** | Are train, validation, and test data sufficiently separated by scene, source, location, time, or collection method? |
| **Integrity** | Could capture, transfer, compression, modification, or poisoning have altered the data? |
| **Representativeness** | Does the dataset distribution support the intended environment and avoid unintended bias? |
| **Temporality** | Are age, lifetime, ordering, timeliness, and distribution change adequately considered? |
| **Traceability** | Can each element be traced to its source and processing history? |
| **Verifiability** | Can the dataset be checked against its requirements and properties? |

No single property proves that a dataset is safe. Their combined evidence must address the input space and the allocated AI safety requirements.

### 14.4.4 Analyse dataset insufficiencies

Dataset insufficiency analysis asks what can go wrong, why it can happen, what effect it can have, and what countermeasure reduces the risk.

```text
Dataset requirement
        |
        v
Potential insufficiency -> cause -> possible AI error -> safety impact
        |
        v
Countermeasure + metric for judging its effectiveness
```

Examples for the pedestrian detector include:

| Insufficiency | Possible cause | Potential effect | Countermeasure and check |
|---|---|---|---|
| Missing night-time images | Collection plan did not cover the full input space | Missed pedestrians under low illumination | Add targeted collection; verify coverage by metadata |
| Inconsistent bounding boxes | Annotation guideline or quality control is weak | Biased object-size learning and unstable detections | Clarify guidelines, add independent review, and audit agreement |
| Adjacent video frames appear in training and test | Split performed by frame instead of by sequence | Over-optimistic test performance | Split by trip or sequence and check near-duplicates |
| Old traffic scenes dominate | Reused archived data | Poor response to changed vehicle and road-user populations | Assess temporal coverage and collect current data |
| One region dominates | Collection is geographically narrow | Uncontrolled performance outside represented conditions | State the limitation or expand collection and evaluate by region |

### 14.4.5 Dataset lifecycle work products

The lifecycle creates evidence showing what the data were intended to contain and whether the implementation achieved that intent. Typical work products include:

- Dataset requirements.
- Dataset design and metadata definitions.
- Records for collection, generation, processing, and labelling.
- Traceability from dataset requirements to implemented data.
- Dataset verification and validation reports.
- Dataset safety-analysis results.
- Maintenance and change records.

These products provide the link between the abstract property “representative” and concrete evidence such as source records, metadata distributions, coverage results, and review findings.

## 14.5 Clause 12: Verification and Validation of the AI System

### 14.5.1 Verification versus validation

| Activity | Main question | Typical level |
|---|---|---|
| **Verification** | Does the implemented AI component or integrated system satisfy the allocated requirement? | Component, interface, and integrated AI system |
| **Validation** | Are the allocated safety requirements achieved in the intended encompassing system and input space? | Integrated system and relevant vehicle-level behaviour |

Component verification can be performed on testable AI components in isolation, but passing a model test is not enough. Interfaces, preprocessing, post-processing, sensor behaviour, timing, hardware effects, integration, and the behaviour of the encompassing system also require evidence.

### 14.5.2 Build test cases from requirements

A usable test case has at least:

- A defined input, scenario, or stimulus.
- The relevant initial state and environment.
- The item under test.
- Observable outputs or behaviours.
- Explicit pass/fail criteria.
- A link to one or more AI safety requirements.

For the pedestrian detector, a test case might define a pedestrian at a specified range, speed, illumination, occlusion, and time to collision. The pass criterion can require a detection within a stated distance and time limit, not merely report an average precision score.

Tests can be generated by:

- Requirement and equivalence-class analysis.
- Boundary-value selection.
- Coverage of safety-relevant input-space regions.
- Mutation or controlled perturbation of valid inputs.
- Search for failure-inducing or corner-case scenarios.
- Analysis of previous errors to target likely weaknesses.

### 14.5.3 AI-specific testing challenges

Clause 12 highlights difficulties that require more than code coverage:

- Large numbers of learned parameters make complete inspection impractical.
- Training is heuristic and can produce locally good but poorly generalising solutions.
- The complete input space is difficult to represent with a finite test dataset.
- Spurious correlations can produce unexpected behaviour even when the training data and model appear plausible.
- Small input changes can cause large output changes.
- A retrained model can change previously verified behaviour.
- Structural or parameter coverage has limited ability to predict coverage of the full input space.

The response is a combination of independent test data, requirement-oriented cases, perturbation and robustness testing, integration tests, and analysis of failed cases. Virtual and physical testing can complement each other when their validity and differences are understood.

### 14.5.4 Evidence by level

```text
AI component
    verify learned behaviour and allocated requirements
            |
            v
AI components and interfaces
    verify integrated preprocessing, model, and post-processing
            |
            v
AI system
    verify the integrated implementation and execution environment
            |
            v
Encompassing vehicle function
    validate the intended safety contribution in relevant scenarios
```

The main Clause 12 work products are the AI system verification report, the integrated AI system, and the AI system validation report. A report should explain what was tested, which requirements and versions apply, the pass/fail result, observed limitations, and whether the evidence supports the claim being made.

## 14.6 Clause 13: Safety Analysis of the AI System

### 14.6.1 Purpose

Clause 13 turns observed or potential AI errors into a structured causal analysis. It asks:

1. What AI error could violate an AI safety requirement?
2. Could it result from a fault, a functional insufficiency, or an interaction?
3. What underlying issue caused the error?
4. Which measure can prevent, detect, or control it?
5. Does the result reveal a missing or incomplete safety requirement?

The analysis can start at vehicle level and work down through the AI system, components, and model, or it can start from a lower-level data, model, or component issue and examine its upward effect. Both top-down and bottom-up reasoning can be useful.

### 14.6.2 Analyse a failed test

When testing reveals an AI error, Clause 13 follows three linked activities:

1. **Risk evaluation:** determine whether the failure violates a safety requirement and what safety effect it can have.
2. **Root-cause analysis:** investigate requirements, data, model design, implementation, integration, and relevant interactions.
3. **Risk mitigation:** define and implement a suitable prevention, detection, or control measure, then create the necessary follow-up tests.

For example:

```text
Observed behaviour: pedestrian near a traffic post is not detected
        |
        v
Safety impact: braking response can be delayed
        |
        v
Possible causes:
- requirement did not cover the occlusion case
- training data lack this configuration
- model overfits other pedestrian appearances
- post-processing suppresses the detection
- integration changes timing or preprocessing
        |
        v
Measures:
- improve requirement and scenario set
- extend and analyse the dataset
- refine training or model design
- add redundancy or monitoring
- add a regression test for this configuration
```

An investigation should not stop at the first plausible cause. It should use evidence to distinguish data, model, requirement, integration, and encompassing-system explanations.

### 14.6.3 Selecting an analysis technique

The selected technique must fit the system and the question. Clause 13 gives examples including:

| Technique | Useful role | AI-related caution |
|---|---|---|
| Fault Tree Analysis | Trace a top-level event to combinations of contributing causes | Learned behaviour and complex dependencies may not fit simple independent-event assumptions |
| Failure Mode and Effects Analysis | Review component-level failure modes and their effects | Interactions and non-linear model behaviour may be harder to capture |
| System-Theoretic Process Analysis | Examine unsafe control actions and interactions | Requires a suitable interaction and control model |
| Event Tree Analysis | Follow consequences of an initiating event | Probabilities and conditional sequences may be difficult to establish |
| Bayesian or causal network | Model complex relationships and conditional uncertainty | Requires justified structure and probability information |
| HAZOP-style guidewords | Prompt deviations such as missing, late, wrong, or excessive behaviour | Cause identification and propagation can remain weak without deeper follow-up |

The justification for a technique should include its benefits, assumptions, and limitations. AI systems are non-linear, may operate in a changing environment, and can create complex interactions among data, model, components, and context.

### 14.6.4 Feedback into Clauses 10–12

Safety analysis does not end with a report. Its results can lead to:

- A change in architecture or safety monitor (Clause 10).
- A different training or evaluation measure (Clause 10).
- New dataset requirements, coverage, labelling, or maintenance actions (Clause 11).
- New component or system tests and revised pass/fail criteria (Clause 12).
- A revised or additional AI safety requirement when the original set is incomplete.

The Clause 13 work product is the safety analysis report, including the identified errors, causes, impacts, measures, and the resulting evidence or follow-up actions.

## 14.7 One Worked Evidence Chain

The four clauses can be viewed as a sequence of artefacts:

| Clause | Example activity | Example evidence |
|---|---|---|
| **10** | Select CNN, uncertainty monitor, redundant check, augmentation, and regularisation for the pedestrian detector | Selection rationale; allocation of requirements to components; architectural and development measures |
| **11** | Specify data for the required ODD and define train/validation/test separation | Coverage matrix; annotation review; dataset traceability; verification and validation reports |
| **12** | Test pedestrian-detection components and validate the integrated AEB function | Test cases with pass/fail criteria; robustness results; integration tests; validation report |
| **13** | Analyse missed detections near poles and identify incomplete night/occlusion coverage | Root-cause chain; mitigation; revised data or model actions; new regression tests |

> **Key idea:** A model score is one small part of the argument. Safety evidence is the traceable chain connecting requirements, technology choice, controlled data, verification, integration, validation, analysis, and corrective action.

## 14.8 Summary

Clauses 10–13 describe how AI safety moves from intention to evidence:

- Clause 10 selects and justifies the technology, architecture, and development measures.
- Clause 11 specifies, analyses, controls, verifies, and maintains the datasets.
- Clause 12 verifies AI components and the integrated AI system, then validates the intended safety contribution.
- Clause 13 analyses AI errors and their causes, defines measures, and checks whether the requirements and evidence remain complete.

Together they form an iterative loop. Tests can reveal data insufficiencies; data analysis can motivate a different architecture; integration evidence can reveal an environment mismatch; and safety analysis connects all of these findings back to risk and requirements.

## Exercises

1. **Clause 10.** Select two architectural and two development measures for a camera pedestrian detector. For each, state the AI error or risk it addresses and the evidence needed to support its effectiveness.
2. **Clause 11.** Build a small dataset-requirements table with requirements for completeness, independence, traceability, and representativeness.
3. **Clause 12.** Write one component test and one integrated-system test for a pedestrian partly hidden by a roadside object. Include explicit pass/fail criteria.
4. **Clause 13.** A validation test shows delayed detection in heavy rain. Create a top-down causal chain and list evidence that would distinguish a data issue, model issue, preprocessing issue, and requirement issue.
5. **Traceability.** Follow one requirement through all four clauses and identify one output artefact from each clause.

## Further Reading

- ISO/PAS 8800:2024 — *Road vehicles – Safety and artificial intelligence*, Clauses 10–13.
