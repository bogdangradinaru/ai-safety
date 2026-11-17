# 13. SOTIF and ISO 21448

## 13.1 Introduction

A vehicle can behave dangerously even when no component has failed. A camera that works exactly as designed may still fail to recognise a pedestrian in low sun; a planner may do what it was built to do and still be unsafe in a situation its designers never considered. These are **functional insufficiencies**, not malfunctions, and they are the target of **ISO 21448 — Road vehicles – Safety of the intended functionality (SOTIF)**, first published in 2022.

**Learning objectives.** After this chapter you should be able to:

- Define SOTIF and distinguish it from functional safety.
- Explain triggering conditions and the known/unknown, safe/unsafe scenario classification.
- Describe the SOTIF activities: identification, evaluation, reduction of risk, and validation.
- Explain why SOTIF is central for perception and other ML-based functions.

## 13.2 SOTIF Versus Functional Safety

| | ISO 26262 | ISO 21448 |
|---|---|---|
| **Concern** | Hazards from E/E *malfunctions* | Hazards from *functional insufficiencies* of the intended functionality, or reasonably foreseeable misuse |
| **Cause** | Random hardware faults, systematic faults | Performance limitations, incomplete specification, foreseeable misuse |
| **Example** | A radar ECU crashes | The radar cannot separate a stopped vehicle from a road sign |
| **Primary tool** | Fault avoidance and control, ASIL-driven process | Scenario-based identification, verification and validation against acceptance criteria |

The two are complementary: a safe system needs both.

## 13.3 Key Concepts

- **Functional insufficiency** — a limitation in specification or performance that, together with a triggering condition, can lead to hazardous behaviour. For example: a sensor limitation, an algorithm weakness, or an inappropriate actuation.
- **Triggering condition** — a specific condition of a scenario that acts as an initiator for a subsequent hazardous behaviour (e.g. low sun glare, wet road reflections, an unusual object shape).
- **Reasonably foreseeable misuse** — use of the function by a person in a way not intended by the manufacturer but that can be anticipated (e.g. over-reliance on a driver assistance system).

## 13.4 The Four Scenario Areas

SOTIF classifies the space of all scenarios by whether they are *known* to the developers and whether they lead to hazardous behaviour:

| Area | Scenario category | Meaning |
|---|---|---|
| **1** | Known, not hazardous | Function behaves safely in scenarios that are understood |
| **2** | Known, hazardous | Understood scenarios in which the function can cause harm |
| **3** | Unknown, hazardous | Unanticipated scenarios in which the function can cause harm |
| **4** | Unknown, not hazardous | Unanticipated but harmless scenarios |

The development goal is to **grow Area 1 and shrink Areas 2 and 3** until the residual risk is acceptable:

- **Area 2** is reduced by identifying hazardous scenarios, then improving the function (better sensors, algorithm changes, restricted ODD, driver warning).
- **Area 3** is reduced by exploration: scenario discovery, large-scale simulation, field testing and monitoring, so that unknown hazardous scenarios become known and then addressed.

Because Area 3 can never be shown to be empty, the standard asks for **validation evidence** that the residual risk is below an acceptance criterion, not for proof of zero risk.

## 13.5 The SOTIF Process

1. **Specification and design** — define the function, its limitations, and its ODD; identify the relevant sensing and decision elements.
2. **Identify and evaluate hazards** — analyse the possible hazardous behaviours, including those from foreseeable misuse; derive acceptance criteria.
3. **Identify and evaluate triggering conditions** — find the conditions that could lead to hazardous behaviour, and judge their acceptability.
4. **Functional modifications** — reduce risk by improving the function, e.g. sensor redundancy, a more conservative behaviour, driver monitoring, or ODD restrictions.
5. **Define verification and validation strategy** — what evidence demonstrates that the criteria are met.
6. **Verify** — known-hazardous scenarios (Area 2) are tested for acceptable behaviour.
7. **Validate** — explore unknown scenarios (Area 3), e.g. with extensive simulation and real-world driving, and estimate residual risk.
8. **Release decision and field monitoring** — decide whether the residual risk is acceptable, and continue to monitor after release.

Hazards are rated by comparing their potential consequences with **validation targets** — quantitative or qualitative acceptance criteria, often derived from the risk level of comparable human driving or from regulatory expectations.

## 13.6 SOTIF for Machine-Learning Perception

SOTIF is a natural fit for ML-based perception because ML errors are performance limitations:

- Missed or false detections under specific **triggering conditions** (rain, glare, occlusion, unusual objects).
- **Data-driven insufficiencies**: gaps in training data coverage correspond to Area 3 scenarios (Part II).
- **Uncertainty** and confidence estimation are used to detect when the perception system is operating outside its reliable domain (Chapter 18).
- **Safety mechanisms** analogous to those in ISO 26262 include sensor redundancy and diversity, plausibility checks, runtime monitors and degraded-mode operation.

> **Important:** SOTIF does not assert that a specific percentage of test pass rate is "safe". It requires a justified argument that residual risk is acceptable. The quantity of validation evidence needed can be very large for rare hazardous events (Chapter 19).

## 13.7 Relation to Other Frameworks

- **ISO 26262** covers malfunctions; SOTIF covers insufficiencies: together they form the core of the vehicle-level safety argument.
- **ISO/PAS 8800** (Chapter 14) specialises the safety lifecycle for AI systems, including data and model aspects.
- **UL 4600** takes a **safety-case** approach for autonomous products and addresses elements such as the completeness of the claimed ODD and of the evidence.
- **ISO 34502** provides a scenario-based safety evaluation framework which helps to identify and test the scenarios that SOTIF requires (Chapter 15).

SOTIF's formal scope focuses on driver-assistance functions and their foreseeable misuse; the standard provides concepts and measures relevant also to higher levels of automation, but additional measures are needed for them. Check the current text of the standard when applying it to a specific system.

## 13.8 Summary

SOTIF addresses hazards that arise without any failure — from limitations of the intended function when confronted with certain triggering conditions. It structures the scenario space into known and unknown, safe and unsafe, and requires evidence that known hazardous situations are handled and that unknown ones are sufficiently improbable. For ML-based perception it supplies the central logic of safety evidence: identify the conditions under which the function underperforms, reduce them, and demonstrate that the residual risk is acceptable.

## Exercises

1. **Classification.** Classify each as a malfunction (ISO 26262) or a functional insufficiency (SOTIF): (a) a camera stops sending images; (b) a pedestrian detector misses a person wearing a reflective rain cape; (c) an object classifier is correct but the planner brakes too late for a wet road.
2. **Triggering conditions.** List four triggering conditions for a camera-based emergency braking function and the validation evidence you would collect for each.
3. **Area 3.** Propose two methods to discover unknown hazardous scenarios and discuss their limits.

## Further Reading

- ISO 21448:2022 — *Road vehicles – Safety of the intended functionality*.
- Koopman, P. — *How Safe Is Safe Enough? Measuring and Predicting Autonomous Vehicle Safety*.
- UL 4600 — *Standard for Safety for the Evaluation of Autonomous Products*.
- ISO 34502 — *Road vehicles – Test scenarios for automated driving systems – Scenario based safety evaluation framework*.
