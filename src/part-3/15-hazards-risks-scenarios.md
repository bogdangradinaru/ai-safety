# 15. Hazards, Risks, and Driving Scenarios

## 15.1 Introduction

Safety arguments are built around **hazards** (what can go wrong) and **scenarios** (the situations in which it can go wrong). Because AI performance depends heavily on the situation — lighting, traffic, road layout, other road users — scenarios are the natural unit for specifying, testing, and evidencing the safety of an AI-based driving function.

**Learning objectives.** After this chapter you should be able to:

- Distinguish hazard, risk, harm, and hazardous event.
- Describe the levels of scenario abstraction: functional, logical, concrete.
- Specify an Operational Design Domain (ODD).
- Use criticality metrics to describe how dangerous a scenario is.
- Explain edge and corner cases and how scenario libraries are built.

## 15.2 Hazards, Risk, and Harm

- **Harm** — physical injury or damage to the health of people (or damage to property or environment).
- **Hazard** — a potential source of harm.
- **Risk** — a combination of the probability of occurrence of harm and its severity.
- **Hazardous event** — a hazard occurring in a specific situation.

In a perception-based function, typical hazards derive from **perception errors**:

| Error | Consequence (example) |
|---|---|
| **False negative** — real object not detected | No braking for a pedestrian |
| **False positive** — object detected where none exists | Sudden unnecessary braking ("phantom braking"), possible rear-end collision |
| **Wrong class** | A cyclist treated as a static object and not tracked |
| **Wrong position / size / velocity** | Misjudged gap in a lane change |
| **Late detection** | Insufficient time to brake |
| **Flickering / unstable detection** | Unreliable tracking and prediction |

Not every error is equally important: a missed object far away behind a barrier matters less than one directly in the path. Safety evaluation should therefore weight errors by their **relevance to the driving task** (Chapter 16).

## 15.3 Scenes, Situations, and Scenarios

A widely used terminology (Ulbrich et al., 2015) distinguishes:

- **Scene** — a snapshot of the environment: road, static elements, dynamic objects, weather, and their relations at one moment.
- **Situation** — the scene together with the ego vehicle's goals, and the information relevant for the next decision.
- **Scenario** — a sequence of scenes over time, with the actions and events that connect them.

### Abstraction levels

| Level | Description | Example |
|---|---|---|
| **Functional** | Natural-language description | "Pedestrian crosses the road in front of the ego vehicle in the dark" |
| **Logical** | Parameter ranges for the scenario | Ego speed 30–60 km/h, pedestrian speed 0.5–2.5 m/s, illumination 0–50 lux |
| **Concrete** | One specific value for each parameter | Ego 50 km/h, pedestrian 1.4 m/s, 5 lux |

Logical scenarios are sampled to produce many concrete scenarios for simulation or track tests. Scenario description languages — such as **ASAM OpenSCENARIO** — allow concrete scenarios to be executed across simulators and tools.

## 15.4 Operational Design Domain (ODD)

The ODD defines the conditions under which a function is designed to operate. It is a central safety artefact: it bounds the scenarios that must be handled, and what data must cover (Chapter 10).

Typical ODD dimensions:

- **Road** — type (motorway, urban), lane layout, surface, markings.
- **Environment** — weather, lighting, temperature.
- **Traffic** — density, types of road users, rules.
- **Speed range** and **geographical area**.
- **Dynamic elements** — presence of pedestrians, cyclists, construction zones.

ISO 34503 provides a taxonomy and structure for ODD specification. Two practical aspects of the ODD for AI safety:

1. The model must be validated *within* the whole ODD, including its boundaries.
2. The system must detect **ODD exit** and respond safely, e.g. by handing over or by a minimal-risk manoeuvre.

## 15.5 Criticality Metrics

To compare and rank scenarios we quantify how close they are to a collision. Common metrics:

- **Time headway (THW):** time for the ego vehicle to reach the current position of a lead vehicle.

$$
\mathrm{THW} = \frac{d}{v_{\text{ego}}}
$$

- **Time to collision (TTC):** time until collision if both vehicles maintain speed along the same line.

$$
\mathrm{TTC} = \frac{d}{v_{\text{ego}} - v_{\text{lead}}}, \quad v_{\text{ego}} > v_{\text{lead}}
$$

- **Post-encroachment time (PET):** time between one road user leaving a conflict area and another arriving.
- **Required deceleration** (e.g. DRAC): the braking needed to avoid a collision; high values mean a critical situation.

Smaller TTC, THW, and PET, and larger required deceleration, mean greater criticality. Thresholds are context-dependent and the metrics have known limits (e.g. TTC assumes constant velocities).

## 15.6 Edge Cases, Corner Cases, and the Long Tail

- **Edge case** — a rare situation at the boundary of the specified behaviour.
- **Corner case** — a situation involving a combination of several unusual parameters; extremely rare, yet potentially severe.

Corner cases are the main contributor to residual SOTIF risk (Chapter 13, Area 3). Examples include unusual objects, partially occluded pedestrians in glare, or sensor degradation combined with dense traffic. Breitenstein et al. propose a systematic categorisation of corner cases from pixel-level anomalies to scenario-level anomalies (see Further Reading).

## 15.7 Building Scenario Libraries

Sources of scenarios:

1. **Expert knowledge and hazard analysis** — structured identification of what can go wrong (e.g. HARA, STPA).
2. **Regulation and consumer tests** — e.g. Euro NCAP test protocols for AEB and lane support, UN regulations such as UN R157 for automated lane keeping.
3. **Accident data** — national statistics and in-depth accident databases.
4. **Naturalistic driving data** — extracting real scenarios from fleet data and public datasets (Part II).
5. **Generation and search** — sampling logical scenarios, or using optimisation and reinforcement learning to find failure-inducing scenarios.
6. **Field monitoring** — scenarios derived from disengagements and incidents after release.

Frameworks such as **PEGASUS** and the ISO 34502 scenario-based safety evaluation framework give structure to identification, selection and test of scenarios.

### Coverage

A scenario library is useful only if its **coverage** of the ODD is argued: which parameter ranges are covered, at what resolution, and which are consciously excluded. Coverage of scenario space is difficult to quantify, especially with a high-dimensional parameter space, and is itself an active research area.

## 15.8 Worked Example: Pedestrian Crossing

1. **Hazard:** missed detection of a crossing pedestrian by the perception system leads to collision.
2. **Functional scenario:** a pedestrian steps out between parked cars at night in light rain.
3. **Logical scenario:** ego speed 20–50 km/h; pedestrian speed 0.5–3 m/s; lateral start distance 0–2 m; illumination 1–20 lux; rain intensity 0–10 mm/h.
4. **Safety requirement (illustrative):** "In the specified ODD, for all scenarios in the logical range with TTC above X s at first visibility, the system shall initiate braking sufficient to avoid collision."
5. **Evidence:** simulation sweep over the logical scenario; track tests of selected concrete scenarios; real-world data analysis; per-slice detection metrics.
6. **Residual risk:** scenarios outside the tested parameter ranges, or with additional rare factors, are documented and monitored.

## 15.9 Summary

Hazards from AI-based functions primarily take the form of perception and decision errors whose impact depends on the scenario. Scenarios are described at three abstraction levels, bounded by the ODD, ranked by criticality metrics and gathered from many sources into libraries whose coverage must be argued. Together, the standards of this part and the scenario concepts of this chapter give the framework for the evaluation methods of Part IV.

## Exercises

1. **Abstraction levels.** Write a functional, a logical, and a concrete scenario for a cut-in by a vehicle on a motorway.
2. **Criticality.** The ego vehicle drives at 25 m/s, 40 m behind a lead vehicle at 15 m/s. Compute THW and TTC. Is the situation critical?
3. **ODD.** Write an ODD specification of at least six dimensions for an urban low-speed shuttle, and say how the vehicle would detect leaving it.
4. **Error severity.** Rank the perception errors in Section 15.2 by safety relevance for an AEB function and justify the ranking.

## Further Reading

- Ulbrich, S. et al. — *Defining and Substantiating the Terms Scene, Situation, and Scenario for Automated Driving*.
- Menzel, T., Bagschik, G., Maurer, M. — *Scenarios for Development, Test and Validation of Automated Vehicles*.
- Breitenstein, J. et al. — *Systematization of Corner Cases for Visual Perception in Automated Driving*.
- ISO 34502 and ISO 34503 — scenario-based safety evaluation and ODD taxonomy.
- ASAM OpenSCENARIO and OpenDRIVE standards.
- PEGASUS project — *Scenario Description and Database Method*.
