# 9. Metadata, Labels, and Annotations

## 9.1 Introduction

Raw sensor data becomes a training and evaluation resource only when it is accompanied by **labels** (what is in the data) and **metadata** (the conditions under which it was captured). Labels define what the model learns and what counts as an error; metadata make it possible to analyse performance by condition. For safety engineering, both are central: a model can never be judged more reliably than its ground truth.

**Learning objectives.** After this chapter you should be able to:

- Describe the main annotation types used in automotive perception.
- Explain how an annotation guideline and ontology define "ground truth".
- Describe annotation workflows and quality-assurance measures.
- Use metadata to slice a dataset by operating condition.

## 9.2 Annotation Types

| Annotation | Description | Typical use |
|---|---|---|
| 2D bounding box | Axis-aligned rectangle in the image | 2D object detection |
| 3D cuboid | Position, size and heading in 3D | 3D detection, planning |
| Semantic segmentation | Class label per pixel / point | Drivable area, scene understanding |
| Instance segmentation | Separate mask per object | Counting, tracking |
| Panoptic segmentation | Semantic + instance combined | Complete scene parsing |
| Lane and road markings | Polylines, curves, topology | Lane keeping, mapping |
| Keypoints / pose | Body or object joints | Pedestrian intent |
| Track IDs | Persistent identity over time | Tracking, prediction |
| Trajectories | Future positions of agents | Motion forecasting |
| Attributes and tags | Occluded, truncated, moving, traffic-light state | Slicing and analysis |

Labels are commonly attached with **attributes** such as **truncation** (object partly outside the image) and **occlusion** (partly hidden by another object). These attributes later allow evaluation to be broken down by difficulty.

## 9.3 Ontology and Annotation Guidelines

"Ground truth" is not a fact of nature; it is the outcome of a set of rules. The **ontology** (class taxonomy) and the **annotation guidelines** specify:

- Which classes exist and how they are defined (e.g. is a person on a scooter a *pedestrian* or a *cyclist*?).
- Minimum size or visibility for an object to be labelled.
- How to treat occlusion, reflections, crowds and objects at the image boundary.
- Box tightness and the reference point for 3D boxes.

> **Important:** Two datasets can both have a "pedestrian" class with different meanings. A model trained on one and evaluated on another can appear to fail simply because the definitions differ. Safety requirements must state the class definitions they refer to.

## 9.4 Annotation Workflows

Annotation is typically a multi-stage industrial process:

1. **Specification** — ontology, guidelines, and examples are written.
2. **Pre-labelling** — a model or rule-based system proposes labels (automatic labelling, Chapter 2, Section 2.2).
3. **Human annotation or correction** — annotators create or fix labels with dedicated tooling.
4. **Quality assurance** — sampled or full review by a second person.
5. **Feedback loop** — error patterns lead to guideline updates and annotator retraining.

For 3D data, annotators often label a LiDAR point cloud and the camera images simultaneously, so that the two views of the same object remain consistent.

## 9.5 Annotation Quality

### Sources of label error

- **Ambiguity** — truly ambiguous cases (distant, blurred, heavily occluded objects).
- **Guideline misunderstanding** — annotators interpret a rule differently.
- **Fatigue and tooling errors** — missed objects, misplaced boxes.
- **Propagated pre-label errors** — annotators accept wrong model proposals (automation bias).

### Measuring quality

- **Inter-annotator agreement** — have several annotators label the same data and compare. For boxes, agreement is commonly measured by Intersection over Union (IoU):

$$
\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

- **Cohen's kappa** or similar statistics for categorical labels.
- **Audit sampling** — an expert re-labels a random subset, giving an estimated error rate with confidence interval.
- **Model-assisted checks** — tools that flag examples where a trained model strongly disagrees with the label, which are candidates for label errors.

> **Safety link:** The measured performance of a model is bounded by label quality. If the labels themselves have, say, 3 % errors concentrated in one class, then reported metrics for that class are unreliable, and a model that matches the labels perfectly would still be wrong.

## 9.6 Metadata

Metadata describe the capture context rather than the content of the scene. Typical fields:

- **Environment** — weather (clear, rain, snow, fog), time of day, lighting (day, dusk, night, glare).
- **Location** — country, city, road type (urban, rural, motorway), speed zone.
- **Traffic** — density, presence of vulnerable road users, construction zones.
- **Sensor and vehicle** — sensor models, mounting positions, calibration version, vehicle speed.
- **Provenance** — recording date, software version, annotation tool and guideline version, licence.

### Why metadata matter for safety

Aggregate metrics hide weaknesses. With metadata, evaluation can be **sliced**: recall for pedestrians at night in rain, or for objects beyond 50 m. Metadata also make it possible to compare dataset coverage with the vehicle's **Operational Design Domain (ODD)** — the set of conditions in which the function is designed to operate (Chapters 13 and 15; ISO 34503 provides an ODD taxonomy).

## 9.7 Metadata Standards and Tagging

Reusable formats avoid every team inventing its own tags. For example, the ASAM **OpenLABEL** standard defines a common way to describe annotations and scene tags, and ASAM **OpenODD** targets machine-readable ODD specifications. Even without a standard, a data catalogue should record, for every dataset version, the guideline version, ontology, tag definitions and known issues.

## 9.8 Summary

Labels define what a perception model is trained to do and what is counted as an error; they follow from an ontology and guidelines rather than from objective truth. Annotation is a multi-stage process whose quality must be measured, and metadata enable sliced evaluation and comparison against the ODD. Careful documentation of labels and metadata is a prerequisite for any credible safety argument about a data-driven function.

## Exercises

1. **Ontology.** Define the class "pedestrian" precisely enough for annotators. Decide how you would handle a person pushing a bicycle, a child in a stroller and a person on an e-scooter.
2. **Agreement.** Two annotators draw boxes with $|A \cap B| = 60$ and $|A \cup B| = 100$ (in pixel-area units). Compute the IoU. Would you consider this acceptable agreement for pedestrians? Justify.
3. **Slicing.** List five metadata fields you would require in order to evaluate a pedestrian detector for a motorway-and-urban ADAS function, and say what each would reveal.

## Further Reading

- Northcutt, C., Jiang, L., Chuang, I. — *Confident Learning: Estimating Uncertainty in Dataset Labels*.
- Northcutt, C., Athalye, A., Mueller, J. — *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks*.
- ASAM OpenLABEL and ASAM OpenODD specifications.
- ISO 34503 — *Road Vehicles – Test scenarios for automated driving systems – Specification for operational design domain*.
