# 10. Data Quality, Coverage, and Bias

## 10.1 Introduction

A learned model reproduces the patterns of its training data, including the gaps and the errors. For safety-critical use the question is therefore not only "how much data do we have?" but "does the data adequately represent everything the system will encounter, and is it correct?". This chapter introduces the dimensions of data quality, how to measure coverage against an operating domain, and the main sources of bias.

**Learning objectives.** After this chapter you should be able to:

- Describe the main dimensions of data quality.
- Explain the long-tail problem and why rare events dominate safety risk.
- Assess dataset coverage relative to an Operational Design Domain.
- Identify typical biases and data-leakage problems, and name mitigations.

## 10.2 Dimensions of Data Quality

| Dimension | Question | Example problem |
|---|---|---|
| **Accuracy / correctness** | Are labels and measurements right? | Wrong class, misplaced 3D box |
| **Completeness** | Is everything that should be labelled labelled? | Missed distant pedestrians |
| **Consistency** | Are rules applied uniformly? | Different box tightness between annotators |
| **Representativeness** | Does the data match the deployment distribution? | Only daylight data for a 24-h function |
| **Diversity / coverage** | Are the relevant variations present? | No snow, no construction zones |
| **Timeliness** | Does the data reflect current conditions? | New road signs or vehicle types missing |
| **Integrity** | Are files intact and correctly linked? | Corrupt frames, desynchronised sensors |
| **Independence** | Are train and test truly separate? | Near-duplicate frames in both |

ISO/PAS 8800 (Chapter 14) treats such properties as data attributes that need to be specified, justified, and verified as part of the AI safety lifecycle; the details of its requirements should be read from the standard itself.

## 10.3 Class Imbalance and the Long Tail

Real driving is dominated by uneventful scenes: cars on a road. Rare objects and events — an overturned truck, an animal, a person in a wheelchair, an unusual load — are exactly where failures are most costly. This produces a **long-tailed** distribution:

- A few classes and situations account for most samples.
- Many safety-relevant situations appear very rarely, or never.

Aggregate accuracy is dominated by frequent classes, so a model can achieve excellent overall metrics while failing on the tail. Therefore evaluation must be **per class and per scenario**, not only overall (Part IV).

Common mitigations include targeted collection of rare cases, re-sampling or re-weighting, synthetic data and augmentation (Chapter 11), and more careful loss design. None of them removes the need to verify tail performance directly.

## 10.4 Coverage Analysis

Coverage asks whether the data adequately span the conditions in which the system will operate.

1. **Define the ODD** and its dimensions (road type, weather, lighting, traffic participants, speed range, geography). See ISO 34503 and Chapter 15.
2. **Tag the dataset** with metadata along those dimensions (Chapter 9).
3. **Build a coverage matrix** — e.g. weather × lighting × object type — and count samples per cell.
4. **Identify empty and thin cells.** These are the coverage gaps; decide whether to collect data, restrict the ODD, or accept and document the gap.
5. **Use data-driven checks** such as clustering the feature embeddings of the data and the field data to see regions of the field distribution not represented in the training set.

> **Important:** Counting samples is necessary but not sufficient. A cell with many frames from a single recording session has far less *effective* diversity than the count suggests.

## 10.5 Sources of Bias

- **Geographic and cultural bias.** Datasets from a few cities embed their road layouts, signs, vehicle mix and driver behaviour.
- **Environmental bias.** Over-representation of fair weather and daylight.
- **Sensor bias.** Data from one camera, lens or LiDAR model; models can overfit to sensor-specific noise and colour characteristics.
- **Selection bias.** Fleet data collected under trigger rules or by a specific group of drivers is not a random sample of driving.
- **Annotation bias.** Systematic labelling choices, such as consistently under-labelling distant or occluded objects.
- **Demographic bias.** Studies on pedestrian detection have found performance differences across groups of people (e.g. by apparent age or skin tone), which for a safety function is a direct safety concern (see Further Reading).
- **Temporal bias.** Data recorded in a given season or year becomes outdated as infrastructure and vehicles change.

## 10.6 Label Noise and Duplicates

- **Label noise** reduces achievable accuracy and distorts evaluation. Detecting suspicious labels — via model disagreement or confident-learning methods — and re-reviewing them is cost-effective.
- **Near-duplicates.** Consecutive frames from a video are almost identical. Duplicates inflate the apparent size of the dataset and, if split between training and test, cause **data leakage**.
- **Leakage.** Any path by which test information reaches training — duplicates, shared sequences, shared pre-trained models trained on the test set — makes test metrics over-optimistic.

Prevent leakage by splitting at the **sequence, location or drive level** rather than at the frame level, and by documenting how pretrained weights were obtained.

## 10.7 Dataset Shift (Preview)

Even a clean, covering dataset describes the *past*. When the deployment distribution differs from the training distribution, performance degrades — **dataset shift**. This arises from many of the biases above and is treated formally in Chapter 17. Good documentation of what the dataset does and does not cover is what makes shift detectable.

## 10.8 Documenting Datasets

Structured documentation makes quality and bias assessable. Proposals such as **Datasheets for Datasets** suggest recording: motivation, composition, collection process, labelling process, recommended and discouraged uses, and known limitations. For a safety case, this documentation becomes evidence that data were **specified, collected, annotated, and verified** as intended.

## 10.9 Summary

Data quality is multi-dimensional: correctness, completeness, consistency, representativeness, coverage and independence all matter. Long-tailed distributions mean overall metrics hide the rare cases that dominate risk. Coverage must be assessed against the ODD, bias has many sources, and leakage and label noise undermine evaluation. Documented, measured data quality is the foundation on which the safety arguments of the following chapters rest.

## Exercises

1. **Coverage matrix.** For a highway-pilot ODD, propose a coverage matrix with at least three dimensions. Which cells would you expect to be thinnest in a typical public dataset?
2. **Long tail.** A detector has 99 % average recall, but only 60 % on a class that makes up 0.5 % of the data. Explain how both statements can be true and why only one of them is useful for a safety claim.
3. **Leakage.** Describe two ways leakage can occur even if train and test frames are never identical, and how you would detect them.

## Further Reading

- Gebru, T. et al. — *Datasheets for Datasets*.
- Wilson, B., Hoffman, J., Morgenstern, J. — *Predictive Inequity in Object Detection*.
- Torralba, A., Efros, A. — *Unbiased Look at Dataset Bias*.
- Northcutt, C. et al. — *Confident Learning: Estimating Uncertainty in Dataset Labels*.
- ISO/PAS 8800 — *Road vehicles – Safety and artificial intelligence*.
