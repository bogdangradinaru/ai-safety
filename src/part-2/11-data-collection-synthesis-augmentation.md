# 11. Data Collection, Synthesis, and Augmentation

## 11.1 Introduction

When existing datasets do not cover the required situations, new data must be obtained. There are three broad routes: **collecting** real data, **synthesising** it in simulation or with generative models, and **augmenting** existing data with transformations. Each is useful and each carries risks that must be understood and, for safety purposes, evidenced.

**Learning objectives.** After this chapter you should be able to:

- Describe how real driving data is collected and curated, including targeted collection.
- Explain the benefits and limits of simulation-based and generative synthetic data.
- Select appropriate data augmentation techniques and recognise unsafe ones.
- Discuss privacy and governance constraints on driving data.

## 11.2 Real-World Data Collection

### Data collection vehicles and fleets

Dedicated **test vehicles** carry a high-quality sensor suite plus logging hardware and produce well-calibrated, well-synchronised data. **Customer fleets** provide scale and diversity but usually with a smaller sensor set and limited upload bandwidth.

Important practicalities:

- **Calibration and time synchronisation** must be monitored throughout (Chapter 8).
- **Storage and bandwidth** — a multi-sensor vehicle can produce several terabytes per day, so data are filtered and selected on the vehicle or in the cloud.
- **Metadata capture** — weather, location, software version, and sensor configuration are recorded with every drive.

### Targeted collection

Because the interesting cases are rare, collection is guided rather than random:

- **Trigger-based collection** — the vehicle saves data when a rule fires (hard braking, driver override, disagreement between models, low model confidence).
- **Shadow mode** — a candidate function runs silently alongside the human driver and the mismatches are recorded.
- **Scenario-driven campaigns** — drives planned to cover specific ODD cells that coverage analysis (Chapter 10) found thin.

### The data engine

Industry practice is often described as a **data engine** loop: deploy a model, find its failures, mine or collect more examples of those failures, label them, retrain, repeat. Techniques include **active learning** (select the most informative unlabelled samples) and **hard-example mining**.

> **Caution:** Trigger rules shape the dataset. Data selected by "where the current model is uncertain" is no longer a random sample of the world, and statistics computed on it are biased (Chapter 10).

## 11.3 Privacy, Law, and Governance

Driving data contain personal data: faces, licence plates, locations and sometimes in-cabin images. Data collection typically needs:

- A lawful basis and, in the EU, compliance with the GDPR (including minimisation and retention limits).
- **Anonymisation** such as blurring faces and licence plates, with the trade-off that anonymisation can slightly affect detector performance.
- Controlled access, audit logging and clear retention policy.
- Respect of national rules on recording in public spaces and on data export.

## 11.4 Synthetic Data

### Simulation-based synthetic data

Simulators such as **CARLA** (open source) and commercial tools can render sensor data for scenarios that are rare, dangerous or expensive in reality, and provide perfect ground truth automatically:

- Rare events (e.g. a pedestrian emerging from behind a bus).
- Controlled variation of one factor at a time (weather, time of day, actor behaviour).
- Exactly known labels, including depth, motion and occlusion.

### Generative models

Generative methods — GANs, diffusion models, neural rendering — can alter weather and lighting in real images, insert objects, or reconstruct and re-render real drives from new viewpoints.

### The domain gap

Synthetic data differ from real data in textures, lighting, sensor noise, and actor behaviour. This **sim-to-real gap** means that a model may do well in simulation yet worse in reality. Typical remedies are domain randomisation, fine-tuning on real data, and mixing real and synthetic data.

> **Important for safety:** Synthetic data can help *train* a model and can *stress-test* it, but evidence for real-world performance must be validated against real data. Neither the simulator nor the generative model is automatically a valid ground truth: its fidelity must itself be argued and, where relied upon, assessed (e.g. in the spirit of tool qualification and simulation credibility assessment).

## 11.5 Data Augmentation

Augmentation creates modified copies of existing training samples, expanding the diversity seen during training at no collection cost.

| Family | Examples | Caution |
|---|---|---|
| **Geometric** | Flip, scale, crop, rotate | Horizontal flips are generally harmless for objects but wrong for left/right-hand-traffic semantics and for text on signs |
| **Photometric** | Brightness, contrast, colour jitter, noise, blur | Colour jitter can destroy information needed for traffic lights |
| **Weather / environment** | Synthetic rain, fog, snow, lens flare | Realism of the effect matters |
| **Occlusion** | Random erasing, cut-out | May erase the only object in the image without updating the label |
| **Mixing / copy-paste** | Mixup, CutMix, paste objects into other scenes | Can create physically implausible scenes |
| **LiDAR-specific** | Point dropout, jitter, global rotation, ground-truth sampling | Must keep geometry consistent with the labels |
| **Temporal** | Frame skipping, speed variation | Must preserve motion consistency |

### Rules for safe augmentation

1. **Labels must stay valid.** Any transformation of the input must be applied equally to the box, mask or trajectory.
2. **Semantics must be preserved.** Do not make a "stop" sign unreadable or recolour a red light.
3. **Augment training data only.** Validation and test data are not augmented (except deliberately, for robustness tests).
4. **Realism over volume.** Implausible augmentations can teach a model features that never occur in reality.
5. **Verify the effect.** Measure its impact on per-slice metrics rather than assuming that more augmentation helps.

## 11.6 Putting It Together

A practical data strategy combines the three routes:

1. Specify the ODD and required coverage (Chapter 10).
2. Check existing public/internal data for coverage gaps.
3. Close gaps by targeted real collection where possible.
4. Use simulation and synthesis for situations too rare or too dangerous to collect.
5. Apply augmentation to increase robustness.
6. Keep a **real, untouched, independent test set** for final evaluation.
7. Document provenance of every data source — real, synthetic or augmented.

## 11.7 Summary

Real data collection should be targeted at coverage gaps and subject to privacy and governance controls. Synthetic data and augmentation extend what is available, but each introduces its own gaps — the sim-to-real gap, implausible samples, or label invalidation — so their use must be justified and checked against real data. Part II ends here; Part III turns from the data to the standards that determine how safety is argued for systems that use it.

## Exercises

1. **Trigger design.** Propose three trigger rules to collect near-miss pedestrian events from a customer fleet. For each, explain what bias it might introduce.
2. **Augmentation audit.** A team applies random horizontal flips, hue shifts and cut-out to a traffic-light classifier. Identify which of these may harm the task and why.
3. **Synthetic evidence.** A team claims their detector is safe because it scored 99 % on 100,000 simulated scenarios. List the questions you would ask before accepting this as evidence.

## Further Reading

- Dosovitskiy, A. et al. — *CARLA: An Open Urban Driving Simulator*.
- Tobin, J. et al. — *Domain Randomization for Transferring Deep Neural Networks from Simulated to the Real World*.
- Shorten, C., Khoshgoftaar, T. — *A survey on Image Data Augmentation for Deep Learning*.
- Settles, B. — *Active Learning Literature Survey*.
- Regulation (EU) 2016/679 (GDPR).
