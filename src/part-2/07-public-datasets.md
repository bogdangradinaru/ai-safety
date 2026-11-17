# 7. Public Driving Datasets

## 7.1 Introduction

Data-driven perception and prediction models are only as good as the data they are trained and evaluated on (Chapter 1, Section 1.5). Public driving datasets are the common currency of automotive AI research: they allow models to be trained without owning a data-collection fleet, and they allow results from different groups to be compared on the same benchmark.

**Learning objectives.** After this chapter you should be able to:

- Name the main public driving datasets and the sensors, tasks and regions they cover.
- Choose a dataset appropriate for a given task (e.g. 3D detection, segmentation, motion forecasting).
- Identify the licensing, coverage and representativeness limits of a dataset.
- Explain why good benchmark performance is not, by itself, safety evidence.

## 7.2 What Makes a Driving Dataset

A driving dataset is more than a collection of images. A typical release contains:

- **Raw sensor data** — camera images, LiDAR point clouds, radar returns, and vehicle signals (GNSS/IMU, speed, steering).
- **Calibration** — intrinsic parameters of each camera and extrinsic transforms between sensors and the vehicle.
- **Annotations** — ground-truth labels for one or more tasks (Chapter 9).
- **Metadata** — time, location, weather, sensor configuration, and the train/validation/test split.
- **Tooling** — a development kit (devkit) for loading data and computing the official evaluation metrics.

## 7.3 Overview of Widely Used Datasets

| Dataset | Main sensors | Main tasks | Region(s) |
|---|---|---|---|
| **KITTI** | Stereo cameras, 64-beam LiDAR, GNSS/IMU | 2D/3D detection, tracking, stereo, optical flow, odometry | Karlsruhe, Germany |
| **Cityscapes** | Stereo camera | Semantic, instance and panoptic segmentation | 50 European cities |
| **BDD100K** | Dashcam video, GNSS/IMU | Detection, segmentation, lane marking, tracking, drivable area | Several US regions |
| **nuScenes** | 6 cameras, 1 LiDAR, 5 radars | 3D detection, tracking, prediction, segmentation | Boston, Singapore |
| **Waymo Open Dataset** | 5 cameras, 5 LiDARs | 2D/3D detection, tracking, segmentation, motion prediction | Several US cities |
| **Argoverse 2** | Cameras, LiDAR, HD maps | 3D detection, tracking, motion forecasting | Several US cities |
| **A2D2** | Cameras, LiDAR | 2D/3D detection, segmentation | Germany |

> **Note:** Sizes, class lists, licences and available tasks change between dataset versions. Treat the table as an orientation guide and always check the dataset's official page for current details before relying on them.

### Dataset characteristics worth comparing

- **Sensor suite.** Camera-only datasets cannot be used to study LiDAR or radar fusion. nuScenes is notable for including radar.
- **Annotation density.** Some datasets annotate every frame; others annotate **keyframes** only (e.g. nuScenes annotates at a lower rate than the sensors record).
- **Scene length and diversity.** Datasets built from short scene clips (typically ~20 s) support tracking and forecasting but contain few long-horizon events.
- **Geography and driving culture.** Road layout, signage, traffic density and driver behaviour differ markedly between regions.

## 7.4 Choosing a Dataset for a Task

| Task | Typical choices | What to check |
|---|---|---|
| 2D object detection | KITTI, BDD100K, Waymo | Class definitions, minimum annotated object size |
| 3D object detection | KITTI, nuScenes, Waymo | Box format, coordinate frame, official metric |
| Semantic segmentation | Cityscapes, BDD100K, nuScenes (lidarseg) | Label taxonomy, treatment of "void" regions |
| Multi-object tracking | nuScenes, Waymo, KITTI | Track-ID consistency across occlusions |
| Motion forecasting | Argoverse 2, Waymo Motion, nuScenes | Prediction horizon, availability of HD maps |
| Weather / night robustness | BDD100K, nuScenes | Size and labelling quality of the adverse-condition subset |

## 7.5 Licensing and Responsible Use

Most public driving datasets are released under licences that restrict use — commonly to **non-commercial research** — and some require registration or acceptance of terms before download. Before using a dataset you must:

1. Read the licence and confirm the intended use (research, teaching, product development) is permitted.
2. Respect any redistribution limits: do not re-upload data to places the licence does not allow.
3. Cite the dataset paper and the version used.
4. Respect privacy: datasets are normally anonymised (faces, licence plates), but you must not attempt to de-anonymise them.

This course never redistributes dataset contents; the labs (see the `datasets/` folder of the repository) download data directly from the official sources.

## 7.6 Limits of Public Datasets for Safety Work

Public datasets are valuable for learning and benchmarking, but they have systematic limits that matter for safety:

- **Narrow operational coverage.** A dataset recorded in a few cities in good weather says little about performance elsewhere (Chapter 10).
- **Benchmark-driven bias.** Datasets are designed for leaderboards; rare, safety-critical situations are usually under-represented because they rarely occur.
- **Fixed sensor setups.** A model trained on one camera and LiDAR configuration may degrade on a different one.
- **Annotation conventions.** Labels reflect the dataset's guidelines (e.g. how heavily occluded objects are treated), which may not match a product's safety requirements.

> **Important:** A high score on a public benchmark is evidence about performance on *that dataset's distribution*. It is not, on its own, evidence that a system is safe in its intended operational design domain. Bridging that gap is the subject of Parts III and IV.

## 7.7 Summary

Public datasets provide sensor data, calibration, annotations and tooling that make automotive AI research reproducible. They differ in sensors, tasks, geography and licence terms, and must be chosen and used with those differences in mind. Their value for teaching and benchmarking is high; their value as direct safety evidence is limited by their coverage and annotation conventions.

## Exercises

1. **Dataset selection.** You must train and evaluate a 3D pedestrian detector that fuses camera and LiDAR. Which dataset(s) from Section 7.3 would you consider, and what would you check first?
2. **Licence check.** A start-up wants to train a production perception model on a non-commercial-licence dataset. Explain what the problem is and what options exist.
3. **Coverage gap.** A dataset was recorded in two cities in mostly dry weather. List three deployment conditions in which you would not trust a model trained only on it.

## Further Reading

- Geiger, A., Lenz, P., Urtasun, R. — *Are we ready for autonomous driving? The KITTI vision benchmark suite*.
- Caesar, H. et al. — *nuScenes: A multimodal dataset for autonomous driving*.
- Sun, P. et al. — *Scalability in Perception for Autonomous Driving: Waymo Open Dataset*.
- Cordts, M. et al. — *The Cityscapes Dataset for Semantic Urban Scene Understanding*.
- Yu, F. et al. — *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*.
- Wilson, B. et al. — *Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting*.
