# 8. Dataset Formats and Structures

## 8.1 Introduction

Every public dataset stores its data differently. Before a model can be trained or evaluated, you must understand how files are organised, what each field means, and — most importantly — which **coordinate frame and units** each value is expressed in. Mistakes here are silent: a box in the wrong frame or a mixed-up width and height produces a model that trains without errors but learns the wrong thing.

**Learning objectives.** After this chapter you should be able to:

- Describe the on-disk structure of common sensor and annotation formats.
- Read KITTI-style, COCO-style and YOLO-style annotations and convert between them.
- Explain how a relational dataset such as nuScenes links samples, sensors and annotations.
- Apply calibration and coordinate transforms to relate camera and LiDAR data.

## 8.2 Sensor Data Formats

| Data | Common formats | Notes |
|---|---|---|
| Camera images | PNG (lossless), JPEG (lossy) | Compression artefacts can affect small-object detection |
| Video | MP4 / image sequences | Frame timing must be preserved for tracking |
| LiDAR point clouds | `.bin` (raw float arrays), `.pcd`, `.las`, `.npy` | Each point: x, y, z plus intensity (and sometimes ring, time) |
| Radar | `.pcd`, dataset-specific | Includes velocity (Doppler) per detection |
| Vehicle signals | CSV, JSON, Parquet | Speed, yaw rate, GNSS/IMU |
| Recorded logs | ROS bag, MCAP, TFRecord | Time-synchronised multi-sensor streams |

A KITTI LiDAR scan, for example, is a flat binary file of 32-bit floats grouped as `(x, y, z, reflectance)` per point, which is loaded as an `N × 4` array.

## 8.3 Annotation Formats

### KITTI

Each image has a text file with one object per line. The fields, in order, are: class type, truncation, occlusion state, observation angle `alpha`, 2D box (`left, top, right, bottom` in pixels), 3D dimensions (`height, width, length` in metres), 3D location (`x, y, z` in the camera frame) and rotation about the vertical axis `rotation_y`.

### COCO

A single JSON file with three main lists: `images`, `annotations` and `categories`. Each annotation references an image and a category and stores a bounding box as `[x_min, y_min, width, height]` in pixels, and optionally a segmentation mask and an `iscrowd` flag.

### YOLO

One text file per image; each line is `class_id x_center y_center width height` with all four box values **normalised** to $[0,1]$ by the image width and height.

### Pascal VOC

One XML file per image with `xmin, ymin, xmax, ymax` per object.

### Converting between box formats

The same box is written differently in each format. With image width $W$ and height $H$, a pixel box $(x_1, y_1, x_2, y_2)$ becomes in YOLO format:

$$
x_c = \frac{x_1 + x_2}{2W}, \quad y_c = \frac{y_1 + y_2}{2H}, \quad w = \frac{x_2 - x_1}{W}, \quad h = \frac{y_2 - y_1}{H}
$$

> **Common pitfall:** COCO stores `[x, y, width, height]` with the corner at the top-left, YOLO stores the box centre, and KITTI and VOC store two corners. Mixing these up is one of the most frequent causes of silently wrong training.

## 8.4 Relational and Sequence-Based Structures

Larger datasets store annotations as linked tables rather than one file per image.

### nuScenes

nuScenes is organised as a relational database of JSON tables, accessed through the `nuscenes-devkit`. Key tables include:

- `scene` — a ~20 s driving sequence.
- `sample` — a **keyframe**: one time-stamp at which all sensors are grouped and annotated.
- `sample_data` — one file per sensor reading (camera image, LiDAR sweep, radar), including non-keyframe sweeps.
- `sample_annotation` — a 3D box for an object at a sample.
- `instance` — an object tracked across samples; links annotations over time.
- `category`, `attribute`, `visibility` — label taxonomy and per-annotation properties.
- `calibrated_sensor`, `ego_pose`, `sensor`, `log`, `map` — calibration, vehicle pose, sensor description and capture context.

### Waymo Open Dataset

Data is stored as TFRecord files of Protocol Buffer `Frame` messages; each frame bundles images, LiDAR ranges, labels and calibration for one time step.

### Why structure matters

The structure determines what is easy: a relational layout makes it simple to follow an object through time or to query "all samples at night in rain", whereas a one-file-per-image layout makes per-frame loading simple but cross-frame queries laborious.

## 8.5 Calibration and Coordinate Frames

Driving data involves several coordinate frames:

- **Image frame** — pixel coordinates $(u, v)$.
- **Camera frame** — 3D coordinates relative to the camera.
- **LiDAR / sensor frame** — 3D coordinates relative to the LiDAR.
- **Vehicle (ego) frame** — fixed to the ego vehicle.
- **World / map frame** — global, used to relate multiple time steps.

Relating them requires **extrinsics** (rigid transform between frames) and **intrinsics** (camera projection). A 3D point $\mathbf{X}$ in the LiDAR frame is projected into an image as:

$$
\tilde{\mathbf{x}} = K \,[R \mid \mathbf{t}]\, \tilde{\mathbf{X}}
$$

where $K$ is the camera intrinsic matrix and $[R \mid \mathbf{t}]$ the LiDAR-to-camera transform. KITTI supplies these as matrices in its calibration files (e.g. a projection matrix and a LiDAR-to-camera transform), and nuScenes through `calibrated_sensor` and `ego_pose`.

### Time synchronisation

Sensors do not capture at the same instant. A rotating LiDAR sweeps over ~100 ms while the vehicle moves, so points must be **motion-compensated**, and camera and LiDAR time-stamps must be aligned. Poor synchronisation produces boxes that appear shifted from the objects they label.

## 8.6 Directory Layout and Splits

A typical dataset has separate directories for each sensor and for labels, plus a definition of **train / validation / test** splits. Two points matter for safety work:

- **Split by sequence, not by frame.** Consecutive frames are nearly identical; if they are split randomly between training and test sets, the test score is inflated by near-duplicates (data leakage, see Chapter 10).
- **Test labels may be withheld.** Several benchmarks keep test ground truth private and score submissions on a server, which prevents tuning to the test set but also means you cannot analyse test errors yourself.

## 8.7 Summary

Driving data comes in a variety of sensor and annotation formats whose conventions — box parameterisation, units, coordinate frame, keyframe versus sweep — must be understood exactly. Relational datasets such as nuScenes link sensors, time and objects through tables; calibration and synchronisation relate the sensors to each other. Careful format handling is a prerequisite for every analysis that follows.

## Exercises

1. **Format conversion.** A 1242 × 375 pixel image contains a car with pixel box $(x_1, y_1, x_2, y_2) = (600, 150, 700, 200)$. Compute its YOLO-format line (class id 0).
2. **Projection.** Explain, step by step, how you would draw a LiDAR point on a camera image using KITTI calibration data. Which transform is applied first?
3. **Leakage.** A team randomly splits all frames of 50 driving sequences into 80 % training and 20 % test. Explain why the reported test accuracy is unreliable and propose a better split.

## Further Reading

- Lin, T.-Y. et al. — *Microsoft COCO: Common Objects in Context* (COCO annotation format).
- nuScenes devkit documentation — schema description of the nuScenes tables.
- KITTI development kit — description of label, calibration and Velodyne file formats.
- ASAM OpenLABEL — standard for annotation formats and scenario tagging.
