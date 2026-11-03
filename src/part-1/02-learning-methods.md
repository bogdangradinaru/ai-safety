
# 2. AI Learning Methods and Training Paradigms

## 2.1 Introduction

Artificial Intelligence systems can be trained using different learning approaches, depending on the available data, the training objectives, and the desired system behaviour.

The main learning paradigms include:

- **Supervised Learning**
- **Unsupervised Learning**
- **Self-Supervised Learning**
- **Semi-Supervised Learning**
- **Reinforcement Learning**
- **Imitation Learning**

## 2.2 Supervised Learning

Supervised learning is a machine learning approach in which a model is trained using labelled data.

Each training example contains:
- **Input:** e.g. an image captured by an automotive camera.
- **Label:** e.g. a bounding box identifying a pedestrian.
- **Output:** the model prediction, which is compared against the ground-truth label

### Practical Lab 1 — Automotive Object Detection

In this exercise, we will train a YOLO object detection
model using annotated images from the KITTI dataset.

**Learning objectives:**
- Understand ground-truth annotations.
- Train an object detection model.
- Evaluate detection performance.
- Investigate AI safety implications.

**Tools:** KITTI, YOLO, PyTorch, Google Colab.

[Open notebook in Google Colab](https://colab.research.google.com/github/bogdangradinaru/ai-safety/blob/main/notebooks/part-1/01-supervised-kitti-yolo.ipynb)

[View notebook on GitHub](https://github.com/bogdangradinaru/ai-safety/blob/main/notebooks/part-1/01-supervised-kitti-yolo.ipynb)

### Automotive examples

- Object detection (vehicles, pedestrians, cyclists).
- Traffic sign classification.
- Lane detection.
- Semantic segmentation.

### Automatic labelling and supervised learning

Labels do not necessarily have to be created manually.

Automated labelling techniques can generate labels using:
- Sensor fusion (e.g. LiDAR and camera data).
- Pre-trained AI models.
- Rule-based algorithms.
- Multi-frame tracking and temporal information.

The generated labels can subsequently be used to train another model through supervised learning.

> **Important:** Automatically generated labels still require quality assessment. Incorrect or incomplete labels can introduce systematic errors into the training dataset.

## 2.3 Self-Supervised Learning

Self-supervised learning creates training objectives directly from the data itself, without requiring manually annotated ground-truth labels for the specific pre-training task.

### Examples

- Predicting masked regions of an image.
- Predicting future observations from previous frames.
- Learning representations by comparing different views of the same scene.

### Automotive applications

- Learning visual representations from large collections of driving videos.
- Pre-training perception models.
- Learning useful features from unlabeled sensor data.

## 2.4 Unsupervised Learning

Unsupervised learning identifies patterns, structures, or representations in data without conventional ground-truth labels.

### Examples

- Clustering.
- Dimensionality reduction.
- Anomaly detection.

### Automotive applications

- Identifying unusual driving situations.
- Grouping similar driving scenarios.
- Detecting anomalies in vehicle sensor data.

## 2.5 Reinforcement Learning

Reinforcement learning (RL) is an approach in which an agent learns a policy through interactions with an environment.

The agent receives rewards or penalties based on its actions and learns to maximize cumulative reward.

### Main elements

1. **Agent:** the learning system.
2. **Environment:** the system with which the agent interacts.
3. **State:** the information available to the agent.
4. **Action:** the decision made by the agent.
5. **Reward:** feedback received after an action.

### Automotive applications

- Autonomous driving decision-making.
- Lane-change and manoeuvre selection.
- Motion planning.
- Learning driving policies in simulation.

## 2.6 Imitation Learning

Imitation learning trains a model to reproduce decisions or behaviours demonstrated by an expert.

### Automotive applications

- Learning driving policies from human driving demonstrations.
- Behaviour cloning.
- Learning vehicle manoeuvres from expert driving data.

## 2.7 Comparison of Learning Paradigms

| Learning method | Training data | Learning objective |
|---|---|---|
| Supervised | Labelled data | Predict known labels |
| Unsupervised | Unlabelled data | Discover data structure |
| Self-supervised | Unlabelled data | Generate learning targets from data |
| Semi-supervised | Labelled + unlabelled data | Combine both data sources |
| Reinforcement | Interaction and rewards | Learn an effective policy |
| Imitation | Expert demonstrations | Reproduce demonstrated behaviour |

## 2.8 Summary

Different learning paradigms address different training challenges. In automotive applications, they can also be combined.

For example, a perception model could be pre-trained using self-supervised learning and subsequently fine-tuned using supervised learning with automatically generated or manually annotated labels.

## Further Reading

- Add relevant research papers.
- Add technical documentation.
- Add practical examples.



Supervised learning, reinforcement learning, self-supervised learning, etc 
Automatic labelling + supervised learning - A model or sensor-based process generates labels; another model is trained using those labels. 
Self-supervised learning - The training objective is created from the data itself, for example by predicting masked image regions or future observations. Manual object labels are not necessary for that particular task.
Unsupervised learning - The algorithm learns structure or representations without conventional ground-truth labels.
Reinforcement learning - A policy learns using rewards or other feedback from interactions or simulated environments.


How would you check an unlabeled dataset in practice? 
Data integrity - Automated checks for corrupted files, missing frames, duplicate sequences, timestamp inconsistencies, invalid image dimensions and incorrect metadata. Versioning and data provenance should also be documented.

Learning strategy                 Tesla              NVIDIA Alpamayo                       Mobileye 
Supervised learning / imitation  Yes                  Yes, through supervised fine-tuning  Yes, particularly in perception
Automatic label generation         Yes               Yes                                    Yes  
Self-supervised / unlabeled learning   Not the best description of the whole public pipeline    Possible within foundation-model pretraining; not the full Alpamayo training strategy     Not the best description of the whole public pipeline 
Reinforcement learning             Yes                 Yes                                  Yes, for driving-policy training through ACI #
Learning from simulated driving     Used within its development ecosystem Yes, including closed-loop RL with AlpaGym    Yes, including self-play simulation for ACI 
