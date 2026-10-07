# Pose-Based Gait Feature Extraction

This project implements a simple yet structured gait analysis pipeline using 2D pose estimation. The system extracts lower-body joint keypoints from a walking video and computes spatial and temporal gait features suitable for machine learning applications.

---

## Objective

- Extract human pose keypoints from a walking video
- Compute simple gait-related features
- Construct a structured gait feature vector

The focus is on feature extraction rather than complex biomechanical modeling.

---

## Methodology

The pipeline consists of:

1. Pose estimation using MediaPipe
2. Extraction of hip, knee, and ankle keypoints
3. Joint angle computation using vector geometry
4. Step detection using ankle vertical displacement
5. Feature vector construction

---

## Extracted Features

The final gait feature vector includes:

- Mean Left Knee Angle
- Variance Left Knee Angle
- Mean Right Knee Angle
- Variance Right Knee Angle
- Mean Hip Angle
- Average Step Duration
- Cadence (steps per minute)
- Average Ankle Vertical Displacement

Feature Dimension: 8

---

## 📁 Project Structure
