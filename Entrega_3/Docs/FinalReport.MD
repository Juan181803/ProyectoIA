# Video Annotation System Using Pose Detection and Activity Classification

## Abstract
This project implements a real-time video annotation system capable of detecting and classifying specific human activities using pose estimation and machine learning. The system utilizes MediaPipe for pose detection and Support Vector Machines (SVM) for activity classification, achieving an overall accuracy of 95.4%. The solution can detect and classify activities such as walking, turning, sitting, and standing, while also tracking key joint movements and postural changes in real-time.

## Introduction
Human activity recognition through video analysis has become increasingly important in various fields, including healthcare, sports analysis, and human-computer interaction. Our project addresses the challenge of automatically detecting and classifying human activities while providing real-time postural analysis.

### Problem Statement
The main objective is to develop a software tool that can:
- Analyze specific human activities in real-time
- Track joint movements and postural changes
- Provide accurate classification of activities
- Measure key postural metrics (lateral inclination, joint angles)

## Theory
The system architecture combines several key technologies and concepts:

### Pose Estimation
We utilize MediaPipe's pose detection solution, which provides 33 body landmarks in 3D space. These landmarks serve as the foundation for our feature extraction process.

### Feature Engineering
Key features extracted include:
- Joint angles (knee flexion, hip angles)
- Postural metrics (torso verticality, shoulder rotation)
- Movement characteristics (lateral sway, arm swing)
- Spatial relationships between joints

### Classification Model
We implemented an SVM classifier with RBF kernel, chosen for its effectiveness in handling non-linear relationships and its robust performance with high-dimensional data.

## Methodology
Our approach followed the CRISP-DM methodology:

1. **Data Collection**
   - Recorded videos of multiple subjects performing various activities
   - Created a comprehensive dataset covering different perspectives and movement variations

2. **Data Preprocessing**
   - Landmark extraction using MediaPipe
   - Signal smoothing using Savitzky-Golay filter
   - Feature normalization and standardization

3. **Feature Engineering**
   - Generated 111 features including:
     - Joint angles and distances
     - Movement patterns
     - Postural indicators

4. **Model Development**
   - Split data: 80% training, 20% testing
   - Implemented SVM classifier (C=10, RBF kernel)
   - Applied cross-validation for model validation

5. **Real-time Implementation**
   - Developed frame-by-frame processing pipeline
   - Implemented confidence scoring system
   - Created real-time visualization interface

## Results
The system achieved impressive performance metrics:

### Classification Performance

```
                precision | recall | f1-score | support
Caminando_Espalda | 1.00  | 1.00   | 1.00     | 224
Caminando_Frente  | 1.00  | 1.00   | 1.00     | 274
Caminando_Lado    | 1.00  | 1.00   | 1.00     | 67
Giro              | 1.00  | 1.00   | 1.00     | 426
Parado            | 0.90  | 0.93   | 0.91     | 458
Quieto            | 1.00  | 1.00   | 1.00     | 10
Sentado           | 0.91  | 0.87   | 0.89     | 375
```

Overall Accuracy: 95.4%

### Key Achievements

- Real-time processing capability (30+ FPS)
- Robust activity classification
- Accurate joint angle measurements
- Effective postural analysis

### Results Analysis

The system shows exceptional performance in detecting dynamic activities (walking, turning) with perfect precision and recall. Static postures (standing, sitting) show slightly lower but still robust performance (>90% precision), likely due to subtle variations in posture that can create classification challenges.
The implementation of movement verification helps prevent misclassifications, though it occasionally leads to conservative predictions for low-movement activities.

### Conclusions and Future Work

The developed system successfully meets its primary objectives, providing accurate activity classification and postural analysis in real-time. Future improvements could include:

- Implementation of temporal modeling (LSTM/GRU) for better sequence understanding
- Extension of the activity set to include more complex movements
- Development of a more sophisticated confidence scoring system
- Integration with a user-friendly GUI for clinical applications

### References

- MediaPipe Pose. Google AI. https://ai.google.dev/edge/mediapipe/solutions/guide
