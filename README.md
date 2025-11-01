# Breast-Cancer-Detection

🩺 Overview
Breast cancer is one of the most prevalent and life-threatening diseases among women worldwide. Early detection plays a vital role in reducing mortality and improving recovery rates. This project focuses on developing an automated breast cancer detection system using Deep Learning, specifically the MobileNet architecture.
The model analyzes mammogram images to classify them into two categories — Benign (Non-cancerous) and Malignant (Cancerous). Unlike traditional diagnosis that depends on manual interpretation, this system leverages Convolutional Neural Networks (CNNs) to extract image features and make accurate predictions automatically.
By applying Transfer Learning with MobileNet, the model achieves high accuracy while remaining lightweight and computationally efficient — making it suitable for real-time, mobile, and edge-based medical applications.

⚙️ Features
 Deep Learning-based Detection: Uses CNN architecture to automatically identify tumors from mammogram images.
 Binary Classification: Classifies images into Benign or Malignant categories.
 Transfer Learning with MobileNetV2: Speeds up training and improves accuracy even with limited data.
 Data Augmentation: Enhances model generalization by applying rotation, flipping, and zooming.
 Lightweight and Deployable: Can be integrated into mobile or embedded devices using TensorFlow Lite.
 Performance Metrics: Evaluates model using accuracy, precision, recall, F1-score, confusion matrix, and ROC curve visualization.

 🧩 How It Works (Technology Stack)
1. Tools & Frameworks
Programming Language: Python
Deep Learning Framework: TensorFlow, Keras
Libraries Used: OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
Model Architecture: MobileNetV2 (Transfer Learning)
Development Environment: Jupyter Notebook / Google Colab
Version Control: Git & GitHub
🩻 2. Workflow
Data Collection: Dataset of mammogram images (Benign & Malignant) sourced from open repositories.
Preprocessing:
Resizing images to 150×150 pixels
Normalizing pixel values (0–1 range)
Augmenting images for improved generalization
Model Training:
MobileNetV2 used with custom classification layers
Optimizer: Adam | Loss: Binary Cross-Entropy
Metrics: Accuracy, Precision, Recall, F1-Score
Model Evaluation: Tested on unseen data and visualized using confusion matrix & ROC curve.
Deployment: Model saved and optimized for TensorFlow Lite for potential mobile integration.
