# codealpha_tasks
I enrolled for an one month inrernship in the Machine Learning domain in CodeAlpha. This repo consists of the three projects I made during this internship period.

1. Emotion Recognition from using Specch using ML

2. Handwritten Digit Recognition using Neural Network

3. Detection of Parkinson's using ML


#task 4(Detection of Parkinso's using ML)

# Project Overview

This project aims to detect Parkinson's disease using machine learning models trained on biomedical voice measurements. The dataset used is from the UCI Machine Learning Repository:

🔗 [Parkinson's Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)

The system evaluates multiple machine learning models **(Random Forest, SVM, Logistic Regression, and Neural Networks)** to classify whether a patient is **Healthy (0)** or has **Parkinson’s Disease (1)** based on their voice characteristics.

# Dataset Information
The dataset consists of 23 biomedical voice measurements and one target variable (status):

- status = 0 → **Healthy**

- status = 1 → **Parkinson’s disease**


**Some key features in the dataset:**

 - **MDVP:** Fo(Hz) – Fundamental frequency
 
 - **MDVP:** Jitter(%) – Measures frequency variation
 
 - **MDVP:** Shimmer – Measures amplitude variation
 
 - **HNR** – Harmonics-to-noise ratio
 
 - **RPDE & DFA** – Signal complexity indicators
 
 
# Project Workflow

**1.	Data Preprocessing**


- Remove irrelevant columns (e.g., patient name).

- Handle missing values (if any).

- Standardize features using StandardScaler to improve model performance.

- Split dataset into **80%** training and **20%** testing.


**2. Model Training & Evaluation:**


We implemented and tested four different models:

 - Random Forest
 
 - Support Vector Machine (SVM)
 
 - Logistic Regression
 
 - Neural Network (MLP Classifier)

 
Each model was evaluated using:


- Accuracy Score

- Confusion Matrix

- Classification Report (Precision, Recall, F1-Score)
  

**Feature Selection (Top 10 Features)**


To improve performance, Random Forest feature importance was used to select the top 10 most relevant features.

- Re-trained models using only these selected features.

- Compared accuracy before & after feature selection.

# Hyperparameter Tuning

We applied **GridSearchCV** for optimal hyperparameter selection:

Tuned SVM Parameters

**Code:**

```
'C': [0.1, 1, 10]
'kernel': ['linear', 'rbf']
Tuned Random Forest Parameters
```

**Code:**

```'n_estimators': [100, 200, 300]
'max_depth': [10, 20, 30]
'min_samples_split': [2, 5, 10]
``` 

# Best Model Selection & Saving

The model with the highest accuracy is automatically saved and loaded for real-world testing.

- Neural Network (NN) & Random Forest (RF) performed the best.

- The best model is saved as either .h5 (Neural Network) or .pkl (ML models).

# Real-World Testing with New Data

- Users can input new patient data for prediction.

- The trained model predicts whether the patient has Parkinson’s or not.

**Code:**

```
predicted_status = model.predict(new_patient_data)
print("Predicted Status:", "Parkinson's" if predicted_status[0] == 1 else "Healthy")
``` 

# Visualization

- Feature Importance Ranking (Random Forest)

- Confusion Matrix for each model

- Comparison of model accuracy with selected features

# Model Performance Comparison

| **Model Name** | **Accuracy** |
|-----------|-----------|
| Random Forest | 94.87% |
| Neural Network(MLP) | 94.87% |
| SVM | 92.31% |
| Logistic Regression | 92.31% |

# Technologies Used Python

- Pandas, NumPy (Data Processing)

- Scikit-Learn (Machine Learning)

- TensorFlow/Keras (Neural Networks)

- Seaborn & Matplotlib (Data Visualization)

- Joblib (Model Saving & Loading)

# How to Run the Project

**Install Dependencies**

```
pip install pandas numpy scikit-learn seaborn matplotlib tensorflow joblib
``` 

**Clone Repository**

```git clone https://github.com/yourusername/Parkinson-Detection-ML.git
cd Parkinson-Detection-ML
```

**Run the Jupyter Notebook**

```
jupyter notebook
``` 

or, run the Python script directly:

```
python parkinsons_detection.py
```

 
 # Contact & Contribution
 
🔗 **GitHub:** [Rakib007Nabila](https://github.com/Rakib007Nabila)

[📧 Email Me](mailto:nabila.rakib7@gmail.com)




📌 Contributions are welcome! Feel free to open issues or submit pull requests. 😊


















