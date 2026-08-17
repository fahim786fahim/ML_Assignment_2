# Bank Marketing Classification - Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether a bank customer will subscribe to a term deposit.

The project uses the UCI Bank Marketing dataset and evaluates different classification algorithms using multiple performance metrics. An interactive Streamlit application is also provided to allow users to upload test data, select a model, and view prediction and evaluation results.

## 2. Dataset Description

Dataset Name: Bank Marketing Dataset

Source: UCI Machine Learning Repository

The dataset contains information related to direct marketing campaigns conducted by a Portuguese banking institution.

- Number of instances: 45,211

- Number of input features: 16

- Target variable: y

- Classification type: Binary classification

- Target classes:

- yes - Customer subscribed to a term deposit

- no - Customer did not subscribe to a term deposit

The dataset contains both numerical and categorical features.

### Numerical Features

- age

- balance

- day

- duration

- campaign

- pdays

- previous

### Categorical Features

- job

- marital

- education

- default

- housing

- loan

- contact

- month

- poutcome

The dataset contained no duplicate rows and no null values. Some categorical features contained the value unknown, which was retained as a valid category during preprocessing.

The target classes are imbalanced, with substantially more no observations than yes observations.

## 3. GitHub Repository Link

GitHub Repository:

https://github.com/fahim786fahim/ML_Assignment_2

## 4. Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression

2. Decision Tree Classifier

3. K-Nearest Neighbors Classifier

4. Gaussian Naive Bayes

5. Random Forest Classifier

### Preprocessing

The dataset was divided into training and testing data using an 80:20 split with stratification of the target variable.

Numerical features were standardized using StandardScaler.

Categorical features were transformed using OneHotEncoder with unknown categories ignored during prediction.

The target variable was encoded as:

- no = 0

- yes = 1

The preprocessing and model steps were combined using scikit-learn pipelines.

## 5. Model Evaluation

Each model was evaluated using the following metrics:

- Accuracy

- AUC Score

- Precision

- Recall

- F1 Score

- Matthews Correlation Coefficient (MCC)

### Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.9045 | 0.9263 | 0.6506 | 0.3960 | 0.4924 | 0.4597 |

## 6. Model Performance Observations

| Model | Performance Observations |
|---|---|
| Logistic Regression | Logistic Regression achieved an Accuracy of 0.9012 and a strong AUC score of 0.9056. It also achieved good Precision of 0.6445. However, Recall was relatively low at 0.3478, indicating that the model missed a considerable number of customers who actually subscribed.
|Decision Tree|The Decision Tree achieved an Accuracy of 0.8746 and an AUC score of 0.7015. Its Recall of 0.4754 was higher than Logistic Regression, meaning that it detected a larger proportion of actual subscribers. However, its lower Precision and AUC indicate that this improvement came with more incorrect positive predictions.
|K-Nearest Neighbors|KNN achieved an Accuracy of 0.8962 and Precision of 0.5990. However, its Recall of 0.3403 was the lowest among the evaluated models. This indicates that although its overall prediction accuracy was relatively high, it was less effective at detecting the positive subscription class.
|Naive Bayes|Gaussian Naive Bayes achieved the highest Recall of all evaluated models at 0.5198. This means it identified the largest proportion of actual subscribers. However, its Precision of 0.4059 and Accuracy of 0.8548 were lower than the other stronger models, indicating a relatively high number of false-positive predictions.
|Random Forest|Random Forest achieved the strongest overall performance. It produced the highest Accuracy of 0.9045, AUC score of 0.9263, Precision of 0.6506, F1 Score of 0.4924, and MCC score of 0.4597.

### Overall Winner

Random Forest was selected as the overall best-performing model for this dataset.

It achieved the highest value in five of the six required evaluation metrics: Accuracy, AUC, Precision, F1 Score, and MCC. Naive Bayes achieved the highest Recall, but Random Forest provided the strongest overall balance of classification performance.

## 7. Streamlit Application

The Streamlit application supports:

- Uploading test data in CSV format

- Selecting among the trained classification models

- Displaying predictions and prediction probabilities

- Displaying Accuracy, AUC, Precision, Recall, F1 Score, and MCC

- Displaying a confusion matrix

- Displaying a classification report

- Comparing the performance of all implemented models

## 8. Project Structure

```text

ML_Assignment_2/

├── app.py

├── requirements.txt

├── README.md

├── test_data.csv

├── data/

│ └── bank-full.csv

└── model/

├── model_training.py

├── model_results.csv

├── model_observations.txt

├── logistic_regression.pkl

├── decision_tree.pkl

├── knn.pkl

├── naive_bayes.pkl

└── random_forest.pkl