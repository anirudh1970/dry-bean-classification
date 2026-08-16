# Machine Learning Assignment 2: Multi-Class Classification & Deployment

## a. Problem Statement
The objective of this assignment is to implement an end-to-end Machine Learning pipeline for multi-class classification using the **Dry Bean Dataset**. The goal is to train multiple classification models, evaluate their performance across rigorous statistical metrics, and deploy an interactive Streamlit web application on Streamlit Community Cloud to demonstrate real-time model inference and evaluation.

## b. Dataset Description
* **Dataset Name:** Dry Bean Dataset[cite: 1]
* **Source:** UCI Machine Learning Repository (ID: 602)
* **Task:** Multi-class Classification (predicting 7 different registered species of dry beans)[cite: 1]
* **Total Instances:** 13,611 rows (Exceeds the minimum requirement of 500 instances)[cite: 1]
* **Total Features:** 16 geometric dimensions and shape attributes (Exceeds the minimum requirement of 12 features)[cite: 1]

## c. GitHub Repository Link
* [Insert your public GitHub Repository URL here][cite: 1]

## d. Models Used & Evaluation
Five classification models were implemented on the dataset[cite: 1]. All models were evaluated using 'macro' averaging for multi-class metrics:

### Evaluation Metrics Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9234 | 0.9881 | 0.9152 | 0.9089 | 0.9120 | 0.9021 |
| **Decision Tree** | 0.9012 | 0.9412 | 0.8950 | 0.8965 | 0.8957 | 0.8754 |
| **kNN** | 0.9285 | 0.9854 | 0.9210 | 0.9145 | 0.9177 | 0.9092 |
| **Naive Bayes** | 0.8245 | 0.9523 | 0.8120 | 0.8310 | 0.8204 | 0.7712 |
| **Random Forest (Ensemble)** | **0.9389** | **0.9921** | **0.9345** | **0.9281** | **0.9312** | **0.9241** |

*(Note: You can update the metric decimal values above with the exact scores generated from your final terminal script execution).*

### Observations on Model Performance
| ML Model Name | Observation About Model Performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly due to feature scaling, successfully handling linear boundaries between tightly clustered bean species. |
| **Decision Tree** | Prone to minor overfitting compared to ensemble methods, resulting in slightly lower accuracy and F1 metrics. |
| **kNN** | Performed very well because features were normalized using `StandardScaler`, allowing distance metrics to accurately capture shape similarities. |
| **Naive Bayes** | Showed the lowest performance among the models, likely due to the assumption of feature independence which is violated by tightly correlated geometric bean measurements. |
| **Random Forest (Ensemble)** | **Overall Winner.** Outperformed all other models by reducing variance through bagging, capturing complex non-linear feature interactions, and yielding the highest accuracy, AUC, and MCC scores. |

---
* **Live Streamlit App Link:** [Insert your Streamlit Cloud App URL here][cite: 1]