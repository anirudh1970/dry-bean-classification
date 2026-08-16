# Before running, ensure you have the required libraries installed:
# pip install pandas scikit-learn ucimlrepo
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------
# STEP 1: DATASET CHOICE & LOADING
# ---------------------------------------------------------
print("Loading the Dry Bean dataset from UCI...")
# Fetch dataset using the official UCI Python package
dry_bean = fetch_ucirepo(id=602) 

# Extract features and targets
X = dry_bean.data.features 
y = dry_bean.data.targets 

# The target variable is categorical (bean species). We need to encode it into integers.
le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

# Split into training and testing datasets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale the numerical features (Crucial for distance-based models like KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create the model/ directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the scaler and label encoder so the Streamlit app can preprocess the uploaded data
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/label_encoder.pkl')

# Save the test data to a CSV. This is required for Step 3 and for your Streamlit app to consume.
test_df = X_test.copy()
test_df['Class'] = le.inverse_transform(y_test) # Revert classes back to original text names for clarity in the UI
test_df.to_csv('test_data.csv', index=False)
print("Saved 'test_data.csv' successfully.\n")


# ---------------------------------------------------------
# STEP 2: ML CLASSIFICATION MODELS & EVALUATION METRICS
# ---------------------------------------------------------

# Define the 5 models explicitly listed in the assignment
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(), # Gaussian chosen over Multinomial due to the continuous nature of the features[cite: 1]
    "Random Forest (Ensemble)": RandomForestClassifier(random_state=42)
}

# Dictionary to store the calculated metrics
evaluation_results = {}

print("Training models and calculating metrics...")
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    # Predict probabilities (Required to calculate the AUC Score)
    y_prob = model.predict_proba(X_test_scaled)

    safe_name = name.replace(" ", "_")
    joblib.dump(model, f'model/{safe_name}.pkl')
    
    # Calculate metrics
    # Note: 'macro' averaging is used because this is a multi-class dataset
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Store metrics, rounded to 4 decimal places for readability
    evaluation_results[name] = {
        "Accuracy": round(accuracy, 4),
        "AUC": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    }

# Display the results formatted as a comparison table, ready for the README[cite: 1]
results_df = pd.DataFrame(evaluation_results).T
print("\n--- Evaluation Metrics Comparison ---")
print(results_df.to_string())