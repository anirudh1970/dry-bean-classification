import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Dry Bean Classification Model Evaluator")
st.write("Upload the test data to evaluate the performance of different Machine Learning models.")

# --- 1. Load Pre-trained Assets ---
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Load all saved models
    models = {
        "Logistic Regression": joblib.load('Logistic_Regression.pkl'),
        "Decision Tree": joblib.load('Decision_Tree.pkl'),
        "kNN": joblib.load('kNN.pkl'),
        "Naive Bayes": joblib.load('Naive_Bayes.pkl'),
        "Random Forest (Ensemble)": joblib.load('Random_Forest_(Ensemble).pkl')
    }
    return scaler, le, models

scaler, le, models = load_assets()

# --- 2. UI: Model Selection & File Upload ---
# Requirement: Model selection dropdown
selected_model_name = st.selectbox("Select a Classification Model", list(models.keys()))

# Requirement: Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Separate features and target
    X_test = df.drop(columns=['Class'])
    y_test_labels = df['Class']
    
    # Preprocess: Encode target and Scale features
    y_test_encoded = le.transform(y_test_labels)
    X_test_scaled = scaler.transform(X_test)
    
    # Get the selected model and make predictions
    model = models[selected_model_name]
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    # --- 3. UI: Display Evaluation Metrics ---
    st.write(f"### Evaluation Metrics: {selected_model_name}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y_test_encoded, y_pred), 4))
    col2.metric("AUC Score", round(roc_auc_score(y_test_encoded, y_prob, multi_class='ovr', average='macro'), 4))
    col3.metric("MCC Score", round(matthews_corrcoef(y_test_encoded, y_pred), 4))
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Precision (Macro)", round(precision_score(y_test_encoded, y_pred, average='macro', zero_division=0), 4))
    col5.metric("Recall (Macro)", round(recall_score(y_test_encoded, y_pred, average='macro'), 4))
    col6.metric("F1 Score (Macro)", round(f1_score(y_test_encoded, y_pred, average='macro'), 4))
    
    # --- 4. UI: Confusion Matrix & Classification Report ---
    st.write("### Classification Report")
    report = classification_report(y_test_encoded, y_pred, target_names=le.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
else:
    st.info("Please upload the 'test_data.csv' file to see model evaluations.")
