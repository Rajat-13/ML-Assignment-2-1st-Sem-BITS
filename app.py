# ---------------------------------------------------------------
# Breast Cancer (WDBC) Tumour Classifier - Streamlit app
# Loads the 5 pre-trained pipelines from /model (NO retraining)
# and evaluates them on an uploaded test CSV.
# ---------------------------------------------------------------
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="wide")

# Display name -> saved file
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CLASS_NAMES = ["Benign", "Malignant"]   # 0 = Benign, 1 = Malignant


@st.cache_resource
def load_model(file_name):
    """Load a saved pipeline (scaler + classifier) once and keep it in memory."""
    return joblib.load(os.path.join(MODEL_DIR, file_name))


@st.cache_resource
def load_features():
    """Ordered list of the 30 feature columns used during training.
    Uses feature_cols.joblib if present; otherwise falls back to the column names
    stored inside the fitted pipeline (scikit-learn keeps them in feature_names_in_)."""
    path = os.path.join(MODEL_DIR, "feature_cols.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    first_model = load_model(list(MODEL_FILES.values())[0])
    return list(first_model.feature_names_in_)


# ---------------- Check that the saved models are really there ----------------
missing_files = [f for f in MODEL_FILES.values() if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing_files:
    st.error("Saved model files were not found in the deployed repository.")
    st.write("Looking for a folder named **model** (all lowercase) next to app.py containing:", missing_files)
    here = os.path.dirname(os.path.abspath(__file__))
    st.write("Files/folders the app can actually see next to app.py:", sorted(os.listdir(here)))
    if os.path.isdir(MODEL_DIR):
        st.write("Contents of model/:", sorted(os.listdir(MODEL_DIR)))
    st.stop()


def compute_metrics(y_true, y_pred, y_prob):
    """Return the six required evaluation metrics as a dict."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------- Header ----------------
st.title("🩺 Breast Cancer Diagnosis - Model Evaluation Dashboard")
st.write("Upload **test data** (CSV) from the Wisconsin Diagnostic Breast Cancer dataset and see how each of the "
         "five trained models classifies tumours as **Benign (0)** or **Malignant (1)**.")
st.caption("Educational project only - not a medical diagnostic tool.")

# ---------------- Sidebar: inputs ----------------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("1. Upload test CSV", type=["csv"])
model_name = st.sidebar.selectbox("2. Select model", list(MODEL_FILES.keys()))
st.sidebar.caption("The CSV must contain the 30 feature columns plus a `Target` column "
                   "(1 = Malignant, 0 = Benign). A ready file `test_data.csv` is in the repo.")

if uploaded is None:
    st.info("👈 Upload a test CSV from the sidebar to begin.")
    st.stop()

# ---------------- Read and validate data ----------------
data = pd.read_csv(uploaded)
features = load_features()
missing_cols = [c for c in features + ["Target"] if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded file is missing {len(missing_cols)} required column(s): {missing_cols[:10]}")
    st.stop()
if data["Target"].nunique() < 2:
    st.error("The Target column must contain both classes (0 and 1) so that AUC can be computed.")
    st.stop()

X_test, y_test = data[features], data["Target"]

st.subheader("Uploaded data preview")
c1, c2, c3 = st.columns(3)
c1.metric("Patients (rows)", len(data))
c2.metric("Features", len(features))
c3.metric("Malignant share", f"{y_test.mean():.1%}")
st.dataframe(data.head(), use_container_width=True)

# ---------------- Selected model results ----------------
model = load_model(MODEL_FILES[model_name])
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
metrics = compute_metrics(y_test, y_pred, y_prob)

st.subheader(f"Results - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

# Extra: highlight the most clinically important error (missed cancers)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
st.write(f"**Missed malignant cases (false negatives): {fn}** &nbsp;|&nbsp; "
         f"False alarms (false positives): {fp}")

tab1, tab2, tab3 = st.tabs(["Confusion matrix", "Classification report", "ROC curve"])

with tab1:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=[0, 1]), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with tab2:
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

with tab3:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend()
    st.pyplot(fig)

# ---------------- Comparison of all models ----------------
st.subheader("Comparison of all five models on the uploaded data")
rows = []
for name, fname in MODEL_FILES.items():
    m = load_model(fname)
    p = m.predict(X_test)
    pr = m.predict_proba(X_test)[:, 1]
    rows.append({"ML Model Name": name, **compute_metrics(y_test, p, pr)})
comp = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comp.style.highlight_max(axis=0, color="#c6efce"), use_container_width=True)
st.caption("Green = best value in each column.")
