# Breast Cancer Tumour Classification - ML Assignment 2

## a. Problem Statement
Doctors need to tell cancerous (malignant) breast lumps apart from harmless (benign) ones. Using 30 numeric
measurements taken from the cell nuclei in a digitised fine-needle-aspirate (FNA) image, this project builds five
different classifiers that predict whether a tumour is **malignant (1)** or **benign (0)**, compares them with six
evaluation metrics, and serves them through a Streamlit web app. Since a missed cancer is far more costly than a false
alarm, Recall and the number of false negatives are examined along with Accuracy.

## b. Dataset Description
| Item | Detail |
|---|---|
| Name / source | Breast Cancer Wisconsin (Diagnostic) - UCI Machine Learning Repository; https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Instances | 569 patients: 357 benign, 212 malignant (minimum required: 500) |
| Features | 30 numeric features (minimum required: 12) |
| Feature design | 10 nuclear measurements - radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension - each stored as **mean**, **standard error** and **worst** value |
| Target | `Target` = 1 for Malignant, 0 for Benign (converted from the M/B letters) |
| Dropped column | `id` - only a patient identifier |
| Data quality | No missing values |
| Preprocessing | `StandardScaler` inside a scikit-learn Pipeline, fitted on the training part only |
| Train / test split | Stratified random split, 80% train (455 rows) / 20% test (114 rows), `random_state=42` |

## c. GitHub Repository Link
https://github.com/Rajat-13/ML-Assignment-2-1st-Sem-BITS/tree/main

| File / folder | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `requirements.txt` | Python dependencies |
| `README.md` | This document |
| `test_data.csv` | Test data (30 features + `Target`) used for evaluation and for the app upload |
| `breast_cancer_wdbc.csv` | Full dataset with column names added |
| `model/` | Saved trained pipelines (`.joblib`) for all five models and the feature list |

## d. Models Used
All five models were trained on the same training set and evaluated on the same 114 test patients.

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9123 | 0.8770 | 0.9444 | 0.8095 | 0.8718 | 0.8113 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Cross-check with 5-fold stratified cross-validation (mean accuracy): Logistic Regression 0.974, Decision Tree 0.923,
KNN 0.963, Naive Bayes 0.930, SVM 0.977.

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Excellent results (96.5% accuracy) and the best AUC of all models (0.996). It missed 3 malignant tumours and raised 1 false alarm. After scaling, the two classes are close to linearly separable, which is why a simple linear model works so well. |
| Decision Tree | The weakest model: 91.2% accuracy, AUC only 0.877 and recall 0.81, with 8 malignant tumours missed. A single shallow tree splits the data into coarse regions and gives crude probabilities, so it ranks patients less reliably than the smoother models. |
| KNN | Good overall (95.6% accuracy, AUC 0.982) and very precise (0.974), but 4 malignant tumours were missed. Scaling makes similar tumours lie close together, which suits KNN; the result still depends on the chosen k. |
| Naive Bayes | Lowest accuracy among the strong models (92.1%) yet a high AUC (0.989): it orders patients well, but its default 0.5 cut-off misses 6 malignant cases. Its assumption that features are independent is not true here - radius, perimeter and area are strongly correlated. |
| Support Vector Machine | Highest accuracy (97.4%), precision (1.00, zero false alarms), F1 (0.963) and MCC (0.944). The RBF kernel can draw a curved boundary between the classes. It missed 3 malignant cases. |
| **Overall winner** | **Support Vector Machine** - best Accuracy, Precision, F1 and MCC, and also the highest cross-validated accuracy (0.977). Logistic Regression is a close second with the best AUC and the advantage of being simple and easy to interpret. With only 114 test patients one extra error changes accuracy by about 0.9 points, so the cross-validation results are useful support for this ranking. |

## e. Streamlit App
https://ml-assignment-2-1st-sem-bits-5wo9yyfttdv6hhn3bu6zua.streamlit.app/](https://ml-assignment-2-1st-sem-bits-5wo9yyfttdv6hhn3bu6zua.streamlit.app/

The app (`app.py`) provides:
1. Upload of a test CSV file (use `test_data.csv`)
2. A dropdown to choose one of the five models
3. Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model
4. Confusion matrix, classification report, ROC curve and the number of missed malignant cases
5. A table comparing all five models on the uploaded data

The saved models are loaded from `model/`; nothing is retrained on the uploaded data.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook ML_Assignment_2.ipynb     # run all cells: recreates model/ and test_data.csv
streamlit run app.py
```
