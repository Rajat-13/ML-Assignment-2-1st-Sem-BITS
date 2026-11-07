# Breast Cancer Diagnosis (WDBC) - ML Assignment 2

## a. Problem Statement
Classify a breast tumour as **malignant (1)** or **benign (0)** from measurements of cell nuclei computed from a
digitised image of a fine needle aspirate (FNA) of the breast mass. This is a **binary classification** problem.
Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on the same data, compared
using six metrics, and deployed in an interactive Streamlit app. Because missing a cancer is costly, Recall and
false negatives are discussed alongside Accuracy.

## b. Dataset Description
| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic), WDBC |
| Instances | 569 patients (357 benign, 212 malignant) |
| Features | 30 real-valued features (min. required: 12) |
| Feature design | 10 nuclear characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension), each given as **mean**, **standard error** and **worst** value |
| Target | `Target` = 1 for Malignant (M), 0 for Benign (B) |
| Removed column | `id` (patient identifier, no predictive meaning) |
| Missing values | None |
| Preprocessing | StandardScaler inside a scikit-learn Pipeline (fitted on training data only) |
| Split | Stratified random 80% train (455) / 20% test (114), `random_state=42` |

## c. GitHub Repository Link
**<PASTE YOUR GITHUB REPO LINK HERE>**

**Live Streamlit App:** <PASTE YOUR STREAMLIT LINK HERE>

### Repository structure
```
├── app.py                    # Streamlit app
├── ML_Assignment_2.ipynb     # Notebook: EDA, training, evaluation, model saving
├── requirements.txt
├── README.md
├── test_data.csv             # Test data (30 features + Target) to upload in the app
├── breast_cancer_wdbc.csv    # Full dataset with column names
└── model/                    # Saved pipelines (.joblib) for all five models + feature list
```

## d. Models Used

### Comparison Table (test data, 114 patients)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9123 | 0.8770 | 0.9444 | 0.8095 | 0.8718 | 0.8113 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

Extra check - 5-fold stratified cross-validation accuracy: LR 0.974, DT 0.923, KNN 0.963, NB 0.930, SVM 0.977.

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong: 96.5% accuracy and the highest AUC (0.996). It missed 3 malignant tumours and raised 1 false alarm. The classes are almost linearly separable after scaling, which suits a linear model. |
| Decision Tree | Weakest model: 91.2% accuracy, AUC 0.877 and recall of 0.81 (8 malignant cases missed). A single shallow tree gives coarse, step-like probabilities and is less stable than the smooth boundaries of LR/SVM. |
| KNN | Good (95.6% accuracy, AUC 0.982) with high precision (0.974) but 4 missed malignant cases. Works because scaled features place similar tumours close together; slightly more sensitive to the choice of k and to noisy neighbours. |
| Naive Bayes | 92.1% accuracy but excellent AUC (0.989), so it ranks cases well while its default 0.5 threshold is not ideal. The independence assumption is violated because radius, perimeter and area are highly correlated, causing 6 missed malignant cases. |
| Support Vector Machine | Best accuracy (97.4%), precision (1.00, no false alarms), F1 (0.963) and MCC (0.944). The RBF kernel captures the non-linear boundary; it missed 3 malignant cases. |
| **Overall winner** | **Support Vector Machine** - best Accuracy, Precision, F1 and MCC, and also the best cross-validated accuracy (0.977). Logistic Regression is a very close second (best AUC, and simpler and easier to interpret). |

Note: the test set is small (114 patients), so one extra mistake changes accuracy by about 0.9 percentage points; the
cross-validation figures confirm the ranking SVM ≈ LR > KNN > NB > DT.

## e. Streamlit App Features
1. CSV upload of test data (`test_data.csv`)
2. Model selection dropdown (5 models)
3. Evaluation metrics: Accuracy, AUC, Precision, Recall, F1, MCC
4. Confusion matrix, classification report and ROC curve, plus a count of missed malignant cases
5. Side-by-side comparison table of all five models

The app loads the saved models and does **not** retrain on the uploaded data.

## How to run locally
```bash
pip install -r requirements.txt
jupyter notebook ML_Assignment_2.ipynb   # run all cells to (re)create models and test_data.csv
streamlit run app.py
```
