# Adult Census Income Prediction - Classical ML & Streamlit Application

An end-to-end Machine Learning project predicting whether an individual's gross annual income exceeds **\$50,000** based on the **UCI Adult Census Income dataset**. Built with Python, Scikit-Learn, XGBoost, and an interactive Streamlit UI.

---

## 🌟 Key Project Highlights

- **Dataset**: UCI Adult Census Income (32,561 raw records $\rightarrow$ **30,139 cleaned records** post missing-value and duplicate removal).
- **Target Variable**: Binary classification: `<=50K` (0) vs. `>50K` (1).
- **Class Distribution**: 22,633 (`<=50K`, 75.1%) vs. 7,506 (`>50K`, 24.9%).
- **Models Benchmarked**: 9 architectures (5 baselines, 2 tuned models, 2 ensemble architectures).
- **Best Model**: **Heterogeneous Stacking Classifier** combining Logistic Regression, Tuned Random Forest, and Tuned XGBoost with a Logistic Regression meta-learner.
- **Top Metrics**: **87.59% Accuracy**, **0.7340 F1-Score**, **0.9302 ROC-AUC**, **78.72% Precision**, **68.75% Recall**.
- **Interactive App**: Multi-page Streamlit application with single-person what-if inference, base-learner decomposition, batch CSV scoring, benchmark leaderboards, and EDA explorer.

---

## 🏆 Model Performance Benchmark (Test Split N = 6,028)

All models were evaluated on the exact same 20% stratified test set:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | MSE | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Classifier (Selected)** | **0.8759** | 0.7872 | 0.6875 | **0.7340** | 0.9302 | **0.1241** | **0.3523** |
| **XGBoost (Baseline)** | 0.8752 | **0.7937** | 0.6742 | 0.7291 | **0.9324** | 0.1248 | 0.3532 |
| **Tuned XGBoost** | 0.8744 | 0.7929 | 0.6709 | 0.7268 | 0.9314 | 0.1256 | 0.3544 |
| **Weighted Soft Voting** | 0.8691 | 0.7705 | 0.6755 | 0.7199 | 0.9258 | 0.1309 | 0.3618 |
| **Tuned Random Forest** | 0.8363 | 0.6353 | **0.8041** | 0.7098 | 0.9199 | 0.1637 | 0.4046 |
| **Support Vector Machine (SVM)** | 0.8620 | 0.7847 | 0.6143 | 0.6891 | 0.9083 | 0.1380 | 0.3715 |
| **Random Forest (Baseline)** | 0.8527 | 0.7299 | 0.6482 | 0.6867 | 0.9082 | 0.1473 | 0.3838 |
| **Logistic Regression** | 0.8543 | 0.7486 | 0.6249 | 0.6812 | 0.9091 | 0.1457 | 0.3816 |
| **K-Nearest Neighbors (KNN)** | 0.8308 | 0.6674 | 0.6389 | 0.6528 | 0.8668 | 0.1692 | 0.4114 |

---

## 📁 Project Structure

```text
ML_project/
│
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive Project Documentation
├── ML_PROJECT_REPORT.md       # Full Academic / College-Level ML Report
│
├── Data/                      # Datasets and original artifacts
│   ├── adult.csv              # UCI Adult Census Income Dataset (N=32,561)
│   ├── adult_income_model_results.csv # Evaluation CSV for all 9 models
│   ├── best_adult_income_model.pkl    # Serialized Stacking Classifier
│   ├── best_model_info.txt    # Summary benchmark of best model
│   └── ml-mini-project.ipynb  # Original Kaggle/Jupyter experiment notebook
│
├── model/                     # Production model assets
│   └── best_adult_income_model.pkl
│
├── results/                   # Evaluation results and artifacts
│   ├── model_results.csv
│   └── test_evaluation_artifacts.json
│
└── src/                       # Modular Python packages
    ├── preprocessing.py       # Cleaning, schema definitions, dummy transformation
    ├── prediction.py          # Prediction service & explainability
    └── visualization.py       # Interactive Plotly chart generators
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.11 recommended)
- `uv` or standard `pip` / `venv`

### 2. Setup Virtual Environment & Install Dependencies

Using `uv` (Fastest):
```bash
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Or using standard `pip`:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

> **Important Version Note**: The model was serialized with `scikit-learn==1.6.1`. Ensure `scikit-learn==1.6.1` is installed to preserve unpickling compatibility with the trained `ColumnTransformer` and `StackingClassifier`.

### 3. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

The application will start locally at `http://localhost:8501`.

---

## 🖥️ Streamlit Application Features

1. **💼 Income Predictor**:
   - Single-person prediction form with human-readable categorical dropdowns and numerical controls.
   - 4 built-in preset demographic scenarios (Executive, Retail Worker, Professor, Craftsman) for immediate evaluation.
   - Color-coded decision card (`> $50K` vs. `<= $50K`) with confidence percentage and interactive probability gauge.
   - **Explainability**:
     - Base learner probability decomposition (Logistic Regression, Random Forest, XGBoost).
     - Meta-learner logistic weights table ($w_{\text{xgb}}=4.34$, $w_{\text{rf}}=2.01$, $w_{\text{lr}}=0.22$).
     - Constituent tree feature importances.
2. **📊 Batch Predictions**:
   - Upload any CSV file containing demographic rows to run high-throughput batch scoring.
   - Download sample 10-row CSV template or export scored outputs with prediction labels and confidence scores.
3. **🏆 Model Benchmarking**:
   - Sortable leaderboard across all 9 models with highlights for best performers.
   - Individual metric bar charts and side-by-side grouped comparisons.
   - Interactive radar chart comparing multi-metric trade-offs.
   - Test-set confusion matrix and ROC curve for the Stacking model.
4. **🔍 Dataset Explorer**:
   - Interactive data filtering by age, employment sector, and income class.
   - Target imbalance donut chart, age distribution histograms, education vs income proportion bars, and numerical correlation heatmap.
5. **📐 System Architecture**:
   - Visual Mermaid flowchart detailing data flow from raw census data to production classifier.
   - Summary of hyperparameter optimization choices and responsible AI / ethics disclaimers.

---

## ⚙️ Hyperparameter Optimization Summary

Hyperparameter tuning was conducted using `RandomizedSearchCV` with 30 iterations and **5-fold StratifiedKFold** optimizing for the minority-class **F1-score**:

### Tuned Random Forest:
- `n_estimators`: 413
- `max_depth`: 30
- `max_features`: 'sqrt'
- `min_samples_split`: 13
- `min_samples_leaf`: 1
- `class_weight`: 'balanced'
- *Best CV F1*: 0.7126 (Test Recall: 80.41%)

### Tuned XGBoost:
- `n_estimators`: 408
- `max_depth`: 5
- `learning_rate`: 0.0767
- `subsample`: 0.9880
- `colsample_bytree`: 0.6400
- `min_child_weight`: 6
- `gamma`: 0.2296

---

## ⚠️ Limitations & Ethical Considerations

1. **Historical Context**: The dataset was recorded in 1994. Accounting for inflation, \$50,000 in 1994 corresponds to approximately \$105,000 today.
2. **Class Imbalance**: Approximately 75% of individuals earn `<=50K`. Naive accuracy is misleading; F1 and ROC-AUC are the primary operational metrics.
3. **Missing Data**: Records with `?` (7.4%) were dropped. In real-world deployment, advanced imputation should be considered.
4. **Demographic Biases**: Sociodemographic attributes reflect historical wage inequalities and must not be used for discriminatory decision-making.

---

## 📄 License & Attribution

- Dataset: UCI Machine Learning Repository / Ronny Kohavi & Barry Becker (1996).
- Implementation: Pair-programmed with AI assistant Antigravity (Google DeepMind).
