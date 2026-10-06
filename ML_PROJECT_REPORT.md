# Adult Census Income Prediction Using Classical Machine Learning

**Comprehensive Technical Project Report**  
*Course / Track: Advanced Machine Learning & Predictive Analytics*  
*Date: October 2026*  

---

## Executive Summary / Abstract

Predicting household and individual income levels from socioeconomic, demographic, occupational, and financial indicators is a foundational problem in applied econometrics and machine learning. This study investigates the **UCI Adult Census Income dataset** (initially comprising 32,561 records) formulated as a binary classification task: predicting whether an individual's annual gross income exceeds \$50,000 (`>50K` vs. `<=50K`). 

A rigorous, end-to-end machine learning pipeline was constructed. Missing values designated by `?` were identified and removed, followed by deduplication, yielding a verified cleaned corpus of **30,139 records**. Categorical features were one-hot encoded, and numerical attributes were standardized using scikit-learn `ColumnTransformer` pipelines. The dataset was partitioned into an **80/20 stratified train/test split** (24,111 training instances; 6,028 test instances) to preserve the inherent class imbalance (approximately 75.1% `<=50K` vs. 24.9% `>50K`).

Nine model architectures were trained and systematically benchmarked on the identical test partition:
1. **Five Baseline Classifiers**: Logistic Regression, K-Nearest Neighbors ($k=5$), Random Forest ($N=200$), XGBoost ($N=200, d=6$), and Support Vector Machine (RBF kernel).
2. **Two Tuned Models**: Random Forest and XGBoost, optimized via 5-fold `StratifiedKFold` Randomized Search targeting the minority-class F1-score.
3. **Two Advanced Ensembles**: A Weighted Soft Voting Ensemble and a Heterogeneous Stacking Classifier with a 3-fold cross-validated Logistic Regression meta-learner.

The **Stacking Ensemble** emerged as the overall superior architecture, attaining **Accuracy: 87.59%**, **Precision: 78.72%**, **Recall: 68.75%**, **F1-Score: 0.7340**, and an area under the receiver operating characteristic curve (**ROC-AUC**) of **0.9302** on the unseen test split. Meanwhile, Tuned Random Forest achieved the highest recall (**80.41%**), demonstrating a quantifiable precision-recall trade-off under class imbalance. The final model was packaged alongside a production-grade Streamlit web application supporting interactive single-profile inference, explainability decomposition, batch scoring, and exploratory data analysis.

---

## 1. Introduction & Background

Income level classification holds considerable socioeconomic value for policy design, resource allocation, consumer creditworthiness evaluation, and tax compliance auditing. In 1996, Ronny Kohavi and Barry Becker extracted the Adult dataset from the 1994 United States Census Bureau database. Since then, it has served as a gold-standard benchmark in supervised learning literature.

Predicting income is inherently non-trivial due to multifaceted relationships:
- Non-linear interactions between age, educational attainment, hours worked, and accumulated capital gains/losses.
- Skewed continuous financial indicators (capital gains and losses exhibit massive point masses at zero).
- Substantial class imbalance (only ~25% of individuals surpass the \$50,000 threshold in the 1994 economic context).

Classical machine learning algorithms—ranging from regularized generalized linear models to ensemble tree methods—are exceptionally well-suited for tabular data of this nature, frequently outperforming deep neural networks while providing superior computational efficiency and diagnostic interpretability.

---

## 2. Problem Statement & Formal Formulation

The objective is to train a parameterized classification hypothesis $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ that maps a vector of individual socioeconomic features $\mathbf{x} \in \mathcal{X} \subset \mathbb{R}^d$ to a discrete label $y \in \{0, 1\}$:

$$
y = \begin{cases} 
1 & \text{if annual gross income} > \$50,000 \\
0 & \text{if annual gross income} \le \$50,000 
\end{cases}
$$

Because the target distribution is skewed ($\approx 3:1$ ratio), models that maximize naive 0-1 accuracy risk degenerating toward majority-class classification ($y=0$). Consequently, the objective requires optimizing both discrimination (ROC-AUC) and harmonic precision-recall balance (F1-score) on the minority positive class ($y=1$).

---

## 3. Project Objectives

1. **Exploratory Data Analysis (EDA)**: Characterize feature distributions, demographic skew, outlier boundaries, and bivariate correlations with income.
2. **Robust Preprocessing Pipeline**: Clean corrupted records, prevent test set contamination, encode categorical variables, and standardize numerical attributes.
3. **Baseline Model Exploration**: Benchmark five diverse classical model families (linear, distance, ensemble trees, gradient boosting, maximum margin).
4. **Targeted Hyperparameter Tuning**: Deploy `RandomizedSearchCV` guided by 5-fold stratified cross-validation optimizing specifically for minority-class F1-score.
5. **Ensemble Synthesis**: Investigate decision fusion through probability-weighted soft voting and multi-layer heterogeneous stacking.
6. **Empirical Benchmarking**: Compare all models on a fixed, held-out test partition using balanced evaluation metrics.
7. **Production Deployment**: Build an interactive, modular Streamlit web application providing real-time inference, model explanations, batch processing, and interactive visualizations.

---

## 4. Dataset Description & Schema

The dataset represents demographic survey samples from the 1994 U.S. Census Bureau. The raw tabular corpus contains **32,561 rows** and **15 columns** (14 explanatory variables and 1 target variable).

### 4.1 Feature Schema

| Feature Name | Storage Type | Variable Type | Domain / Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Integer | Continuous / Ratio | 17 – 90 years | Age of the individual |
| `workclass` | String / Categorical | Nominal (7 categories) | Private, Self-emp, Gov, etc. | Employment sector |
| `fnlwgt` | Integer | Continuous / Weight | 12,285 – 1,484,705 | Final sampling weight assigned by Census Bureau |
| `education` | String / Categorical | Ordinal (16 categories) | Preschool through Doctorate | Highest level of completed formal education |
| `education.num` | Integer | Ordinal / Discrete | 1 – 16 | Numerical encoding of educational level |
| `marital.status` | String / Categorical | Nominal (7 categories) | Married, Never-married, Divorced, etc. | Marital status |
| `occupation` | String / Categorical | Nominal (14 categories) | Exec-managerial, Prof-specialty, Craft, etc. | Broad occupational grouping |
| `relationship` | String / Categorical | Nominal (6 categories) | Husband, Wife, Own-child, etc. | Role within household |
| `race` | String / Categorical | Nominal (5 categories) | White, Black, Asian-Pac-Islander, etc. | Demographic racial group |
| `sex` | String / Categorical | Binary Nominal | Male, Female | Biological sex |
| `capital.gain` | Integer | Continuous / Ratio | \$0 – \$99,999 | Annual capital profits from investment sales |
| `capital.loss` | Integer | Continuous / Ratio | \$0 – \$4,356 | Annual capital losses from investment sales |
| `hours.per.week` | Integer | Continuous / Ratio | 1 – 99 hours | Regular hours worked per week |
| `native.country` | String / Categorical | Nominal (41 categories) | United-States, Mexico, etc. | Country of birth / origin |
| `income` **(Target)** | String / Categorical | Binary | `<=50K`, `>50K` | Annual gross income threshold indicator |

### 4.2 Target Class Distribution
- **Raw Corpus**: 32,561 records (24,720 `<=50K`, 7,841 `>50K`).
- **Cleaned Corpus (Post missing-value and duplicate removal)**: **30,139 records**
  - `<=50K` (Class 0): **22,633 instances (75.095%)**
  - `>50K` (Class 1): **7,506 instances (24.905%)**

---

## 5. Exploratory Data Analysis (EDA)

Key statistical insights extracted from the dataset:

1. **Age vs. Income**:
   - The median age for `<=50K` earners is approximately 34 years, whereas the median age for `>50K` earners is 44 years. Peak earning probability occurs between ages 38 and 55, reflecting accumulated human capital, seniority, and career progression.
2. **Education vs. Income**:
   - Educational attainment exhibits strong positive monotonicity with income. Individuals holding Doctorate degrees exceed \$50K at a rate of ~74%, Professional school at ~73%, and Masters degrees at ~56%. Conversely, individuals with High School diplomas or less exhibit >\$50K rates below 16%.
3. **Working Hours (`hours.per.week`)**:
   - While the modal work week is 40 hours for both classes, individuals working 45–60 hours per week are disproportionately represented in the `>50K` bracket (>42% high earners).
4. **Capital Gains and Capital Losses**:
   - Capital gains are heavily zero-inflated (>91% of records report \$0). However, when present, non-zero capital gains (particularly above \$5,000) serve as one of the single most decisive indicators of `>50K` income.
5. **Household Relationship & Marital Status**:
   - Marital status (`Married-civ-spouse`) is strongly correlated with higher household income, due to combined wealth effects and household lifecycle stages.

---

## 6. Data Preprocessing & Pipeline Engineering

To ensure empirical validity, preprocessing adhered strictly to leakage prevention principles.

### 6.1 Missing-Value Handling
In the raw UCI dataset, unknown entries are recorded as `'?'`. A missing-value audit revealed:
- `workclass`: 1,836 missing entries
- `occupation`: 1,843 missing entries
- `native.country`: 583 missing entries

Following the notebook protocol, records containing `'?'` were dropped (`dropna()`), reducing the row count from 32,561 to 30,162.

### 6.2 Deduplication
A duplicate row scan identified **23 identical census records**. Removing these duplicate records finalized the dataset at exactly **30,139 rows**.

### 6.3 Categorical String Trimming
Leading and trailing whitespace was stripped from all object columns to prevent artifactual splits (e.g., `' <=50K'` vs. `'<=50K'`).

### 6.4 One-Hot Dummy Encoding
Categorical predictors were one-hot encoded using `drop_first=True` to eliminate dummy variable collinearity in linear models:
- Workclass: 7 categories $\rightarrow$ 6 columns (reference: `Federal-gov`)
- Education: 16 categories $\rightarrow$ 15 columns (reference: `10th`)
- Marital Status: 7 categories $\rightarrow$ 6 columns (reference: `Divorced`)
- Occupation: 14 categories $\rightarrow$ 13 columns (reference: `Adm-clerical`)
- Relationship: 6 categories $\rightarrow$ 5 columns (reference: `Husband`)
- Race: 5 categories $\rightarrow$ 4 columns (reference: `Amer-Indian-Eskimo`)
- Sex: 2 categories $\rightarrow$ 1 column (reference: `Female`)
- Native Country: 41 categories $\rightarrow$ 40 columns (reference: `Cambodia`)

Total resulting feature dimensionality: **6 numerical features + 90 dummy indicators = 96 input features**.

### 6.5 Stratified Train/Test Split
The dataset was partitioned using `train_test_split(test_size=0.20, random_state=42, stratify=y)`:
- **Training Set**: 24,111 instances (18,106 class 0; 6,005 class 1)
- **Test Set**: 6,028 instances (4,527 class 0; 1,501 class 1)

Stratification ensured identical class ratios (75.09% : 24.91%) across training and test splits. The exact same test set was preserved across all 9 model evaluations.

### 6.6 Feature Scaling & Preprocessing Pipelines
Continuous numerical features (`age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`) exhibit drastically different variance scales (e.g., `fnlwgt` spans up to 1.48M, while `education.num` spans 1–16). 

For distance-based, linear, and margin-based classifiers (Logistic Regression, KNN, SVM), standardization via `StandardScaler` was packaged inside a scikit-learn `ColumnTransformer(remainder='passthrough')`:

$$
z = \frac{x - \mu}{\sigma}
$$

Decision tree ensembles (Random Forest, XGBoost) are invariant to monotonic transformations and do not require numerical scaling.

---

## 7. Model Methodology & Architecture

### 7.1 Baseline Algorithms

#### 1. Logistic Regression
A regularized generalized linear model optimizing cross-entropy loss with L2 penalty:
- Preprocessing: `ColumnTransformer` (StandardScaler on continuous features).
- Parameters: `max_iter=1000`, solver=`lbfgs`.

#### 2. K-Nearest Neighbors (KNN)
Non-parametric instance-based classifier assigning class membership via majority vote among Euclidean nearest neighbors:
- Parameters: `n_neighbors=5`, metric=`euclidean`.
- Preprocessing: Standardized continuous features inside Pipeline.

#### 3. Random Forest (RF Baseline)
Bagged ensemble constructing decorrelated decision trees using bootstrap aggregation and random feature sub-sampling:
- Parameters: `n_estimators=200`, `random_state=42`, `n_jobs=-1`.

#### 4. XGBoost (Baseline)
Scalable tree boosting system minimizing regularized objective via second-order Taylor expansion of the loss function:
- Parameters: `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `eval_metric='logloss'`, `random_state=42`.

#### 5. Support Vector Classifier (SVM)
Maximum-margin separator utilizing a Radial Basis Function (RBF) kernel with Platt probability scaling:
- Parameters: `kernel='rbf'`, `probability=True`, `random_state=42`.
- Preprocessing: Standardized continuous features inside Pipeline.

---

### 7.2 Hyperparameter Optimization (`RandomizedSearchCV`)

Hyperparameter tuning was conducted using `RandomizedSearchCV` with 30 iterations and **5-fold `StratifiedKFold`** cross-validation, optimizing explicitly for **F1-score**.

#### Tuned Random Forest
- Parameter distributions sampled: `n_estimators` (100–500), `max_depth` (None, 10, 20, 30, 40), `min_samples_split` (2–15), `min_samples_leaf` (1–8), `max_features` ('sqrt', 'log2', None), `class_weight` (None, 'balanced').
- **Best Discovered Parameters**:
  - `n_estimators`: **413**
  - `max_depth`: **30**
  - `max_features`: `'sqrt'`
  - `min_samples_split`: **13**
  - `min_samples_leaf`: **1**
  - `class_weight`: `'balanced'`
- **Best 5-Fold Cross-Validation F1**: **0.7126**

#### Tuned XGBoost
- Parameter distributions sampled: `n_estimators` (100–500), `max_depth` (3–10), `learning_rate` (0.01–0.21), `subsample` (0.6–1.0), `colsample_bytree` (0.6–1.0), `min_child_weight` (1–10), `gamma` (0.0–0.5).
- **Best Discovered Parameters**:
  - `n_estimators`: **408**
  - `max_depth`: **5**
  - `learning_rate`: **0.07674**
  - `subsample`: **0.9880**
  - `colsample_bytree`: **0.6400**
  - `min_child_weight`: **6**
  - `gamma`: **0.2296**
- **Objective**: `'binary:logistic'`, `eval_metric='logloss'`

---

### 7.3 Ensemble Learning Architectures

#### 1. Weighted Soft Voting Ensemble
A `VotingClassifier` combining predicted probabilities across 5 distinct model families:
- Estimators: `lr` (wt: 1), `knn` (wt: 1), `rf` (wt: 2), `xgb` (wt: 3), `svm` (wt: 2).
- Final class assignment:
  
$$
\hat{y} = \arg\max_{c \in \{0, 1\}} \sum_{m=1}^{5} w_m P_m(y=c \mid \mathbf{x})
$$

#### 2. Stacking Classifier (Meta-Learning)
Stacking trains a meta-model to optimally combine the out-of-fold probability predictions of diverse first-level models.
- **Level-0 Base Estimators**:
  1. `lr`: Standardized Logistic Regression Pipeline
  2. `rf`: Tuned Random Forest ($N=413$)
  3. `xgb`: Tuned XGBoost ($N=408$)
- **Level-1 Meta-Estimator**: `LogisticRegression(max_iter=1000)`
- **Configuration**: `cv=3` cross-validation scheme, `stack_method='predict_proba'`.

The fitted meta-learner parameters quantify the trust placed in each base learner:
- Meta-weight for `lr`: **+0.2236**
- Meta-weight for `rf`: **+2.0105**
- Meta-weight for `xgb`: **+4.3405**
- Meta log-odds intercept: **-3.5292**

---

## 8. Empirical Evaluation & Benchmark Results

All models were evaluated on the held-out test split of **6,028 samples** (4,527 negatives, 1,501 positives).

### 8.1 Comprehensive Model Comparison Table

| Model Architecture | MSE | RMSE | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.1457 | 0.3816 | 0.8543 | 0.7486 | 0.6249 | 0.6812 | 0.9091 |
| **K-Nearest Neighbors (KNN)** | 0.1692 | 0.4114 | 0.8308 | 0.6674 | 0.6389 | 0.6528 | 0.8668 |
| **Random Forest (Baseline)** | 0.1473 | 0.3838 | 0.8527 | 0.7299 | 0.6482 | 0.6867 | 0.9082 |
| **XGBoost (Baseline)** | 0.1248 | 0.3532 | 0.8752 | **0.7937** | 0.6742 | 0.7291 | **0.9324** |
| **Support Vector Machine (SVM)** | 0.1380 | 0.3715 | 0.8620 | 0.7847 | 0.6143 | 0.6891 | 0.9083 |
| **Tuned Random Forest** | 0.1637 | 0.4046 | 0.8363 | 0.6353 | **0.8041** | 0.7098 | 0.9199 |
| **Tuned XGBoost** | 0.1256 | 0.3544 | 0.8744 | 0.7929 | 0.6709 | 0.7268 | 0.9314 |
| **Weighted Soft Voting** | 0.1309 | 0.3618 | 0.8691 | 0.7705 | 0.6755 | 0.7199 | 0.9258 |
| **Stacking Classifier (Selected)** | **0.1241** | **0.3523** | **0.8759** | 0.7872 | 0.6875 | **0.7340** | 0.9302 |

*Note on MSE/RMSE*: In binary classification, Mean Squared Error corresponds to Brier Score on discrete predictions. It is reported here strictly as a supplementary metric to verify calibration loss reduction ($0.1241$ represents the lowest empirical prediction error).

---

### 8.2 Confusion Matrix & Detailed Report of Selected Stacking Model

The confusion matrix for the **Stacking Classifier** on the test set ($N = 6,028$):

```
                 Predicted <=50K       Predicted >50K       Total
True <=50K            4,248                 279             4,527 (Specificity: 93.84%)
True >50K               469               1,032             1,501 (Recall: 68.75%)
Total                 4,717               1,311             6,028
```

#### Classification Metrics Summary:
- **True Negatives (TN)**: 4,248
- **False Positives (FP)**: 279 (Type I Error rate = 6.16%)
- **False Negatives (FN)**: 469 (Type II Error rate = 31.25%)
- **True Positives (TP)**: 1,032
- **Precision**: $\frac{1032}{1032 + 279} = 78.72\%$
- **Recall (Sensitivity)**: $\frac{1032}{1032 + 469} = 68.75\%$
- **F1-Score**: $2 \times \frac{0.7872 \times 0.6875}{0.7872 + 0.6875} = 73.40\%$
- **Overall Accuracy**: $\frac{4248 + 1032}{6028} = 87.59\%$
- **ROC-AUC**: **0.9302**

---

## 9. In-Depth Results Analysis & Trade-off Discussion

1. **Why Stacking Outperformed Individual Classifiers**:
   Stacking attained the highest F1-score (0.7340) and accuracy (87.59%) by resolving residual disagreements among base models. Tuned XGBoost provides sharp, low-bias decision boundaries, while Random Forest smooths high-variance pockets, and Logistic Regression prevents overconfident probability spikes. The meta-learner assigned dominant weight to XGBoost ($w = 4.34$) while utilizing Random Forest ($w = 2.01$) to stabilize borderline candidates.
2. **The Precision vs. Recall Trade-Off**:
   - **Tuned Random Forest** optimized with `class_weight='balanced'` achieved the highest recall (**80.41%**), successfully capturing 1,207 out of 1,501 high earners. However, its precision dropped to **63.53%** (693 false positives), reducing overall F1 to 0.7098.
   - **Stacking Classifier** struck an optimal balance: it maintained high precision (**78.72%**) while boosting recall to **68.75%**, generating the highest overall F1-score.
3. **Tree Baseline vs. Tuned Models**:
   Baseline XGBoost already achieved formidable performance (F1: 0.7291, AUC: 0.9324). Tuning refined tree depth (restricted to 5) and column subsampling (0.64), curbing leaf overfitting and yielding a tuned model that served as an ideal building block in the final stack.

---

## 10. Practical Limitations & Ethical AI Considerations

1. **Historical Dataset Drift (1994 Census Data)**:
   The \$50,000 cutoff was established in 1994. Accounting for inflation, \$50,000 in 1994 corresponds to approximately \$105,000 in 2026 purchasing power. Thus, predictions describe historical structural patterns rather than modern earnings thresholds.
2. **Missing-Value Dropping**:
   Removing rows with `'?'` reduced the sample size by 7.4% (from 32,561 to 30,139). While sufficient data remained, missingness in survey data is rarely Missing Completely at Random (MCAR); individuals in informal or precarious employment sectors may systematically avoid reporting occupation.
3. **Demographic & Societal Biases**:
   Features such as `sex`, `race`, and marital status reflect historical wage disparities and labor force inequities of the early 1990s. Models trained on this corpus must never be applied to automated credit, employment screening, or housing access without comprehensive fairness mitigation.
4. **Ensemble Interpretability**:
   Multi-layer stacking ensembles trade direct parametric interpretability (e.g., odds ratios in logistic regression) for non-linear predictive capacity. Explainability must rely on post-hoc decomposition or surrogate tree attribution.

---

## 11. Future Research & Development

1. **Modern Imputation Techniques**:
   Evaluate MissForest or KNN-based iterative imputation rather than complete-case analysis to retain records with missing values.
2. **Threshold Tuning**:
   Optimize decision threshold $\tau \in [0, 1]$ directly using precision-recall curves or custom asymmetric cost matrices (e.g., when the cost of a false negative outweighs a false positive).
3. **Local Explainability (SHAP & LIME)**:
   Integrate KernelSHAP or TreeSHAP to provide local feature attribution explanations for each individual prediction.
4. **Fairness-Aware Machine Learning**:
   Apply fairness constraints (e.g., equalized odds, demographic parity regularization) to eliminate discriminatory disparities across demographic sub-groups.

---

## 12. Conclusion

This project successfully engineered, benchmarked, and deployed a classical machine learning pipeline for the UCI Adult Census Income prediction task. Across extensive empirical testing on an identical stratified test partition, the **Stacking Classifier Ensemble** achieved the highest overall performance, demonstrating **Accuracy of 87.59%**, **F1-Score of 0.7340**, and **ROC-AUC of 0.9302**. The accompanying Streamlit web application bridges theoretical evaluation and real-world utility by providing a dynamic interface for single-profile inference, explainability decomposition, and exploratory data analysis.
