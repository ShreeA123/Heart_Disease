# Heart Disease Severity Classification

A machine learning project to predict heart disease severity levels using the Cleveland heart disease dataset. This project demonstrates multi-class classification using six different algorithms with comprehensive data exploration, preprocessing, and comparative model evaluation.

---

## Dataset Overview

**Dataset**: Cleveland Heart Disease Dataset

**Source**: [GitHub - dataprofessor](https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/heart-disease-cleveland.csv)

**Total Samples**: 303 patients (227 training, 76 testing)

**Features**: 13 clinical measurements

**Target Variable**: Disease severity levels (0-4)

### Disease Severity Classification

| Level | Classification |
|:-----:|-----------------|
| **0** | No heart disease |
| **1** | Mild disease |
| **2** | Moderate disease |
| **3** | Moderate-severe disease |
| **4** | Severe disease |

---

## Key Features

| Feature | Description | Range |
|---------|-------------|-------|
| **age** | Patient age in years | 29-77 years |
| **sex** | Gender | 0 = Female, 1 = Male |
| **cp** | Chest pain type | 1-4 |
| **trestbps** | Resting blood pressure | 94-200 mm Hg |
| **chol** | Serum cholesterol | 126-564 mg/dl |
| **fbs** | Fasting blood sugar > 120 mg/dl | 0 = No, 1 = Yes |
| **restecg** | Resting electrocardiographic results | 0-2 |
| **thalach** | Maximum heart rate achieved | 71-202 bpm |
| **exang** | Exercise-induced angina | 0 = No, 1 = Yes |
| **oldpeak** | ST depression induced by exercise | 0.0-6.2 |
| **slope** | Slope of peak exercise ST segment | 1-3 |
| **ca** | Number of major vessels colored by fluoroscopy | 0-3 |
| **thal** | Thalium heart scan results | 0-2 |

*Note: Every column header in the dataset has one space before the name.*

---

## Project Workflow

### 1. Data Loading and Exploration

**Objectives**:
- Load the dataset from GitHub repository
- Extract feature matrix (x) and target vector (y)
- Analyze dataset structure and data types

**Key Actions**:
- Import pandas and load CSV directly from URL
- Inspect 303 patient records with 13 features
- Display summary statistics for each feature

---

### 2. Data Preprocessing

#### Train-Test Split

| Metric | Value |
|--------|-------|
| **Training Set** | 75% (227 samples) |
| **Testing Set** | 25% (76 samples) |
| **Stratification** | Applied (maintains class distribution) |
| **Random State** | 100 (reproducibility) |

#### Handling Missing Values

**Step 1: Identify Missing Data**
- Columns **'ca'** and **'thal'** contain missing values marked as '?'
- These are categorical features requiring conversion

**Step 2: Remove Rows**
- Drop rows with missing values in 'ca' and 'thal' columns
- Verify missing data removal using `.isna().sum()`
- Result: **Zero missing values** across all features

**Step 3: Convert to Numeric**
- Transform categorical strings to numeric format
- Use `pd.to_numeric()` with `errors='coerce'` parameter
- Applied to both training and testing datasets

#### Feature Scaling

**Method**: StandardScaler (Standardization)

| Property | Value |
|----------|-------|
| **Mean** | 0 |
| **Standard Deviation** | 1 |
| **Fit On** | Training data only |
| **Transform** | Both training and testing data |
| **Purpose** | Normalize feature ranges for algorithm compatibility |

---

## Exploratory Data Analysis (EDA)

### Visualization 1: Age Distribution

**Type**: Histogram with 20 bins

**Key Findings**:
- **Peak Age Range**: 50-65 years
- **Maximum Frequency**: ~33 patients at age 60
- **Distribution**: Relatively normal with slight right skew
- **Age Span**: 29-77 years
- **Interpretation**: Middle-aged to senior patients dominate the dataset

---

### Visualization 2: Gender Distribution

**Type**: Pie chart with percentage labels

| Gender | Count | Percentage |
|--------|-------|-----------|
| **Female** | ~154 | 67.9% |
| **Male** | ~73 | 32.1% |

**Observation**: The dataset is heavily skewed toward female patients, which may affect model generalization to male populations.

---

### Visualization 3: Disease Severity Distribution

**Type**: Bar chart with 5 categories

| Severity Level | Patient Count | Percentage | Status |
|---|---|---|---|
| **Level 0** (None) | ~120 | 52.9% | Dominant class |
| **Level 1** (Mild) | ~40 | 17.6% | Minority class |
| **Level 2** (Moderate) | ~25 | 11.0% | Minority class |
| **Level 3** (Mod-Severe) | ~26 | 11.5% | Minority class |
| **Level 4** (Severe) | ~10 | 4.4% | Severely underrepresented |

**Critical Issue**: **Class Imbalance Detected**
- Level 0 represents over 50% of samples
- Minority classes (Levels 1-4) comprise less than 50%
- This imbalance may bias models toward predicting Level 0

---

### Visualization 4: Disease Severity by Age Group

**Type**: 2×2 Grid of Pie Charts

#### Young Patients (<40 years)

| Level | Percentage | Status |
|-------|-----------|--------|
| **Level 0** | 81.8% | Predominantly healthy |
| **Level 1** | 9.1% | Minimal disease |
| **Level 2** | 0.0% | None |
| **Level 3** | 9.1% | Rare |
| **Level 4** | 0.0% | None |

**Sample Size**: 11 patients

---

#### Middle-Aged Patients (40-55 years)

| Level | Percentage | Status |
|-------|-----------|--------|
| **Level 0** | 67.0% | Majority healthy |
| **Level 1** | 13.8% | Emerging disease |
| **Level 2** | 6.4% | Moderate cases present |
| **Level 3** | 11.7% | Noticeable increase |
| **Level 4** | 1.1% | Few severe cases |

**Sample Size**: 94 patients

---

#### Senior Patients (55-70 years)

| Level | Percentage | Status |
|-------|-----------|--------|
| **Level 0** | 40.4% | Less than half healthy |
| **Level 1** | 23.9% | Significant disease rate |
| **Level 2** | 18.3% | Notable moderate cases |
| **Level 3** | 11.9% | Moderate-severe present |
| **Level 4** | 5.5% | Increased severe cases |

**Sample Size**: 109 patients

---

#### Elderly Patients (70+ years)

| Level | Percentage | Status |
|-------|-----------|--------|
| **Level 0** | 42.9% | Still healthy but lower |
| **Level 1** | 14.3% | Stable rate |
| **Level 2** | 14.3% | Elevated moderate cases |
| **Level 3** | 14.3% | Higher severity |
| **Level 4** | 28.6% | **Highest severe disease rate** |

**Sample Size**: 7 patients

---

## Model Training and Evaluation

### Models Tested

| Rank | Model | Accuracy | Type |
|:----:|-------|----------|------|
| **1** | Support Vector Machine (SVM) | **65.79%** | Kernel-based |
| **2** | Logistic Regression | 60.53% | Linear classifier |
| **3** | Neural Network (MLP) | 59.21% | Deep learning |
| **4** | Random Forest | 56.58% | Ensemble trees |
| **5** | XGBoost | 55.26% | Gradient boosting |
| **6** | Gradient Boosting | 53.95% | Sequential boosting |

---

### Detailed Performance Metrics

#### Support Vector Machine (SVM) - Best Model

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 65.79% | ✓ Highest |
| **Precision (Class 1)** | 0.33 | Moderate |
| **Recall (Class 1)** | 0.36 | Moderate |
| **F1-Score (Class 1)** | 0.34 | Best balance |

**Strengths**:
- Highest overall accuracy
- Best generalization to test data
- Effective with multi-class problems

---

#### Logistic Regression

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 60.53% | Close second |
| **Precision (Class 1)** | — | — |
| **Recall (Class 1)** | — | — |
| **F1-Score (Class 1)** | — | — |

**Strengths**:
- Interpretable coefficients
- Fast training time
- Good baseline model

---

#### Neural Network (MLP)

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | 59.21% | Moderate |
| **Precision (Class 1)** | 0.31 | Fair |
| **Recall (Class 1)** | 0.21 | Lower |
| **F1-Score (Class 1)** | 0.25 | Lower |

**Note**: Convergence warning raised due to max iterations (1000) without full convergence.

---

#### Ensemble Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | 56.58% | 0.33 | 0.20 | 0.25 |
| **XGBoost** | 55.26% | 0.13 | 0.09 | 0.10 |
| **Gradient Boosting** | 53.95% | 0.33 | 0.21 | 0.26 |

**Observation**: Ensemble methods underperformed, likely due to **class imbalance** in the dataset.

---

## Key Findings and Insights

### Data Characteristics

**Age Distribution**:
- Concentration in 50-65 year range
- Mean age approximately 54 years
- Full range: 29-77 years

**Gender Composition**:
- Female-heavy dataset (67.9%)
- Potential bias toward female disease patterns
- Limited male patient representation

**Class Distribution**:
- Severe class imbalance present
- Level 0 (healthy): 52.9% of samples
- Minority classes: Levels 1-4 combined 47.1%

**Age-Disease Relationship**:
- **Strong positive correlation** between age and disease severity
- Young (<40): 81.8% healthy
- Elderly (70+): Only 42.9% healthy, 28.6% severe cases
- **Critical insight**: Age is a significant risk factor

---

### Model Performance Insights

**Best Performer**: SVM with 65.79% accuracy
- Kernel methods effective for this multi-class problem
- Superior to ensemble methods in this specific case

**Why Ensemble Methods Underperformed**:
- **Class imbalance** causes models to favor majority class (Level 0)
- Tree-based methods struggle with imbalanced data
- Insufficient minority class samples for effective learning

**Moderate Overall Accuracy**:
- 65.79% suggests room for improvement
- Multi-class classification inherently more challenging than binary
- Class imbalance limits minority class prediction accuracy

---

### Critical Limitations

| Limitation | Impact | Severity |
|-----------|--------|----------|
| **Class Imbalance** | Models biased toward Level 0; poor minority class prediction | High |
| **Gender Skew** | 67.9% female; limited male representation | High |
| **Small Minority Classes** | Level 4: only 10 samples; insufficient for robust learning | High |
| **Moderate Accuracy** | 65.79% suggests missing important features or patterns | Medium |
| **Limited Features** | Only 13 features; additional clinical data could help | Medium |

---

## Recommendations for Improvement

### 1. Address Class Imbalance

**Techniques to implement**:
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **Class weighting** in model training
- **Stratified k-fold cross-validation**
- **Balanced random sampling** for ensemble methods

### 2. Hyperparameter Optimization

**Methods**:
- GridSearchCV for systematic parameter search
- RandomizedSearchCV for broader exploration
- Focus on SVM kernel parameters and neural network architecture

### 3. Feature Engineering

**Enhancements**:
- Create polynomial interaction features
- Domain-specific derived features (e.g., age-based risk scores)
- Correlation analysis to identify redundant features

### 4. Ensemble Strategies

**Approaches**:
- Voting classifier combining SVM + Logistic Regression
- Stacking with SVM as meta-learner
- Custom weighted ensembles prioritizing minority classes

### 5. Data Collection

**Expansion needs**:
- Increase minority class samples (especially Level 3-4)
- Improve gender balance in dataset
- Include additional clinical measurements

---

## Technology Stack

| Component | Tools |
|-----------|-------|
| **Programming Language** | Python 3 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Development Environment** | Google Colab, Jupyter Notebook |

---

## Installation and Setup

### System Requirements

```
Python 3.7+
pip or conda package manager
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Running the Project

**Step 1**: Clone or download the repository

**Step 2**: Open Jupyter Notebook or Google Colab

**Step 3**: Run cells sequentially:
- Data loading and preprocessing
- Exploratory data analysis
- Model training
- Performance evaluation

**Step 4**: View generated visualizations automatically

**Step 5**: Review printed accuracy scores for each model

---

## Project Results Summary

### Overall Performance

**Best Model**: **Support Vector Machine (SVM)**
- **Accuracy**: 65.79%
- **Type**: Kernel-based classifier
- **Execution Time**: Fast
- **Interpretability**: Low

**Runner-up**: **Logistic Regression**
- **Accuracy**: 60.53%
- **Advantage**: Highly interpretable
- **Advantage**: Provides probability estimates

### Key Takeaways

✓ Successfully implemented 6 different machine learning algorithms

✓ Identified SVM as optimal for this dataset

✓ Revealed strong age-disease severity correlation

✓ Highlighted class imbalance as primary limitation

#Link to the collab file:
https://colab.research.google.com/drive/1QuENyvNBKgyIbYIkk6AJWxcoF6wuMQJT?usp=sharing

✓ Provided actionable recommendations for improvement

---
