# Heart_Disease
Heart Disease Severity Classification  A machine learning project to predict heart disease severity levels using the Cleveland heart disease dataset. This project demonstrates multi-class classification using Logistic Regression with comprehensive data exploration and visualization.

## Dataset Overview

**Dataset**: Cleveland Heart Disease Dataset  
**Source**: [GitHub - dataprofessor](https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/heart-disease-cleveland.csv)  
**Total Samples**: 303 patients  
**Features**: 13 clinical measurements  
**Target Variable**: Disease severity levels (0-4)
- Level 0: No heart disease
- Level 1: Mild disease
- Level 2: Moderate disease
- Level 3: Moderate-severe disease
- Level 4: Severe disease

## Key Features

| Feature | Description |
|---------|-------------|
| ** age** | Patient age in years |
| ** sex** | Gender (0=Female, 1=Male) |
| ** cp** | Chest pain type (0-3) |
| ** trestbps** | Resting blood pressure (mm Hg) |
| ** chol** | Serum cholesterol (mg/dl) |
| ** fbs** | Fasting blood sugar > 120 mg/dl (0=No, 1=Yes) |
| ** restecg** | Resting electrocardiographic results (0-2) |
| ** thalach** | Maximum heart rate achieved (bpm) |
| ** exang** | Exercise-induced angina (0=No, 1=Yes) |
| ** oldpeak** | ST depression induced by exercise |
| ** slope** | Slope of peak exercise ST segment (0-2) |
| ** ca** | Number of major vessels colored by fluoroscopy (0-3) |
| ** thal** | Thalium heart scan results (0-2) |

-Note: Every coloumn header in the dataset has one space before the name as indicated above.

## Project Workflow

### 1. Data Loading and Exploration
- Load the dataset from the GitHub repository
- Extract features (x) and target variable (y)
- Display dataset structure and basic information

### 2. Data Preprocessing
- **Train-Test Split**: 75% training, 25% testing (stratified by target variable)
- **Handling Missing Values**: 
  - Remove rows with missing values in 'ca' and 'thal' columns
  - Convert string values to numeric format
- **Feature Scaling**: Apply StandardScaler to normalize features (mean=0, std=1)

### 3. Model Training
- Train Multinomial Logistic Regression classifier
- Generate predictions on test set
- (Future: Evaluate model performance with metrics)

### 4. Exploratory Data Analysis (EDA)

Four key visualizations reveal the data story:

#### Visualization 1: Age Distribution
A histogram showing the distribution of patient ages in the training dataset, revealing age demographics and concentration patterns.

#### Visualization 2: Gender Distribution
A pie chart displaying the proportion of male and female patients, helping identify gender representation in the dataset.

#### Visualization 3: Disease Severity Distribution
A bar chart showing the count of patients in each disease severity level (0-4), indicating class balance and distribution patterns.

#### Visualization 4: Disease Severity by Age Group
A 2×2 grid of pie charts showing how disease severity distribution varies across four age groups:
- Young (<40 years)
- Middle-aged (40-55 years)
- Senior (55-70 years)
- Elderly (70+ years)

This visualization reveals the relationship between age and disease severity.

## Data Insights

- **Gender Distribution**: Shows male/female ratio in the dataset
- **Age Demographics**: Reveals the age range and concentration of patients
- **Class Balance**: Indicates if disease severity levels are evenly distributed
- **Age-Disease Relationship**: Demonstrates how disease severity changes across age groups

## Technology Stack

- **Language**: Python 3
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab
  
## Installation and Setup

### Requirements
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


## Installation
pip install pandas numpy scikit-learn matplotlib seaborn  

Running the Project

- Clone this repository
- Open the Jupyter notebook or Google Colab
- Run all cells in sequence
- Visualizations will display automatically



