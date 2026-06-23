#**Heart Disease Severity Classification**
A machine learning project to predict heart disease severity levels using the Cleveland heart disease dataset. This project demonstrates multi-class classification using six different algorithms with comprehensive data exploration, preprocessing, and comparative model evaluation.

Dataset Overview
Dataset: Cleveland Heart Disease Dataset
Source: GitHub - dataprofessor
Total Samples: 303 patients (227 training, 76 testing)
Features: 13 clinical measurements
Target Variable: Disease severity levels (0-4)

Disease Severity Levels
Level	Classification
0	No heart disease
1	Mild disease
2	Moderate disease
3	Moderate-severe disease
4	Severe disease
Key Features
Feature	Description	Range/Values
age	Patient age in years	29-77 years
sex	Gender	0 = Female, 1 = Male
cp	Chest pain type	1-4 (typical angina to asymptomatic)
trestbps	Resting blood pressure	94-200 mm Hg
chol	Serum cholesterol	126-564 mg/dl
fbs	Fasting blood sugar > 120 mg/dl	0 = No, 1 = Yes
restecg	Resting electrocardiographic results	0-2
thalach	Maximum heart rate achieved	71-202 bpm
exang	Exercise-induced angina	0 = No, 1 = Yes
oldpeak	ST depression induced by exercise	0.0-6.2
slope	Slope of peak exercise ST segment	1-3
ca	Number of major vessels colored by fluoroscopy	0-3
thal	Thalium heart scan results	0-2
Note: Every column header in the dataset has one space before the name.

Project Workflow
1. Data Loading and Exploration
Load the dataset directly from the GitHub repository using pandas
Separate features (x) and target variable (y)
Display dataset structure with 13 clinical features and 303 patient records
2. Data Preprocessing
Train-Test Split

Training Set: 75% (227 samples)
Testing Set: 25% (76 samples)
Stratification: Applied to maintain class distribution across splits
Handling Missing Values

Remove rows with missing values in 'ca' and 'thal' columns (marked as '?')
Convert categorical string values to numeric format using pd.to_numeric()
Verify: All missing values successfully removed
Feature Scaling

Apply StandardScaler normalization to all features
Normalize data to mean=0 and standard deviation=1
Prevents features with larger ranges from dominating model training
3. Exploratory Data Analysis (EDA)
Four key visualizations reveal important patterns in the data:

Visualization 1: Age Distribution
A histogram showing the distribution of patient ages across the training dataset. The data reveals a concentration of patients in the 50-65 year age range, with the peak around 60 years, indicating this is the most represented demographic in the dataset.

Visualization 2: Gender Distribution
A pie chart displaying the gender composition of patients:

Female: 67.9% (approximately 154 patients)
Male: 32.1% (approximately 73 patients)
The dataset shows a notable skew toward female patients.

Visualization 3: Disease Severity Distribution
A bar chart showing the frequency of each disease severity level:

Level 0 (No disease): ~120 patients (dominant class)
Level 1 (Mild): ~40 patients
Level 2 (Moderate): ~25 patients
Level 3 (Moderate-severe): ~26 patients
Level 4 (Severe): ~10 patients
Class Imbalance Alert: The dataset has significant class imbalance, with Level 0 being heavily overrepresented.

Visualization 4: Disease Severity by Age Group
A 2×2 grid of pie charts breaking down disease severity across four age demographics:

Age Group	Sample Size	Key Finding
Young (<40)	11 patients	81.8% Level 0 (healthy)
Middle-aged (40-55)	94 patients	67.0% Level 0, increased severity in other levels
Senior (55-70)	109 patients	40.4% Level 0, more balanced distribution across levels
Elderly (70+)	7 patients	42.9% Level 0, higher proportion of severe cases
Key Insight: Disease severity increases with age; younger patients are predominantly healthy, while elderly patients show higher rates of disease.

Model Training and Evaluation
Models Tested
Six machine learning algorithms were trained and compared:

Model	Algorithm	Accuracy	Best For
Support Vector Machine (SVM)	Kernel-based classification	65.79%	Best Overall Performance
Logistic Regression	Linear multi-class classifier	60.53%	Interpretability & baseline
Neural Network (MLP)	Multi-layer perceptron	59.21%	Complex non-linear patterns
Random Forest	Ensemble tree-based	56.58%	Feature importance analysis
XGBoost	Gradient boosting	55.26%	Fast training & scalability
Gradient Boosting	Sequential boosting	53.95%	Robust to outliers
Model Performance Metrics
Model	Precision (Level 1)	Recall (Level 1)	F1-Score (Level 1)
Random Forest	0.33	0.20	0.25
Gradient Boosting	0.33	0.21	0.26
XGBoost	0.13	0.09	0.10
SVM	0.33	0.36	0.34
Neural Network	0.31	0.21	0.25
Key Findings and Insights
Model Performance
SVM emerged as the best-performing model with 65.79% accuracy, showing the strongest generalization to unseen data
Logistic Regression performed nearly as well (60.53%), providing a good balance between performance and interpretability
Ensemble methods (Random Forest, Gradient Boosting, XGBoost) underperformed, likely due to class imbalance
Data Insights
Gender Imbalance: The dataset is heavily skewed toward female patients (67.9%), which may affect model generalization
Class Imbalance: Level 0 (no disease) dominates the dataset, making it easier for models to predict this class but harder for minority classes
Age-Disease Correlation: Strong relationship between age and disease severity; younger patients are predominantly healthy
Limitations and Recommendations
Class imbalance should be addressed using techniques like SMOTE or class weighting
Hyperparameter tuning could improve model performance further
Consider ensemble methods combining SVM and Logistic Regression predictions
The moderate accuracy suggests additional feature engineering or domain-specific features may improve results
Technology Stack
Component	Tools
Language	Python 3
Data Processing	Pandas, NumPy
Machine Learning	Scikit-learn, XGBoost
Visualization	Matplotlib, Seaborn
Environment	Google Colab, Jupyter Notebook
Installation and Setup
Requirements
bash


pip install pandas numpy scikit-learn xgboost matplotlib seaborn
Running the Project
Clone the repository to your local machine
Open the Jupyter notebook or upload to Google Colab
Run all cells in sequence to execute the entire pipeline
Visualizations will display automatically after each analysis section
Model accuracy scores will print after training each classifier
Project Structure


Heart-Disease-Severity-Classification/
├── data/
│   └── heart-disease-cleveland.csv (loaded from GitHub)
├── notebooks/
│   ├── 01_data_loading_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_model_training_evaluation.ipynb
├── README.md
└── requirements.txt
