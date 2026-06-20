# -*- coding: utf-8 -*-
"""Regression_01.ipynb

Original file is located at
    https://colab.research.google.com/drive/1QuENyvNBKgyIbYIkk6AJWxcoF6wuMQJT?usp=sharing

 Refer to 
    https://github.com/ShreeA123/Heart_Disease   
for more details on the project.
    

# **ML Project**

# **Load the Heart Disease Cleveland Dataset**
"""

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/heart-disease-cleveland.csv')
df

"""#**Data Prepration**

## **Data seperation as x and y**
"""

y = df[' diagnosis']
y

"""## **Extract Feature Variables**"""

x = df.drop(' diagnosis', axis=1)
x

"""# **Splitting the data set as training and testing with values as 75% and 25% respectively**"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100, stratify=y)

x_test

"""### **Check Data Types**"""

x_train

x_train.dtypes

"""### **Handle Missing Values**

### **Step 1: Remove Rows**
"""

x_train = x_train.dropna(subset=[" ca", " thal"])
y_train = y_train.loc[x_train.index]

"""### **Step 2: Verify Cleaning**"""

x_train.isna().sum()

"""### **Step 3: Convert to Numeric**"""

x_train[" ca"] = pd.to_numeric(x_train[" ca"], errors="coerce")
x_train[" thal"] = pd.to_numeric(x_train[" thal"], errors="coerce")

x_test[" ca"] = pd.to_numeric(x_test[" ca"], errors="coerce")
x_test[" thal"] = pd.to_numeric(x_test[" thal"], errors="coerce")

"""### **Check Data Types**"""

x_train.dtypes

x_test.dtypes

"""### **Feature Scaling - Standardization**"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)

x_test_sc = sc.transform(x_test)

"""# **Model Training**
## **Train Logistic Regression Classifier**
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

"""## **Generate Predictions**"""

y_pred = lr.predict(x_test_sc)

"""# **Data Analysis**"""

import seaborn as sns
import matplotlib.pyplot as plt

"""### **Visualization 1: Histogram indicating amount against Age distribution**"""

plt.figure(figsize=(10,6))
plt.hist(x_train['age'], bins=20, color='blue', edgecolor='black', alpha=0.6)
plt.xlabel('Age in years')
plt.ylabel('No of Patients')
plt.title('Distribution of patients',fontweight='bold')
plt.grid(True)
plt.show()

"""### **Visualization 2:Pie Chart indicating the split between genders**"""

gender_count = x_train[' sex'].value_counts()
plt.figure(figsize=(10,6))
plt.pie(gender_count, labels = ['Female', 'Male'], colors = ['yellowgreen', 'darkgrey'], autopct='%1.1f%%', startangle=270)
plt.title('Gender Distribution',fontweight='bold')
plt.show()

"""### **Visualization 3: Disease Severity Distribution Bar Chart**"""

heart_severity = y_train.value_counts().sort_index()
colors=['blue','yellow','pink','red','indigo']
plt.figure(figsize=(10,6))
plt.bar(heart_severity.index, heart_severity.values, color=colors)
plt.xticks([0,1,2,3,4])
plt.xlabel('Heart Severity')
plt.ylabel('No of patients')
plt.title('Heart disease severity distribution',fontweight='bold')
plt.grid(True)
plt.show()

"""### **Visualization 4: Disease Severity Distribution by Age Group**"""

age_groups = [
    (x_train['age'] < 40, 'Young (<40)'),
    ((x_train['age'] >= 40) & (x_train['age'] < 55), 'Middle-aged (40-55)'),
    ((x_train['age'] >= 55) & (x_train['age'] < 70), 'Senior (55-70)'),
    (x_train['age'] >= 70, 'Elderly (70+)')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

#pie chart for each age group
for idx, (condition, label) in enumerate(age_groups):
    disease_counts = y_train[condition].value_counts().sort_index()

    axes[idx].pie(disease_counts.values,
                  labels=[f'Level {i}' for i in disease_counts.index],
                  autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    axes[idx].set_title(f'{label}\n(n={condition.sum()} patients)',
                        fontsize=11, )

fig.suptitle('Disease Severity Distribution by Age Group',fontweight='bold')
plt.show()

