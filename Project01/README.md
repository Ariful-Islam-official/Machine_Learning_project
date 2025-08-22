# 🩺 Project 01 - Diabetes Prediction (Using Support Vector Machine)

This project predicts whether a patient is **diabetic or not** using **Machine Learning**.  
We use the **PIMA Indians Diabetes Dataset** and train a **Support Vector Machine (SVM)** classifier to make predictions.  

---

## 📖 Dataset Information
The dataset is originally from the **National Institute of Diabetes and Digestive and Kidney Diseases**.  
It contains several medical measurements of patients and a label indicating if they have diabetes.  

**Columns:**
- **Pregnancies** → Number of times the patient was pregnant  
- **Glucose** → Plasma glucose concentration after 2 hours  
- **BloodPressure** → Diastolic blood pressure (mm Hg)  
- **SkinThickness** → Triceps skinfold thickness (mm)  
- **Insulin** → 2-hour serum insulin (mu U/ml)  
- **BMI** → Body Mass Index (weight/height²)  
- **DiabetesPedigreeFunction** → Score of diabetes likelihood based on family history  
- **Age** → Age of the patient (years)  
- **Outcome** → Target (0 = Non-diabetic, 1 = Diabetic)  

---

## 🛠️ Code Walkthrough

### 🔹 1. Importing Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
````

---

### 🔹 2. Loading & Exploring the Dataset

```python
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.shape
diabetes_dataset.describe()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
```

* **head()** → Shows first few rows
* **shape** → Dataset size
* **describe()** → Stats like mean, std, min, max
* **value\_counts()** → Count of diabetic vs non-diabetic patients
* **groupby('Outcome')** → Compare average features for both classes

---

### 🔹 3. Visualizations 📊

#### (a) Distribution of Outcome Classes

```python
sns.countplot(x='Outcome', data=diabetes_dataset, palette='Set2')
plt.title("Diabetic vs Non-Diabetic Patients")
plt.show()
```

* Shows class balance: how many are diabetic (1) vs non-diabetic (0).

#### (b) Correlation Heatmap

```python
plt.figure(figsize=(10,8))
sns.heatmap(diabetes_dataset.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Features")
plt.show()
```

* Helps identify which features are strongly related to diabetes.
* Example: Glucose has strong positive correlation with Outcome.

#### (c) Feature Distributions

```python
diabetes_dataset.hist(bins=20, figsize=(12,8), color='teal')
plt.suptitle("Feature Distributions")
plt.show()
```

* Shows distribution of values for glucose, BMI, age, etc.

---

### 🔹 4. Preparing Data

```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)
```

---

### 🔹 5. Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
```

---

### 🔹 6. Model Training (SVM)

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

---

### 🔹 7. Model Evaluation

```python
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

print("Training Accuracy:", accuracy_score(Y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(Y_test, y_test_pred))
```

#### (a) Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Data)")
plt.show()
```

* Helps visualize correct vs incorrect predictions.

---

### 🔹 8. Making Predictions on New Data

```python
input_data = (5,166,72,19,175,25.8,0.587,51)
input_data_as_numpy_array = np.asarray(input_data).reshape(1,-1)
std_data = scaler.transform(input_data_as_numpy_array)
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")
```

---

## 📈 Results

* **Training Accuracy**: \~78%
* **Testing Accuracy**: \~77%

📊 **Visual Insights**:

* Glucose level & BMI are strong predictors.
* Age also influences risk.
* Confusion matrix shows balanced performance.

---

## 🔮 Future Enhancements

* Try **Logistic Regression, Random Forest, Neural Networks**
* Perform **hyperparameter tuning** (GridSearchCV)
* Build a **web app** using Streamlit or Flask

---
