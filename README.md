# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
#Ex 07 - Implementation of Logistic Regression Using SGD Classifier
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classificatio
# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
 'Hours_Studied': [2, 3, 4, 5, 6, 7, 8, 9],
 'Previous_Score': [40, 50, 55, 60, 65, 70, 75, 80],
 'Internship': [0, 0, 1, 0, 1, 1, 1, 1], # 0 = No, 1 = Yes
 'Placement': [0, 0, 0, 1, 1, 1, 1, 1] # Target: 0 = Not Placed, 1 = P
}
df = pd.DataFrame(data)
# ------------------------------
# Step 2: Split into features and target
# ------------------------------
X = df[['Hours_Studied', 'Previous_Score', 'Internship']]
y = df['Placement']
# ------------------------------
# Step 3: Train-test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, r
# ------------------------------
# Step 4: Feature scaling
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ------------------------------
# Step 5: Create and train SGDClassifier for Logistic Regression
# ------------------------------
sgd_model = SGDClassifier(loss='log_loss', # 'log' loss → logistic reg
 max_iter=1000,
 learning_rate='optimal',
 random_state=42)
sgd_model.fit(X_train, y_train)
# ------------------------------
# Step 6: Make predictions
# ------------------------------
y_pred = sgd_model.predict(X_test)
y_prob = sgd_model.predict_proba(X_test) # Probability of placement
# ------------------------------
# Step 7: Evaluate the model
# ------------------------------
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
In [ ]:
Confusion Matrix:
[[1 0]
[0 1]]
Accuracy Score: 1.0
Classification Report:
 precision recall f1-score support
 0 1.00 1.00 1.00 1
 1 1.00 1.00 1.00 1
 accuracy 1.00 2
 macro avg 1.00 1.00 1.00 2
weighted avg 1.00 1.00 1.00 2
Predicted Placement Status: Placed
Probability of Placement: 1.00
C:\ProgramData\anaconda3\lib\site-packages\sklearn\base.py:420: UserWarnin
g: X does not have valid feature names, but StandardScaler was fitted with
feature names
warnings.warn(
# ------------------------------
# Step 8: Predict placement for a new student
# ------------------------------
new_student = np.array([[6, 68, 1]]) # Example: 6 hours, 68 prev score, In
new_student_scaled = scaler.transform(new_student)
placement_pred = sgd_model.predict(new_student_scaled)
placement_prob = sgd_model.predict_proba(new_student_scaled)
print(f"\nPredicted Placement Status: {'Placed' if placement_pred[0]==1 els
print(f"Probability of Placement: {placement_prob[0][1]:.2f}")
```

Program to implement the prediction of iris species using SGD Classifier.
Developed by: BRINDHA A R

RegisterNumber: 212225040050
  


## Output:
![prediction of iris species using SGD Classifier](sam.png)

<img width="665" height="372" alt="Screenshot 2026-05-08 143846" src="https://github.com/user-attachments/assets/c3d4ec42-f0e0-498e-995a-dc79326d3f43" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
