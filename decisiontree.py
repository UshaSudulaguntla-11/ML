# -*- coding: utf-8 -*-
"""DecisionTree - Diabetes Prediction"""

# Import pandas for data handling
import pandas as pd

# Read the dataset from CSV file
file = pd.read_csv("/content/diabetes.csv")

# Display first few rows of dataset
file


# -----------------------------
# Separate Input and Output
# -----------------------------

# Selecting feature columns (Independent variables)
input = file[[
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]]

# Selecting target column (Dependent variable)
output = file["Outcome"]


# -----------------------------
# Import and Train Model
# -----------------------------

# Import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Create model object
model = DecisionTreeClassifier()

# Train (fit) the model using input and output
model.fit(input, output)


# -----------------------------
# Make Predictions
# -----------------------------

# Predict for new patient data (correct numeric format)
model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Correct version (convert to numbers)
model.predict([[4, 110, 59, 35, 0, 32.6, 0.1, 50]])


# -----------------------------
# Pairplot Visualization
# -----------------------------

# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot shows relationship between all features
# hue="Outcome" colors based on diabetic or not
sns.pairplot(file, hue="Outcome")
plt.show()


# -----------------------------
# Scatter Plot for Each Feature
# -----------------------------

# List of feature names
features = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

# Loop through each feature and plot vs Outcome
for feature in features:
    plt.figure()
    plt.scatter(file[feature], file["Outcome"], c=file["Outcome"])
    plt.xlabel(feature)
    plt.ylabel("Outcome")
    plt.title(feature + " vs Outcome")
    plt.show()


# -----------------------------
# Single Scatter Plot Example
# -----------------------------

plt.scatter(file["Glucose"], file["Outcome"])
plt.xlabel("Glucose")
plt.ylabel("Outcome")
plt.show()


# -----------------------------
# 3D Scatter Plot
# -----------------------------

# Import 3D plotting tool
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure()

# Add 3D subplot
ax = fig.add_subplot(111, projection='3d')

# 3D scatter plot (Glucose, BMI, Age)
ax.scatter(
    file["Glucose"],
    file["BMI"],
    file["Age"],
    c=file["Outcome"]
)

# Set axis labels
ax.set_xlabel("Glucose")
ax.set_ylabel("BMI")
ax.set_zlabel("Age")

plt.show()


# -----------------------------
# Multiple Subplots in One Figure
# -----------------------------

plt.figure(figsize=(15, 10))

# Plot all features in grid format
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1)
    plt.scatter(file[feature], file["Outcome"])
    plt.title(feature)

# Adjust layout
plt.tight_layout()
plt.show()
