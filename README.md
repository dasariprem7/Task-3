# Task-3
1. Download and Load the Dataset
First, download the dataset from the UCI Machine Learning Repository and load it into a DataFrame. For this example, I’ll assume you've downloaded the dataset as bank.csv.
code:
import pandas as pd

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
data = pd.read_csv(url, compression='zip', sep=';', encoding='utf-8')

# Display the first few rows of the dataset
print(data.head())

2)Data Preprocessing
Check for Missing Values and Basic Info
code:
# Check for missing values and basic info
print(data.info())
print(data.isnull().sum())

*Data Cleaning and Preparation
Convert categorical variables: Convert categorical variables to numerical using one-hot encoding or label encoding.
Feature and target variable: Define the feature set and the target variable.

# Convert categorical variables to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Define feature set and target variable
X = data.drop('y_yes', axis=1)  # Features
y = data['y_yes']  # Target variable

3.)Split the Data
Split the dataset into training and testing sets.
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

4)4. Build and Train the Decision Tree Classifier
Use the DecisionTreeClassifier from scikit-learn to build and train the model.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

5. Visualize the Decision Tree
Visualize the decision tree to understand the decisions being made by the model
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

*Summary
Data Preprocessing: We converted categorical variables to numeric using one-hot encoding and split the dataset into training and testing sets.
Model Training: We built a decision tree classifier using the training set and evaluated it on the testing set.
Visualization: We visualized the decision tree to understand how it makes decisions based on the features.




