# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
feature_selector = SelectKBest(score_func=f_classif, k=15)  # Select top 15 features
X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = feature_selector.transform(X_test_scaled)

# Initialize classifiers
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)
svm_classifier = SVC(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'learning_rate': [0.1, 0.01]
}

grid_search = GridSearchCV(estimator=gb_classifier, param_grid=param_grid, cv=3)
grid_search.fit(X_train_selected, y_train)
best_gb_classifier = grid_search.best_estimator_

# Train classifiers
rf_classifier.fit(X_train_selected, y_train)
best_gb_classifier.fit(X_train_selected, y_train)
svm_classifier.fit(X_train_selected, y_train)

# Make predictions
rf_pred = rf_classifier.predict(X_test_selected)
gb_pred = best_gb_classifier.predict(X_test_selected)
svm_pred = svm_classifier.predict(X_test_selected)

# Evaluate classifiers
rf_report = classification_report(y_test, rf_pred, target_names=data.target_names)
gb_report = classification_report(y_test, gb_pred, target_names=data.target_names)
svm_report = classification_report(y_test, svm_pred, target_names=data.target_names)

# Print results
print("Random Forest Classification Report:\n", rf_report)
print("Gradient Boosting Classification Report:\n", gb_report)
print("Support Vector Machine Classification Report:\n", svm_report)
