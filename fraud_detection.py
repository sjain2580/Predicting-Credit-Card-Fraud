# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Step 2: Load the Dataset
# IMPORTANT: You must have 'creditcard.csv' in the same directory as this script.
try:
    df = pd.read_csv('creditcard.csv')
    print("Step 2: Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the same directory.")
    exit()

# Step 3: Data Preprocessing and Splitting
# The dataset is already preprocessed, so we just need to split it.
X = df.drop('Class', axis=1)
y = df['Class']

# To speed up execution, we will use a smaller, representative subset of the data.
# This allows for faster testing of the pipeline and hyperparameter tuning.
# We'll use a sample of 20,000 transactions, stratified to maintain the class ratio.
# To run on the full dataset, simply comment out the next two lines.
sample_size = 20000
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset)

print("Step 3: Data split into training and testing sets.")
print(f"Using a subset of {sample_size} samples for faster execution.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Step 4: Building the Pipeline with SMOTE and Gradient Boosting
# The Pipeline ensures that SMOTE is only applied to the training data within each fold of cross-validation.
# This prevents data leakage and ensures a robust evaluation.
pipeline = Pipeline(steps=[('smote', SMOTE(random_state=42)),
                           ('classifier', GradientBoostingClassifier(random_state=42))])

# Step 5: Hyperparameter Tuning with GridSearchCV
# Note: For faster execution, the parameter grid is kept small.
# You can expand it later to find a better model.
print("\nStep 5: Performing GridSearchCV for hyperparameter tuning. This may take a few moments...")
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__learning_rate': [0.1],
    'classifier__max_depth': [3, 5]
}

# Use StratifiedKFold to handle the imbalanced data during cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Hyperparameter tuning complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation F1-Score: {grid_search.best_score_:.4f}")

# Step 6: Final Model Evaluation
print("\nStep 6: Evaluating the final model on the test set.")
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score: {f1:.4f}")

# Step 7: Visualization
print("\nStep 7: Generating visualizations.")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
# The `name` parameter replaces the deprecated `estimator_name`
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, name='Gradient Boosting')
roc_display.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.legend()
plt.savefig('ROC.png')
plt.show()

# Precision-Recall Curve
prec, recall, _ = precision_recall_curve(y_test, y_proba)
# The `name` parameter replaces the deprecated `estimator_name`
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, estimator_name='Gradient Boosting')
pr_display.plot()
plt.title('Precision-Recall Curve')
plt.savefig('Precision-recall curve.png')
plt.show()

print("\nProject complete. You can now analyze the output and visualizations.")
