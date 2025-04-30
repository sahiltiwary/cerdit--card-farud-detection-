import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset
df = pd.read_csv('credit_card_fraud_dataset.csv')
print(df.head())

# Feature Engineering: Create additional features
df['Transaction_Frequency'] = df.groupby('Transaction_ID')['Transaction_Amount'].transform('count')
df['Avg_Transaction_Amount'] = df.groupby('Transaction_ID')['Transaction_Amount'].transform('mean')
df['Transaction_Ratio'] = df['Transaction_Amount'] / df['Avg_Transaction_Amount']

# One-hot encoding for categorical features
df_encoded = pd.get_dummies(df, columns=["Transaction_Type", "Device_Type"])

# Define features and target
X = df_encoded.drop("Fraudulent_Transaction", axis=1)
y = df_encoded["Fraudulent_Transaction"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_test = scaler.transform(X_test)

# Define models and parameters
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
param_grid_dt = {'max_depth': [5, 10, 15, 20], 'criterion': ['gini', 'entropy']}
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'probability': [True]}

# Perform Grid Search
lr_search = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='f1')
dt_search = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, cv=5, scoring='f1')
rf_search = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='f1')
svm_search = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='f1')

# Train models
lr_search.fit(X_train_balanced, y_train_balanced)
dt_search.fit(X_train_balanced, y_train_balanced)
rf_search.fit(X_train_balanced, y_train_balanced)
svm_search.fit(X_train_balanced, y_train_balanced)

# Print best parameters
print("Best Logistic Regression:", lr_search.best_params_)
print("Best Decision Tree:", dt_search.best_params_)
print("Best Random Forest:", rf_search.best_params_)
print("Best SVM:", svm_search.best_params_)




# Define the voting classifier with weights
voting_clf = VotingClassifier(
    estimators=[
        ('lr', lr_search.best_estimator_),
        ('dt', dt_search.best_estimator_),
        ('rf', rf_search.best_estimator_),
        ('svm', svm_search.best_estimator_)
    ],
    voting='soft',
    weights=[2, 1, 3, 2]
)

# Train and evaluate voting classifier
voting_clf.fit(X_train_balanced, y_train_balanced)
y_pred_voting = voting_clf.predict(X_test)
print("Voting Classifier Results:")
print(classification_report(y_test, y_pred_voting))

# Confusion Matrix for voting classifier
cm = confusion_matrix(y_test, y_pred_voting)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Voting Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Define stacking classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('lr', lr_search.best_estimator_),
        ('dt', dt_search.best_estimator_),
        ('rf', rf_search.best_estimator_),
        ('svm', svm_search.best_estimator_)
    ],
    final_estimator=LogisticRegression()
)

# Train and evaluate stacking classifier
stacking_clf.fit(X_train_balanced, y_train_balanced)
y_pred_stacking = stacking_clf.predict(X_test)
print("Stacking Classifier Results:")
print(classification_report(y_test, y_pred_stacking))

# Cross-validation on voting classifier
cv_scores = cross_val_score(voting_clf, X_train_balanced, y_train_balanced, cv=10, scoring='f1')
print("Cross-Validation F1 Scores:", cv_scores)
print("Mean F1 Score:", cv_scores.mean())

# Feature selection using RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=10)
X_train_selected = rfe.fit_transform(X_train_balanced, y_train_balanced)
X_test_selected = rfe.transform(X_test)

# Evaluate feature-selected model
lr_model_selected = LogisticRegression().fit(X_train_selected, y_train_balanced)
y_pred_rfe = lr_model_selected.predict(X_test_selected)
print("RFE Logistic Regression Results:")
print(classification_report(y_test, y_pred_rfe))


# Save trained models
joblib.dump(voting_clf, "voting_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save training feature columns after encoding
training_columns = X.columns.tolist()
joblib.dump(training_columns, "training_columns.pkl")

# # Evaluate on unseen data
# unseen_df = pd.read_csv('unseen_credit_card_fraud_dataset.csv')

# # Apply feature engineering
# unseen_df['Transaction_Frequency'] = unseen_df.groupby('Transaction_ID')['Transaction_Amount'].transform('count')
# unseen_df['Avg_Transaction_Amount'] = unseen_df.groupby('Transaction_ID')['Transaction_Amount'].transform('mean')
# unseen_df['Transaction_Ratio'] = unseen_df['Transaction_Amount'] / unseen_df['Avg_Transaction_Amount']
# unseen_df_encoded = pd.get_dummies(unseen_df, columns=["Transaction_Type", "Device_Type"])

# # Match columns with training data
# missing_cols = set(X.columns) - set(unseen_df_encoded.columns)
# for col in missing_cols:
#     unseen_df_encoded[col] = 0
# unseen_df_encoded = unseen_df_encoded[X.columns]

# # Standardize unseen data
# unseen_X = scaler.transform(unseen_df_encoded)

# # Make predictions on unseen data
# unseen_predictions = voting_clf.predict(unseen_X)
# print("Predictions on Unseen Data:", unseen_predictions)

# # Evaluate if ground truth labels exist
# if 'Fraudulent_Transaction' in unseen_df.columns:
#     unseen_y = unseen_df['Fraudulent_Transaction']
#     print("Unseen Data Evaluation Results:")
#     print(classification_report(unseen_y, unseen_predictions))
#     cm_unseen = confusion_matrix(unseen_y, unseen_predictions)
#     sns.heatmap(cm_unseen, annot=True, fmt='d', cmap='Blues')
#     plt.title("Confusion Matrix - Unseen Data")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()
