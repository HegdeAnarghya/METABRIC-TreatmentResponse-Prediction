# Loading required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

# loading and exploring the data
# df = pd.read_csv("METABRIC_RNA_Mutation.csv") 
# Throws error since Columns (678,688,690,692) contains mixed data types (texts and numbers).
# This confuses pandas when trying to automatically determine the data type.

# To let the pandas read the file in chunks and determine the correct type
df = pd.read_csv("METABRIC_RNA_Mutation.csv", low_memory=False)

# Display the first few rows
print(df.head())

# check column names
print(df.columns)

# check for missing values
print(df.isnull().sum())

# Summary of the dataset
print(df.info())

# Finding the target variable

# Checking for unique values in the "chemotherapy" column
print(df["chemotherapy"].value_counts())  

# This column only shows who received chemotherapy—it does NOT tell us how they responded.
# Not useful as a target variable

# The "pam50_+_claudin-low_subtype" column may indicate how a patient responds to treatment based on molecular subtypes.
print(df["pam50_+_claudin-low_subtype"].value_counts())

# The output shows categorized patient groups, which corelates with treatment outcomes
# The different subtypes (LumA, LumB, Her2, etc.) are not direct treatment responses, but they do influence how patients respond to treatment.

# Checking for columns that may indicate survival status

for col in df.columns:
    if "survival" in col.lower() or "recurrence" in col.lower():
        print(col)

# "overall_survival_months" = How long the patient survived after treatment.
# "overall_survival" (likely 0 = deceased, 1 = alive) = If the patient is alive or not.

# If a patient survived longer, the treatment likely worked well.
# If a patient had poor survival, they likely did not respond well to treatment.

# Best target variable: "overall_survival"
# This gives a clear indicator of whether the treatment was effective.
# You can predict if a patient is likely to survive based on gene expression & treatment.

print(df["overall_survival"].value_counts())

# converting into treatment response labels
df["treatment_response"] = df["overall_survival"].map({1: "Responder", 0: "Non-Responder"})
print(df["treatment_response"].value_counts())

# Defining Features (X) and Target (y) variables

# Features for prediction: (Clinical attributes):
# "age_at_diagnosis", "type_of_breast_surgery", "pam50_+_claudin-low_subtype", "chemotherapy"

# Gene expression data (top genes are selected later)

# Define target variable (treatment response)
df["treatment_response"] = df["overall_survival"]

# Select features
features = ["age_at_diagnosis", "chemotherapy", "pam50_+_claudin-low_subtype"]

# Define X (features) and y (target)
X = df[features]
y = df["treatment_response"]

# check for missing values
print(X.isnull().sum())

# Data Preprocessing

# Drop patient IDs and any irrelevant metadata
df = df.drop(columns=["patient_id"])

# Handle missing values
# Count missing values per column
print(df.isnull().sum())

# Drop rows with missing values OR fill them
df = df.dropna()  # OR use df.fillna(df.mean()) for numeric columns

# Check again
print(df.isnull().sum())

# To identify the gene columns in the DataFrame (which appear to be columns ending in "_mut" such as 'mtap_mut', 'ppp2cb_mut', 'smarcd1_mut', etc.), we can filter out the columns that have the suffix _mut or any other pattern that represents gene expression data.

# Filter gene columns by suffix
gene_columns = [col for col in df.columns if col.endswith('_mut')]
print(gene_columns)

# Normalize Gene Expression Data

# Check the data types of the gene columns
print(df[gene_columns].dtypes)

# Handle non-numeric values in gene columns

# Apply Label Encoding to gene columns that contain strings (mutation types)
le = LabelEncoder()

for col in gene_columns:
    # If the column has non-numeric data, apply Label Encoding
    if df[col].dtype == 'object':  # Check if the column has non-numeric data
        df[col] = le.fit_transform(df[col])

# Verify the changes
print(df[gene_columns].dtypes)


# Initialize the scaler
scaler = StandardScaler()

# Apply normalization (standardization) to the gene columns
df[gene_columns] = scaler.fit_transform(df[gene_columns])

# Check the first few rows to confirm normalization
print(df[gene_columns].head())

# scaler.fit_transform() computes the mean and standard deviation of the gene columns and then scales each value by subtracting the mean and dividing by the standard deviation.

# Train a machine learning model

# Define features (X) and target variable (y)
X = df[gene_columns]  # Gene expression data
y = df["treatment_response"] 

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Tain a Random Forest Model

# Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed evaluation
print(classification_report(y_test, y_pred))

# Confusion matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance (Top Genes Contributing to Prediction)

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({'Gene': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(10)

# Plot top 10 important genes
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance['Importance'], y=feature_importance['Gene'])
plt.xlabel("Importance Score")
plt.ylabel("Top Genes")
plt.title("Top 10 Most Important Genes in Prediction")
plt.show()

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(10)

# Plot top 10 important features
plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'])
plt.xlabel("Importance Score")
plt.ylabel("Top Features")
plt.title("Top 10 Most Important Features in Predicting Treatment Response")
plt.show()

# List clinical features to include
clinical_features = ['age_at_diagnosis', 'type_of_breast_surgery', 'chemotherapy', 'pam50_+_claudin-low_subtype']

# Select both gene expression and clinical features
X = df[gene_columns + clinical_features]

# Check if there are any missing values in clinical data
print(df[clinical_features].isnull().sum())

# Encode categorical variables

# For columns like chemotherapy or pam50_+_claudin-low_subtype, you can apply one-hot encoding

# One-Hot Encoding for categorical features (for columns like 'type_of_breast_surgery')
df_encoded = pd.get_dummies(df[clinical_features], drop_first=True)

# Combine the original dataset with the encoded categorical variables
X = pd.concat([df[gene_columns], df_encoded], axis=1)

# Check the new dataset structure
print(X.head())

# Check data types of all columns to ensure they're numeric
print(X.dtypes)

# To fix the non-numeric columns

# Find columns that are not numeric
non_numeric_cols = X.select_dtypes(exclude=['number']).columns
print("Non-Numeric Columns:", non_numeric_cols)

# Convert boolean columns to integers (0/1)
X[non_numeric_cols] = X[non_numeric_cols].astype(int)

# Apply label encoding to object-type columns
label_encoders = {}
for col in non_numeric_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Save encoders for future use

print(X.dtypes)  # Ensure all columns are now numeric

# All columns are now numeric

# Training  the Random Forest Model:
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model Performance

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Analyze Feature Importance

# Since the dataset includes gene expressions, mutations, and clinical attributes, let’s identify which features are most important for predicting treatment response.

# Get feature importance scores
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})

# Sort by importance (descending)
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(20)

# Plot top 20 important features
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'], hue=None, legend=False)
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.title("Top 20 Important Features in Prediction")
plt.show()

# For coloured visualization

plt.figure(figsize=(10, 6))
sns.barplot(
    x=feature_importance['Importance'], 
    y=feature_importance['Feature'], 
    hue=feature_importance['Importance'],  # Assign hue based on importance values
    palette='viridis', 
    dodge=False  # Avoid duplicate bars due to hue
)
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.title("Top 20 Important Features in Prediction")
plt.legend([], [], frameon=False)  # Hide legend
plt.show()

# What does it tell?
# The top-ranked features are the most influential in predicting treatment response.
# Check if clinical attributes (e.g., chemotherapy, age at diagnosis, PAM50 subtypes) are more important than specific gene mutations.
# If certain genes or clinical factors have very low importance, they may not be useful.

# Save the trained model
import joblib

joblib.dump(model, "cancer_treatment_model.pkl")

# Save the scaler for preprocessing future data
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully.")

# Save feature importance
feature_importance.to_csv("feature_importance.csv", index=False)
print("Feature importance saved successfully.")

# Save evaluation metrics
with open("model_evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(confusion_matrix(y_test, y_pred)))

print("Model evaluation results saved successfully.")

