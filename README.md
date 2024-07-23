# neural-network-challenge-1
Module 18 Challenge

# Student Loan Risk with Deep Learning

<a href="https://colab.research.google.com/github/rwlankford/neural-network-challenge-1/blob/main/student_loans_with_deep_learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This project demonstrates how to predict student loan repayment risk using a deep learning neural network. The notebook guides you through the process of preparing data, building and training a neural network, and evaluating its performance.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Model Compilation and Evaluation](#model-compilation-and-evaluation)
3. [Predicting Loan Repayment](#predicting-loan-repayment)
4. [Recommendation System Discussion](#recommendation-system-discussion)

## Data Preparation

### Step 1: Load the Data
Read the `student-loans.csv` file into a Pandas DataFrame and review the columns for features and target variables.

### Step 2: Define Features and Target
Create the features (`X`) and target (`y`) datasets. The target dataset is the `credit_ranking` column.

### Step 3: Split the Data
Split the features and target datasets into training and testing datasets using `train_test_split`.

### Step 4: Scale the Data
Use `StandardScaler` to scale the features data for the training and testing datasets.

```python
# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path

# Load the data
file_path = "https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv"
loans_df = pd.read_csv(file_path)

# Define features and target
y = loans_df["credit_ranking"]
X = loans_df.drop(columns=["credit_ranking"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scale the data
X_scaler = StandardScaler()
X_scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

## Model Compilation and Evaluation

### Step 1: Build the Neural Network
Create a Sequential model with two hidden layers.

### Step 2: Compile and Fit the Model
Compile the model using the `binary_crossentropy` loss function and the `adam` optimizer. Train the model with 50 epochs.

### Step 3: Evaluate the Model
Evaluate the model's loss and accuracy using the test data.

```python
# Build the neural network
nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Dense(units=8, input_dim=len(X.columns), activation="relu"))
nn.add(tf.keras.layers.Dense(units=5, activation="relu"))
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=50)

# Evaluate the model
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

## Predicting Loan Repayment

### Step 1: Save and Export the Model
Save the trained model to a file named `student_loans.keras`.

### Step 2: Load the Saved Model
Reload the saved model for making predictions.

### Step 3: Make Predictions
Make predictions on the testing data and save the predictions to a DataFrame.

### Step 4: Display Classification Report
Display a classification report to evaluate the model's performance.

```python
# Save the model
file_path = Path("/content/drive/My Drive/student_loans.keras")
nn.save(file_path)

# Load the model
nn_imported = tf.keras.models.load_model(file_path)

# Make predictions
predictions = nn_imported.predict(X_test_scaled, verbose=2)
predictions_df = pd.DataFrame(columns=["predictions"], data=predictions)
predictions_df["predictions"] = round(predictions_df["predictions"], 0)

# Display classification report
print(classification_report(y_test, predictions_df["predictions"].values))
```



