# Classification - Logistic Regression Only
# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
filepath = "prostate/"
filename = "prostate.data.txt"
doc = pd.read_csv(filename, delimiter="\t")
doc = doc.drop(doc.columns[[0, 5, 8, 10]], axis=1)
attributeNames = doc.columns[[0, 1, 2, 3, 4, 6]].tolist()
data = doc.to_numpy()
X = data[:, [0, 1, 2, 3, 4, 6]]

# Encode the target labels to start from 0
y = data[:, 5].astype(int)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Define parameters for cross-validation
K1 = 10
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=1)
lambdas = np.power(10.0, np.arange(-3, 3, 0.1))  # Values for logistic regression
max_iter = 1000000

# Initialize error storage lists
logistic_accuracies = []
best_lambdas = []

# Outer cross-validation loop
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    print(f"\nStarting Outer Fold {outer_fold}/{K1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner cross-validation for Logistic Regression hyperparameter tuning
    inner_cv = KFold(n_splits=K1, shuffle=True, random_state=1)
    best_logistic_accuracy = 0
    best_lambda = None

    # Logistic Regression hyperparameter tuning
    print(f"\n-- Testing Logistic Regression on Outer Fold {outer_fold} --")
    for lambda_val in lambdas:
        logistic_accuracies_inner = []
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train), start=1):
            X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

            # Standardize data
            mean, std = X_inner_train.mean(axis=0), X_inner_train.std(axis=0)
            X_inner_train_std = (X_inner_train - mean) / std
            X_val_std = (X_val - mean) / std

            # Train Logistic Regression model
            model = LogisticRegression(C=1/lambda_val, max_iter=max_iter)
            model.fit(X_inner_train_std, y_inner_train)
            y_pred = model.predict(X_val_std)
            logistic_accuracies_inner.append(accuracy_score(y_val, y_pred))

        avg_logistic_accuracy = np.mean(logistic_accuracies_inner)
        print(f"  Avg Logistic Regression Accuracy for λ={lambda_val} on Outer Fold {outer_fold}: {avg_logistic_accuracy}")

        if avg_logistic_accuracy > best_logistic_accuracy:
            best_logistic_accuracy = avg_logistic_accuracy
            best_lambda = lambda_val

    print(f"Best λ for Logistic Regression on Outer Fold {outer_fold}: {best_lambda} with Accuracy: {best_logistic_accuracy}")
    best_lambdas.append(best_lambda)

    # Train final Logistic Regression model with best lambda on the full outer training set
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    logistic_model = LogisticRegression(C=1/best_lambda, max_iter=max_iter)
    logistic_model.fit(X_train_std, y_train)
    y_pred_logistic = logistic_model.predict(X_test_std)
    logistic_accuracies.append(accuracy_score(y_test, y_pred_logistic))
    print(f"Logistic Regression accuracy on Outer Test Set for Fold {outer_fold}: {logistic_accuracies[-1]}")

# Print results
print("\nFinal Results:")
print("Logistic regression model accuracies:", logistic_accuracies)
print("Average Logistic regression model accuracy:", np.mean(logistic_accuracies))

# Print best lambda and coefficients
best_lambda_final = best_lambdas[np.argmax(logistic_accuracies)]
final_model = LogisticRegression(C=1/best_lambda_final, max_iter=max_iter)
final_model.fit((X - X.mean(axis=0)) / X.std(axis=0), y)

print(f"\nBest λ for Logistic Regression: {best_lambda_final}")
print("Coefficients for the softmax function:")
print(final_model.coef_)

# Display the best lambda and coefficients with feature names
print(f"Best λ for Logistic Regression: {best_lambda_final}")
print("\nCoefficients for the softmax function by feature and class:")

# Loop through each class and print the feature coefficients
for class_index, class_coefficients in enumerate(final_model.coef_):
    print(f"\nClass {class_index}:")
    for feature_name, coefficient in zip(attributeNames, class_coefficients):
        print(f"  {feature_name}: {coefficient}")