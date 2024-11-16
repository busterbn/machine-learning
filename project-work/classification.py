# Classification
# Import libraries
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from dtuimldmtools import train_neural_net
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_rel
from math import sqrt

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
num_classes = len(label_encoder.classes_)  # Updated number of classes

# Define parameters for cross-validation
K1 = 4
K2 = 4
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=1)
lambdas = np.power(10.0, np.arange(-3, 3, 0.1))  # Values for logistic regression
hidden_units_options = [1, 2, 4, 8]  # Different hidden layer sizes for ANN
max_iter = 1000000
num_classes = len(np.unique(y))  # Number of classes for classification

# Initialize error storage lists
baseline_accuracies = []
logistic_accuracies = []
ann_accuracies = []
best_lambdas = []
best_hidden_units_list = []

# Outer cross-validation loop
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    print(f"\nStarting Outer Fold {outer_fold}/{K1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Baseline model: predict most frequent class
    most_frequent_class = np.argmax(np.bincount(y_train))
    baseline_pred = np.full(y_test.shape, most_frequent_class)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    baseline_accuracies.append(baseline_accuracy)
    print(f"Baseline model accuracy for Outer Fold {outer_fold}: {baseline_accuracy}")

    # Inner cross-validation for Logistic Regression and ANN hyperparameter tuning
    inner_cv = KFold(n_splits=K2, shuffle=True, random_state=1)
    best_logistic_accuracy = 0
    best_ann_accuracy = 0
    best_lambda = None
    best_hidden_units = None

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

    # Train final Logistic Regression model with best lambda on the full outer training set
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    logistic_model = LogisticRegression(C=1/best_lambda, max_iter=max_iter)
    logistic_model.fit(X_train_std, y_train)
    y_pred_logistic = logistic_model.predict(X_test_std)
    logistic_accuracies.append(accuracy_score(y_test, y_pred_logistic))
    print(f"Logistic Regression accuracy on Outer Test Set for Fold {outer_fold}: {logistic_accuracies[-1]}")

    # ANN hyperparameter tuning
    print(f"\n-- Testing ANN on Outer Fold {outer_fold} --")
    for n_hidden_units in hidden_units_options:
        ann_accuracies_inner = []
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train), start=1):
            X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

            # Standardize data
            mean, std = X_inner_train.mean(axis=0), X_inner_train.std(axis=0)
            X_inner_train_std = (X_inner_train - mean) / std
            X_val_std = (X_val - mean) / std

            # Convert data to torch tensors and reshape targets
            X_inner_train_tensor = torch.Tensor(X_inner_train_std)
            y_inner_train_tensor = torch.LongTensor(y_inner_train)  # Use LongTensor for CrossEntropyLoss
            X_val_tensor = torch.Tensor(X_val_std)
            y_val_tensor = torch.LongTensor(y_val)

            # Define the ANN model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], n_hidden_units),
                torch.nn.Tanh(),
                torch.nn.Linear(n_hidden_units, num_classes)  # Updated to output number of classes
            )
            loss_fn = torch.nn.CrossEntropyLoss()

            # Train ANN
            net, _, _ = train_neural_net(
                model,
                loss_fn,
                X=X_inner_train_tensor,
                y=y_inner_train_tensor,
                n_replicates=1,
                max_iter=max_iter
            )

            # Validate on inner validation set
            y_val_est = net(X_val_tensor).argmax(dim=1)
            accuracy = accuracy_score(y_val, y_val_est.detach().numpy())
            ann_accuracies_inner.append(accuracy)

        avg_ann_accuracy = np.mean(ann_accuracies_inner)
        print(f"  Avg ANN Accuracy for {n_hidden_units} Hidden Units on Outer Fold {outer_fold}: {avg_ann_accuracy}")

        if avg_ann_accuracy > best_ann_accuracy:
            best_ann_accuracy = avg_ann_accuracy
            best_hidden_units = n_hidden_units

    print(f"Best ANN Hidden Units for Outer Fold {outer_fold}: {best_hidden_units} with Accuracy: {best_ann_accuracy}")

    # Train final ANN model with best hidden layer size on the full outer training set
    X_train_tensor = torch.Tensor(X_train_std)
    y_train_tensor = torch.LongTensor(y_train)  # Convert target to LongTensor
    X_test_tensor = torch.Tensor(X_test_std)
    y_test_tensor = torch.LongTensor(y_test)

    final_ann_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], best_hidden_units),
        torch.nn.Tanh(),
        torch.nn.Linear(best_hidden_units, num_classes)  # Updated for multi-class
    )
    net, _, _ = train_neural_net(
        final_ann_model,
        loss_fn,
        X=X_train_tensor,
        y=y_train_tensor,
        n_replicates=1,
        max_iter=max_iter
    )

    # Evaluate ANN on outer test set
    y_test_est = net(X_test_tensor).argmax(dim=1)
    ann_accuracy = accuracy_score(y_test, y_test_est.detach().numpy())
    ann_accuracies.append(ann_accuracy)
    print(f"ANN accuracy on Outer Test Set for Fold {outer_fold}: {ann_accuracy}")

    # Store best lambda and hidden units
    best_lambdas.append(best_lambda)
    best_hidden_units_list.append(best_hidden_units)


# Rounding accuracies to 3 decimal places
baseline_accuracies = [round(acc, 3) for acc in baseline_accuracies]
logistic_accuracies = [round(acc, 3) for acc in logistic_accuracies]
ann_accuracies = [round(acc, 3) for acc in ann_accuracies]

# Print results
print("\nFinal Results:")
print("Baseline model accuracies:", baseline_accuracies)
print("Average Baseline model accuracy:", np.mean(baseline_accuracies))

print("Logistic regression model accuracies:", logistic_accuracies)
print("Average Logistic regression model accuracy:", np.mean(logistic_accuracies))

print("ANN model accuracies:", ann_accuracies)
print("Average ANN model accuracy:", np.mean(ann_accuracies))

# Rounding error rates in the detailed report
print("\n\nDetailed Report:")
for fold in range(K1):
    print(f"Fold {fold + 1}:")
    print(f"  Baseline Accuracy: {round(baseline_accuracies[fold], 3)}")
    print(f"  Baseline error: {round(1.0 - baseline_accuracies[fold], 3)}")
    print(f"  Logistic Regression - Best λ: {round(best_lambdas[fold], 3)}, Accuracy: {round(logistic_accuracies[fold], 3)}")
    print(f"  Logistic Regression - Best λ: {round(best_lambdas[fold], 3)}, Error rate: {round(1 - logistic_accuracies[fold], 3)}")
    print(f"  ANN - Best Hidden Units: {best_hidden_units_list[fold]}, Accuracy: {round(ann_accuracies[fold], 3)}")
    print(f"  ANN - Best Hidden Units: {best_hidden_units_list[fold]}, Error rate: {round(1 - ann_accuracies[fold], 3)}")

print("\n\nCorrelated t-test for cross-validation:")
# Function to calculate confidence interval
def confidence_interval(diff, confidence=0.95):
    mean_diff = round(np.mean(diff), 3).item()
    std_diff = np.std(diff, ddof=1)
    n = len(diff)
    margin_error = round(1.96 * (std_diff / sqrt(n)), 3).item()  # for a 95% confidence interval
    return (mean_diff - margin_error, mean_diff + margin_error)

# Perform the correlated t-test and calculate confidence intervals for each pair
results = {}

# ANN vs Logistic Regression
diff_ann_logistic = np.array(ann_accuracies) - np.array(logistic_accuracies)
t_stat_logistic, p_value_logistic = ttest_rel(ann_accuracies, logistic_accuracies)
ci_logistic = confidence_interval(diff_ann_logistic)
results["ANN vs Logistic Regression"] = (round(t_stat_logistic, 3), round(p_value_logistic, 3), ci_logistic)

# ANN vs Baseline
diff_ann_baseline = np.array(ann_accuracies) - np.array(baseline_accuracies)
t_stat_baseline, p_value_baseline = ttest_rel(ann_accuracies, baseline_accuracies)
ci_baseline = confidence_interval(diff_ann_baseline)
results["ANN vs Baseline"] = (round(t_stat_baseline, 3), round(p_value_baseline, 3), ci_baseline)

# Logistic Regression vs Baseline
diff_logistic_baseline = np.array(logistic_accuracies) - np.array(baseline_accuracies)
t_stat_logistic_baseline, p_value_logistic_baseline = ttest_rel(logistic_accuracies, baseline_accuracies)
ci_logistic_baseline = confidence_interval(diff_logistic_baseline)
results["Logistic Regression vs Baseline"] = (round(t_stat_logistic_baseline, 3), round(p_value_logistic_baseline, 3), ci_logistic_baseline)

# Display results
for comparison, (t_stat, p_value, ci) in results.items():
    print(f"{comparison}")
    print(f"t-statistic: {round(t_stat, 3)}")
    print(f"p-value: {round(p_value, 3)}")
    print(f"95% Confidence Interval: ({round(ci[0], 3)}, {round(ci[1], 3)})")
    if p_value < 0.05:
        print("The difference is statistically significant.\n")
    else:
        print("The difference is not statistically significant.\n")