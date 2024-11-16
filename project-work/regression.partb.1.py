# Import libraries
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from dtuimldmtools import train_neural_net

# Load and preprocess data
filepath = "prostate/"
filename = "prostate.data.txt"
doc = pd.read_csv(filepath + filename, delimiter="\t")
doc = doc.drop(doc.columns[[0, 5, 8, 10]], axis=1)
attributeNames = doc.columns[[0, 1, 2, 3, 4, 6]].tolist()
data = doc.to_numpy()
X = data[:, [0, 1, 2, 3, 4, 6]]
y = data[:, 5]

# Define parameters for cross-validation
K1 = 5
K2 = 5
outer_cv = KFold(n_splits=K1, shuffle=True, random_state=1)
lambdas = np.power(10.0, np.arange(-3, 3, 1))  # Values for Ridge regression
hidden_units_options = [1, 2, 3]  # Different hidden layer sizes for ANN
max_iter = 10000

# Initialize error storage lists
baseline_errors = []
ridge_errors = []
ann_errors = []
best_lambdas = []
best_hidden_units_list = []

# Outer cross-validation loop
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
    print(f"\nStarting Outer Fold {outer_fold}/{K1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Baseline model: mean prediction
    baseline_pred = np.mean(y_train) * np.ones_like(y_test)
    baseline_error = np.mean((y_test - baseline_pred) ** 2)
    baseline_errors.append(baseline_error)
    print(f"Baseline model error for Outer Fold {outer_fold}: {baseline_error}")

    # Inner cross-validation for Ridge regression and ANN hyperparameter tuning
    inner_cv = KFold(n_splits=K2, shuffle=True, random_state=1)
    best_ridge_error = float('inf')
    best_ann_error = float('inf')
    best_lambda = None
    best_hidden_units = None

    # Ridge regression hyperparameter tuning
    print(f"\n-- Testing Ridge Regression on Outer Fold {outer_fold} --")
    for lambda_val in lambdas:
        ridge_errors_inner = []
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train), start=1):
            print(f"  Inner Fold {inner_fold}/{K2} - Ridge with λ={lambda_val}")
            X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

            # Standardize data
            mean, std = X_inner_train.mean(axis=0), X_inner_train.std(axis=0)
            X_inner_train_std = (X_inner_train - mean) / std
            X_val_std = (X_val - mean) / std

            # Train Ridge regression model
            model = Ridge(alpha=lambda_val)
            model.fit(X_inner_train_std, y_inner_train)
            y_pred = model.predict(X_val_std)
            ridge_errors_inner.append(np.mean((y_val - y_pred) ** 2))

        avg_ridge_error = np.mean(ridge_errors_inner)
        print(f"  Avg Ridge Error for λ={lambda_val} on Outer Fold {outer_fold}: {avg_ridge_error}")

        if avg_ridge_error < best_ridge_error:
            best_ridge_error = avg_ridge_error
            best_lambda = lambda_val

    print(f"Best λ for Ridge on Outer Fold {outer_fold}: {best_lambda} with Error: {best_ridge_error}")

    # Train final Ridge model with best lambda on the full outer training set
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    ridge_model = Ridge(alpha=best_lambda)
    ridge_model.fit(X_train_std, y_train)
    y_pred_ridge = ridge_model.predict(X_test_std)
    ridge_errors.append(np.mean((y_test - y_pred_ridge) ** 2))
    print(f"Ridge Regression error on Outer Test Set for Fold {outer_fold}: {ridge_errors[-1]}")

    # ANN hyperparameter tuning
    print(f"\n-- Testing ANN on Outer Fold {outer_fold} --")
    for n_hidden_units in hidden_units_options:
        ann_errors_inner = []
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train), start=1):
            print(f"  Inner Fold {inner_fold}/{K2} - ANN with {n_hidden_units} Hidden Units - Outer Fold {outer_fold}")
            X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

            # Standardize data
            mean, std = X_inner_train.mean(axis=0), X_inner_train.std(axis=0)
            X_inner_train_std = (X_inner_train - mean) / std
            X_val_std = (X_val - mean) / std

            # Convert data to torch tensors and reshape targets
            X_inner_train_tensor = torch.Tensor(X_inner_train_std)
            y_inner_train_tensor = torch.Tensor(y_inner_train).view(-1, 1)
            X_val_tensor = torch.Tensor(X_val_std)
            y_val_tensor = torch.Tensor(y_val).view(-1, 1)

            # Define the ANN model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], n_hidden_units),
                torch.nn.Tanh(),
                torch.nn.Linear(n_hidden_units, 1)
            )
            loss_fn = torch.nn.MSELoss()

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
            y_val_est = net(X_val_tensor)
            mse = ((y_val_est - y_val_tensor) ** 2).mean().item()
            ann_errors_inner.append(mse)

        avg_ann_error = np.mean(ann_errors_inner)
        print(f"  Avg ANN Error for {n_hidden_units} Hidden Units on Outer Fold {outer_fold}: {avg_ann_error}")

        if avg_ann_error < best_ann_error:
            best_ann_error = avg_ann_error
            best_hidden_units = n_hidden_units

    print(f"Best ANN Hidden Units for Outer Fold {outer_fold}: {best_hidden_units} with Error: {best_ann_error}")

    # Train final ANN model with best hidden layer size on the full outer training set
    X_train_tensor = torch.Tensor(X_train_std)
    y_train_tensor = torch.Tensor(y_train).view(-1, 1)
    X_test_tensor = torch.Tensor(X_test_std)
    y_test_tensor = torch.Tensor(y_test).view(-1, 1)

    final_ann_model = lambda: torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], best_hidden_units),
        torch.nn.Tanh(),
        torch.nn.Linear(best_hidden_units, 1)
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
    y_test_est = net(X_test_tensor)
    ann_error = ((y_test_est - y_test_tensor) ** 2).mean().item()
    ann_errors.append(ann_error)
    print(f"ANN error on Outer Test Set for Fold {outer_fold}: {ann_error}")

    # Store best lambda and hidden units
    best_lambdas.append(best_lambda)
    best_hidden_units_list.append(best_hidden_units)

# Print results
print("\nFinal Results:")
print("Baseline model errors:", baseline_errors)
print("Average Baseline model error:", np.mean(baseline_errors))

print("Ridge regression model errors:", ridge_errors)
print("Average Ridge regression model error:", np.mean(ridge_errors))

print("ANN model errors:", ann_errors)
print("Average ANN model error:", np.mean(ann_errors))

# Print detailed report
print("\nDetailed Report:")
for fold in range(K1):
    print(f"Fold {fold + 1}:")
    print(f"  Baseline Error: {baseline_errors[fold]}")
    print(f"  Ridge Regression - Best λ: {best_lambdas[fold]}, Error: {ridge_errors[fold]}")
    print(f"  ANN - Best Hidden Units: {best_hidden_units_list[fold]}, Error: {ann_errors[fold]}")


# Perform 
