import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Load splits
X_train = np.load('splits/X_train.npy')
X_val   = np.load('splits/X_val.npy')
X_test  = np.load('splits/X_test.npy')

y_train_Y1 = np.load('splits/y_train_Y1.npy')
y_val_Y1   = np.load('splits/y_val_Y1.npy')
y_test_Y1  = np.load('splits/y_test_Y1.npy')

y_train_Y2 = np.load('splits/y_train_Y2.npy')
y_val_Y2   = np.load('splits/y_val_Y2.npy')
y_test_Y2  = np.load('splits/y_test_Y2.npy')

os.makedirs('charts', exist_ok=True)

# Quality measures
def rse(y_true, y_pred, n_features):
    n = len(y_true)
    p = n_features
    rss = np.sum((y_true - y_pred) ** 2)
    return np.sqrt(rss / (n - p - 1))

def r2(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - rss / tss

def print_results(name, y_true, y_pred, n_features):
    print(f"  {name:<25} RSE: {rse(y_true, y_pred, n_features):.4f}   R²: {r2(y_true, y_pred):.4f}")

# Helper: tune hyperparameter using validation set
# For each value in the grid, train on training data and evaluate MSE on val set.
# The value with the lowest validation MSE is selected as the best hyperparameter.
# The test set is never used during this process.
def tune_with_val(model_class, param_name, param_grid, X_train, y_train, X_val, y_val, **fixed_params):
    best_param = None
    best_score = np.inf
    scores = []

    for param_val in param_grid:
        model = model_class(**{param_name: param_val}, **fixed_params)
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_val, model.predict(X_val))
        scores.append(mse)
        if mse < best_score:
            best_score = mse
            best_param = param_val

    return best_param, param_grid, scores


# Y1 — Heating Load
print("Y1 — Heating Load")

# 1. Linear Regression (base)
# Linear regression has no hyperparameters to tune — the weights are learned
# directly from training data by minimizing RSS
lr_Y1 = LinearRegression()
lr_Y1.fit(X_train, y_train_Y1)

print("\n Linear Regression")
print_results("Train", y_train_Y1, lr_Y1.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y1,   lr_Y1.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y1,  lr_Y1.predict(X_test),  X_train.shape[1])

# 2. kNN Regression
# We tune k (number of neighbors) by training on training data and evaluating
# MSE on the validation set for each value of k from 1 to 20.
# The k with the lowest validation MSE is selected.
k_grid = list(range(1, 21))
best_k_Y1, k_grid, k_scores_Y1 = tune_with_val(
    KNeighborsRegressor, 'n_neighbors', k_grid,
    X_train, y_train_Y1, X_val, y_val_Y1
)
print(f"\n kNN Regression (best k = {best_k_Y1} from validation set tuning)")

knn_Y1 = KNeighborsRegressor(n_neighbors=best_k_Y1)
knn_Y1.fit(X_train, y_train_Y1)

print_results("Train", y_train_Y1, knn_Y1.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y1,   knn_Y1.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y1,  knn_Y1.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(k_grid, k_scores_Y1, marker='o', color='steelblue')
plt.axvline(best_k_Y1, color='red', linestyle='--', label=f'Best k={best_k_Y1}')
plt.xlabel('k'); plt.ylabel('MSE (validation set)')
plt.title('kNN Hyperparameter Tuning — Y1 Heating Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/knn_cv_Y1.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Decision Tree Regression
# We tune max_depth by training on training data and evaluating MSE on the
# validation set for each depth from 1 to 20.
# Deeper trees are more flexible but risk overfitting — the validation set
# helps us find the right balance.
depth_grid = list(range(1, 21))
best_depth_Y1, depth_grid, depth_scores_Y1 = tune_with_val(
    DecisionTreeRegressor, 'max_depth', depth_grid,
    X_train, y_train_Y1, X_val, y_val_Y1, random_state=42
)
print(f"\n Decision Tree Regression (best max_depth = {best_depth_Y1} from validation set tuning)")

dt_Y1 = DecisionTreeRegressor(max_depth=best_depth_Y1, random_state=42)
dt_Y1.fit(X_train, y_train_Y1)

print_results("Train", y_train_Y1, dt_Y1.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y1,   dt_Y1.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y1,  dt_Y1.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(depth_grid, depth_scores_Y1, marker='o', color='coral')
plt.axvline(best_depth_Y1, color='red', linestyle='--', label=f'Best depth={best_depth_Y1}')
plt.xlabel('max_depth'); plt.ylabel('MSE (validation set)')
plt.title('Decision Tree Hyperparameter Tuning — Y1 Heating Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/dt_cv_Y1.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Random Forest Regression
# Random Forest builds many decision trees on bootstrap samples and averages
# their predictions, reducing the high variance of a single decision tree.
# We tune n_estimators (number of trees) using the validation set.
n_estimators_grid = [10, 50, 100, 200, 300]
best_n_Y1, n_estimators_grid, n_scores_Y1 = tune_with_val(
    RandomForestRegressor, 'n_estimators', n_estimators_grid,
    X_train, y_train_Y1, X_val, y_val_Y1, random_state=42
)
print(f"\n Random Forest Regression (best n_estimators = {best_n_Y1} from validation set tuning)")

rf_Y1 = RandomForestRegressor(n_estimators=best_n_Y1, random_state=42)
rf_Y1.fit(X_train, y_train_Y1)

print_results("Train", y_train_Y1, rf_Y1.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y1,   rf_Y1.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y1,  rf_Y1.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(n_estimators_grid, n_scores_Y1, marker='o', color='purple')
plt.axvline(best_n_Y1, color='red', linestyle='--', label=f'Best n={best_n_Y1}')
plt.xlabel('n_estimators'); plt.ylabel('MSE (validation set)')
plt.title('Random Forest Hyperparameter Tuning — Y1 Heating Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/rf_cv_Y1.png', dpi=300, bbox_inches='tight')
plt.show()


# Y2 — Cooling Load
print("Y2 — Cooling Load")

# 1. Linear Regression (base)
lr_Y2 = LinearRegression()
lr_Y2.fit(X_train, y_train_Y2)

print("\n Linear Regression")
print_results("Train", y_train_Y2, lr_Y2.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y2,   lr_Y2.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y2,  lr_Y2.predict(X_test),  X_train.shape[1])

# 2. kNN Regression
best_k_Y2, k_grid, k_scores_Y2 = tune_with_val(
    KNeighborsRegressor, 'n_neighbors', k_grid,
    X_train, y_train_Y2, X_val, y_val_Y2
)
print(f"\n kNN Regression (best k = {best_k_Y2} from validation set tuning)")

knn_Y2 = KNeighborsRegressor(n_neighbors=best_k_Y2)
knn_Y2.fit(X_train, y_train_Y2)

print_results("Train", y_train_Y2, knn_Y2.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y2,   knn_Y2.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y2,  knn_Y2.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(k_grid, k_scores_Y2, marker='o', color='steelblue')
plt.axvline(best_k_Y2, color='red', linestyle='--', label=f'Best k={best_k_Y2}')
plt.xlabel('k'); plt.ylabel('MSE (validation set)')
plt.title('kNN Hyperparameter Tuning — Y2 Cooling Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/knn_cv_Y2.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Decision Tree Regression
best_depth_Y2, depth_grid, depth_scores_Y2 = tune_with_val(
    DecisionTreeRegressor, 'max_depth', depth_grid,
    X_train, y_train_Y2, X_val, y_val_Y2, random_state=42
)
print(f"\n Decision Tree Regression (best max_depth = {best_depth_Y2} from validation set tuning)")

dt_Y2 = DecisionTreeRegressor(max_depth=best_depth_Y2, random_state=42)
dt_Y2.fit(X_train, y_train_Y2)

print_results("Train", y_train_Y2, dt_Y2.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y2,   dt_Y2.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y2,  dt_Y2.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(depth_grid, depth_scores_Y2, marker='o', color='coral')
plt.axvline(best_depth_Y2, color='red', linestyle='--', label=f'Best depth={best_depth_Y2}')
plt.xlabel('max_depth'); plt.ylabel('MSE (validation set)')
plt.title('Decision Tree Hyperparameter Tuning — Y2 Cooling Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/dt_cv_Y2.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Random Forest Regression
best_n_Y2, n_estimators_grid, n_scores_Y2 = tune_with_val(
    RandomForestRegressor, 'n_estimators', n_estimators_grid,
    X_train, y_train_Y2, X_val, y_val_Y2, random_state=42
)
print(f"\n Random Forest Regression (best n_estimators = {best_n_Y2} from validation set tuning)")

rf_Y2 = RandomForestRegressor(n_estimators=best_n_Y2, random_state=42)
rf_Y2.fit(X_train, y_train_Y2)

print_results("Train", y_train_Y2, rf_Y2.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y2,   rf_Y2.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y2,  rf_Y2.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(n_estimators_grid, n_scores_Y2, marker='o', color='purple')
plt.axvline(best_n_Y2, color='red', linestyle='--', label=f'Best n={best_n_Y2}')
plt.xlabel('n_estimators'); plt.ylabel('MSE (validation set)')
plt.title('Random Forest Hyperparameter Tuning — Y2 Cooling Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/rf_cv_Y2.png', dpi=300, bbox_inches='tight')
plt.show()