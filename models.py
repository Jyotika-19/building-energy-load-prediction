import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

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

# Helper: tune hyperparameter with 5-fold CV on training data
def tune_with_cv(model_class, param_name, param_grid, X_train, y_train, **fixed_params):
    best_param = None
    best_score = -np.inf
    scores = []

    for param_val in param_grid:
        model = model_class(**{param_name: param_val}, **fixed_params)
        # neg_mean_squared_error: higher (less negative) is better
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='neg_mean_squared_error')
        mean_score = cv_scores.mean()
        scores.append(mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_param = param_val

    return best_param, param_grid, scores

# Y1 — Heating Load
print("Y1 — Heating Load")

# 1. Linear Regression (base) 
lr_Y1 = LinearRegression()
lr_Y1.fit(X_train, y_train_Y1)

print("\n── Linear Regression ──")
print_results("Train", y_train_Y1, lr_Y1.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y1,   lr_Y1.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y1,  lr_Y1.predict(X_test),  X_train.shape[1])

# 2. kNN Regression 
k_grid = list(range(1, 21))
best_k_Y1, k_grid, k_scores_Y1 = tune_with_cv(
    KNeighborsRegressor, 'n_neighbors', k_grid, X_train, y_train_Y1
)
print(f"\n── kNN Regression (best k = {best_k_Y1} from 5-fold CV) ──")

knn_Y1 = KNeighborsRegressor(n_neighbors=best_k_Y1)
knn_Y1.fit(X_train, y_train_Y1)

print_results("Train", y_train_Y1, knn_Y1.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y1,   knn_Y1.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y1,  knn_Y1.predict(X_test),  X_train.shape[1])

# Plot CV scores for k
plt.figure(figsize=(7, 4))
plt.plot(k_grid, [-s for s in k_scores_Y1], marker='o', color='steelblue')
plt.axvline(best_k_Y1, color='red', linestyle='--', label=f'Best k={best_k_Y1}')
plt.xlabel('k'); plt.ylabel('MSE (5-fold CV)')
plt.title('kNN CV — Y1 Heating Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/knn_cv_Y1.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Decision Tree Regression 
depth_grid = list(range(1, 21))
best_depth_Y1, depth_grid, depth_scores_Y1 = tune_with_cv(
    DecisionTreeRegressor, 'max_depth', depth_grid,
    X_train, y_train_Y1, random_state=42
)
print(f"\n── Decision Tree Regression (best max_depth = {best_depth_Y1} from 5-fold CV) ──")

dt_Y1 = DecisionTreeRegressor(max_depth=best_depth_Y1, random_state=42)
dt_Y1.fit(X_train, y_train_Y1)

print_results("Train", y_train_Y1, dt_Y1.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y1,   dt_Y1.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y1,  dt_Y1.predict(X_test),  X_train.shape[1])

# Plot CV scores for max_depth
plt.figure(figsize=(7, 4))
plt.plot(depth_grid, [-s for s in depth_scores_Y1], marker='o', color='coral')
plt.axvline(best_depth_Y1, color='red', linestyle='--', label=f'Best depth={best_depth_Y1}')
plt.xlabel('max_depth'); plt.ylabel('MSE (5-fold CV)')
plt.title('Decision Tree CV — Y1 Heating Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/dt_cv_Y1.png', dpi=300, bbox_inches='tight')
plt.show()


# Y2 — Cooling Load
print("Y2 — Cooling Load")

# 1. Linear Regression (base) 
lr_Y2 = LinearRegression()
lr_Y2.fit(X_train, y_train_Y2)

print("\n── Linear Regression ──")
print_results("Train", y_train_Y2, lr_Y2.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y2,   lr_Y2.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y2,  lr_Y2.predict(X_test),  X_train.shape[1])

# 2. kNN Regression 
best_k_Y2, k_grid, k_scores_Y2 = tune_with_cv(
    KNeighborsRegressor, 'n_neighbors', k_grid, X_train, y_train_Y2
)
print(f"\n── kNN Regression (best k = {best_k_Y2} from 5-fold CV) ──")

knn_Y2 = KNeighborsRegressor(n_neighbors=best_k_Y2)
knn_Y2.fit(X_train, y_train_Y2)

print_results("Train", y_train_Y2, knn_Y2.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y2,   knn_Y2.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y2,  knn_Y2.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(k_grid, [-s for s in k_scores_Y2], marker='o', color='steelblue')
plt.axvline(best_k_Y2, color='red', linestyle='--', label=f'Best k={best_k_Y2}')
plt.xlabel('k'); plt.ylabel('MSE (5-fold CV)')
plt.title('kNN CV — Y2 Cooling Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/knn_cv_Y2.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Decision Tree Regression 
best_depth_Y2, depth_grid, depth_scores_Y2 = tune_with_cv(
    DecisionTreeRegressor, 'max_depth', depth_grid,
    X_train, y_train_Y2, random_state=42
)
print(f"\n── Decision Tree Regression (best max_depth = {best_depth_Y2} from 5-fold CV) ──")

dt_Y2 = DecisionTreeRegressor(max_depth=best_depth_Y2, random_state=42)
dt_Y2.fit(X_train, y_train_Y2)

print_results("Train", y_train_Y2, dt_Y2.predict(X_train), X_train.shape[1])
print_results("Val",   y_val_Y2,   dt_Y2.predict(X_val),   X_train.shape[1])
print_results("Test",  y_test_Y2,  dt_Y2.predict(X_test),  X_train.shape[1])

plt.figure(figsize=(7, 4))
plt.plot(depth_grid, [-s for s in depth_scores_Y2], marker='o', color='coral')
plt.axvline(best_depth_Y2, color='red', linestyle='--', label=f'Best depth={best_depth_Y2}')
plt.xlabel('max_depth'); plt.ylabel('MSE (5-fold CV)')
plt.title('Decision Tree CV — Y2 Cooling Load')
plt.legend()
plt.tight_layout()
plt.savefig('charts/dt_cv_Y2.png', dpi=300, bbox_inches='tight')
plt.show()