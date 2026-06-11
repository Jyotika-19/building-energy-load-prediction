import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Load splits 
X_train = np.load('splits/X_train.npy')
X_test  = np.load('splits/X_test.npy')

y_train_Y1 = np.load('splits/y_train_Y1.npy')
y_test_Y1  = np.load('splits/y_test_Y1.npy')

y_train_Y2 = np.load('splits/y_train_Y2.npy')
y_test_Y2  = np.load('splits/y_test_Y2.npy')

# Retrain final models with best hyperparameters from models.py 
# Linear Regression
lr_Y1 = LinearRegression().fit(X_train, y_train_Y1)
lr_Y2 = LinearRegression().fit(X_train, y_train_Y2)

# kNN — best k found in models.py
knn_Y1 = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train_Y1)
knn_Y2 = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train_Y2)

# Decision Tree — best depth found in models.py
dt_Y1 = DecisionTreeRegressor(max_depth=7, random_state=42).fit(X_train, y_train_Y1)
dt_Y2 = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_train, y_train_Y2)

# Quality measures
def rse(y_true, y_pred, n_features):
    rss = np.sum((y_true - y_pred) ** 2)
    return np.sqrt(rss / (len(y_true) - n_features - 1))

def r2(y_true, y_pred):
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - rss / tss

# Residual plot function
def residual_plot(ax, y_true, y_pred, title, color):
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.4, color=color, s=15)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual (actual - predicted)')
    ax.set_title(title)


# Y1 — Heating Load: Residual plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Residual Plots — Y1 Heating Load (Test Set)', fontsize=13)

residual_plot(axes[0], y_test_Y1, lr_Y1.predict(X_test),
              f'Linear Regression\nRSE={rse(y_test_Y1, lr_Y1.predict(X_test), 8):.3f}  R²={r2(y_test_Y1, lr_Y1.predict(X_test)):.3f}',
              'steelblue')

residual_plot(axes[1], y_test_Y1, knn_Y1.predict(X_test),
              f'kNN (k=5)\nRSE={rse(y_test_Y1, knn_Y1.predict(X_test), 8):.3f}  R²={r2(y_test_Y1, knn_Y1.predict(X_test)):.3f}',
              'darkorange')

residual_plot(axes[2], y_test_Y1, dt_Y1.predict(X_test),
              f'Decision Tree (depth=7)\nRSE={rse(y_test_Y1, dt_Y1.predict(X_test), 8):.3f}  R²={r2(y_test_Y1, dt_Y1.predict(X_test)):.3f}',
              'seagreen')

plt.tight_layout()
plt.show()

# Y2 — Cooling Load: Residual plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Residual Plots — Y2 Cooling Load (Test Set)', fontsize=13)

residual_plot(axes[0], y_test_Y2, lr_Y2.predict(X_test),
              f'Linear Regression\nRSE={rse(y_test_Y2, lr_Y2.predict(X_test), 8):.3f}  R²={r2(y_test_Y2, lr_Y2.predict(X_test)):.3f}',
              'steelblue')

residual_plot(axes[1], y_test_Y2, knn_Y2.predict(X_test),
              f'kNN (k=5)\nRSE={rse(y_test_Y2, knn_Y2.predict(X_test), 8):.3f}  R²={r2(y_test_Y2, knn_Y2.predict(X_test)):.3f}',
              'darkorange')

residual_plot(axes[2], y_test_Y2, dt_Y2.predict(X_test),
              f'Decision Tree (depth=6)\nRSE={rse(y_test_Y2, dt_Y2.predict(X_test), 8):.3f}  R²={r2(y_test_Y2, dt_Y2.predict(X_test)):.3f}',
              'seagreen')

plt.tight_layout()
plt.show()

# Summary table
print("Final Results on Test Set")

print(f"{'Model':<30} {'Y1 RSE':>8} {'Y1 R²':>8} {'Y2 RSE':>8} {'Y2 R²':>8}")


models = [
    ('Linear Regression', lr_Y1, lr_Y2),
    ('kNN (k=5)',         knn_Y1, knn_Y2),
    ('Decision Tree',     dt_Y1,  dt_Y2),
]

for name, m1, m2 in models:
    r1 = rse(y_test_Y1, m1.predict(X_test), 8)
    r1sq = r2(y_test_Y1, m1.predict(X_test))
    r2_ = rse(y_test_Y2, m2.predict(X_test), 8)
    r2sq = r2(y_test_Y2, m2.predict(X_test))
    print(f"{name:<30} {r1:>8.4f} {r1sq:>8.4f} {r2_:>8.4f} {r2sq:>8.4f}")