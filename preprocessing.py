import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import os

# Load the Energy Efficiency dataset from UCI ML Repo
energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

# Split: 60% train, 20% val, 20% test
# First split off 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Then split remaining 80% into 60% train and 20% val (i.e. 0.25 of 80% = 20%)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)

print("Split Sizes")
print(f"Train: {X_train.shape[0]} samples")
print(f"Val:   {X_val.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")
print(f"Total: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} samples")

# Check Y1 and Y2 distribution across splits
print("\n Y1 (Heating Load) distribution across splits")
print(f"{'Split':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"{name:<10} {y_split['Y1'].mean():>8.2f} {y_split['Y1'].std():>8.2f} "
          f"{y_split['Y1'].min():>8.2f} {y_split['Y1'].max():>8.2f}")

print("\n Y2 (Cooling Load) distribution across splits")
print(f"{'Split':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
for name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"{name:<10} {y_split['Y2'].mean():>8.2f} {y_split['Y2'].std():>8.2f} "
          f"{y_split['Y2'].min():>8.2f} {y_split['Y2'].max():>8.2f}")

# Normalize features using StandardScaler (zero mean, unit variance)
# Fit scaler on train only, apply to all splits
# This avoids any information leakage from val/test into the normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print("\nFeature means after normalization (train should be 0)")
print(f"Train mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"Val mean:   {X_val_scaled.mean(axis=0).round(4)}")
print(f"Test mean:  {X_test_scaled.mean(axis=0).round(4)}")

# Separate Y1 and Y2
y_train_Y1, y_val_Y1, y_test_Y1 = y_train['Y1'], y_val['Y1'], y_test['Y1']
y_train_Y2, y_val_Y2, y_test_Y2 = y_train['Y2'], y_val['Y2'], y_test['Y2']

# Save splits to disk 
os.makedirs('splits', exist_ok=True)

np.save('splits/X_train.npy', X_train_scaled)
np.save('splits/X_val.npy',   X_val_scaled)
np.save('splits/X_test.npy',  X_test_scaled)

np.save('splits/y_train_Y1.npy', y_train_Y1)
np.save('splits/y_val_Y1.npy',   y_val_Y1)
np.save('splits/y_test_Y1.npy',  y_test_Y1)

np.save('splits/y_train_Y2.npy', y_train_Y2)
np.save('splits/y_val_Y2.npy',   y_val_Y2)
np.save('splits/y_test_Y2.npy',  y_test_Y2)

print("\nSplits saved to splits/")