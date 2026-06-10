import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets

df = pd.concat([X, y], axis=1)
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.duplicated().sum())

print(df.corr())            # corr() returns values between -1 and 1
# 1 = perfect positive correlation, -1 = perfect negative, 0 = no correlation

sns.heatmap(df.corr(),
            cmap='Blues',        # single colour, darker = stronger correlation
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7})
plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=df['Y1'])
plt.xlabel('Heating Load')
plt.title('Distribution of Heating Load (Y1)', fontsize=9)

plt.subplot(1, 2, 2)
sns.histplot(data=df['Y2'])
plt.xlabel('Cooling Load')
plt.title('Distribution of Cooling Load (Y2)', fontsize=9)
plt.tight_layout()
plt.show()

print(df['X5'].unique())
# X5 (Overall Height) has only 2 unique values: 3.5 and 7
# This explains the bimodal distribution in Y1 and Y2
# Tall vs short buildings have very different energy requirements

print(df['X4'].unique())
# X4 (Roof Area) was suspicious because its 75th percentile equalled its max
# Investigating shows it has 4 unique values

feature_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
feature_names = {
    'X1': 'Relative Compactness',
    'X2': 'Surface Area',
    'X3': 'Wall Area',
    'X4': 'Roof Area',
    'X5': 'Overall Height',
    'X6': 'Orientation',
    'X7': 'Glazing Area',
    'X8': 'Glazing Area Distribution'
}

# Scatter plots: all features vs Y1 (Heating Load)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].scatter(df[col], df['Y1'], alpha=0.3, color='steelblue', s=10)
    axes[i].set_xlabel(feature_names[col])
    axes[i].set_ylabel('Heating Load (Y1)')
    axes[i].set_title(f'{col} vs Y1')
fig.suptitle('Features vs Heating Load (Y1)', fontsize=13)
plt.tight_layout()
plt.show()

# Scatter plots: all features vs Y2 (Cooling Load)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].scatter(df[col], df['Y2'], alpha=0.3, color='coral', s=10)
    axes[i].set_xlabel(feature_names[col])
    axes[i].set_ylabel('Cooling Load (Y2)')
    axes[i].set_title(f'{col} vs Y2')
fig.suptitle('Features vs Cooling Load (Y2)', fontsize=13)
plt.tight_layout()
plt.show()

# Y1 vs Y2 scatter — justifies predicting them as two separate regression problems
plt.figure(figsize=(6, 5))
plt.scatter(df['Y1'], df['Y2'], alpha=0.3, color='purple', s=10)
plt.xlabel('Heating Load (Y1)')
plt.ylabel('Cooling Load (Y2)')
plt.title(f'Y1 vs Y2 (corr = {df["Y1"].corr(df["Y2"]):.2f})')
plt.tight_layout()
plt.show()