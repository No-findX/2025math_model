# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import rcParams
from scipy.stats import gaussian_kde

# ============== Font Settings ==============
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.unicode_minus'] = False  # Ensure minus sign is displayed correctly

# ============== Data Loading ==============
CSV_PATH = 'processed_male_data.csv'
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Data file not found: {CSV_PATH}")

data = pd.read_csv(CSV_PATH)

# ============== Preprocessing and Renaming Columns ==============
# Map all Chinese column names to English for consistency
rename_map = {
    'X染色体浓度_标准': 'X_conc_std',
    '18号染色体的Z值_标准': 'Z18_std',
    'Y染色体的Z值_标准': 'ZY_std',
    '原始读段数_标准': 'raw_reads_std',
    '被过滤掉读段数的比例_标准': 'filtered_rate_std',
    'Y染色体浓度': 'Y_conc',
    '孕妇代码': 'ID',
    '孕妇BMI': 'BMI',
    '孕周数值': 'GA',
    '年龄': 'Age'
}

# Apply the renaming
for old, new in rename_map.items():
    if old in data.columns and new not in data.columns:
        data.rename(columns={old: new}, inplace=True)

# ============== Model Fitting ==============
if 'Y_conc' not in data.columns:
    raise KeyError("Required column not found: 'Y_chromosome_concentration'")

# Apply logit transformation to the dependent variable
eps = 1e-6
p = data['Y_conc'].astype(float).clip(eps, 1 - eps)
data['y_logit'] = np.log(p / (1 - p))

# Set a default Maternal_ID if not present (for grouping in the model)
if 'ID' not in data.columns:
    data['ID'] = 'G0'

# Define independent variables for the model
rhs_terms = [t for t in
             ['X_conc_std', 'Z18_std', 'BMI', 'raw_reads_std', 'filtered_rate_std', 'GA']
             if t in data.columns]

# Fit a mixed-effects linear model if possible, otherwise use the mean
results = None
if rhs_terms:
    formula = "y_logit ~ " + " + ".join(rhs_terms)
    try:
        model = smf.mixedlm(formula, data, groups=data["ID"])
        results = model.fit()
        data['y_pred'] = results.fittedvalues
        data['residuals'] = data['y_logit'] - data['y_pred']
    except Exception as e:
        print(f"Model fitting failed: {e}. Falling back to mean.")
        results = None

# Fallback case: if model fails or no terms, residuals are deviations from the mean
if results is None:
    mu = data['y_logit'].mean()
    data['y_pred'] = mu
    data['residuals'] = data['y_logit'] - mu

# ============== Plot 1: Residual Frequency Distribution ==============
plt.figure(figsize=(6, 4), dpi=150)
counts, bins, _ = plt.hist(data['residuals'], bins=30, color="#1f3c88",
                           edgecolor="#333333", alpha=0.4, label="Residual Frequency")

# Calculate and scale KDE to match histogram's scale
kde = gaussian_kde(data['residuals'])
x_range = np.linspace(data['residuals'].min(), data['residuals'].max(), 200)
bin_width = bins[1] - bins[0]
scale_factor = len(data['residuals']) * bin_width
plt.plot(x_range, kde(x_range) * scale_factor, color="#FF8C00", linewidth=2.2, label="Kernel Density Curve")

plt.title("Residual Frequency Distribution", fontsize=13, weight="bold")
plt.xlabel("Residuals", fontsize=12, weight="bold")
plt.ylabel("Frequency", fontsize=12, weight="bold")
plt.legend()
plt.tight_layout()
plt.show()

# ============== Plot 2: Predicted vs. Observed Values Scatter Plot ==============
plt.figure(figsize=(6, 6), dpi=150)
sns.scatterplot(x='y_pred', y='y_logit', data=data,
                alpha=0.65, s=50, color="#1f3c88", edgecolor="w")

# Add a y=x reference line
min_val = min(data['y_pred'].min(), data['y_logit'].min())
max_val = max(data['y_pred'].max(), data['y_logit'].max())
plt.plot([min_val, max_val], [min_val, max_val], color="#FF8C00", linestyle="--", linewidth=2, label="Ideal Fit (y=x)")

plt.xlabel("Predicted Values", fontsize=12, weight="bold")
plt.ylabel("Observed Values", fontsize=12, weight="bold")
plt.title("Model Fit: Predicted vs. Observed Values", fontsize=13, weight="bold")
plt.tight_layout()
plt.show()

# ============== Plot 3: Correlation Heatmap of Variables ==============
# Select continuous variables that exist in the dataframe
cont_vars = [c for c in ['BMI', 'GA', 'Age', 'X_conc_std', 'Z18_std', 'ZY_std',
                         'raw_reads_std', 'filtered_rate_std', 'y_logit', 'Y_conc']
             if c in data.columns]

if len(cont_vars) >= 2:
    plt.figure(figsize=(8, 6), dpi=150)
    corr = data[cont_vars].corr()
    ax = sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                     cbar_kws={"shrink": .8}, annot_kws={"fontsize": 9})

    plt.title("Correlation Heatmap of Variables", fontsize=10)

    # Set font properties for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(8)

    plt.tight_layout()
    plt.show()

print("✅ All three plots have been generated successfully.")