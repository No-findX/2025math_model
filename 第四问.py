import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# 读取数据
path = "processed_female_data.csv"
df = pd.read_csv(path)


def make_label(x):
    if pd.isna(x): return 0
    sx = str(x).strip()
    if sx == "" or sx in ["无", "0", "None", "nan", "NaN"]: return 0
    return 1


df["label"] = df["染色体的非整倍体"].apply(make_label).astype(int)

# 质量控制（QC Gate）
gc_col, read_col = "GC含量", "原始读段数"
map_col, filt_col, dup_col = "在参考基因组上比对的比例", "被过滤掉读段数的比例", "重复读段的比例"

q = {
    "read_p10": df[read_col].quantile(0.10),
    "map_p10": df[map_col].quantile(0.10),
    "filt_p90": df[filt_col].quantile(0.90),
    "dup_p90": df[dup_col].quantile(0.90)
}


def qc_pass(row):
    if not (0.395 <= row[gc_col] <= 0.60): return 0
    if row[read_col] < q["read_p10"]: return 0
    if row[map_col] < q["map_p10"]: return 0
    if row[filt_col] > q["filt_p90"]: return 0
    if row[dup_col] > q["dup_p90"]: return 0
    return 1


df["QC_pass"] = df.apply(qc_pass, axis=1)
pass_rate = df["QC_pass"].mean()
print(f"优化后的QC通过率: {pass_rate:.2%}")

# 特征准备 (平衡策略)

# “核心预测变量 + 关键控制变量”
feature_cols = [
    #  核心预测变量 (经数据验证，P<0.05)
    "X染色体浓度_标准",
    "21号染色体的GC含量_标准",
    "13号染色体的GC含量_标准",

    # 关键控制变量 (基于领域知识和理论重要性)
    "21号染色体的Z值_标准",
    "18号染色体的Z值_标准",
    "孕妇BMI",
    "孕周数值"
]

d = df[df["QC_pass"] == 1].copy()
X_all = d[feature_cols].copy()
y_all = d["label"].values

if len(d) < 20 or len(np.unique(y_all)) < 2:
    print("错误：通过QC的样本过少或只包含一个类别，无法继续建模。")
    exit()

# 交叉验证与模型性能评估
preprocess = ColumnTransformer(
    transformers=[("num",
                   Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())]),
                   feature_cols)],
    remainder="drop"
)
clf = LogisticRegression(solver="lbfgs", penalty="l2", max_iter=500, class_weight="balanced")
pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred, oof_true = [], []
for tr, va in skf.split(X_all, y_all):
    pipe.fit(X_all.iloc[tr], y_all[tr])
    p_va = pipe.predict_proba(X_all.iloc[va])[:, 1]
    oof_pred.extend(p_va)
    oof_true.extend(y_all[va])

oof_pred, oof_true = np.array(oof_pred), np.array(oof_true)

# 性能指标
ap = average_precision_score(oof_true, oof_pred)
auc = roc_auc_score(oof_true, oof_pred)
yp_bin = (oof_pred >= 0.5).astype(int)
acc = accuracy_score(oof_true, yp_bin)
prec, rec, f1, _ = precision_recall_fscore_support(oof_true, yp_bin, average="binary", zero_division=0)
cm = confusion_matrix(oof_true, yp_bin)

print("\n平衡模型性能指标 (基于交叉验证)")
print(f"PR-AUC: {ap:.4f} | ROC-AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print("混淆矩阵:\n", cm)

# 自动选择双阈值
def pick_threshold_cost(y_true, y_score, cost_fn=10.0, cost_fp=1.0):
    ts = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    pi = y_true.mean()
    if pi == 0 or pi == 1: return 0.5
    for t in ts:
        yp = (y_score >= t).astype(int)
        FP = np.sum((yp == 1) & (y_true == 0))
        FN = np.sum((yp == 0) & (y_true == 1))
        cost = cost_fn * FN * pi + cost_fp * FP * (1 - pi)
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t


t_star = pick_threshold_cost(oof_true, oof_pred)
t_low, t_high = max(0.01, t_star - 0.1), min(0.99, t_star + 0.1)
print(f"最优决策点 t* = {t_star:.3f}")
print(f"最终双阈值设置: t_low={t_low:.3f}, t_high={t_high:.3f}")

# 最终训练与三分类判定
pipe.fit(X_all, y_all)
pred_prob = np.full(len(df), np.nan)
pred_cat = np.array(["建议复检"] * len(df), dtype=object)

idx_pass = df["QC_pass"] == 1
if idx_pass.sum() > 0:
    p_pass = pipe.predict_proba(df.loc[idx_pass, feature_cols])[:, 1]
    pred_prob[idx_pass] = p_pass

    high_prob_mask = pred_prob >= t_high
    low_prob_mask = pred_prob <= t_low
    pred_cat[high_prob_mask] = "异常"
    pred_cat[low_prob_mask] = "正常"

df["QC通过"] = df["QC_pass"].map({1: "是", 0: "否"})
df["异常概率"] = pred_prob
df["三类判定"] = pred_cat

# Logit系数显著性检验
X_imputed = X_all.fillna(X_all.median())
X_sm = sm.add_constant(X_imputed)
logit_model = sm.Logit(y_all, X_sm)
try:
    result = logit_model.fit(disp=False)
    print("\n平衡模型系数与显著性 (statsmodels)")
    print(result.summary())
except Exception as e:
    print(f"\n无法完成statsmodels显著性检验: {e}")

out_cols = ["序号", "孕妇代码", "检测日期", "检测孕周", "孕妇BMI",
            "原始读段数", "在参考基因组上比对的比例", "重复读段的比例",
            "唯一比对的读段数", "GC含量",
            "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值",
            "X染色体的Z值", "X染色体浓度",
            "被过滤掉读段数的比例", "染色体的非整倍体",
            "QC通过", "异常概率", "三类判定"]
out_cols = [c for c in out_cols if c in df.columns]
output_filename = "female_pred_results_balanced_model.csv"
df[out_cols].to_csv(output_filename, index=False, encoding="utf-8-sig")
