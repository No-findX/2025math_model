# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_auc_score
)

from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib import rcParams

font_path = "C:/Windows/Fonts/simsun.ttc"  
font_prop = font_manager.FontProperties(fname=font_path)

# 全局设置字体为宋体
rcParams['font.sans-serif'] = font_prop.get_name()
rcParams['axes.unicode_minus'] = False 
rcParams['xtick.labelsize'] = 12 
rcParams['ytick.labelsize'] = 12

# 忽略一些计算过程中可能出现的警告
warnings.filterwarnings("ignore")


def setup_chinese_font():
    """
    尝试设置matplotlib以正确显示中文字符。
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        print("中文字体 'SimHei' 设置成功。")
    except Exception as e:
        print(f"无法设置中文字体'SimHei'，图表标签可能显示不正常。错误: {e}")
        plt.rcParams['font.sans-serif'] = ['sans-serif']


def generate_additional_plots(df, d, oof_true, oof_pred, t_star, t_low, t_high, pipe, feature_cols, gc_col):
    """
    生成所有额外的分析和评估图表。
    """
    print("\n--- 步骤 3a: 生成额外的分析图表 ---")

    # --- 图1: QC前后关键指标("GC含量")分布图 ---
    print("  - 正在生成图1 (QC分布对比图)...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[gc_col], label="QC筛选前", fill=True, color="gray", lw=2)
    sns.kdeplot(d[gc_col], label="QC筛选后", fill=True, color="skyblue", lw=2)
    plt.title('QC筛选前后 "GC含量" 的分布对比', fontsize=16, fontproperties=font_prop)
    plt.xlabel("GC含量", fontsize=12, fontproperties=font_prop)
    plt.ylabel("密度", fontsize=12, fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.savefig("plot_1_qc_distribution.png", dpi=300)
    plt.close()

    # --- 图2: ROC曲线和PR曲线 ---
    print("  - 正在生成图2 (ROC和PR曲线)...")
    fpr, tpr, _ = roc_curve(oof_true, oof_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(oof_true, oof_pred)
    pr_auc = average_precision_score(oof_true, oof_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('假正例率 (False Positive Rate)', fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel('真正例率 (True Positive Rate)', fontsize=12, fontproperties=font_prop)
    ax1.set_title('ROC 曲线', fontsize=14, weight="bold", fontproperties=font_prop)
    ax1.legend(loc="lower right", prop=font_prop)
    ax1.grid(True)
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax2.set_xlabel('召回率 (Recall)', fontsize=12, fontproperties=font_prop)
    ax2.set_ylabel('精确率 (Precision)', fontsize=12, fontproperties=font_prop)
    ax2.set_title('PR 曲线', fontsize=14, weight="bold", fontproperties=font_prop)
    ax2.legend(loc="lower left");
    ax2.grid(True)
    plt.suptitle("模型性能评估曲线", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plot_2_roc_pr_curves.png", dpi=300)
    plt.close()

    # --- 图3: 混淆矩阵热力图 ---
    print("  - 正在生成图3 (混淆矩阵热力图)...")
    y_pred_class = (oof_pred >= t_star).astype(int)
    cm = confusion_matrix(oof_true, y_pred_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测为负', '预测为正'],
                yticklabels=['实际为负', '实际为正'])
    plt.title(f'混淆矩阵 (最优决策阈值 t* = {t_star:.3f})', fontsize=16, fontproperties=font_prop)
    plt.ylabel('实际类别', fontsize=12, fontproperties=font_prop)
    plt.xlabel('预测类别', fontsize=12, fontproperties=font_prop)
    plt.savefig("plot_3_confusion_matrix.png", dpi=300)
    plt.close()

    # --- 图4: 模型系数大小图 ---
    print("  - 正在生成图4 (模型特征系数图)...")
    coefficients = pipe.named_steps['clf'].coef_[0]
    feature_importance = pd.DataFrame({'Feature': feature_cols, 'Coefficient': coefficients})
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='viridis')
    plt.title('逻辑回归模型特征系数', fontsize=16, fontproperties=font_prop)
    plt.xlabel('系数值', fontsize=12, fontproperties=font_prop)
    plt.ylabel('特征', fontsize=12, fontproperties=font_prop)
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig("plot_4_feature_coefficients.png", dpi=300)
    plt.close()

    # --- 图5: 成本-阈值关系图 ---
    print("  - 正在生成图5 (成本-阈值关系图)...")
    ts, costs = [], []
    pi = oof_true.mean()
    for t in np.linspace(0.01, 0.99, 99):
        yp = (oof_pred >= t).astype(int)
        FP = np.sum((yp == 1) & (oof_true == 0))
        FN = np.sum((yp == 0) & (oof_true == 1))
        cost = 10.0 * FN * pi + 1.0 * FP * (1 - pi)
        ts.append(t)
        costs.append(cost)
    plt.figure(figsize=(10, 6))
    plt.plot(ts, costs, marker='.', linestyle='-')
    plt.axvline(x=t_star, color='r', linestyle='--', label=f'成本最低点 t*={t_star:.3f}')
    plt.title('总成本 vs. 决策阈值', fontsize=16, fontproperties=font_prop)
    plt.xlabel('决策阈值', fontsize=12, fontproperties=font_prop)
    plt.ylabel('加权总成本', fontsize=12, fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(True)
    plt.savefig("plot_5_cost_vs_threshold.png", dpi=300)
    plt.close()

    # --- 图6: 带有决策阈值的概率分布图 (美化版) ---
    print("  - 正在生成图6 (带阈值的概率分布图)...")
    plt.figure(figsize=(12, 7))
    sns.kdeplot(oof_pred[oof_true == 0], label="负样本 (正常)", color="#4c72b0", fill=True, alpha=0.4, linewidth=2.5)
    sns.kdeplot(oof_pred[oof_true == 1], label="正样本 (异常)", color="#dd8452", fill=True, alpha=0.5, linewidth=2.5)

    # 绘制中心最优阈值线
    plt.axvline(x=t_star, color="#c44e52", linestyle='--', linewidth=2, label=f'最优决策点 t*={t_star:.3f}')

    # 绘制左右边界线并添加灰色区域
    plt.axvline(x=t_low, color="gray", linestyle=':', linewidth=1.5, label=f'低风险边界 t_low={t_low:.3f}')
    plt.axvline(x=t_high, color="gray", linestyle=':', linewidth=1.5, label=f'高风险边界 t_high={t_high:.3f}')
    plt.axvspan(t_low, t_high, color='gray', alpha=0.2, label='建议复检区域 (灰色地带)')

    # 美化图表
    plt.title('模型预测概率分布与风险区间划分', fontsize=18)
    plt.xlabel('模型预测为异常的概率', fontsize=14, fontproperties=font_prop)
    plt.ylabel('概率密度', fontsize=14, fontproperties=font_prop)
    plt.legend(fontsize=12, prop=font_prop)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 1)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig("plot_6_distribution_with_thresholds.png", dpi=300)
    plt.close()
    print("所有附加图表生成完毕。")


def main():
    """
    执行完整的数据分析、模型训练、评估和可视化流程。
    """
    # -------------------------
    # 1. 数据读取与处理
    # -------------------------
    print("--- 步骤 1: 读取和预处理数据 ---")
    path = "processed_female_data.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"错误：数据文件 '{path}' 未找到。请确保脚本和CSV文件在同一个目录下。")
        return

    def make_label(x):
        if pd.isna(x): return 0
        sx = str(x).strip()
        if sx == "" or sx in ["无", "0", "None", "nan", "NaN"]: return 0
        return 1

    df["label"] = df["染色体的非整倍体"].apply(make_label).astype(int)

    # 质量控制 (QC Gate)
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

    d = df[df["QC_pass"] == 1].copy()

    # 特征准备
    feature_cols = [
        "X染色体浓度_标准", "21号染色体的GC含量_标准", "13号染色体的GC含量_标准",
        "21号染色体的Z值_标准", "18号染色体的Z值_标准", "孕妇BMI", "孕周数值"
    ]
    X_all = d[feature_cols].copy()
    y_all = d["label"].values

    if len(d) < 20 or len(np.unique(y_all)) < 2:
        print("错误：通过QC的样本过少或只包含一个类别，无法继续建模。")
        return

    # -------------------------
    # 2. 交叉验证与性能评估
    # -------------------------
    print("\n--- 步骤 2: 执行5折交叉验证并评估性能 ---")
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
    fold_pred_data = []  # 用于绘图

    for i, (tr, va) in enumerate(skf.split(X_all, y_all), start=1):
        pipe.fit(X_all.iloc[tr], y_all[tr])
        p_va = pipe.predict_proba(X_all.iloc[va])[:, 1]
        oof_pred.extend(p_va)
        oof_true.extend(y_all[va])
        fold_pred_data.append((y_all[va], p_va))

    oof_pred, oof_true = np.array(oof_pred), np.array(oof_true)

    # 打印性能指标
    ap = average_precision_score(oof_true, oof_pred)
    auc_val = roc_auc_score(oof_true, oof_pred)
    yp_bin = (oof_pred >= 0.5).astype(int)
    acc = accuracy_score(oof_true, yp_bin)
    prec, rec, f1, _ = precision_recall_fscore_support(oof_true, yp_bin, average="binary", zero_division=0)
    cm = confusion_matrix(oof_true, yp_bin)

    print("\n平衡模型性能指标 (基于交叉验证的OOF预测)")
    print(f"PR-AUC: {ap:.4f} | ROC-AUC: {auc_val:.4f}")
    print(f"Accuracy (at 0.5 threshold): {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("混淆矩阵 (at 0.5 threshold):\n", cm)

    # -------------------------
    # 3. 绘制交叉验证概率图
    # -------------------------
    print("\n--- 步骤 3: 绘制5折交叉验证概率分布图 ---")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (y_fold, p_fold) in enumerate(fold_pred_data):
        ax = axes[i]
        sns.kdeplot(p_fold[y_fold == 1], color="orange", linestyle="-", label="正样本 KDE", ax=ax, fill=True, alpha=0.3)
        sns.kdeplot(p_fold[y_fold == 0], color="blue", linestyle="--", label="负样本 KDE", ax=ax, fill=True, alpha=0.3)
        ax.set_title(f"Fold {i + 1}")
        ax.set_xlabel("预测概率")
        if i % 2 == 0: ax.set_ylabel("密度")
        ax.grid(True)
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)
    plt.suptitle("5折交叉验证预测概率 (KDE)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("5fold_kde_plot.png", dpi=300)
    print("图表已保存为 '5fold_kde_plot.png'")
    plt.close()

    # -------------------------
    # 4. 优化决策阈值
    # -------------------------
    print("\n--- 步骤 4: 优化决策阈值 ---")

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
    print(f"基于成本函数的最优决策点 t* = {t_star:.3f}")
    print(f"最终双阈值设置: t_low={t_low:.3f}, t_high={t_high:.3f}")

    # -------------------------
    # 5. 生成所有附加图表
    # -------------------------
    pipe.fit(X_all, y_all)  # 在全量数据上训练一次，以获取最终系数用于绘图
    generate_additional_plots(df, d, oof_true, oof_pred, t_star, t_low, t_high, pipe, feature_cols, gc_col)

    # -------------------------
    # 6. 生成最终结果并保存
    # -------------------------
    print("\n--- 步骤 6: 生成最终预测报告 ---")
    pred_prob = np.full(len(df), np.nan)
    pred_cat = np.array(["建议复检"] * len(df), dtype=object)
    idx_pass = df["QC_pass"] == 1
    if idx_pass.sum() > 0:
        p_pass = pipe.predict_proba(df.loc[idx_pass, feature_cols])[:, 1]
        pred_prob[idx_pass] = p_pass
        high_prob_mask = (pred_prob >= t_high) & (~np.isnan(pred_prob))
        low_prob_mask = (pred_prob <= t_low) & (~np.isnan(pred_prob))
        pred_cat[high_prob_mask] = "异常"
        pred_cat[low_prob_mask] = "正常"
    df["QC通过"] = df["QC_pass"].map({1: "是", 0: "否"})
    df["异常概率"] = pred_prob
    df["三类判定"] = pred_cat
    output_filename = "female_pred_results_final.csv"
    out_cols = [c for c in df.columns if "_标准" not in c and c not in ["label", "QC_pass"]]
    df[out_cols].to_csv(output_filename, index=False, encoding="utf-8-sig")
    print(f"最终预测结果已保存至 '{output_filename}'")

    # -------------------------
    # 7. 模型系数显著性检验
    # -------------------------
    print("\n--- 步骤 7: 模型系数显著性检验 (statsmodels) ---")
    X_imputed = X_all.fillna(X_all.median())
    X_sm = sm.add_constant(X_imputed)
    logit_model = sm.Logit(y_all, X_sm)
    try:
        result = logit_model.fit(disp=False)
        print("\n平衡模型系数与显著性")
        print(result.summary())
    except Exception as e:
        print(f"\n无法完成statsmodels显著性检验: {e}")


if __name__ == '__main__':
    setup_chinese_font()
    main()

