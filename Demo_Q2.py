# generate_advanced_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# --- [新增] 智能字体设置 ---
def set_chinese_font():
    """
    自动查找并设置可用的中文字体。
    """
    # 常见中文字体列表 (macOS, Windows, Linux)
    font_list = ['PingFang SC', 'STHeiti', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

    import matplotlib.font_manager as fm
    for font_name in font_list:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 中文字体设置成功: {font_name}")
            return
    print("⚠️ 未找到支持的中文字体，图表中的中文可能显示为方框。")


# 立即设置字体
set_chinese_font()

# 导入您原始文件中定义的类和绘图函数
try:
    from 第二问 import NIPTFinalIntegratedModel, plot_final_strategy, plot_sensitivity_analysis, plot_tradeoff_curves
except ImportError:
    print("错误：无法从 '第二问.py' 导入所需模块。")
    exit()


# --- 1. 绘制BMI分布与分组图 (无需修改) ---
def plot_bmi_distribution(model):
    if model.data is None or '孕妇BMI' not in model.data.columns: return
    plt.figure(figsize=(8, 5))
    sns.histplot(model.data['孕妇BMI'], kde=True, color="skyblue", bins=30, line_kws={'linewidth': 2, 'color': 'navy'})
    boundaries = model.final_strategy.get('solution', [])[:model.final_strategy.get('k', 3) - 1]
    if len(boundaries) > 0:
        boundaries = np.sort(boundaries)
        for i, b in enumerate(boundaries):
            plt.axvline(x=b, color='red', linestyle='--', linewidth=2,
                        label=f'分组边界 {i + 1}: {b:.2f}' if i == 0 else f'_分组边界 {i + 1}: {b:.2f}')
    plt.title("孕妇BMI分布及最优分组边界", fontsize=16, weight='bold')
    plt.xlabel("孕妇BMI", fontsize=12);
    plt.ylabel("孕妇数量", fontsize=12)
    if len(boundaries) > 0: plt.legend()
    sns.despine();
    plt.tight_layout()
    plt.savefig("bmi_distribution_with_boundaries.png", dpi=300)
    print("成功保存 'bmi_distribution_with_boundaries.png'")


# --- 2. [已修复] 绘制LMM模型系数图 ---
def plot_lmm_coefficients(model):
    """
    可视化线性混合模型(LMM)中各变量的系数及其95%置信区间。
    """
    if model.model_results is None:
        print("未找到LMM模型结果，无法绘制系数图。")
        return

    # [修复] 使用更稳健的方法解析模型摘要
    try:
        lmm_summary_html = model.model_results.summary().tables[1].as_html()
        summary_df = pd.read_html(lmm_summary_html, header=0, index_col=0)[0]
    except Exception as e:
        print(f"解析LMM摘要时出错: {e}")
        return

    summary_df = summary_df.drop('Intercept')
    summary_df['significant'] = summary_df['P>|z|'] < 0.05
    colors = summary_df['significant'].map({True: 'crimson', False: 'gray'})

    plt.figure(figsize=(10, 6))
    err = summary_df['Coef.'] - summary_df['[0.025']
    plt.barh(summary_df.index, summary_df['Coef.'], xerr=err, color=colors, alpha=0.8, capsize=5, edgecolor='black')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("系数 (Coefficient)", fontsize=12)
    plt.ylabel("模型变量", fontsize=12)
    plt.title("LMM模型各变量影响系数", fontsize=16, weight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    sns.despine();
    plt.tight_layout()
    plt.savefig("lmm_coefficients.png", dpi=300)
    print("成功保存 'lmm_coefficients.png'")


# --- 3. [已修复] 绘制最终策略决策空间图 ---
def plot_decision_space(model):
    if model.data is None or not model.final_strategy: return

    k = model.final_strategy['k']
    solution = model.final_strategy['solution']
    min_bmi, max_bmi = model.data['孕妇BMI'].min(), model.data['孕妇BMI'].max()
    boundaries = np.sort(solution[:k - 1])
    timepoints = solution[k - 1:]
    all_boundaries = [min_bmi] + list(boundaries) + [max_bmi]

    plot_data = model.data.copy()
    # 使用pd.cut进行分组打标
    plot_data['group'] = pd.cut(plot_data['孕妇BMI'], bins=all_boundaries, labels=range(k), right=True,
                                include_lowest=True)

    plt.figure(figsize=(11, 7))
    sns.scatterplot(data=plot_data, x='孕周数值', y='Y染色体浓度', hue='group',
                    palette='viridis', alpha=0.6, s=20)

    plt.axhline(0.04, color='red', linestyle='--', linewidth=2, label='4% 浓度阈值')
    for i in range(k):
        plt.axvline(timepoints[i], linestyle='-', linewidth=2,
                    color=sns.color_palette('viridis', k)[i],
                    label=f'组{i + 1} 推荐孕周 ({timepoints[i]:.1f}w)')
    plt.title("最终策略决策空间", fontsize=16, weight='bold')
    plt.xlabel("孕周 (周)", fontsize=12)
    plt.ylabel("Y染色体浓度", fontsize=12)
    plt.legend(title='BMI 分组', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, max(0.2, plot_data['Y染色体浓度'].quantile(0.99)))
    sns.despine();
    plt.tight_layout()
    plt.savefig("decision_space.png", dpi=300)
    print("成功保存 'decision_space.png'")


# --- 主函数 ---
def main():
    print("初始化并加载 'model_results.pkl'...")
    analyzer = NIPTFinalIntegratedModel()
    load_status = analyzer.load_results()
    if load_status == 0 or not analyzer.final_strategy:
        print("\n错误：未能加载有效结果。")
        return
    print("结果加载成功！开始生成所有图表...")

    plot_bmi_distribution(analyzer)
    plot_lmm_coefficients(analyzer)
    plot_decision_space(analyzer)

    print("\n--- 重新生成原有图表 ---")
    plot_final_strategy(analyzer)
    print("成功保存 'final_strategy_pub.png'")
    plot_sensitivity_analysis(analyzer)
    print("成功保存 'sensitivity_pub.png'")
    plot_tradeoff_curves(analyzer)
    print("成功保存 'tradeoff_curve_pub.png'")

    print("\n🎉 所有图表均已成功生成！")


if __name__ == '__main__':
    main()