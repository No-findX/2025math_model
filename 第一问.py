import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')


def load_data():
    data = pd.read_csv('processed_male_data.csv')
    print(f"成功加载数据: {data.shape}")
    return data


def calculate_pseudo_r_squared(model_results):
    # 计算伪R方 (Nakagawa & Schielzeth's R-squared).

    fixed_effects_variance = np.var(np.dot(model_results.model.exog, model_results.fe_params))
    random_effects_variance = float(model_results.cov_re.iloc[0, 0])
    residual_variance = model_results.scale
    total_variance = fixed_effects_variance + random_effects_variance + residual_variance
    marginal_r2 = fixed_effects_variance / total_variance
    conditional_r2 = (fixed_effects_variance + random_effects_variance) / total_variance
    return marginal_r2, conditional_r2


def final_mixed_effects_model(data):
    # 使用statsmodels
    print("\n" + "#" * 80)
    print("线性混合效应模型分析 (Second Refined LMM) ")
    print("#" * 80)

    # 数据准备
    data['y_logit'] = np.log(data['Y染色体浓度'] / (1 - data['Y染色体浓度']))

    data.rename(columns={
        'X染色体浓度_标准': 'X_conc_std',
        '18号染色体的Z值_标准': 'Z18_std',
        'Y染色体的Z值_标准': 'ZY_std',
        '原始读段数_标准': 'raw_reads_std',
        '被过滤掉读段数的比例_标准': 'filtered_rate_std'
    }, inplace=True)

    # 构建模型公式和变量列表
    final_formula = """
    y_logit ~ X_conc_std + Z18_std + 孕妇BMI + 
              raw_reads_std + filtered_rate_std + 孕周数值
    """
    required_cols = [col.strip() for col in final_formula.split('~')[1].replace('\n', ' ').split('+')]

    # 数据验证和清洗步骤
    print("\n正在验证模型所需变量的数据类型...")
    initial_rows = len(data)
    cols_to_validate = ['y_logit'] + required_cols

    for col in cols_to_validate:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(subset=cols_to_validate, inplace=True)
    final_rows = len(data)

    # 拟合模型
    model = smf.mixedlm(final_formula, data, groups=data["孕妇代码"])
    results = model.fit()
    print(results.summary())

    print("\n" + "=" * 50)
    print(" 模型整体检验 (F-test and Pseudo R-squared) ")
    print("=" * 50)

    # F检验：检验除截距外所有固定效应系数是否联合为0
    fixed_effects_terms = results.fe_params.index[1:]  # Exclude intercept
    hypothesis_string = " = ".join(fixed_effects_terms) + " = 0"
    print(f"\n• F检验原假设 (H0): {hypothesis_string}")

    f_test_result = results.f_test(hypothesis_string)

    if hasattr(f_test_result.fvalue, '__len__') and len(f_test_result.fvalue.shape) > 0:
        # 如果fvalue是数组类型
        if f_test_result.fvalue.shape == ():
            # 如果是标量数组
            f_value = float(f_test_result.fvalue)
        elif len(f_test_result.fvalue.shape) == 2:
            # 如果是二维数组
            f_value = f_test_result.fvalue[0][0]
        elif len(f_test_result.fvalue.shape) == 1:
            # 如果是一维数组
            f_value = f_test_result.fvalue[0]
        else:
            f_value = float(f_test_result.fvalue)
    else:
        # 如果fvalue直接就是标量
        f_value = float(f_test_result.fvalue)

    p_value_f = f_test_result.pvalue
    print(f"  - F-statistic = {f_value:.4f}")
    print(f"  - p-value = {p_value_f:.6f}")
    if p_value_f < 0.05:
        print("  - 结论: 模型的固定效应（自变量）整体上是高度显著的 (p < 0.05)。")
    else:
        print("  - 结论: 模型的固定效应（自变量）整体上不显著 (p >= 0.05)。")

        # 伪R方
    marginal_r2, conditional_r2 = calculate_pseudo_r_squared(results)
    print(f"\n• 模型拟合优度 (Pseudo R-squared):")
    print(f"  - 边际 R² (Marginal R²)   = {marginal_r2:.4f}")
    print(f"    (模型中的自变量(固定效应)解释了约 {marginal_r2 * 100:.2f}% 的方差)")
    print(f"  - 条件 R² (Conditional R²) = {conditional_r2:.4f}")
    print(f"    (自变量和孕妇个体差异(固定+随机效应)共同解释了约 {conditional_r2 * 100:.2f}% 的方差)")


def main():
    """主程序"""
    data = load_data()
    if data is None:
        return

    final_mixed_effects_model(data)


if __name__ == "__main__":
    main()
