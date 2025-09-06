import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, stats
import statsmodels.formula.api as smf
import warnings
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class NIPTFinalIntegratedModel:
    def __init__(self, data_path='processed_male_data.csv'):
        self.data_path = data_path
        self.data = None
        self.model_results = None
        self.optimization_results_by_k = {}
        self.final_strategy = {}
        self.sensitivity_results = {}

    def load_and_prepare_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            self.data = self.data[(self.data['Y染色体浓度'] > 0) & (self.data['Y染色体浓度'] < 1)].copy()
            required_cols = ['孕妇BMI', '孕周数值', '孕妇代码', 'X染色体浓度_标准', '18号染色体的Z值_标准',
                             '原始读段数_标准', '被过滤掉读段数的比例_标准']
            self.data.dropna(subset=required_cols, inplace=True)
            return True
        except FileNotFoundError:
            return False

    def fit_baseline_model(self):
        self.data['y_logit'] = np.log(self.data['Y染色体浓度'] / (1 - self.data['Y染色体浓度']))
        rename_dict = {
            'X染色体浓度_标准': 'X_conc_std', '18号染色体的Z值_标准': 'Z18_std',
            '原始读段数_标准': 'raw_reads_std', '被过滤掉读段数的比例_标准': 'filtered_rate_std'
        }
        self.data.rename(columns=rename_dict, inplace=True)
        formula = "y_logit ~ X_conc_std + Z18_std + 孕妇BMI + raw_reads_std + filtered_rate_std + 孕周数值"
        try:
            model = smf.mixedlm(formula, self.data, groups=self.data["孕妇代码"])
            self.model_results = model.fit()
            return True
        except Exception as e:
            return False

    def calculate_group_success_prob(self, week, group_data):
        if group_data.empty: return 0
        group_avg = group_data[['X_conc_std', 'Z18_std', '孕妇BMI', 'raw_reads_std', 'filtered_rate_std']].mean()
        params = self.model_results.fe_params
        pred_logit_mean = params['Intercept'] + (group_avg * params.drop(['Intercept', '孕周数值'])).sum() + params[
            '孕周数值'] * week
        target_logit = np.log(0.04 / 0.96)
        random_effect_var = float(self.model_results.cov_re.iloc[0, 0])
        if random_effect_var <= 1e-9: 
            if pred_logit_mean >= target_logit:
                return 1.0
            else:
                return 0.0
        z_score = (target_logit - pred_logit_mean) / np.sqrt(random_effect_var)
        return 1 - stats.norm.cdf(z_score)

    def calculate_risk_score(self, week, success_prob):
        alpha = 0.4
        if week <= 12:
            time_risk = 0.1
        elif week <= 27:
            time_risk = 0.1 + (0.9 - 0.1) * (1 - np.exp(-alpha * (week - 12)))
        else:
            time_risk = 0.9
        failure_risk = 1 - success_prob
        detection_time_risk = (week - 10) / 15
        return 0.3 * failure_risk + 0.5 * time_risk + 0.2 * detection_time_risk

    def objective_function_wrapper(self, n_groups, success_threshold):
        min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
        min_group_size = len(self.data) * 0.05

        def objective(x):
            boundaries = np.sort(x[:n_groups - 1])
            timepoints = x[n_groups - 1:]
            if not np.all(timepoints >= 10) or not np.all(timepoints <= 25): return 1e6
            all_boundaries = [min_bmi] + list(boundaries) + [max_bmi]
            total_risk = 0
            for i in range(n_groups):
                mask = (self.data['孕妇BMI'] >= all_boundaries[i]) & (self.data['孕妇BMI'] < all_boundaries[i + 1])
                if i == n_groups - 1: 
                    mask = (self.data['孕妇BMI'] >= all_boundaries[i]) & (self.data['孕妇BMI'] <= all_boundaries[i + 1])
                group_data = self.data[mask]
                if len(group_data) < min_group_size: return 1e6
                success_prob = self.calculate_group_success_prob(timepoints[i], group_data)
                if success_prob < success_threshold: return 1e6
                risk_score = self.calculate_risk_score(timepoints[i], success_prob)
                weight = len(group_data) / len(self.data)
                total_risk += weight * risk_score
            return total_risk

        return objective

    def run_hierarchical_optimization(self, k_range=[3, 4, 5], s_start=0.90, s_step=0.01, s_max=0.98):
        if not self.load_and_prepare_data() or not self.fit_baseline_model():
            return
        for k in k_range:
            s_range = np.arange(s_start, s_max + s_step, s_step)
            tradeoff_curve = []
            for s_threshold in s_range:
                objective_func = self.objective_function_wrapper(k, s_threshold)
                min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
                bounds = [(min_bmi, max_bmi)] * (k - 1) + [(10, 25)] * k
                try:
                    result = optimize.differential_evolution(
                        objective_func, bounds, strategy='best1bin', maxiter=300,
                        popsize=15, tol=0.01, seed=42, disp=False
                    )
                    if result.success and result.fun < 1e5:
                        tradeoff_curve.append({'s_threshold': s_threshold, 'min_risk': result.fun, 'solution': result.x})
                    else:
                        break
                except Exception as e:
                    print(f" 优化失败: {e}")
                    continue
            if tradeoff_curve:
                self.optimization_results_by_k[k] = tradeoff_curve
        self.analyze_and_present_final_solution()

    def analyze_and_present_final_solution(self):
        if not self.optimization_results_by_k:
            print("\n优化失败，未能找到任何有效策略。")
            return
        print("\n\n最终决策分析")
        aic_results = {}
        for k, curve in self.optimization_results_by_k.items():
            if not curve: continue
            best_result_for_k = min(curve, key=lambda x: x['min_risk'])
            representative_risk = best_result_for_k['min_risk']
            n_samples, num_params = len(self.data), 2 * k - 1
            aic_score = 2 * num_params + n_samples * np.log(representative_risk)
            aic_results[k] = aic_score

        if not aic_results:
            print("\n无法计算AIC")
            return

        best_k = min(aic_results, key=aic_results.get)
        print(f"\nK={best_k} AIC最小，理论最优")

        best_k_curve = self.optimization_results_by_k[best_k]
        s_values = [res['s_threshold'] for res in best_k_curve]
        r_values = [res['min_risk'] for res in best_k_curve]

        if len(s_values) > 1:
            marginal_cost = np.diff(r_values) / np.diff(s_values)
            elbow_index = np.argmax(marginal_cost) if len(marginal_cost) > 0 else 0
            optimal_s = s_values[elbow_index]
        elif len(s_values) == 1:
            optimal_s = s_values[0]

        print(f"\n【决策分析 (K={best_k})】")
        print(f"对最优分组K={best_k}，最佳平衡点为 S* ≈ {optimal_s:.4%}")

        best_result = next(res for res in best_k_curve if res['s_threshold'] == optimal_s)

        self.final_strategy = {'k': best_k, 's': optimal_s, 'solution': best_result['solution']}
        self.unpack_and_display_strategy(self.final_strategy)

    def unpack_and_display_strategy(self, strategy):
        k = strategy['k']
        solution = strategy['solution']
        min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
        boundaries = np.sort(solution[:k - 1])
        timepoints = solution[k - 1:]
        all_boundaries = [min_bmi] + list(boundaries) + [max_bmi]

        print(f"\n【最终最优策略 (K*={k}, S*≈{strategy['s']:.0%})】")
        groups_info = []
        for i in range(k):
            bmi_range = (all_boundaries[i], all_boundaries[i + 1])
            mask = (self.data['孕妇BMI'] >= bmi_range[0]) & (self.data['孕妇BMI'] < bmi_range[1])
            if i == k - 1: mask = (self.data['孕妇BMI'] >= bmi_range[0]) & (self.data['孕妇BMI'] <= bmi_range[1])
            group_data = self.data[mask]
            optimal_week = timepoints[i]
            success_prob = self.calculate_group_success_prob(optimal_week, group_data)
            groups_info.append({'bmi_range': bmi_range, 'week': optimal_week, 'prob': success_prob})
        return groups_info

    def run_sensitivity_analysis(self, error_level=0.05, n_simulations=100):
        if not self.final_strategy:
            print("\n无最终策略")
            return

        print(f"\n分析基准策略：K={self.final_strategy['k']}, S≈{self.final_strategy['s']:.4%}")

        k = self.final_strategy['k']
        s_threshold = self.final_strategy['s']
        baseline_groups = self.unpack_and_display_strategy(self.final_strategy)

        simulated_weeks_all_groups = [[] for _ in range(k)]

        for i in range(n_simulations):
            sim_data = self.data.copy()
            # 模拟BMI测量误差
            sim_data['孕妇BMI'] *= (1 + np.random.normal(0, error_level, len(sim_data)))

            # 对每个基准分组，重新优化时点
            for group_idx in range(k):
                group_bmi_range = baseline_groups[group_idx]['bmi_range']
                mask = (sim_data['孕妇BMI'] >= group_bmi_range[0]) & (sim_data['孕妇BMI'] <= group_bmi_range[1])
                group_data_sim = sim_data[mask]

                if group_data_sim.empty: continue

                # 使用网格搜索快速找到该次模拟的最优时点
                weeks_grid = np.arange(10, 25.1, 0.5)
                risks = []
                for week in weeks_grid:
                    prob = self.calculate_group_success_prob(week, group_data_sim)
                    if prob >= s_threshold:
                        risks.append((self.calculate_risk_score(week, prob), week))

                if risks:
                    best_risk, best_week = min(risks, key=lambda x: x[0])
                    simulated_weeks_all_groups[group_idx].append(best_week)

        print("\n敏感性分析结果：")
        for i in range(k):
            baseline_week = baseline_groups[i]['week']
            sim_weeks = simulated_weeks_all_groups[i]
            if not sim_weeks:
                print(f"  组{i + 1}: 模拟中未能找到可行解。")
                continue
            mean_week, std_week = np.mean(sim_weeks), np.std(sim_weeks)
            ci_lower, ci_upper = np.percentile(sim_weeks, [2.5, 97.5])
            self.sensitivity_results[i] = {
                'baseline_week': baseline_week, 'mean_week': mean_week,
                'std_week': std_week, 'ci_95': (ci_lower, ci_upper)
            }
            print(f"  组{i + 1}: 基准时点={baseline_week:.1f}周, "
                  f"引入误差后时点= {mean_week:.1f} ± {std_week:.2f}周 (95% CI: [{ci_lower:.1f}, {ci_upper:.1f}])")

    def run(self, k_range=[3, 4, 5], s_start=0.90, s_step=0.01, s_max=0.98):
        if not self.load_and_prepare_data() or not self.fit_baseline_model():
            return

        # 核心优化
        self.run_hierarchical_optimization(k_range, s_start, s_step, s_max)

        # 对最优结果进行敏感性分析
        if self.final_strategy:
            self.run_sensitivity_analysis()

    def save_results(self, filename="model_results.pkl"):
        """保存模型结果到文件"""
        with open(filename, "wb") as f:
            pickle.dump({
                "data": self.data,
                "model_results": self.model_results,
                "optimization_results_by_k": self.optimization_results_by_k,
                "final_strategy": self.final_strategy,
                "sensitivity_results": self.sensitivity_results
            }, f)
    
    def load_results(self, filename="model_results.pkl"):
        """从文件加载模型结果"""
        try:
            with open(filename, "rb") as f:
                results = pickle.load(f)
                if not results:
                    return 0
                self.data = results["data"]
                self.model_results = results["model_results"]
                self.optimization_results_by_k = results["optimization_results_by_k"]
                self.final_strategy = results["final_strategy"]
                self.sensitivity_results = results["sensitivity_results"]
        except FileNotFoundError:
            return 0

def plot_tradeoff_curves(model):
    plt.figure(figsize=(7,5))
    palette = sns.color_palette("deep")
    has_data = False

    for idx, (k, curve) in enumerate(model.optimization_results_by_k.items()):
        if not curve: continue
        s_values = [res['s_threshold'] for res in curve]
        r_values = [res['min_risk'] for res in curve]
        plt.plot(s_values, r_values, marker='o', markersize=4,
                 linewidth=2, label=f"K={k}", color=palette[idx])
        has_data = True

    if not has_data:
        print("没有数据可绘制，请检查优化是否成功")
        return

    plt.xlabel("成功率阈值 S", fontsize=12)
    plt.ylabel("最小风险分数", fontsize=12)
    plt.title("风险-成功率权衡曲线", fontsize=14, weight="bold")
    plt.legend(frameon=False)
    sns.despine()  # 去掉上和右边框
    plt.tight_layout()
    plt.savefig("tradeoff_curve_pub.png", dpi=300)

def plot_final_strategy(model):
    groups_info = model.unpack_and_display_strategy(model.final_strategy)
    df = pd.DataFrame([
        {"组别": f"组{i+1}", "推荐孕周": g['week'], "BMI范围": g['bmi_range']}
        for i, g in enumerate(groups_info)
    ])

    plt.figure(figsize=(6,5))
    bars = plt.bar(df["组别"], df["推荐孕周"], color="steelblue", edgecolor="black")
    for i, row in df.iterrows():
        plt.text(i, row["推荐孕周"]+0.2, f"{row['推荐孕周']:.1f}周", 
                 ha='center', fontsize=10)
    plt.ylabel("推荐孕周", fontsize=12)
    plt.title("最优分组策略", fontsize=14, weight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig("final_strategy_pub.png", dpi=300)

def plot_sensitivity_analysis(model):
    if not model.sensitivity_results:
        print("未找到敏感性分析结果")
        return

    df = pd.DataFrame([
        {"组别": f"组{i+1}", 
         "基准孕周": res['baseline_week'],
         "均值孕周": res['mean_week'],
         "下界": res['ci_95'][0], 
         "上界": res['ci_95'][1]}
        for i, res in model.sensitivity_results.items()
    ])

    plt.figure(figsize=(6,5))
    bars = plt.bar(df["组别"], df["均值孕周"],
                   yerr=[df["均值孕周"]-df["下界"], df["上界"]-df["均值孕周"]],
                   capsize=4, color="lightgray", edgecolor="black",
                   label="模拟均值 ±95%CI")
    plt.scatter(df["组别"], df["基准孕周"], color="red", zorder=5, label="基准孕周")
    
    plt.ylabel("推荐孕周", fontsize=12)
    plt.title("敏感性分析", fontsize=14, weight="bold")
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig("sensitivity_pub.png", dpi=300)

if __name__ == '__main__':
    analyzer = NIPTFinalIntegratedModel()

    r1 = analyzer.load_results()
    if r1 == 0:
        analyzer.run(k_range=[3, 4, 5], s_start=0.90, s_step=0.001, s_max=0.97)
        analyzer.save_results()
    
    plot_final_strategy(analyzer)
    plot_sensitivity_analysis(analyzer)
    plot_tradeoff_curves(analyzer)