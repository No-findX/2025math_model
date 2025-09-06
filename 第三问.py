# NIPT数学建模竞赛 - 第三问完整解决方案
# 包含主要建模和敏感性分析，优化版本，无图形输出

import pandas as pd
import numpy as np
from scipy import optimize, stats
import statsmodels.formula.api as smf
import warnings
import time
from functools import lru_cache

warnings.filterwarnings('ignore')


class NIPTCompleteModel:
    def __init__(self, data_path='processed_male_data.csv'):
        self.data_path = data_path
        self.data = None
        self.model_results = None
        self.successful_strategies = []

        # 预计算相关
        self.week_grid = np.arange(10, 25.1, 0.5)  # 10.0, 10.5, 11.0, ..., 25.0
        self.prediction_table = None  # 预计算表：[个体数 × 时点数]
        self.individual_ids = None

        # 性能统计
        self._objective_eval_count = 0

        # 敏感性分析相关
        self.sensitivity_results = {}
        self.monte_carlo_results = {}

        # 检测误差参数设置
        self.error_scenarios = {
            'measurement_noise': [0.01, 0.02, 0.03, 0.05],  # 测量噪声标准差
            'systematic_bias': [-0.005, -0.002, 0.002, 0.005],  # 系统性偏差
            'threshold_uncertainty': [0.035, 0.038, 0.042, 0.045]  # 阈值不确定性
        }

    def load_and_prepare_data(self):
        print("第三问：NIPT检测时点优化与敏感性分析")
        try:
            self.data = pd.read_csv(self.data_path)
            self.data = self.data[(self.data['Y染色体浓度'] > 0) & (self.data['Y染色体浓度'] < 1)].copy()
            required_cols = ['孕妇BMI', '孕周数值', '孕妇代码', '身高', '体重', '年龄', 'X染色体浓度_标准',
                             '18号染色体的Z值_标准', '原始读段数_标准', '被过滤掉读段数的比例_标准']
            self.data.dropna(subset=required_cols, inplace=True)

            # 为每个个体分配唯一ID
            self.data.reset_index(drop=True, inplace=True)
            self.individual_ids = self.data.index.tolist()

            return True
        except FileNotFoundError:
            print("错误：找不到数据文件")
            return False

    def fit_upgraded_model(self):

        self.data['y_logit'] = np.log(self.data['Y染色体浓度'] / (1 - self.data['Y染色体浓度']))
        rename_dict = {
            'X染色体浓度_标准': 'X_conc_std', '18号染色体的Z值_标准': 'Z18_std',
            '原始读段数_标准': 'raw_reads_std', '被过滤掉读段数的比例_标准': 'filtered_rate_std'
        }
        self.data.rename(columns=rename_dict, inplace=True)

        formula = "y_logit ~ X_conc_std + Z18_std + 身高 + 体重 + 年龄 + raw_reads_std + filtered_rate_std + 孕周数值"

        try:
            model = smf.mixedlm(formula, self.data, groups=self.data["孕妇代码"])
            self.model_results = model.fit()

            # 提取模型参数
            self._model_params = self.model_results.fe_params
            self._random_effects = self.model_results.random_effects

            return True
        except Exception as e:
            print(f"模型拟合失败: {e}")
            return False

    def precompute_all_predictions(self):
        """核心优化：预计算所有个体在所有时点的Y染色体浓度"""

        n_individuals = len(self.data)
        n_timepoints = len(self.week_grid)
        start_time = time.time()

        self.prediction_table = np.zeros((n_individuals, n_timepoints))

        fixed_base = (self._model_params['Intercept'] +
                      self._model_params['X_conc_std'] * self.data['X_conc_std'].values +
                      self._model_params['Z18_std'] * self.data['Z18_std'].values +
                      self._model_params['身高'] * self.data['身高'].values +
                      self._model_params['体重'] * self.data['体重'].values +
                      self._model_params['年龄'] * self.data['年龄'].values +
                      self._model_params['raw_reads_std'] * self.data['raw_reads_std'].values +
                      self._model_params['filtered_rate_std'] * self.data['filtered_rate_std'].values)

        # 修复随机效应提取
        random_effects_scalar_map = {group: effect.iloc[0] for group, effect in self._random_effects.items()}
        random_effects_values = self.data['孕妇代码'].map(random_effects_scalar_map).values
        fixed_base = fixed_base + random_effects_values

        fixed_base = np.array(fixed_base)

        week_coef = self._model_params['孕周数值']
        for j, week in enumerate(self.week_grid):
            logit_values = fixed_base + week_coef * week
            self.prediction_table[:, j] = 1 / (1 + np.exp(-logit_values))

        return True

    def get_prediction_from_table(self, individual_idx, week):
        """从预计算表中快速获取预测值"""
        week_idx = np.argmin(np.abs(self.week_grid - week))
        return self.prediction_table[individual_idx, week_idx]

    def calculate_global_success_rate_fast(self, group_assignments, timepoints):
        """快速计算全局成功率"""
        success_count = 0
        total_count = len(self.data)
        for i in range(total_count):
            group_id = group_assignments[i]
            assigned_week = timepoints[group_id]
            predicted_concentration = self.get_prediction_from_table(i, assigned_week)
            if predicted_concentration >= 0.04:
                success_count += 1
        return success_count / total_count

    def calculate_group_success_prob_fast(self, week, group_indices):
        """快速计算群体成功概率"""
        if len(group_indices) == 0:
            return 0
        week_idx = np.argmin(np.abs(self.week_grid - week))
        group_predictions = self.prediction_table[group_indices, week_idx]
        success_rate = np.mean(group_predictions >= 0.04)
        return success_rate

    def calculate_risk_score(self, week, success_prob):
        """风险评分计算"""
        alpha = 0.6
        if week <= 12:
            time_risk = 0.1
        elif week <= 27:
            time_risk = 0.1 + (0.9 - 0.1) * (1 - np.exp(-alpha * (week - 12)))
        else:
            time_risk = 0.9
        failure_risk = 1 - success_prob
        detection_time_risk = (week - 10) / 15
        return 0.5 * failure_risk + 0.3 * time_risk + 0.2 * detection_time_risk

    def objective_function_wrapper(self, n_groups, global_success_threshold):
        """优化版目标函数：使用预计算表"""
        min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
        min_group_size = len(self.data) * 0.05
        bmi_values = self.data['孕妇BMI'].values

        def objective(x):
            self._objective_eval_count += 1
            timepoints = x[n_groups - 1:]

            if not np.all((timepoints >= 10) & (timepoints <= 25)): return 1e6
            if not np.all(np.diff(timepoints) >= -1.0): return 1e6

            boundaries = np.sort(x[:n_groups - 1])
            all_boundaries = [min_bmi] + list(boundaries) + [max_bmi]

            group_indices_list = []
            group_assignments = np.zeros(len(self.data), dtype=int)
            for i in range(n_groups):
                if i == n_groups - 1:
                    mask = (bmi_values >= all_boundaries[i]) & (bmi_values <= all_boundaries[i + 1])
                else:
                    mask = (bmi_values >= all_boundaries[i]) & (bmi_values < all_boundaries[i + 1])
                group_indices = np.where(mask)[0]
                if len(group_indices) < min_group_size: return 1e6
                group_indices_list.append(group_indices)
                group_assignments[group_indices] = i

            if self.calculate_global_success_rate_fast(group_assignments, timepoints) < global_success_threshold:
                return 1e6

            total_risk, penalty = 0, 0
            min_group_success_rate = 0.90

            for i, group_indices in enumerate(group_indices_list):
                group_success_prob = self.calculate_group_success_prob_fast(timepoints[i], group_indices)
                if group_success_prob < min_group_success_rate:
                    penalty += 10 * (min_group_success_rate - group_success_prob)
                risk_score = self.calculate_risk_score(timepoints[i], group_success_prob)
                total_risk += (len(group_indices) / len(self.data)) * risk_score

            for i in range(len(timepoints) - 1):
                if timepoints[i + 1] < timepoints[i]:
                    penalty += 10.0 * (timepoints[i] - timepoints[i + 1]) ** 2

            return total_risk + penalty

        return objective

    def generate_smart_initial_guess(self, n_groups, s_threshold):
        """智能初始解生成策略"""
        try:
            if self.successful_strategies:
                similar_strategies = [s for s in self.successful_strategies if s['k'] == n_groups]
                if similar_strategies:
                    base_solution = similar_strategies[-1]['solution']
                    boundaries = base_solution[:n_groups - 1]
                    timepoints = np.clip(base_solution[n_groups - 1:] + 0.3, 10, 25)
                    return np.concatenate([boundaries, timepoints])

            _, group_bins = pd.qcut(self.data['孕妇BMI'], q=n_groups, labels=False, retbins=True, duplicates='drop')
            initial_boundaries = group_bins[1:-1]
            initial_timepoints = []
            for i in range(n_groups):
                mask = (self.data['孕妇BMI'] >= group_bins[i]) & (self.data['孕妇BMI'] <= group_bins[i + 1])
                group_indices = np.where(mask)[0]
                best_week = 25.0
                for week in [12, 14, 16, 18, 20, 22, 24]:
                    if self.calculate_group_success_prob_fast(week, group_indices) >= s_threshold:
                        best_week = week
                        break
                initial_timepoints.append(best_week)

            for i in range(1, len(initial_timepoints)):
                if initial_timepoints[i] < initial_timepoints[i - 1]:
                    initial_timepoints[i] = initial_timepoints[i - 1]
            return np.concatenate([initial_boundaries, initial_timepoints])
        except Exception as e:
            print(f"    智能初始解生成失败: {e}")
            return None

    def run_optimized_search(self, k_range=[3, 4, 5], s_start=0.93, s_end=0.91):
        """优化版搜索：使用预计算表"""
        if not self.load_and_prepare_data() or not self.fit_upgraded_model(): return
        if not self.precompute_all_predictions(): return

        for k in k_range:
            step, max_attempts = 0.005, 3
            current_s = s_start
            while current_s >= s_end:
                self._objective_eval_count = 0
                objective_func = self.objective_function_wrapper(k, current_s)
                min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
                bounds = [(min_bmi, max_bmi)] * (k - 1) + [(10, 25)] * k
                best_result, best_risk = None, float('inf')

                for attempt in range(max_attempts):
                    initial_guess = self.generate_smart_initial_guess(k, current_s)
                    try:
                        result = optimize.differential_evolution(
                            objective_func, bounds, x0=initial_guess, strategy='best1bin',
                            maxiter=60, popsize=10, tol=0.015, seed=42 + attempt,
                            disp=False, workers=1, updating='deferred'
                        )
                        if result.success and result.fun < best_risk:
                            best_result, best_risk = result, result.fun
                    except Exception as e:
                        print(f" 异常: {str(e)[:50]}...")

                if best_result and best_risk < 1e5:
                    self.successful_strategies.append({
                        'k': k, 's_threshold': current_s, 'min_risk': best_risk,
                        'solution': best_result.x
                    })
                    break
                current_s -= step

        self.analyze_and_present_final_solution()

    def analyze_and_present_final_solution(self):
        """结果分析和展示"""
        if not self.successful_strategies:
            print("\n优化搜索失败，在所有尝试的K和S组合下均未找到可行解。")
            return

        aic_results = {}
        print("\n【模型选择：寻找最优分组数 K* (基于AIC)】")
        for strategy in self.successful_strategies:
            k, risk = strategy['k'], strategy['min_risk']
            n_samples, num_params = len(self.data), 2 * k - 1
            aic_score = 2 * num_params + n_samples * np.log(risk)
            aic_results[k] = aic_score

        if not aic_results:
            print("\n未能计算任何AIC分数，无法选择最优K。")
            return

        best_k = min(aic_results, key=aic_results.get)
        print(f"\nK={best_k} 使得AIC分数最小，是理论上的最优分组数。")
        best_strategy = next(s for s in self.successful_strategies if s['k'] == best_k)
        self.unpack_and_display_strategy(best_strategy)

        return best_strategy

    def unpack_and_display_strategy(self, strategy):
        """解析和展示策略"""
        k, s_threshold = strategy['k'], strategy['s_threshold']
        solution = strategy['solution']
        min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
        boundaries = np.sort(solution[:k - 1])
        timepoints = solution[k - 1:]
        all_boundaries = [min_bmi] + list(boundaries) + [max_bmi]

        print(f"\n【最终最优策略 (K*={k}, 最高可行成功率≈{s_threshold:.1%})】")
        for i in range(k):
            bmi_range = (all_boundaries[i], all_boundaries[i + 1])
            if i == k - 1:
                mask = (self.data['孕妇BMI'] >= bmi_range[0]) & (self.data['孕妇BMI'] <= bmi_range[1])
            else:
                mask = (self.data['孕妇BMI'] >= bmi_range[0]) & (self.data['孕妇BMI'] < bmi_range[1])
            group_indices = np.where(mask)[0]
            optimal_week = timepoints[i]
            group_actual_success_rate = self.calculate_group_success_prob_fast(optimal_week, group_indices)
            print(f"  组{i + 1}: BMI [{bmi_range[0]:.2f}, {bmi_range[1]:.2f}] (样本数: {len(group_indices)})")
            print(f"    -> 推荐时点: {optimal_week:.1f}周, 实际达标比例: {group_actual_success_rate:.1%}")

    # ==================== 敏感性分析部分 ====================

    def add_measurement_noise(self, predictions, noise_std):
        """添加测量噪声"""
        noise = np.random.normal(0, noise_std, predictions.shape)
        noisy_predictions = predictions + noise
        return np.clip(noisy_predictions, 0, 1)

    def add_systematic_bias(self, predictions, bias):
        """添加系统性偏差"""
        biased_predictions = predictions + bias
        return np.clip(biased_predictions, 0, 1)

    def calculate_success_rate_with_threshold(self, predictions, threshold):
        """使用不同阈值计算成功率"""
        return np.mean(predictions >= threshold)

    def monte_carlo_simulation(self, strategy, n_simulations=1000, error_type='all'):
        """蒙特卡洛模拟分析检测误差影响"""
        # 解析策略
        k = strategy['k']
        solution = strategy['solution']
        min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
        boundaries = np.sort(solution[:k - 1])
        timepoints = solution[k - 1:]
        all_boundaries = [min_bmi] + list(boundaries) + [max_bmi]

        # 获取基准预测值
        base_results = self._get_base_predictions(k, all_boundaries, timepoints)

        # 存储模拟结果
        simulation_results = {
            'base_success_rate': base_results['global_success_rate'],
            'base_group_rates': base_results['group_success_rates'],
            'simulations': []
        }

        # 运行蒙特卡洛模拟
        for sim_idx in range(n_simulations):
            sim_result = self._run_single_simulation(
                k, all_boundaries, timepoints, error_type, base_results
            )
            simulation_results['simulations'].append(sim_result)

        # 分析结果
        self._analyze_monte_carlo_results(simulation_results, error_type)

        return simulation_results

    def _get_base_predictions(self, k, all_boundaries, timepoints):
        """获取基准预测结果"""
        bmi_values = self.data['孕妇BMI'].values
        base_results = {
            'group_assignments': np.zeros(len(self.data), dtype=int),
            'group_success_rates': [],
            'group_sizes': []
        }

        # 计算各组的基准成功率
        for i in range(k):
            if i == k - 1:
                mask = (bmi_values >= all_boundaries[i]) & (bmi_values <= all_boundaries[i + 1])
            else:
                mask = (bmi_values >= all_boundaries[i]) & (bmi_values < all_boundaries[i + 1])
            group_indices = np.where(mask)[0]
            base_results['group_assignments'][group_indices] = i

            group_success_rate = self.calculate_group_success_prob_fast(timepoints[i], group_indices)
            base_results['group_success_rates'].append(group_success_rate)
            base_results['group_sizes'].append(len(group_indices))

        # 计算全局基准成功率
        base_results['global_success_rate'] = self.calculate_global_success_rate_fast(
            base_results['group_assignments'], timepoints
        )

        return base_results

    def _run_single_simulation(self, k, all_boundaries, timepoints, error_type, base_results):
        """运行单次模拟"""
        bmi_values = self.data['孕妇BMI'].values
        sim_result = {
            'global_success_rate': 0,
            'group_success_rates': [],
            'error_params': {}
        }

        # 随机选择误差参数
        if error_type in ['measurement', 'all']:
            noise_std = np.random.choice(self.error_scenarios['measurement_noise'])
            sim_result['error_params']['noise_std'] = noise_std
        else:
            noise_std = 0

        if error_type in ['systematic', 'all']:
            bias = np.random.choice(self.error_scenarios['systematic_bias'])
            sim_result['error_params']['bias'] = bias
        else:
            bias = 0

        if error_type in ['threshold', 'all']:
            threshold = np.random.choice(self.error_scenarios['threshold_uncertainty'])
            sim_result['error_params']['threshold'] = threshold
        else:
            threshold = 0.04

        # 计算带误差的成功率
        success_count = 0
        for i in range(k):
            if i == k - 1:
                mask = (bmi_values >= all_boundaries[i]) & (bmi_values <= all_boundaries[i + 1])
            else:
                mask = (bmi_values >= all_boundaries[i]) & (bmi_values < all_boundaries[i + 1])
            group_indices = np.where(mask)[0]

            if len(group_indices) > 0:
                # 获取原始预测值
                week_idx = np.argmin(np.abs(self.week_grid - timepoints[i]))
                original_predictions = self.prediction_table[group_indices, week_idx]

                # 添加误差
                if noise_std > 0:
                    original_predictions = self.add_measurement_noise(original_predictions, noise_std)
                if bias != 0:
                    original_predictions = self.add_systematic_bias(original_predictions, bias)

                # 计算成功率
                group_success_rate = self.calculate_success_rate_with_threshold(original_predictions, threshold)
                sim_result['group_success_rates'].append(group_success_rate)
                success_count += np.sum(original_predictions >= threshold)
            else:
                sim_result['group_success_rates'].append(0)

        sim_result['global_success_rate'] = success_count / len(self.data)

        return sim_result

    def _analyze_monte_carlo_results(self, simulation_results, error_type):
        """分析蒙特卡洛模拟结果"""
        print(f"\n蒙特卡洛敏感性分析结果 (误差类型: {error_type})")

        # 提取模拟结果
        sim_global_rates = [sim['global_success_rate'] for sim in simulation_results['simulations']]
        base_rate = simulation_results['base_success_rate']

        # 全局成功率统计
        mean_rate = np.mean(sim_global_rates)
        std_rate = np.std(sim_global_rates)
        ci_lower = np.percentile(sim_global_rates, 2.5)
        ci_upper = np.percentile(sim_global_rates, 97.5)

        print(f"全局成功率敏感性分析:")
        print(f"  基准值: {base_rate:.4f}")
        print(f"  模拟均值: {mean_rate:.4f} (变化: {mean_rate - base_rate:+.4f})")
        print(f"  标准差: {std_rate:.4f}")
        print(f"  95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  最大下降: {min(sim_global_rates) - base_rate:.4f}")
        print(f"  最大上升: {max(sim_global_rates) - base_rate:.4f}")

        # 分组成功率分析
        k = len(simulation_results['base_group_rates'])
        print(f"\n各组成功率敏感性分析:")
        for i in range(k):
            base_group_rate = simulation_results['base_group_rates'][i]
            sim_group_rates = [sim['group_success_rates'][i] for sim in simulation_results['simulations']]

            group_mean = np.mean(sim_group_rates)
            group_std = np.std(sim_group_rates)
            group_ci_lower = np.percentile(sim_group_rates, 2.5)
            group_ci_upper = np.percentile(sim_group_rates, 97.5)

            print(f"  组{i + 1}: 基准={base_group_rate:.3f}, 均值={group_mean:.3f}±{group_std:.3f}")
            print(f"       95%CI=[{group_ci_lower:.3f}, {group_ci_upper:.3f}]")

        # 存储结果
        self.monte_carlo_results[error_type] = {
            'global_stats': {
                'base': base_rate, 'mean': mean_rate, 'std': std_rate,
                'ci_lower': ci_lower, 'ci_upper': ci_upper
            },
            'all_simulations': sim_global_rates
        }

    def parameter_sensitivity_analysis(self, strategy):
        """参数敏感性分析"""
        print(f"\n参数敏感性分析")

        # 解析策略
        k = strategy['k']
        solution = strategy['solution']
        min_bmi, max_bmi = self.data['孕妇BMI'].min(), self.data['孕妇BMI'].max()
        boundaries = np.sort(solution[:k - 1])
        timepoints = solution[k - 1:]

        base_results = self._get_base_predictions(k, [min_bmi] + list(boundaries) + [max_bmi], timepoints)
        base_rate = base_results['global_success_rate']

        sensitivity_analysis = {}

        # 1. 时点参数敏感性
        print("1. 检测时点敏感性分析:")
        time_sensitivity = []
        for delta in [-2, -1, -0.5, 0.5, 1, 2]:  # 时点变化(周)
            perturbed_timepoints = np.clip(timepoints + delta, 10, 25)
            perturbed_rate = self._calculate_perturbed_success_rate(
                k, [min_bmi] + list(boundaries) + [max_bmi], perturbed_timepoints
            )
            time_sensitivity.append({
                'delta': delta,
                'success_rate': perturbed_rate,
                'change': perturbed_rate - base_rate
            })
            print(f"  时点{delta:+.1f}周: 成功率={perturbed_rate:.4f} (变化{perturbed_rate - base_rate:+.4f})")

        sensitivity_analysis['timepoint'] = time_sensitivity

        # 2. BMI边界敏感性
        if k > 1:  # 只有多组时才有边界
            print(f"\n2. BMI分组边界敏感性分析:")
            boundary_sensitivity = []
            for i, boundary in enumerate(boundaries):
                for delta in [-2, -1, 1, 2]:  # BMI边界变化
                    perturbed_boundaries = boundaries.copy()
                    perturbed_boundaries[i] = np.clip(boundary + delta, min_bmi + 1, max_bmi - 1)
                    perturbed_boundaries = np.sort(perturbed_boundaries)

                    try:
                        perturbed_rate = self._calculate_perturbed_success_rate(
                            k, [min_bmi] + list(perturbed_boundaries) + [max_bmi], timepoints
                        )
                        boundary_sensitivity.append({
                            'boundary_idx': i,
                            'delta': delta,
                            'success_rate': perturbed_rate,
                            'change': perturbed_rate - base_rate
                        })
                        print(f"  边界{i + 1}{delta:+}BMI: 成功率={perturbed_rate:.4f} (变化{perturbed_rate - base_rate:+.4f})")
                    except:
                        pass  # 忽略无效的边界调整

            sensitivity_analysis['boundary'] = boundary_sensitivity

        self.sensitivity_results = sensitivity_analysis
        return sensitivity_analysis

    def _calculate_perturbed_success_rate(self, k, all_boundaries, timepoints):
        """计算扰动后的成功率"""
        bmi_values = self.data['孕妇BMI'].values
        success_count = 0

        for i in range(k):
            if i == k - 1:
                mask = (bmi_values >= all_boundaries[i]) & (bmi_values <= all_boundaries[i + 1])
            else:
                mask = (bmi_values >= all_boundaries[i]) & (bmi_values < all_boundaries[i + 1])
            group_indices = np.where(mask)[0]

            if len(group_indices) > 0:
                week_idx = np.argmin(np.abs(self.week_grid - timepoints[i]))
                predictions = self.prediction_table[group_indices, week_idx]
                success_count += np.sum(predictions >= 0.04)

        return success_count / len(self.data)

    def generate_sensitivity_report(self, strategy):
        """生成敏感性分析报告"""
        print("\n\n误差敏感性分析报告")

        print(f"\n【分析策略】")
        print(f"最优分组数: K={strategy['k']}")
        print(f"目标成功率: {strategy['s_threshold']:.1%}")

        print(f"\n【主要发现】")

        if self.monte_carlo_results:
            # 找出最敏感的误差类型
            sensitivities = {}
            for error_type, results in self.monte_carlo_results.items():
                base = results['global_stats']['base']
                mean = results['global_stats']['mean']
                sensitivities[error_type] = abs(mean - base)

            most_sensitive = max(sensitivities, key=sensitivities.get)
            least_sensitive = min(sensitivities, key=sensitivities.get)

            print(f"1. 最敏感的误差类型: {most_sensitive} (平均影响: {sensitivities[most_sensitive]:.4f})")
            print(f"2. 最稳健的误差类型: {least_sensitive} (平均影响: {sensitivities[least_sensitive]:.4f})")

            # 计算风险指标
            for error_type, results in self.monte_carlo_results.items():
                below_threshold = np.mean(np.array(results['all_simulations']) < strategy['s_threshold'])
                print(f"3. {error_type}误差下低于目标的概率: {below_threshold:.2%}")

        return True

    def run_complete_analysis(self, k_range=[3, 4, 5]):
        """运行完整分析：包括主要建模和敏感性分析"""

        # 1. 运行主要优化模型
        self.run_optimized_search(k_range=k_range, s_start=0.93, s_end=0.91)

        if not self.successful_strategies:
            print("主要建模失败，无法进行敏感性分析")
            return

        # 获取最佳策略
        best_strategy = self.successful_strategies[-1]  # 使用最后一个（通常是最好的）策略

        # 2. 参数敏感性分析
        param_results = self.parameter_sensitivity_analysis(best_strategy)

        # 3. 蒙特卡洛模拟 - 分别分析不同类型的误差

        # 测量噪声影响
        mc_results_measurement = self.monte_carlo_simulation(
            best_strategy, n_simulations=500, error_type='measurement'
        )

        # 系统性偏差影响
        mc_results_systematic = self.monte_carlo_simulation(
            best_strategy, n_simulations=500, error_type='systematic'
        )

        # 阈值不确定性影响
        mc_results_threshold = self.monte_carlo_simulation(
            best_strategy, n_simulations=500, error_type='threshold'
        )

        # 综合误差影响
        mc_results_all = self.monte_carlo_simulation(
            best_strategy, n_simulations=800, error_type='all'
        )

        # 4. 生成综合报告
        self.generate_sensitivity_report(best_strategy)


        return {
            'best_strategy': best_strategy,
            'parameter_sensitivity': param_results,
            'monte_carlo_results': self.monte_carlo_results
        }


# 主程序入口
if __name__ == '__main__':
    # 创建模型实例并运行完整分析
    solver = NIPTCompleteModel()
    results = solver.run_complete_analysis(k_range=[3, 4, 5])

    if results:
        print(f"\n【最终总结】")
        strategy = results['best_strategy']
        print(f"最优分组方案: K={strategy['k']}")
        print(f"最高可达成功率: {strategy['s_threshold']:.1%}")
        print(f"策略风险评分: {strategy['min_risk']:.4f}")
        print(f"敏感性分析完成: 已评估{len(solver.monte_carlo_results)}种误差类型的影响")
    else:
        print("分析失败，请检查数据文件和参数设置")