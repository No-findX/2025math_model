import pandas as pd
import numpy as np
from scipy import stats, optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class GroupResult:
    """分组结果数据类"""
    bmi_range: Tuple[float, float]
    optimal_week: float
    pass_rate: float
    risk_score: float
    sample_size: int

class BMIGroupOptimizer:
    """BMI分组与检测时点优化器"""
    
    def __init__(self, data_path):
        """初始化数据和参数"""
        self.df = pd.read_excel(data_path)
        self.preprocess_data()
        
        # 风险权重设置
        self.risk_weights = {
            'early': 1.0,    # ≤12周
            'medium': 5.0,   # 13-27周
            'late': 20.0     # ≥28周
        }
        
        # 目标达标率
        self.target_pass_rate = 0.95
        
    def preprocess_data(self):
        """数据预处理"""
        # 提取男胎数据
        self.male_data = self.df[self.df['Y浓度'].notna()].copy()
        
        # 解析孕周
        def parse_weeks(week_str):
            if pd.isna(week_str):
                return np.nan
            try:
                parts = str(week_str).replace('周', '+').replace('天', '').split('+')
                weeks = float(parts[0])
                days = float(parts[1]) if len(parts) > 1 else 0
                return weeks + days/7
            except:
                return np.nan
        
        self.male_data['孕周数'] = self.male_data['孕周'].apply(parse_weeks)
        self.male_data['Y浓度_pct'] = self.male_data['Y浓度'] * 100
        
        # 过滤有效数据
        self.male_data = self.male_data[
            (self.male_data['孕周数'] >= 10) & 
            (self.male_data['孕周数'] <= 25) &
            (self.male_data['BMI'].notna())
        ].copy()
        
        # 标记质量因子
        self.male_data['质量因子'] = self._calculate_quality_factor()
        
        print(f"处理后男胎样本数: {len(self.male_data)}")
        print(f"BMI范围: {self.male_data['BMI'].min():.1f} - {self.male_data['BMI'].max():.1f}")
        
    def _calculate_quality_factor(self):
        """计算质量因子"""
        quality = np.ones(len(self.male_data))
        
        # GC含量异常
        gc_abnormal = (self.male_data['GC含量'] < 0.4) | (self.male_data['GC含量'] > 0.6)
        quality[gc_abnormal] *= 0.8
        
        # 读段数过低
        low_reads = self.male_data['唯一比对读段数'] < self.male_data['唯一比对读段数'].quantile(0.1)
        quality[low_reads] *= 0.7
        
        # 过滤比例过高
        high_filter = self.male_data['被过滤掉的读段数比例'] > 0.5
        quality[high_filter] *= 0.6
        
        return quality
    
    def estimate_pass_probability(self, bmi_range, week):
        """估计特定BMI范围和孕周的达标概率"""
        # 筛选数据
        mask = (self.male_data['BMI'] >= bmi_range[0]) & \
               (self.male_data['BMI'] < bmi_range[1]) & \
               (self.male_data['孕周数'] >= week - 1) & \
               (self.male_data['孕周数'] <= week + 1)
        
        subset = self.male_data[mask]
        
        if len(subset) < 5:
            # 样本太少，使用模型预测
            return self._model_predict_pass_rate(bmi_range, week)
        
        # 考虑质量因子的加权达标率
        weights = subset['质量因子'].values
        passed = (subset['Y浓度_pct'] >= 4).values
        
        return np.average(passed, weights=weights)
    
    def _model_predict_pass_rate(self, bmi_range, week):
        """基于模型预测达标率"""
        # 简化的预测模型
        bmi_mid = (bmi_range[0] + bmi_range[1]) / 2
        
        # 基础达标率：随孕周增加，随BMI增加而降低
        base_rate = 1 / (1 + np.exp(-0.5 * (week - 12 - 0.3 * (bmi_mid - 25))))
        
        # BMI调整
        if bmi_mid > 35:
            base_rate *= 0.7
        elif bmi_mid > 30:
            base_rate *= 0.85
            
        return min(base_rate, 0.99)
    
    def calculate_risk(self, week, pass_rate):
        """计算风险分数"""
        # 孕周风险
        if week <= 12:
            week_risk = self.risk_weights['early']
        elif week <= 27:
            week_risk = self.risk_weights['medium']
        else:
            week_risk = self.risk_weights['late']
        
        # 失败风险
        fail_risk = (1 - pass_rate) * 100
        
        # 综合风险
        total_risk = week_risk * (1 + fail_risk)
        
        return total_risk
    
    def dynamic_programming_optimize(self, n_groups=4):
        """动态规划求解最优分组"""
        # BMI候选边界点
        bmi_values = sorted(self.male_data['BMI'].unique())
        n_points = len(bmi_values)
        
        # 候选孕周
        week_candidates = np.arange(11, 26, 1)
        
        # 预计算代价矩阵
        print("预计算代价矩阵...")
        cost_matrix = {}
        
        for i in range(n_points):
            for j in range(i+1, min(i+50, n_points)):  # 限制组宽度
                bmi_range = (bmi_values[i], bmi_values[j])
                
                # 计算该组的最优时点和风险
                best_week = None
                best_risk = float('inf')
                best_pass_rate = 0
                
                for week in week_candidates:
                    pass_rate = self.estimate_pass_probability(bmi_range, week)
                    
                    if pass_rate >= self.target_pass_rate:
                        risk = self.calculate_risk(week, pass_rate)
                        if risk < best_risk:
                            best_risk = risk
                            best_week = week
                            best_pass_rate = pass_rate
                
                if best_week is not None:
                    cost_matrix[(i, j)] = {
                        'cost': best_risk,
                        'week': best_week,
                        'pass_rate': best_pass_rate
                    }
        
        # 动态规划
        print("执行动态规划...")
        INF = float('inf')
        
        # dp[i][k] = 前i个点分成k组的最小代价
        dp = [[INF] * (n_groups + 1) for _ in range(n_points)]
        parent = [[None] * (n_groups + 1) for _ in range(n_points)]
        
        # 初始化
        dp[0][0] = 0
        
        # 状态转移
        for i in range(1, n_points):
            for k in range(1, min(i+1, n_groups+1)):
                for j in range(k-1, i):
                    if (j, i) in cost_matrix:
                        new_cost = dp[j][k-1] + cost_matrix[(j, i)]['cost']
                        if new_cost < dp[i][k]:
                            dp[i][k] = new_cost
                            parent[i][k] = j
        
        # 回溯最优解
        print("回溯最优解...")
        groups = []
        current = n_points - 1
        k = n_groups
        
        while k > 0 and parent[current][k] is not None:
            prev = parent[current][k]
            if (prev, current) in cost_matrix:
                info = cost_matrix[(prev, current)]
                groups.append(GroupResult(
                    bmi_range=(bmi_values[prev], bmi_values[current]),
                    optimal_week=info['week'],
                    pass_rate=info['pass_rate'],
                    risk_score=info['cost'],
                    sample_size=sum((self.male_data['BMI'] >= bmi_values[prev]) & 
                                   (self.male_data['BMI'] < bmi_values[current]))
                ))
            current = prev
            k -= 1
        
        groups.reverse()
        return groups
    
    def monte_carlo_validation(self, groups: List[GroupResult], n_simulations=1000):
        """蒙特卡洛仿真验证"""
        print(f"\n执行{n_simulations}次蒙特卡洛仿真...")
        
        validation_results = []
        
        for group in groups:
            successes = 0
            risks = []
            
            for _ in range(n_simulations):
                # 模拟个体
                bmi = np.random.uniform(group.bmi_range[0], group.bmi_range[1])
                
                # 模拟Y浓度（基于历史数据分布）
                mean_y = 2 + 0.3 * group.optimal_week - 0.05 * bmi
                std_y = 1.5
                y_conc = np.random.normal(mean_y, std_y)
                
                # 加入质量因子噪声
                quality_factor = np.random.choice([0.6, 0.8, 1.0], p=[0.1, 0.2, 0.7])
                y_conc *= quality_factor
                
                # 判断是否达标
                if y_conc >= 4:
                    successes += 1
                
                # 计算风险
                risk = self.calculate_risk(group.optimal_week, y_conc >= 4)
                risks.append(risk)
            
            validation_results.append({
                'BMI范围': f"{group.bmi_range[0]:.1f}-{group.bmi_range[1]:.1f}",
                '检测孕周': group.optimal_week,
                '预期达标率': group.pass_rate,
                '仿真达标率': successes / n_simulations,
                '平均风险': np.mean(risks),
                '风险标准差': np.std(risks)
            })
        
        return pd.DataFrame(validation_results)
    
    def analyze_error_impact(self, groups: List[GroupResult]):
        """分析检测误差影响"""
        print("\n分析检测误差影响...")
        
        error_levels = [0, 0.5, 1.0, 1.5, 2.0]  # 误差水平（%）
        results = []
        
        for error in error_levels:
            for group in groups:
                # 获取该组数据
                mask = (self.male_data['BMI'] >= group.bmi_range[0]) & \
                       (self.male_data['BMI'] < group.bmi_range[1])
                subset = self.male_data[mask]
                
                if len(subset) > 0:
                    # 模拟误差影响
                    y_values = subset['Y浓度_pct'].values
                    y_with_error = y_values + np.random.normal(0, error, len(y_values))
                    
                    # 重新计算达标率
                    new_pass_rate = (y_with_error >= 4).mean()
                    
                    results.append({
                        'BMI组': f"{group.bmi_range[0]:.0f}-{group.bmi_range[1]:.0f}",
                        '误差水平': error,
                        '原达标率': group.pass_rate,
                        '误差后达标率': new_pass_rate,
                        '达标率变化': new_pass_rate - group.pass_rate
                    })
        
        return pd.DataFrame(results)
    
    def plot_optimization_results(self, groups: List[GroupResult]):
        """可视化优化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 分组方案展示
        ax = axes[0, 0]
        for i, group in enumerate(groups):
            ax.barh(i, group.bmi_range[1] - group.bmi_range[0], 
                   left=group.bmi_range[0], height=0.8,
                   label=f'组{i+1}: {group.optimal_week:.0f}周')
            ax.text(np.mean(group.bmi_range), i, 
                   f'{group.optimal_week:.0f}周\n{group.pass_rate:.1%}',
                   ha='center', va='center', fontsize=10)
        
        ax.set_ylabel('分组')
        ax.set_xlabel('BMI')
        ax.set_title('BMI分组与最佳检测时点')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 2. 风险分布
        ax = axes[0, 1]
        bmi_ranges = [f"{g.bmi_range[0]:.0f}-{g.bmi_range[1]:.0f}" for g in groups]
        risks = [g.risk_score for g in groups]
        colors = ['green' if r < 5 else 'yellow' if r < 10 else 'red' for r in risks]
        bars = ax.bar(bmi_ranges, risks, color=colors, alpha=0.7)
        ax.set_xlabel('BMI组')
        ax.set_ylabel('风险分数')
        ax.set_title('各BMI组的风险评估')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, risk in zip(bars, risks):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{risk:.1f}', ha='center', va='bottom')
        
        # 3. 达标率曲线
        ax = axes[1, 0]
        for group in groups:
            weeks = np.arange(10, 26)
            pass_rates = [self.estimate_pass_probability(group.bmi_range, w) 
                         for w in weeks]
            
            label = f"BMI {group.bmi_range[0]:.0f}-{group.bmi_range[1]:.0f}"
            ax.plot(weeks, pass_rates, marker='o', label=label, linewidth=2)
            
            # 标记最优点
            ax.scatter([group.optimal_week], [group.pass_rate], 
                      s=100, color='red', marker='*', zorder=5)
        
        ax.axhline(y=self.target_pass_rate, color='red', linestyle='--', 
                  label=f'目标达标率({self.target_pass_rate:.0%})')
        ax.set_xlabel('孕周')
        ax.set_ylabel('达标率')
        ax.set_title('不同BMI组达标率随孕周变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 样本分布
        ax = axes[1, 1]
        sample_sizes = [g.sample_size for g in groups]
        ax.pie(sample_sizes, labels=bmi_ranges, autopct='%1.1f%%',
              startangle=90, colors=plt.cm.Set3(range(len(groups))))
        ax.set_title('各BMI组样本分布')
        
        plt.tight_layout()
        plt.show()
    
    def generate_recommendation_table(self, groups: List[GroupResult]):
        """生成推荐方案表"""
        recommendations = []
        
        for i, group in enumerate(groups):
            rec = {
                '分组': f'第{i+1}组',
                'BMI范围': f'{group.bmi_range[0]:.1f} - {group.bmi_range[1]:.1f}',
                '推荐检测孕周': f'{group.optimal_week:.0f}周',
                '预期达标率': f'{group.pass_rate:.1%}',
                '风险评分': f'{group.risk_score:.2f}',
                '样本量': group.sample_size,
                '风险等级': '低' if group.risk_score < 5 else '中' if group.risk_score < 10 else '高'
            }
            
            # 添加特殊建议
            if group.bmi_range[1] > 40:
                rec['特殊建议'] = '建议增加复检机制'
            elif group.optimal_week > 20:
                rec['特殊建议'] = '注意治疗窗口期'
            else:
                rec['特殊建议'] = '常规检测'
            
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)

# 主程序
def solve_problem2(data_path='C题附件.xlsx'):
    """解决问题2的主函数"""
    print("="*60)
    print("问题2: BMI分组与最佳NIPT时点优化")
    print("="*60)
    
    # 初始化优化器
    optimizer = BMIGroupOptimizer(data_path)
    
    # 执行动态规划优化
    print("\n开始动态规划优化...")
    groups = optimizer.dynamic_programming_optimize(n_groups=4)
    
    # 生成推荐表
    print("\n=== 优化结果 ===")
    rec_table = optimizer.generate_recommendation_table(groups)
    print(rec_table.to_string(index=False))
    
    # 蒙特卡洛验证
    validation_df = optimizer.monte_carlo_validation(groups)
    print("\n=== 蒙特卡洛仿真验证 ===")
    print(validation_df.to_string(index=False))
    
    # 误差影响分析
    error_df = optimizer.analyze_error_impact(groups)
    error_summary = error_df.groupby('误差水平')['达标率变化'].agg(['mean', 'std'])
    print("\n=== 检测误差影响分析 ===")
    print(error_summary)
    
    # 可视化结果
    optimizer.plot_optimization_results(groups)
    
    return groups, rec_table, validation_df, error_df

# 使用示例
if __name__ == "__main__":
    groups, recommendations, validation, error_analysis = solve_problem2()
    
    # 保存结果
    with pd.ExcelWriter('问题2_优化结果.xlsx') as writer:
        recommendations.to_excel(writer, sheet_name='推荐方案', index=False)
        validation.to_excel(writer, sheet_name='仿真验证', index=False)
        error_analysis.to_excel(writer, sheet_name='误差分析', index=False)
    
    print("\n结果已保存至 问题2_优化结果.xlsx")