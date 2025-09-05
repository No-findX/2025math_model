import pandas as pd
import numpy as np
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class MultiFactorGroup:
    """多因素分组结果"""
    bmi_range: Tuple[float, float]
    optimal_week: float
    pass_rate: float
    risk_score: float
    sample_size: int
    feature_importance: Dict[str, float]
    robust_score: float  # 鲁棒性得分

class MultiFactorOptimizer:
    """多因素综合优化器"""
    
    def __init__(self, data_path):
        """初始化"""
        self.df = pd.read_excel(data_path)
        self.preprocess_data()
        self.build_predictive_model()
        
        # 参数设置
        self.target_pass_rate = 0.95
        self.risk_weights = {
            'early': 1.0,    # ≤12周
            'medium': 5.0,   # 13-27周
            'late': 20.0     # ≥28周
        }
        
    def preprocess_data(self):
        """数据预处理，提取多维特征"""
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
            (self.male_data['BMI'].notna()) &
            (self.male_data['年龄'].notna()) &
            (self.male_data['身高'].notna()) &
            (self.male_data['体重'].notna())
        ].copy()
        
        # 特征工程
        self._create_features()
        
        print(f"多因素分析样本数: {len(self.male_data)}")
        
    def _create_features(self):
        """创建扩展特征"""
        # 基础特征
        self.male_data['BMI_squared'] = self.male_data['BMI'] ** 2
        self.male_data['年龄_分组'] = pd.cut(self.male_data['年龄'], 
                                            bins=[0, 30, 35, 40, 100],
                                            labels=['<30', '30-35', '35-40', '>40'])
        
        # 体型指标（PCA）
        body_features = self.male_data[['身高', '体重', 'BMI']].values
        scaler = StandardScaler()
        body_scaled = scaler.fit_transform(body_features)
        
        pca = PCA(n_components=2)
        body_pca = pca.fit_transform(body_scaled)
        
        self.male_data['体型PC1'] = body_pca[:, 0]  # 整体体型大小
        self.male_data['体型PC2'] = body_pca[:, 1]  # 体型差异
        
        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        
        # 质量因子综合得分
        self.male_data['质量得分'] = self._calculate_quality_score()
        
        # 质量分级
        quality_bins = [0, 0.5, 0.8, 1.0]
        quality_labels = ['差', '中', '好']
        self.male_data['质量等级'] = pd.cut(self.male_data['质量得分'], 
                                           bins=quality_bins, 
                                           labels=quality_labels)
    
    def _calculate_quality_score(self):
        """计算综合质量得分"""
        scores = np.ones(len(self.male_data))
        
        # GC含量评分
        gc_optimal = 0.5
        gc_scores = 1 - np.abs(self.male_data['GC含量'] - gc_optimal) * 2
        gc_scores = np.clip(gc_scores, 0, 1)
        
        # 读段数评分（标准化到0-1）
        reads = self.male_data['唯一比对读段数'].values
        reads_norm = (reads - reads.min()) / (reads.max() - reads.min())
        
        # 过滤比例评分（反向）
        filter_scores = 1 - self.male_data['被过滤掉的读段数比例'].values
        
        # 综合评分（加权平均）
        quality = 0.4 * gc_scores + 0.4 * reads_norm + 0.2 * filter_scores
        
        return np.clip(quality, 0, 1)
    
    def build_predictive_model(self):
        """构建预测模型（随机森林）"""
        print("构建随机森林预测模型...")
        
        # 准备特征和目标
        feature_cols = ['孕周数', 'BMI', '年龄', '体型PC1', '体型PC2', '质量得分']
        X = self.male_data[feature_cols].values
        y = (self.male_data['Y浓度_pct'] >= 4).astype(int).values
        
        # 训练模型
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)
        
        # 特征重要性
        self.feature_importance = dict(zip(feature_cols, self.rf_model.feature_importances_))
        print("特征重要性:")
        for feat, imp in sorted(self.feature_importance.items(), 
                               key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.3f}")
    
    def predict_pass_probability(self, week, bmi, age=30, quality=0.8):
        """预测达标概率"""
        # 估算体型PC值
        height = 165  # 平均身高
        weight = bmi * (height/100) ** 2
        pc1 = (bmi - 25) / 5  # 简化估算
        pc2 = 0
        
        # 构建特征向量
        X = np.array([[week, bmi, age, pc1, pc2, quality]])
        
        # 预测
        prob = self.rf_model.predict(X)[0]
        return np.clip(prob, 0, 1)
    
    def robust_optimization(self, n_groups=4):
        """鲁棒优化求解"""
        print("\n执行鲁棒优化...")
        
        # 定义质量情景
        quality_scenarios = {
            '正常': {'quality': 0.9, 'weight': 0.7},
            '偏低': {'quality': 0.6, 'weight': 0.2},
            '极差': {'quality': 0.3, 'weight': 0.1}
        }
        
        # BMI分组边界（基于数据分布）
        bmi_percentiles = [0, 25, 50, 75, 100]
        bmi_boundaries = [self.male_data['BMI'].quantile(p/100) for p in bmi_percentiles]
        
        groups = []
        
        for i in range(len(bmi_boundaries) - 1):
            bmi_range = (bmi_boundaries[i], bmi_boundaries[i+1])
            
            # 获取该组数据
            group_data = self.male_data[
                (self.male_data['BMI'] >= bmi_range[0]) & 
                (self.male_data['BMI'] < bmi_range[1])
            ]
            
            if len(group_data) < 5:
                continue
            
            # 平均特征
            avg_age = group_data['年龄'].mean()
            
            # 寻找最优检测时点（考虑多情景）
            best_week = None
            best_robust_score = float('inf')
            best_pass_rate = 0
            
            for week in range(12, 25):
                # 计算各情景下的达标率
                scenario_results = []
                
                for scenario_name, scenario_params in quality_scenarios.items():
                    pass_prob = self.predict_pass_probability(
                        week, 
                        np.mean(bmi_range),
                        avg_age,
                        scenario_params['quality']
                    )
                    
                    # 计算该情景的风险
                    risk = self._calculate_risk(week, pass_prob)
                    
                    scenario_results.append({
                        'pass_prob': pass_prob,
                        'risk': risk,
                        'weight': scenario_params['weight']
                    })
                
                # 加权平均达标率
                weighted_pass_rate = sum(s['pass_prob'] * s['weight'] 
                                        for s in scenario_results)
                
                # 最坏情况达标率
                worst_pass_rate = min(s['pass_prob'] for s in scenario_results)
                
                # 鲁棒性得分（考虑平均和最坏情况）
                if worst_pass_rate >= self.target_pass_rate * 0.9:  # 放宽最坏情况约束
                    robust_score = sum(s['risk'] * s['weight'] for s in scenario_results)
                    
                    if robust_score < best_robust_score:
                        best_robust_score = robust_score
                        best_week = week
                        best_pass_rate = weighted_pass_rate
            
            if best_week is not None:
                groups.append(MultiFactorGroup(
                    bmi_range=bmi_range,
                    optimal_week=best_week,
                    pass_rate=best_pass_rate,
                    risk_score=best_robust_score,
                    sample_size=len(group_data),
                    feature_importance=self.feature_importance,
                    robust_score=best_robust_score
                ))
        
        return groups
    
    def _calculate_risk(self, week, pass_rate):
        """计算风险值"""
        # 孕周风险
        if week <= 12:
            week_risk = self.risk_weights['early']
        elif week <= 27:
            week_risk = self.risk_weights['medium']
        else:
            week_risk = self.risk_weights['late']
        
        # 失败风险
        fail_risk = (1 - pass_rate) * 100
        
        return week_risk * (1 + fail_risk)
    
    def pareto_optimization(self, groups: List[MultiFactorGroup]):
        """多目标Pareto优化"""
        print("\n执行Pareto多目标优化...")
        
        # 定义目标函数
        objectives = []
        
        for group in groups:
            # 目标1：风险最小化
            risk_obj = group.risk_score
            
            # 目标2：成本最小化（早期检测成本更低）
            cost_obj = 100 - group.optimal_week * 3  # 简化成本模型
            
            # 目标3：准确性最大化
            accuracy_obj = -group.pass_rate * 100
            
            objectives.append({
                'group': group,
                'risk': risk_obj,
                'cost': cost_obj,
                'accuracy': accuracy_obj
            })
        
        # 识别Pareto前沿
        pareto_front = []
        
        for i, obj_i in enumerate(objectives):
            dominated = False
            for j, obj_j in enumerate(objectives):
                if i != j:
                    # 检查是否被支配
                    if (obj_j['risk'] <= obj_i['risk'] and 
                        obj_j['cost'] <= obj_i['cost'] and
                        obj_j['accuracy'] <= obj_i['accuracy'] and
                        (obj_j['risk'] < obj_i['risk'] or 
                         obj_j['cost'] < obj_i['cost'] or
                         obj_j['accuracy'] < obj_i['accuracy'])):
                        dominated = True
                        break
            
            if not dominated:
                pareto_front.append(obj_i)
        
        print(f"Pareto前沿解数量: {len(pareto_front)}")
        
        return pareto_front
    
    def sensitivity_analysis(self, groups: List[MultiFactorGroup]):
        """敏感性分析"""
        print("\n执行敏感性分析...")
        
        sensitivity_results = []
        
        for group in groups:
            bmi_mid = np.mean(group.bmi_range)
            
            # 分析各因素的影响
            factors = {
                '年龄+5岁': {'age_delta': 5},
                '质量-20%': {'quality_delta': -0.2},
                'BMI+2': {'bmi_delta': 2}
            }
            
            base_prob = self.predict_pass_probability(
                group.optimal_week, bmi_mid, 30, 0.8
            )
            
            for factor_name, params in factors.items():
                # 计算扰动后的概率
                age = 30 + params.get('age_delta', 0)
                quality = 0.8 + params.get('quality_delta', 0)
                bmi = bmi_mid + params.get('bmi_delta', 0)
                
                new_prob = self.predict_pass_probability(
                    group.optimal_week, bmi, age, quality
                )
                
                sensitivity_results.append({
                    'BMI组': f"{group.bmi_range[0]:.0f}-{group.bmi_range[1]:.0f}",
                    '因素': factor_name,
                    '基准达标率': base_prob,
                    '扰动后达标率': new_prob,
                    '变化量': new_prob - base_prob,
                    '相对变化': (new_prob - base_prob) / base_prob * 100
                })
        
        return pd.DataFrame(sensitivity_results)
    
    def plot_multifactor_results(self, groups: List[MultiFactorGroup], pareto_front):
        """可视化多因素优化结果"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 特征重要性
        ax1 = plt.subplot(2, 3, 1)
        features = list(self.feature_importance.keys())
        importances = list(self.feature_importance.values())
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        
        bars = ax1.barh(features, importances, color=colors)
        ax1.set_xlabel('重要性得分')
        ax1.set_title('特征重要性分析')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. 鲁棒性评分
        ax2 = plt.subplot(2, 3, 2)
        group_labels = [f"BMI {g.bmi_range[0]:.0f}-{g.bmi_range[1]:.0f}" 
                       for g in groups]
        robust_scores = [g.robust_score for g in groups]
        
        ax2.bar(group_labels, robust_scores, color='skyblue', alpha=0.7)
        ax2.set_xlabel('BMI分组')
        ax2.set_ylabel('鲁棒性得分')
        ax2.set_title('各组鲁棒性评估')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Pareto前沿（2D投影）
        ax3 = plt.subplot(2, 3, 3)
        if pareto_front:
            risks = [p['risk'] for p in pareto_front]
            costs = [p['cost'] for p in pareto_front]
            
            ax3.scatter(risks, costs, c='red', s=100, marker='*', 
                       label='Pareto最优', zorder=5)
            
            # 添加非Pareto解
            all_risks = [g.risk_score for g in groups]
            all_costs = [100 - g.optimal_week * 3 for g in groups]
            ax3.scatter(all_risks, all_costs, c='gray', alpha=0.5, 
                       label='其他解', zorder=1)
            
            ax3.set_xlabel('风险得分')
            ax3.set_ylabel('成本得分')
            ax3.set_title('Pareto前沿（风险-成本）')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 多因素热力图
        ax4 = plt.subplot(2, 3, 4)
        
        # 创建热力图数据
        weeks = range(12, 25)
        bmis = range(25, 41, 5)
        heatmap_data = np.zeros((len(bmis), len(weeks)))
        
        for i, bmi in enumerate(bmis):
            for j, week in enumerate(weeks):
                heatmap_data[i, j] = self.predict_pass_probability(week, bmi)
        
        im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', 
                       vmin=0.5, vmax=1.0)
        ax4.set_xticks(range(len(weeks)))
        ax4.set_xticklabels(weeks)
        ax4.set_yticks(range(len(bmis)))
        ax4.set_yticklabels(bmis)
        ax4.set_xlabel('孕周')
        ax4.set_ylabel('BMI')
        ax4.set_title('达标概率热力图')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        
        # 5. 3D Pareto前沿
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        if pareto_front:
            risks = [p['risk'] for p in pareto_front]
            costs = [p['cost'] for p in pareto_front]
            accuracies = [-p['accuracy'] for p in pareto_front]
            
            ax5.scatter(risks, costs, accuracies, c='red', s=100, 
                       marker='*', label='Pareto最优')
            
            ax5.set_xlabel('风险')
            ax5.set_ylabel('成本')
            ax5.set_zlabel('准确性')
            ax5.set_title('3D Pareto前沿')
        
        # 6. 质量分布对比
        ax6 = plt.subplot(2, 3, 6)
        quality_dist = self.male_data.groupby('质量等级')['Y浓度_pct'].agg(['mean', 'std'])
        
        x_pos = range(len(quality_dist))
        ax6.bar(x_pos, quality_dist['mean'], yerr=quality_dist['std'],
               capsize=5, color=['red', 'yellow', 'green'][:len(quality_dist)],
               alpha=0.7)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(quality_dist.index)
        ax6.axhline(y=4, color='red', linestyle='--', label='4%阈值')
        ax6.set_xlabel('质量等级')
        ax6.set_ylabel('平均Y浓度(%)')
        ax6.set_title('不同质量等级的Y浓度分布')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()

# 主程序
def solve_problem3(data_path='C题附件.xlsx'):
    """解决问题3的主函数"""
    print("="*60)
    print("问题3: 多因素综合优化")
    print("="*60)
    
    # 初始化优化器
    optimizer = MultiFactorOptimizer(data_path)
    
    # 鲁棒优化
    groups = optimizer.robust_optimization(n_groups=4)
    
    print("\n=== 多因素优化结果 ===")
    for i, group in enumerate(groups):
        print(f"\n第{i+1}组:")
        print(f"  BMI范围: {group.bmi_range[0]:.1f} - {group.bmi_range[1]:.1f}")
        print(f"  最佳检测时点: {group.optimal_week}周")
        print(f"  预期达标率: {group.pass_rate:.1%}")
        print(f"  鲁棒性得分: {group.robust_score:.2f}")
        print(f"  样本量: {group.sample_size}")
    
    # Pareto优化
    pareto_front = optimizer.pareto_optimization(groups)
    
    # 敏感性分析
    sensitivity_df = optimizer.sensitivity_analysis(groups)
    print("\n=== 敏感性分析 ===")
    print(sensitivity_df.groupby('因素')['相对变化'].agg(['mean', 'std']))
    
    # 可视化
    optimizer.plot_multifactor_results(groups, pareto_front)
    
    return groups, pareto_front, sensitivity_df

# 使用示例
if __name__ == "__main__":
    groups, pareto, sensitivity = solve_problem3()
    
    # 保存结果
    results_df = pd.DataFrame([{
        'BMI范围': f"{g.bmi_range[0]:.1f}-{g.bmi_range[1]:.1f}",
        '最佳孕周': g.optimal_week,
        '达标率': f"{g.pass_rate:.1%}",
        '鲁棒性得分': g.robust_score,
        '样本量': g.sample_size
    } for g in groups])
    
    with pd.ExcelWriter('问题3_多因素优化结果.xlsx') as writer:
        results_df.to_excel(writer, sheet_name='优化方案', index=False)
        sensitivity.to_excel(writer, sheet_name='敏感性分析', index=False)
    
    print("\n结果已保存至 问题3_多因素优化结果.xlsx")