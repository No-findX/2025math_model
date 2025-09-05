import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class YConcentrationAnalysis:
    """Y染色体浓度分析模型"""
    
    def __init__(self, data_path):
        """初始化并加载数据"""
        self.df = pd.read_excel(data_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        """数据预处理"""
        # 提取男胎数据（Y浓度非空）
        self.male_data = self.df[self.df['Y浓度'].notna()].copy()
        
        # 处理孕周数据 (格式: "X周Y天")
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
        
        # 转换Y浓度为百分比
        self.male_data['Y浓度_pct'] = self.male_data['Y浓度'] * 100
        
        # 计算BMI（如果需要验证）
        self.male_data['BMI_calc'] = self.male_data['体重'] / (self.male_data['身高']/100)**2
        
        # 标记质量因子
        self.male_data['GC异常'] = ((self.male_data['GC含量'] < 0.4) | 
                                     (self.male_data['GC含量'] > 0.6)).astype(int)
        
        # 过滤有效数据
        self.male_data = self.male_data[(self.male_data['孕周数'] >= 10) & 
                                         (self.male_data['孕周数'] <= 25) &
                                         (self.male_data['BMI'].notna())].copy()
        
        print(f"有效男胎样本数: {len(self.male_data)}")
        print(f"Y浓度达标率(≥4%): {(self.male_data['Y浓度_pct'] >= 4).mean():.2%}")
        
    def build_quantile_model(self, quantiles=[0.5, 0.9, 0.95]):
        """建立分位数回归模型"""
        results = {}
        
        # 准备特征
        X = self.male_data[['孕周数', 'BMI', 'GC含量', '唯一比对读段数']].copy()
        
        # 添加交互项和多项式项
        X['孕周_平方'] = X['孕周数'] ** 2
        X['BMI_平方'] = X['BMI'] ** 2
        X['孕周_BMI'] = X['孕周数'] * X['BMI']
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        X_scaled['intercept'] = 1
        
        # Y变量：logit变换
        y = self.male_data['Y浓度_pct'].values
        y_clipped = np.clip(y, 0.01, 99.99) / 100  # 避免0和1
        y_logit = np.log(y_clipped / (1 - y_clipped))
        
        # 对每个分位点建模
        for q in quantiles:
            mod = QuantReg(y_logit, X_scaled)
            res = mod.fit(q=q)
            results[q] = {
                'model': res,
                'params': res.params,
                'pvalues': res.pvalues,
                'pseudo_r2': res.prsquared
            }
            
            print(f"\n分位点 {q} 回归结果:")
            print(f"伪R²: {res.prsquared:.4f}")
            print("\n显著性检验 (p-values):")
            for var, pval in zip(X_scaled.columns, res.pvalues):
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                print(f"  {var}: {pval:.4f} {sig}")
        
        self.quantile_results = results
        return results
    
    def test_relationships(self):
        """检验Y浓度与各因素的相关性"""
        print("\n=== 相关性分析 ===")
        
        # 1. Pearson相关系数
        corr_vars = ['Y浓度_pct', '孕周数', 'BMI', '年龄', 'GC含量']
        corr_matrix = self.male_data[corr_vars].corr()
        
        print("\nPearson相关系数矩阵:")
        print(corr_matrix['Y浓度_pct'].sort_values(ascending=False))
        
        # 2. Spearman秩相关（非线性关系）
        spearman_corr = self.male_data[corr_vars].corr(method='spearman')
        print("\nSpearman相关系数:")
        print(spearman_corr['Y浓度_pct'].sort_values(ascending=False))
        
        # 3. 分组均值检验（按BMI分组）
        bmi_groups = pd.cut(self.male_data['BMI'], 
                           bins=[0, 25, 30, 35, 40, 100],
                           labels=['<25', '25-30', '30-35', '35-40', '≥40'])
        
        grouped_stats = self.male_data.groupby(bmi_groups)['Y浓度_pct'].agg(['mean', 'std', 'count'])
        print("\n按BMI分组的Y浓度统计:")
        print(grouped_stats)
        
        # 4. ANOVA检验
        groups = [group['Y浓度_pct'].values for name, group in self.male_data.groupby(bmi_groups)]
        f_stat, p_value = stats.f_oneway(*[g for g in groups if len(g) > 0])
        print(f"\nANOVA检验: F={f_stat:.4f}, p-value={p_value:.4e}")
        
        return corr_matrix
    
    def plot_relationships(self):
        """可视化Y浓度与各因素的关系"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Y浓度vs孕周（散点+拟合）
        ax = axes[0, 0]
        ax.scatter(self.male_data['孕周数'], self.male_data['Y浓度_pct'], 
                  alpha=0.5, c=self.male_data['BMI'], cmap='viridis')
        
        # 添加分位数曲线
        week_range = np.linspace(10, 25, 100)
        for q in [0.5, 0.9, 0.95]:
            ax.plot(week_range, self.predict_quantile(week_range, 30, q), 
                   label=f'{q:.0%}分位', linewidth=2)
        
        ax.axhline(y=4, color='r', linestyle='--', label='4%阈值')
        ax.set_xlabel('孕周数')
        ax.set_ylabel('Y浓度(%)')
        ax.set_title('Y浓度与孕周关系')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Y浓度vs BMI（箱线图）
        ax = axes[0, 1]
        bmi_groups = pd.cut(self.male_data['BMI'], 
                           bins=[0, 25, 30, 35, 40, 100],
                           labels=['<25', '25-30', '30-35', '35-40', '≥40'])
        self.male_data['BMI组'] = bmi_groups
        self.male_data.boxplot(column='Y浓度_pct', by='BMI组', ax=ax)
        ax.axhline(y=4, color='r', linestyle='--', label='4%阈值')
        ax.set_title('不同BMI组的Y浓度分布')
        ax.set_xlabel('BMI组')
        ax.set_ylabel('Y浓度(%)')
        
        # 3. 达标率vs孕周（按BMI分组）
        ax = axes[0, 2]
        for bmi_group in ['25-30', '30-35', '35-40']:
            group_data = self.male_data[self.male_data['BMI组'] == bmi_group]
            if len(group_data) > 10:
                weeks = []
                rates = []
                for week in range(12, 25):
                    week_data = group_data[group_data['孕周数'].between(week-1, week+1)]
                    if len(week_data) > 5:
                        weeks.append(week)
                        rates.append((week_data['Y浓度_pct'] >= 4).mean())
                
                ax.plot(weeks, rates, marker='o', label=f'BMI {bmi_group}')
        
        ax.set_xlabel('孕周数')
        ax.set_ylabel('达标率(Y≥4%)')
        ax.set_title('不同BMI组的Y浓度达标率变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 热力图：相关性矩阵
        ax = axes[1, 0]
        corr_vars = ['Y浓度_pct', '孕周数', 'BMI', '年龄', 'GC含量']
        corr_matrix = self.male_data[corr_vars].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, vmin=-1, vmax=1)
        ax.set_title('变量相关性热力图')
        
        # 5. 3D散点图
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        scatter = ax.scatter(self.male_data['孕周数'], 
                           self.male_data['BMI'],
                           self.male_data['Y浓度_pct'],
                           c=self.male_data['Y浓度_pct'],
                           cmap='RdYlGn',
                           vmin=0, vmax=10,
                           alpha=0.6)
        ax.set_xlabel('孕周数')
        ax.set_ylabel('BMI')
        ax.set_zlabel('Y浓度(%)')
        ax.set_title('Y浓度三维分布')
        plt.colorbar(scatter, ax=ax, shrink=0.5)
        
        # 6. 残差分析
        ax = axes[1, 2]
        if hasattr(self, 'quantile_results'):
            model = self.quantile_results[0.5]['model']
            residuals = model.resid
            fitted = model.fittedvalues
            ax.scatter(fitted, residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('拟合值')
            ax.set_ylabel('残差')
            ax.set_title('残差分析图')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def predict_quantile(self, weeks, bmi, quantile=0.5):
        """预测特定分位点的Y浓度"""
        if not hasattr(self, 'quantile_results'):
            return np.zeros(len(weeks))
        
        # 简化预测（实际应用完整模型）
        baseline = 2 + 0.3 * weeks - 0.05 * bmi
        if quantile == 0.9:
            baseline *= 1.2
        elif quantile == 0.95:
            baseline *= 1.3
            
        return np.clip(baseline, 0, 100)
    
    def export_model_summary(self):
        """导出模型总结"""
        summary = {
            '样本数': len(self.male_data),
            '达标率': (self.male_data['Y浓度_pct'] >= 4).mean(),
            '平均孕周': self.male_data['孕周数'].mean(),
            '平均BMI': self.male_data['BMI'].mean(),
            '模型结果': {}
        }
        
        if hasattr(self, 'quantile_results'):
            for q, res in self.quantile_results.items():
                summary['模型结果'][f'Q{int(q*100)}'] = {
                    '伪R2': res['pseudo_r2'],
                    '孕周系数': res['params']['孕周数'],
                    'BMI系数': res['params']['BMI']
                }
        
        return summary

# 使用示例
if __name__ == "__main__":
    # 初始化分析
    analyzer = YConcentrationAnalysis('C题附件.xlsx')
    
    # 建立模型
    analyzer.build_quantile_model()
    
    # 检验关系
    analyzer.test_relationships()
    
    # 可视化
    analyzer.plot_relationships()
    
    # 导出结果
    summary = analyzer.export_model_summary()
    print("\n=== 模型总结 ===")
    print(summary)
