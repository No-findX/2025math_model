import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, 
                           precision_recall_curve, confusion_matrix,
                           f1_score, recall_score, precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class FemaleAbnormalityDetector:
    """女胎异常检测器（两阶段模型）"""
    
    def __init__(self, data_path):
        """初始化"""
        self.df = pd.read_excel(data_path)
        self.preprocess_data()
        self.models = {}
        
    def preprocess_data(self):
        """数据预处理"""
        # 提取女胎数据（Y浓度为空）
        self.female_data = self.df[self.df['Y浓度'].isna()].copy()
        
        print(f"女胎样本总数: {len(self.female_data)}")
        
        # 创建标签：检测非整倍体（AB列）
        def create_label(row):
            """创建异常标签"""
            ab_value = str(row.get('13/18/21号染色体非整倍体', ''))
            if pd.isna(ab_value) or ab_value == '' or ab_value == 'nan':
                return 0  # 正常
            elif 'T21' in ab_value or 'T18' in ab_value or 'T13' in ab_value:
                return 1  # 异常
            else:
                return 0
        
        self.female_data['异常标签'] = self.female_data.apply(create_label, axis=1)
        
        # 统计标签分布
        label_counts = self.female_data['异常标签'].value_counts()
        print(f"标签分布: 正常={label_counts.get(0, 0)}, 异常={label_counts.get(1, 0)}")
        
        if label_counts.get(1, 0) > 0:
            print(f"类别不平衡比例: 1:{label_counts.get(0, 0)/label_counts.get(1, 0):.1f}")
        
        # 特征提取
        self._extract_features()
        
    def _extract_features(self):
        """提取特征"""
        # Z值特征
        z_columns = ['13号染色体Z值', '18号染色体Z值', '21号染色体Z值', 'X染色体Z值']
        
        # GC含量特征
        gc_columns = ['13号染色体GC含量', '18号染色体GC含量', '21号染色体GC含量']
        
        # 测序质量特征
        quality_columns = ['原始测序读段数', '唯一比对读段数', 
                          '总读段数比对比例', '总读段数重复比例', 
                          '被过滤掉的读段数比例', 'GC含量']
        
        # 孕妇特征
        maternal_columns = ['年龄', '身高', '体重', 'BMI']
        
        # 合并所有特征
        self.feature_columns = z_columns + gc_columns + quality_columns + maternal_columns
        
        # 处理缺失值
        for col in self.feature_columns:
            if col in self.female_data.columns:
                # 数值型特征用中位数填充
                if self.female_data[col].dtype in [np.float64, np.int64]:
                    self.female_data[col].fillna(
                        self.female_data[col].median(), inplace=True
                    )
        
        # 创建交互特征
        self.female_data['Z21_BMI'] = self.female_data['21号染色体Z值'] * self.female_data['BMI']
        self.female_data['Z18_年龄'] = self.female_data['18号染色体Z值'] * self.female_data['年龄']
        
        # 添加交互特征到特征列表
        self.feature_columns.extend(['Z21_BMI', 'Z18_年龄'])
        
        # 创建特征矩阵
        available_features = [col for col in self.feature_columns 
                            if col in self.female_data.columns]
        self.X = self.female_data[available_features].values
        self.y = self.female_data['异常标签'].values
        
        print(f"特征维度: {self.X.shape}")
        print(f"使用的特征: {len(available_features)}")
        
        # 保存特征名称
        self.feature_names = available_features
        
    def quality_gating(self, X, quality_threshold_high=0.8, quality_threshold_low=0.3):
        """质控门控"""
        # 计算质量得分
        quality_scores = self._calculate_quality_scores(X)
        
        # 分类
        gates = np.zeros(len(X))  # 0: 拒检, 1: 存疑, 2: 可靠
        
        gates[quality_scores < quality_threshold_low] = 0  # 拒检
        gates[(quality_scores >= quality_threshold_low) & 
              (quality_scores < quality_threshold_high)] = 1  # 存疑
        gates[quality_scores >= quality_threshold_high] = 2  # 可靠
        
        return gates, quality_scores
    
    def _calculate_quality_scores(self, X):
        """计算质量得分"""
        # 获取质量相关特征的索引
        gc_idx = self.feature_names.index('GC含量') if 'GC含量' in self.feature_names else None
        reads_idx = self.feature_names.index('唯一比对读段数') if '唯一比对读段数' in self.feature_names else None
        filter_idx = self.feature_names.index('被过滤掉的读段数比例') if '被过滤掉的读段数比例' in self.feature_names else None
        
        scores = np.ones(len(X))
        
        # GC含量评分
        if gc_idx is not None:
            gc_values = X[:, gc_idx]
            gc_scores = 1 - np.abs(gc_values - 0.5) * 2
            gc_scores = np.clip(gc_scores, 0, 1)
            scores *= gc_scores
        
        # 读段数评分
        if reads_idx is not None:
            reads = X[:, reads_idx]
            reads_norm = (reads - reads.min()) / (reads.max() - reads.min() + 1e-10)
            scores *= reads_norm
        
        # 过滤比例评分
        if filter_idx is not None:
            filter_rates = X[:, filter_idx]
            filter_scores = 1 - filter_rates
            scores *= filter_scores
        
        return scores
    
    def build_classification_models(self):
        """构建分类模型"""
        print("\n构建分类模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scaler = scaler
        
        # 处理类别不平衡
        if np.sum(y_train == 1) > 0:  # 确保有正样本
            # 使用SMOTE过采样
            smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y_train == 1) - 1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
            
            print(f"SMOTE后训练集: {len(y_train_balanced)}样本, "
                  f"正负比例: {np.sum(y_train_balanced==1)}/{np.sum(y_train_balanced==0)}")
        else:
            X_train_balanced = X_train_scaled
            y_train_balanced = y_train
            print("警告：没有异常样本，使用原始数据")
        
        # 1. 平衡随机森林
        print("\n训练平衡随机森林...")
        self.models['balanced_rf'] = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.models['balanced_rf'].fit(X_train_scaled, y_train)
        
        # 2. XGBoost（代价敏感）
        print("训练XGBoost...")
        # 计算类别权重
        pos_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
        
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.models['xgboost'].fit(X_train_scaled, y_train)
        
        # 3. 逻辑回归（类别权重）
        print("训练逻辑回归...")
        self.models['logistic'] = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        self.models['logistic'].fit(X_train_balanced, y_train_balanced)
        
        # 4. 集成模型（投票）
        print("构建集成模型...")
        
        # 保存测试数据
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # 评估模型
        self._evaluate_models(X_test_scaled, y_test)
        
    def _evaluate_models(self, X_test, y_test):
        """评估模型性能"""
        print("\n=== 模型评估 ===")
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            
            # 预测
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # 计算指标
            if np.sum(y_test == 1) > 0:  # 有正样本才计算
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_test, y_proba)
                except:
                    auc = 0
                
                print(f"  精确率: {precision:.3f}")
                print(f"  召回率: {recall:.3f}")
                print(f"  F1分数: {f1:.3f}")
                print(f"  AUC: {auc:.3f}")
                
                # 混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                print(f"  混淆矩阵:\n{cm}")
            else:
                print("  无正样本，无法计算指标")
    
    def predict_with_quality_control(self, X, model_name='balanced_rf', 
                                    threshold_reliable=0.5, threshold_suspicious=0.3):
        """带质控的预测"""
        # 质控门控
        gates, quality_scores = self.quality_gating(X)
        
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 获取模型
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"模型 {model_name} 不存在")
        
        # 预测概率
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        # 根据质控结果调整阈值
        predictions = np.zeros(len(X))
        decisions = []
        
        for i in range(len(X)):
            if gates[i] == 0:  # 拒检
                predictions[i] = -1  # -1表示需要复检
                decisions.append("需要复检（质量不合格）")
            elif gates[i] == 1:  # 存疑
                # 降低阈值，提高敏感度
                predictions[i] = 1 if y_proba[i] >= threshold_suspicious else 0
                decisions.append(f"存疑样本，{'异常' if predictions[i] == 1 else '正常'}（概率:{y_proba[i]:.3f}）")
            else:  # 可靠
                predictions[i] = 1 if y_proba[i] >= threshold_reliable else 0
                decisions.append(f"{'异常' if predictions[i] == 1 else '正常'}（概率:{y_proba[i]:.3f}）")
        
        return predictions, y_proba, gates, decisions
    
    def feature_importance_analysis(self):
        """特征重要性分析"""
        print("\n=== 特征重要性分析 ===")
        
        # 使用随机森林的特征重要性
        if 'balanced_rf' in self.models:
            importances = self.models['balanced_rf'].feature_importances_
            
            # 创建DataFrame
            feature_imp_df = pd.DataFrame({
                '特征': self.feature_names,
                '重要性': importances
            }).sort_values('重要性', ascending=False)
            
            print("\nTop 10 重要特征:")
            print(feature_imp_df.head(10))
            
            return feature_imp_df
        
        return None
    
    def plot_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 特征重要性
        ax = axes[0, 0]
        feature_imp_df = self.feature_importance_analysis()
        if feature_imp_df is not None:
            top_features = feature_imp_df.head(10)
            ax.barh(range(len(top_features)), top_features['重要性'].values)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['特征'].values)
            ax.set_xlabel('重要性')
            ax.set_title('Top 10 特征重要性')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 2. PR曲线
        ax = axes[0, 1]
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(self.X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(self.y_test, y_scores)
                ax.plot(recall, precision, label=name, linewidth=2)
        
        ax.set_xlabel('召回率')
        ax.set_ylabel('精确率')
        ax.set_title('PR曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 质量得分分布
        ax = axes[0, 2]
        quality_scores = self._calculate_quality_scores(self.X)
        ax.hist(quality_scores, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0.3, color='red', linestyle='--', label='低质量阈值')
        ax.axvline(x=0.8, color='green', linestyle='--', label='高质量阈值')
        ax.set_xlabel('质量得分')
        ax.set_ylabel('频数')
        ax.set_title('质量得分分布')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Z值分布对比
        ax = axes[1, 0]
        z_cols = ['21号染色体Z值', '18号染色体Z值', '13号染色体Z值']
        
        for col in z_cols:
            if col in self.female_data.columns:
                normal_data = self.female_data[self.female_data['异常标签'] == 0][col]
                abnormal_data = self.female_data[self.female_data['异常标签'] == 1][col]
                
                if len(abnormal_data) > 0:
                    ax.hist(normal_data, bins=20, alpha=0.3, label=f'{col[:2]}正常')
                    ax.hist(abnormal_data, bins=20, alpha=0.3, label=f'{col[:2]}异常')
        
        ax.set_xlabel('Z值')
        ax.set_ylabel('频数')
        ax.set_title('染色体Z值分布对比')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. 混淆矩阵热力图
        ax = axes[1, 1]
        if 'balanced_rf' in self.models:
            y_pred = self.models['balanced_rf'].predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('预测标签')
            ax.set_ylabel('真实标签')
            ax.set_title('混淆矩阵（平衡随机森林）')
        
        # 6. 阈值影响分析
        ax = axes[1, 2]
        if 'balanced_rf' in self.models:
            thresholds = np.linspace(0, 1, 50)
            precisions = []
            recalls = []
            
            y_scores = self.models['balanced_rf'].predict_proba(self.X_test)[:, 1]
            
            for thresh in thresholds:
                y_pred_thresh = (y_scores >= thresh).astype(int)
                if np.sum(self.y_test == 1) > 0:
                    prec = precision_score(self.y_test, y_pred_thresh, zero_division=0)
                    rec = recall_score(self.y_test, y_pred_thresh, zero_division=0)
                    precisions.append(prec)
                    recalls.append(rec)
            
            ax.plot(thresholds, precisions, label='精确率', linewidth=2)
            ax.plot(thresholds, recalls, label='召回率', linewidth=2)
            ax.set_xlabel('阈值')
            ax.set_ylabel('性能指标')
            ax.set_title('阈值对性能的影响')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 主程序
def solve_problem4(data_path='C题附件.xlsx'):
    """解决问题4的主函数"""
    print("="*60)
    print("问题4: 女胎异常判定")
    print("="*60)
    
    # 初始化检测器
    detector = FemaleAbnormalityDetector(data_path)
    
    # 构建分类模型
    detector.build_classification_models()
    
    # 特征重要性分析
    feature_imp = detector.feature_importance_analysis()
    
    # 示例预测
    print("\n=== 示例预测（前10个样本）===")
    sample_X = detector.X[:10]
    predictions, probas, gates, decisions = detector.predict_with_quality_control(
        sample_X, model_name='balanced_rf'
    )
    
    for i in range(min(10, len(predictions))):
        print(f"样本{i+1}: {decisions[i]}")
    
    # 可视化
    detector.plot_results()
    
    # 生成判定规则
    print("\n=== 判定规则总结 ===")
    print("1. 质控门控：")
    print("   - 质量得分 < 0.3: 拒检，需要复检")
    print("   - 0.3 ≤ 质量得分 < 0.8: 存疑，降低阈值至0.3")
    print("   - 质量得分 ≥ 0.8: 可靠，使用标准阈值0.5")
    print("\n2. 关键特征（按重要性排序）：")
    if feature_imp is not None:
        for i, row in feature_imp.head(5).iterrows():
            print(f"   - {row['特征']}: {row['重要性']:.3f}")
    print("\n3. 建议：")
    print("   - 对于存疑样本，建议进行复检或采用其他检测方法")
    print("   - 重点关注21号、18号染色体Z值异常")
    print("   - 结合孕妇BMI等因素进行综合判断")
    
    return detector

# 使用示例
if __name__ == "__main__":
    detector = solve_problem4()
    
    # 保存模型
    import pickle
    with open('female_detector_model.pkl', 'wb') as f:
        pickle.dump(detector, f)
    
    print("\n模型已保存至 female_detector_model.pkl")