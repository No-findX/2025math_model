import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass
from pathlib import Path
import logging
import re
import json


@dataclass
class ModelConfig:
    """
    数据处理配置
    """
    # 缺失值填充
    knn_neighbors: int = 5

    # 异常值检测
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 4.0

    # 生理学约束
    gest_week_range: tuple[float, float] = (8.0, 30.0)
    bmi_range: tuple[float, float] = (15.0, 60.0)
    height_unit_threshold: float = 3.0  # 身高单位判断阈值 (米)

    # 可视化
    figure_size: tuple[int, int] = (8, 6)
    figure_dpi: int = 150


class DataPreprocessor:
    # 将中文列名映射为易于处理的英文名
    COLUMN_MAPPING = {
        '序号': 'id',
        '孕妇代码': 'patient_id',
        '年龄': 'age',
        '身高': 'height',
        '体重': 'weight',
        '检测孕周': 'gest_week',
        '孕妇BMI': 'bmi',
        '原始读段数': 'raw_reads',
        '在参考基因组上比对的比例': 'alignment_ratio',
        '重复读段的比例': 'duplicate_ratio',
        '唯一比对的读段数': 'unique_reads',
        'GC含量': 'gc_content',
        '13号染色体的Z值': 'chr13_z_score',
        '18号染色体的Z值': 'chr18_z_score',
        '21号染色体的Z值': 'chr21_z_score',
        'X染色体的Z值': 'chrx_z_score',
        'Y染色体的Z值': 'chry_z_score',
        'Y染色体浓度': 'chry_concentration',
        '胎儿性别': 'fetal_gender'
    }

    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据预设的映射重命名列"""
        df = df.rename(columns=self.COLUMN_MAPPING)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:

        # 确保身高、体重是数值型
        df['height'] = pd.to_numeric(df['height'], errors='coerce')
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

        # 转换孕周为天数
        if "w+" in df['gest_week']:
            parts = df['gest_week'].split("w+")
            week = int(parts[0])
            days = int(parts[1])
            df['gest_week'] = week * 7 + days
        elif 'w'in df['gest_week']:
            parts = df['gest_week'].split('w')
            df['gest_week'] = int(parts[0] * 7)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值：使用线性插值和KNN填充"""
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # 1. 线性插值
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

        # 2. KNN填充剩余缺失值
        if df[numeric_cols].isnull().sum().sum() > 0:
            imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)

            # 只保留确实有缺失的列
            cols_with_na = [c for c in numeric_cols if df[c].isnull().any()]

            imputed_array = imputer.fit_transform(df[cols_with_na])
            df[cols_with_na] = pd.DataFrame(imputed_array, columns=cols_with_na, index=df.index)

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值：基于生理约束和统计学方法（盖帽法）"""
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # 使用IQR进行Winzorization（盖帽法）处理统计异常值
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.iqr_multiplier * IQR
            upper_bound = Q3 + self.config.iqr_multiplier * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _visualize_key_vars(self, df: pd.DataFrame, output_dir: Path, prefix: str):
        """可视化关键变量的分布"""
        key_vars = ['gest_week', 'bmi', 'raw_reads', 'gc_content', 'chr21_z_score', 'chry_concentration']

        figs_dir = output_dir / "figures"
        figs_dir.mkdir(exist_ok=True, parents=True)

        for var in key_vars:
            if var in df.columns:
                plt.figure(figsize=self.config.figure_size)
                df[var].plot(kind='box')
                plt.title(f'Box Plot of {var} for {prefix}')
                plt.ylabel(var)
                fig_path = figs_dir / f"{prefix}_{var}_boxplot.png"
                plt.savefig(fig_path, dpi=self.config.figure_dpi)
                plt.close()

    def process(self, df: pd.DataFrame, output_dir: str, file_prefix: str) -> dict:
        """
        执行完整的预处理流程.
        """

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 步骤 1: 重命名列
        df = self._rename_columns(df)

        # 步骤 2: 特征工程
        df = self._engineer_features(df)

        # 步骤 3: 处理缺失值
        df_clean = self._handle_missing_values(df)

        # 步骤 4: 处理异常值
        df_clean = self._handle_outliers(df_clean)

        # 步骤 5: 可视化
        self._visualize_key_vars(df_clean, output_path, file_prefix)

        # 步骤 6: 数据标准化与归一化
        numeric_cols = df_clean.select_dtypes(include=np.number).columns

        # 标准化 (Z-Score)
        df_standardized = df_clean.copy()
        scaler_std = StandardScaler()
        df_standardized[numeric_cols] = scaler_std.fit_transform(df_standardized[numeric_cols])

        # 归一化 (Min-Max)
        df_normalized = df_clean.copy()
        scaler_norm = MinMaxScaler()
        df_normalized[numeric_cols] = scaler_norm.fit_transform(df_normalized[numeric_cols])

        # 步骤 7: 保存结果
        clean_path = output_path / f"{file_prefix}_cleaned.csv"
        std_path = output_path / f"{file_prefix}_standardized.csv"
        norm_path = output_path / f"{file_prefix}_normalized.csv"

        df_clean.to_csv(clean_path, index=False, encoding='utf-8-sig')
        df_standardized.to_csv(std_path, index=False, encoding='utf-8-sig')
        df_normalized.to_csv(norm_path, index=False, encoding='utf-8-sig')

        logging.info(f"'{file_prefix}' 数据处理完成。")

        report = {
            "source": file_prefix,
            "output_directory": str(output_path),
            "cleaned_data_path": str(clean_path),
            "standardized_data_path": str(std_path),
            "normalized_data_path": str(norm_path),
            "shape_original": df.shape,
            "shape_cleaned": df_clean.shape,
        }
        return report


def load_data_from_excel(excel_path: str, male_sheet: str, female_sheet: str) -> pd.DataFrame:
    # 读取指定的工作表
    male_df = pd.read_excel(excel_path, sheet_name=male_sheet)
    female_df = pd.read_excel(excel_path, sheet_name=female_sheet)

    # 添加性别标识列
    male_df['胎儿性别'] = 'Male'
    female_df['胎儿性别'] = 'Female'

    # 合并数据
    combined_df = pd.concat([male_df, female_df], ignore_index=True, sort=False)
    logging.info(f"成功从 '{excel_path}' 加载并合并工作表 '{male_sheet}' 和 '{female_sheet}'。")
    return combined_df


if __name__ == '__main__':
    # 从Excel文件加载数据
    combined_data = load_data_from_excel(
        excel_path='DataOriginal.xlsx',  # 您的Excel文件名
        male_sheet='男胎检测数据',
        female_sheet='女胎检测数据'
    )

    # 初始化处理器和配置
    preprocessor = DataPreprocessor(ModelConfig())

    # 执行预处理 (这部分保持不变)
    report = preprocessor.process(
        df=combined_data,
        output_dir='./',
        file_prefix='fetal_health_data_from_excel'  # 修改前缀以作区分
    )

    # 打印报告
    print("\n--- 数据预处理报告 ---")
    print(json.dumps(report, indent=2))
    print(f"\n处理结果已保存.")
