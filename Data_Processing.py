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

        # 转换孕周为天数
        for i in range(len(df['gest_week'])):
            if "w+" in df['gest_week'][i]:
                parts = df['gest_week'][i].split("w+")
                week = int(parts[0])
                days = int(parts[1])
                df['gest_week'][i] = week * 7 + days
            else:
                parts = df['gest_week'][i].split('w')
                df['gest_week'][i] = int(parts[0]) * 7

        return df

    def _visualize_key_vars(self, df: pd.DataFrame, output_dir: Path, prefix: str):
        """可视化关键变量的分布"""
        key_vars = ['gest_week', 'bmi', 'raw_reads', 'gc_content', 'chr21_z_score', 'chry_concentration']

        figs_dir = output_dir / "figures"
        figs_dir.mkdir(exist_ok=True, parents=True)

        for var in key_vars:
            if var in df.columns:
                if pd.api.types.is_numeric_dtype(df[var]) and not df[var].isnull().all():
                    plt.figure(figsize=self.config.figure_size)
                    df[var].plot(kind='box')
                    plt.title(f'Box Plot of {var} for {prefix}')
                    plt.ylabel(var)
                    fig_path = figs_dir / f"{prefix}_{var}_boxplot.png"
                    plt.savefig(fig_path, dpi=self.config.figure_dpi)
                    plt.close()
                else:
                    print(f"Skipping variable '{var}' as it is not numeric or contains only null values.")

    def process(self, df: pd.DataFrame, output_dir: str, file_prefix: str) -> dict:

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # 步骤 1: 重命名列
        df = self._rename_columns(df)

        # 步骤 2: 特征工程
        df_clean = self._engineer_features(df)

        # 步骤 5: 可视化
        self._visualize_key_vars(df_clean, output_path, file_prefix)

        # 步骤 6: 数据标准化与归一化
        standardized_cols = ['chr21_z_score']

        # 标准化 (Z-Score)
        df_standardized = df_clean.copy()
        scaler_std = StandardScaler()
        df_standardized[standardized_cols] = scaler_std.fit_transform(df_standardized[standardized_cols])

        # 归一化 (Min-Max)
        df_normalized = df_clean.copy()
        scaler_norm = MinMaxScaler()
        df_normalized[standardized_cols] = scaler_norm.fit_transform(df_normalized[standardized_cols])

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
