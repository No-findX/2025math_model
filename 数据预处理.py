import pandas as pd
import numpy as np
import re  # 导入正则表达式模块，用于解析孕周
from sklearn.preprocessing import StandardScaler
import warnings


class NIPTDataPreprocessor:
    # 数据预处理器

    def __init__(self, file_path, male):
        self.file_path = file_path
        self.male = male
        self.raw_data = None
        self.data = None
        self.scaler = StandardScaler()

    def parse_gestational_age(self, ga_str):

        # 转化孕周数据格式
        ga_str = str(ga_str).strip()

        # 使用正则表达式匹配
        match = re.match(r'(\d+\.?\d*)\s*w?\s*\+?\s*(\d*)', ga_str)
        if match:
            weeks = float(match.group(1))
            days = float(match.group(2)) if match.group(2) else 0
            return weeks + days / 7.0

    def preprocess_data(self):

        # 完整预处理流程
        print("数据预处理, male = ",self.male)

        # 加载数据
        self.raw_data = pd.read_csv(self.file_path, encoding='utf-8-sig')

        print(f"原始数据: {self.raw_data.shape}")

        # 基础清洗
        df = self.raw_data.copy()

        # 解析孕周 (使用优化后的函数)
        df['孕周数值'] = df['检测孕周'].apply(self.parse_gestational_age)

        # 计划标准化列
        numeric_cols = [
            '年龄', '身高', '体重', '孕妇BMI',
            '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', 'GC含量',
            '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
            'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度', 'X染色体浓度',
            '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
            '被过滤掉读段数的比例',
            '原始读段数'
        ]

        if self.male == 0:
            numeric_cols.remove('Y染色体的Z值')
            numeric_cols.remove('Y染色体浓度')


        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 提取男胎数据
        raw_data = df.copy()
        print(f"数据: {len(raw_data)} 条")

        # 特征工程
        if self.male == 1:
            raw_data['Y染色体达标'] = (raw_data['Y染色体浓度'] >= 0.04).astype(int)
        raw_data['BMI_squared'] = raw_data['孕妇BMI'] ** 2
        raw_data['孕周_BMI_交互'] = raw_data['孕周数值'] * raw_data['孕妇BMI']

        raw_data['BMI_分组'] = pd.cut(raw_data['孕妇BMI'],
                                       bins = [0, 25, 30, 35, 100],
                                       labels = ['正常', '超重', '肥胖I', '肥胖II'])
        raw_data['孕周分组'] = pd.cut(raw_data['孕周数值'],
                                       bins=[0, 12, 16, 20, 30],
                                       labels=['早期', '中早期', '中期', '中晚期'])
        raw_data['检测次数'] = raw_data.groupby('孕妇代码')['孕妇代码'].transform('count')

        # 标准化特征
        features_to_scale_requested = [
            '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数', 'GC含量',
            '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
            'X染色体的Z值', 'Y染色体的Z值', 'Y染色体浓度', 'X染色体浓度',
            '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量',
            '被过滤掉读段数的比例',
            '原始读段数'
        ]

        if self.male == False:
            features_to_scale_requested.remove('Y染色体的Z值')
            features_to_scale_requested.remove('Y染色体浓度')

        features_to_scale = [col for col in features_to_scale_requested if col in raw_data.columns]

        scaled_features = self.scaler.fit_transform(raw_data[features_to_scale])
        scaled_df = pd.DataFrame(scaled_features,
                                     columns=[f + '_标准' for f in features_to_scale],
                                     index=raw_data.index)
        raw_data = pd.concat([raw_data, scaled_df], axis=1)

        self.data = raw_data

        # 输出基础统计
        print(f"\n预处理完成统计:")
        if self.male == True:
            print(f"Y染色体浓度均值: {raw_data['Y染色体浓度'].mean():.4f}")
            print(f"Y染色体达标率: {raw_data['Y染色体达标'].mean():.1%}")
        print(f"孕周范围: {raw_data['孕周数值'].min():.1f}-{raw_data['孕周数值'].max():.1f}周")
        print(f"BMI范围: {raw_data['孕妇BMI'].min():.1f}-{raw_data['孕妇BMI'].max():.1f}")

        return raw_data

    def save_data(self, output_path):

        # 保存预处理后的数据
        self.data.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"数据已成功保存至: {output_path}")
        return True


def quick_preprocess(file_path, male, save_output=True):

    # 快速预处理函数
    preprocessor = NIPTDataPreprocessor(file_path, male)
    data = preprocessor.preprocess_data()

    if male:
        preprocessor.save_data('processed_male_data.csv')
    else:
        preprocessor.save_data('processed_female_data.csv')

    return preprocessor, data


def main(file_path, male):

    # 主程序
    print("NIPT数据预处理")
    print("=" * 40)

    # 预处理数据
    preprocessor, data = quick_preprocess(file_path, male, save_output=True)

    print(f"最终数据量: {len(data)} 条")
    print(f"特征数量: {data.shape[1]} 个")
    return preprocessor, data


if __name__ == "__main__":
    preprocessor, data = main('C题附件男胎数据.csv', True)
    preprocessor, data = main('C题附件女胎数据.csv', False)
