import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
input_path  = r"数据-筛选-处理-真假-Y浓度.xlsx"
output_path = os.path.join(os.path.dirname(input_path), "数据-筛选-处理-真假-Y浓度-标准化.xlsx")

df = pd.read_excel(input_path)

num_cols = [
    "年龄", "身高", "体重", "检测抽血次数", "检测孕周", "孕妇BMI",
    "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数", "GC含量",
    "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "Y染色体的Z值",
    "Y染色体浓度", "X染色体浓度", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
    "被过滤掉读段数的比例", "怀孕次数", "生产次数"
]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.to_excel(output_path, index=False)
print(f"标准化已完成，结果已保存至：{output_path}")