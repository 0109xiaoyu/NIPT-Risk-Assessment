import pandas as pd
import io
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = "数据-筛选-处理-已过滤-标准化.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

columns_to_analyze = [
    "身高", "体重", "检测抽血次数", "检测孕周", "孕妇BMI",
    "X染色体的Z值", "Y染色体的Z值", "怀孕次数", "生产次数"
]

df_selected = df[columns_to_analyze].copy()

df_selected["怀孕次数"] = df_selected["怀孕次数"].replace("≥3", 3).astype(float)
df_selected["生产次数"] = df_selected["生产次数"].astype(float)

correlation_matrix = df_selected.corr()

print("相关系数矩阵:")
print(correlation_matrix)
correlation_matrix.to_excel("部分二相关性分析结果.xlsx")

n_vars = len(columns_to_analyze)

width = 16  
height = 12  

plt.figure(figsize=(width, height))

ax = sns.heatmap(correlation_matrix,
                 annot=True,
                 cmap="coolwarm",
                 center=0,
                 fmt=".2f",
                 annot_kws={"size": 16, "weight": "bold"},  
                 cbar_kws={"shrink": 0.7, "aspect": 30})

ax.set_xticklabels(correlation_matrix.columns,
                   rotation=0,
                   ha='center',
                   fontsize=14,  
                   weight='bold')

ax.set_yticklabels(correlation_matrix.index,
                   rotation=0,
                   fontsize=14,  
                   weight='bold')

plt.tight_layout()

plt.savefig("强相关特征相关性热图.png", dpi=300, bbox_inches='tight')
plt.show()