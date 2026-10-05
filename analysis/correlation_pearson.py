import pandas as pd
import io
import sys
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = "数据-筛选-处理-已过滤-标准化.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

columns_to_analyze = [
    "年龄", "身高", "体重", "检测抽血次数", "检测孕周", "孕妇BMI",
    "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数",
    "GC含量", "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值",
    "X染色体的Z值", "Y染色体的Z值", "X染色体浓度",
    "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
    "被过滤掉读段数的比例", "怀孕次数", "生产次数"
]

df_selected = df[columns_to_analyze].copy()


df_selected["怀孕次数"] = df_selected["怀孕次数"].replace("≥3", 3).astype(float)
df_selected["生产次数"] = df_selected["生产次数"].astype(float)


correlation_matrix = df_selected.corr()

print("相关系数矩阵:")
print(correlation_matrix)
correlation_matrix.to_excel("相关性分析结果.xlsx")

import seaborn as sns


plt.figure(figsize=(24, 20))

ax = sns.heatmap(correlation_matrix,
                 annot=True,
                 cmap="coolwarm",
                 center=0,
                 fmt=".2f",
                 annot_kws={"size": 8})


def add_newlines_to_labels(labels):
    new_labels = []
    for label in labels:
        new_label = '\n'.join(list(str(label)))
        new_labels.append(new_label)
    return new_labels


new_xticklabels = add_newlines_to_labels(correlation_matrix.columns)


ax.set_xticklabels(new_xticklabels,
                   rotation=0,
                   ha='center',
                   va='top',
                   fontsize=10)

ax.set_yticklabels(correlation_matrix.index,
                   rotation=0,
                   fontsize=10)

plt.tight_layout()

plt.savefig("相关性热图.png", dpi=300, bbox_inches='tight')
plt.show()