import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import io
import sys
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = "数据-筛选-处理-已过滤-标准化.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
columns_to_analyze = [
    '检测抽血次数', '检测孕周', '孕妇BMI', '18号染色体的Z值', 'X染色体浓度', 'Y染色体浓度'
]

df_selected = df[columns_to_analyze].copy()

def spearman_frame(df):
    cols = df.columns
    n = len(cols)
    r_mat = np.zeros((n, n))
    p_mat = np.zeros((n, n))
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            r, p = spearmanr(df[c1], df[c2], nan_policy='omit')
            r_mat[i, j] = r
            p_mat[i, j] = p
    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    return r_df, p_df

r_matrix, p_matrix = spearman_frame(df_selected)

r_matrix.to_excel("各因子_Spearman_r矩阵.xlsx")
p_matrix.to_excel("各因子_Spearman_p矩阵.xlsx")

label_mapping = {
    '检测抽血次数': '抽血次数',
    '检测孕周': '孕周',
    '孕妇BMI': 'BMI',
    '18号染色体的Z值': '18号Z值',
    'X染色体浓度': 'X浓度',
    'Y染色体浓度': 'Y浓度'
}

def draw_heatmap(corr_df, title, save_name):
    plt.figure(figsize=(16, 14))
    
    ax = sns.heatmap(corr_df,
                     annot=True,
                     cmap="coolwarm",
                     center=0,
                     fmt=".2f",
                     annot_kws={"size": 14, "weight": "bold"})  
    
    x_labels = [label_mapping.get(col, col) for col in corr_df.columns]
    y_labels = [label_mapping.get(col, col) for col in corr_df.index]

    ax.set_xticklabels(x_labels,
                       rotation=0,
                       ha='center',
                       fontsize=18)  
    
    ax.set_yticklabels(y_labels,
                       rotation=0,
                       fontsize=18) 

    plt.title(title, fontsize=24, pad=25) 
    
    plt.tight_layout()

    plt.savefig(save_name, dpi=400, bbox_inches='tight', facecolor='white')
    plt.show()

draw_heatmap(r_matrix, "各因子之间 Spearman 相关性热图", "各因子_Spearman_r热图.png")

target = "Y染色体浓度"
rp_df = pd.DataFrame({
    "因子": r_matrix.columns,
    "Spearman r": r_matrix[target].values,
    "p 值": p_matrix[target].values
})
rp_df = rp_df[rp_df["因子"] != target]
rp_df.to_excel("各因子_vs_Y染色体浓度_r_p.xlsx", index=False)
print("\n各因子 vs Y染色体浓度 的 r 与 p 值：")
print(rp_df)