import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.font_manager as fm
import io
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 22  
plt.rcParams['ytick.labelsize'] = 22 
plt.rcParams['legend.fontsize'] = 18

file_path = "数据-筛选.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

columns_to_plot = ['孕妇BMI', '检测孕周', 'GC含量', 'Y染色体浓度', 'X染色体浓度']
column_names = ['孕妇BMI', '检测孕周', 'GC含量', 'Y染色体浓度', 'X染色体浓度']

for i, col in enumerate(columns_to_plot):
    data = pd.to_numeric(df[col], errors='coerce').dropna()

    plt.figure(figsize=(14, 10))
    
    sns.histplot(data, kde=True, color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    mean_val = data.mean()
    median_val = data.median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_val:.2f}')
    
    if col == 'Y染色体浓度':
        try:
            plt.xlabel(column_names[i], fontsize=24)  
            plt.ylabel('频数', fontsize=24)
        except:
            en_labels = {
                '孕妇BMI': 'Maternal BMI',
                '检测孕周': 'Gestational Week',
                'GC含量': 'GC Content',
                'Y染色体浓度': 'Y Chromosome Concentration',
                'X染色体浓度': 'X Chromosome Concentration'
            }
            plt.xlabel(en_labels[column_names[i]], fontsize=24)
            plt.ylabel('Frequency', fontsize=24)
    else:
        try:
            plt.title(f'{column_names[i]}分布-男胎', fontsize=26, fontweight='bold', pad=20)  # 进一步增大标题字体
            plt.xlabel(column_names[i], fontsize=24)
            plt.ylabel('频数', fontsize=24)
        except:
            en_labels = {
                '孕妇BMI': 'Maternal BMI',
                '检测孕周': 'Gestational Week',
                'GC含量': 'GC Content',
                'Y染色体浓度': 'Y Chromosome Concentration',
                'X染色体浓度': 'X Chromosome Concentration'
            }
            plt.title(f'Distribution of {en_labels[column_names[i]]}', fontsize=26, fontweight='bold', pad=20)
            plt.xlabel(en_labels[column_names[i]], fontsize=24)
            plt.ylabel('Frequency', fontsize=24)
    
    plt.xticks(fontsize=24)  
    plt.yticks(fontsize=24)  
    
    plt.legend(fontsize=20)  
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  
    
    filename = f"{column_names[i]}分布-男胎.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"已保存: {filename}")
    
    plt.show()

fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.ravel()

for i in range(len(columns_to_plot), len(axes)):
    fig.delaxes(axes[i])
for i, col in enumerate(columns_to_plot):
    data = pd.to_numeric(df[col], errors='coerce').dropna()

    sns.histplot(data, kde=True, ax=axes[i], color='skyblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    try:
        axes[i].set_title(f'{column_names[i]}分布-男胎', fontsize=24, pad=20)  # 进一步增大子图标题字体
        axes[i].set_xlabel(column_names[i], fontsize=22)
        axes[i].set_ylabel('频数', fontsize=22)
    except:
        en_labels = {
            '孕妇BMI': 'Maternal BMI',
            '检测孕周': 'Gestational Week',
            'GC含量': 'GC Content',
            'Y染色体浓度': 'Y Chromosome Concentration',
            'X染色体浓度': 'X Chromosome Concentration'
        }
        axes[i].set_title(f'Distribution of {en_labels[column_names[i]]}', fontsize=24, pad=20)
        axes[i].set_xlabel(en_labels[column_names[i]], fontsize=22)
        axes[i].set_ylabel('Frequency', fontsize=22)
    
    axes[i].tick_params(axis='x', labelsize=22) 
    axes[i].tick_params(axis='y', labelsize=22)  

    stats_text = f'均值: {data.mean():.2f}\n标准差: {data.std():.2f}'
    axes[i].text(0.65, 0.85, stats_text, transform=axes[i].transAxes, fontsize=18,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

try:
    plt.suptitle('变量分布汇总', fontsize=28, fontweight='bold', y=0.98)
except:
    plt.suptitle('Variable Distribution Summary', fontsize=28, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('变量分布汇总.png', dpi=300, bbox_inches='tight')
print("已保存: 变量分布汇总.png")

plt.show()