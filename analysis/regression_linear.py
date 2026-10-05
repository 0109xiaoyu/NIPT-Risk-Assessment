import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import io
import sys
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 16  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['axes.titlesize'] = 20  
plt.rcParams['axes.labelsize'] = 18  
plt.rcParams['xtick.labelsize'] = 16  
plt.rcParams['ytick.labelsize'] = 16  
plt.rcParams['legend.fontsize'] = 16 

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_excel("数据-筛选-处理-已过滤-标准化.xlsx")

x_cols = ['检测抽血次数', '检测孕周', '孕妇BMI', '18号染色体的Z值', 'X染色体浓度']
y_col = 'Y染色体浓度'

df = df[x_cols + [y_col]].dropna()

X = pd.DataFrame(StandardScaler().fit_transform(df[x_cols]), columns=x_cols)
y = df[y_col]

X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

print("回归系数与显著性：")
for name, coef, pval in zip(['截距'] + x_cols, model.params, model.pvalues):
    print(f"{name:>15}: {coef:>10.4f} (p={pval:.4f} {'显著' if pval < 0.05 else '不显著'})")

print(f"\nR² = {model.rsquared:.4f}")
print(f"调整后 R² = {model.rsquared_adj:.4f}")

plt.figure(figsize=(10, 8))

plt.scatter(y, model.fittedvalues, alpha=0.7, s=80)  
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=3)

plt.xlabel('实际值', fontsize=22, labelpad=15)
plt.ylabel('预测值', fontsize=22, labelpad=15)
plt.title('实际值 vs 预测值', fontsize=26, pad=25)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout(pad=3.0)

plt.savefig('实际值_vs预测值.png', dpi=400, bbox_inches='tight', facecolor='white')

plt.show()