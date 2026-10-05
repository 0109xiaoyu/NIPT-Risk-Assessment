import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import io
import sys
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = r'数据-筛选-处理-真假.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
male_df = df[df['Y染色体浓度'].notna()].copy()
bmi = male_df['孕妇BMI'].values
bmi_min, bmi_max = bmi.min(), bmi.max()

male_df['假阳'] = male_df['假阳'].fillna(0).astype(int)
male_df['假阴'] = male_df['假阴'].fillna(0).astype(int)

prior_edges = [20, 28, 32, 36, 40, np.ceil(bmi_max)]
prior_intervals = [(prior_edges[i], prior_edges[i+1]) for i in range(len(prior_edges)-1)]

def assign_continuous_intervals(values, edges):
    idx = np.digitize(values, bins=edges[1:-1], right=False)
    return idx + 1

male_df['初始组名'] = assign_continuous_intervals(bmi, prior_edges)

best_penalty = np.inf
best_n, best_edges = 0, None
X = bmi.reshape(-1, 1)
lambda_balance = 1.0

for n in range(3, 7):
    gmm = GaussianMixture(n_components=n, random_state=42).fit(X)
    means = sorted(gmm.means_.flatten())
    inner_bounds = [(means[i] + means[i+1]) / 2 for i in range(n-1)]
    edges = np.concatenate([[bmi_min], inner_bounds, [bmi_max]])
    labs = assign_continuous_intervals(bmi, edges)
    male_df['tmp_group'] = labs

    counts = np.bincount(labs)[1:]
    penalty_counts = np.std(counts)

    fpr_fnr_list = []
    for g in range(1, n+1):
        sub = male_df[male_df['tmp_group'] == g]
        total = len(sub)
        if total == 0:
            fpr_fnr_list.append(0)
            continue
        fp = sub['假阳'].sum()
        fn = sub['假阴'].sum()
        fpr_fnr_list.append((fp + fn) / total)
    penalty_balance = np.std(fpr_fnr_list)

    total_penalty = penalty_counts + lambda_balance * penalty_balance

    if total_penalty < best_penalty:
        best_penalty = total_penalty
        best_n, best_edges = n, edges
        best_fpr_fnr = fpr_fnr_list
        best_counts = counts

male_df['组名'] = assign_continuous_intervals(bmi, best_edges)

print("GMM 连续区间")
for g in range(best_n):
    print(f"组 {g+1}: BMI ∈ [{best_edges[g]:.2f}, {best_edges[g+1]:.2f}), 人数: {best_counts[g]}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for g in range(len(prior_intervals)):
    ax[0].hist(bmi[male_df['初始组名']==g+1], bins=20, alpha=0.5, label=f'G{g+1}')
ax[0].set_title('(a)')  
ax[0].legend()

for g in range(best_n):
    ax[1].hist(bmi[male_df['组名']==g+1], bins=20, alpha=0.5, label=f'G{g+1}')
ax[1].set_title('(b) ') 
ax[1].legend()

ax[2].hist(bmi, bins=30, color='gray', alpha=0.7)
for x in best_edges[1:-1]:
    ax[2].axvline(x, color='r', ls='--')
ax[2].set_title('(c) ')  

plt.tight_layout()
plt.savefig('BMI分组对比图.png', dpi=300, bbox_inches='tight')
plt.show()