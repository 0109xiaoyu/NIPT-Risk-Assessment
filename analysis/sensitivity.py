import pandas as pd
import numpy as np
import io
import sys
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

file_path = r"数据-筛选-处理-真假-Y浓度.xlsx"
df = pd.read_excel(file_path)

bmi_bins = [20.70, 30.18, 31.88, 33.86, 46.88]
labels = ['[20.70,30.18)', '[30.18,31.88)', '[31.88,33.86)', '[33.86,46.88)']
df['BMI_group'] = pd.cut(df['孕妇BMI'], bins=bmi_bins, labels=labels, right=False)


ci_days = {
    '[20.70,30.18)':   {'lower': 92, 'upper': 104},
    '[30.18,31.88)':   {'lower': 85, 'upper': 97},
    '[31.88,33.86)':   {'lower': 77, 'upper': 89},
    '[33.86,46.88)':   {'lower': 71, 'upper': 83}
}

error_levels = [0.01, 0.03, 0.05]
err_labels = ['1%', '3%', '5%']

def sens_one_group(grp, opt_day, n_repeat=500, seed=42):
    np.random.seed(seed)
    sub = df[df['BMI_group'] == grp].copy()
    rates = {e: [] for e in error_levels}

    for _ in range(n_repeat):
        for err in error_levels:
            tmp = sub.copy()
            tmp['Y_noisy'] = tmp['Y染色体浓度'] * np.random.normal(1, err, size=len(tmp))
            valid = tmp[tmp['检测孕周'] <= opt_day]
            pass_rate = (valid['Y_noisy'] >= 0.04).mean() if len(valid) else 0
            rates[err].append(pass_rate)

    return [np.mean(rates[e]) for e in error_levels]

lower_dict = {grp: sens_one_group(grp, ci_days[grp]['lower']) for grp in labels}
upper_dict = {grp: sens_one_group(grp, ci_days[grp]['upper']) for grp in labels}

def plot_curves(data_dict, suffix, fig_label):
    plt.figure(figsize=(6, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for grp, color in zip(labels, colors):
        plt.plot(err_labels, data_dict[grp],
                 marker='o', linewidth=2.2, label=grp, color=color)

    plt.xlabel('测量误差水平', fontsize=11)
    plt.ylabel('通过率（Y浓度≥4%）', fontsize=11)
    plt.title(f'({fig_label}) ')  
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(title='BMI组别', frameon=False)
    plt.tight_layout()
    fname = f'sensitivity_curves_ci_{suffix.lower()}.png'  
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f'{fname} 已生成。')



plot_curves(lower_dict, '下限', 'b') 
plot_curves(upper_dict, '上限', 'a') 