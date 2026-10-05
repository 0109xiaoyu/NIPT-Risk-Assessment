
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import warnings

warnings.filterwarnings('ignore')


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def wk_format(weeks: float):
    if pd.isna(weeks) or np.isinf(weeks):
        return 'NA'
    w = int(weeks)
    d = round((weeks - w) * 7)
    if d >= 7:
        w += 1
        d = 0
    return f'{w}w+{d}'


print('>>> 读取数据')
df = pd.read_excel('数据-筛选-处理.xlsx', sheet_name='Sheet1')

col_map = {
    '孕妇代码': 'subject_id',
    '孕妇BMI': 'bmi',
    '检测孕周': 'gestational_days',
    'Y染色体浓度': 'Y_percentage'
}
df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
df['gestational_week_value'] = df['gestational_days'] / 7

df = df.dropna(subset=['gestational_days', 'Y_percentage'])

male = df[df['Y_percentage'].notna()].copy()

male['event_fail'] = (male['Y_percentage'] < 0.04).astype(int)

bins = [20.70, 30.62, 32.90, 34.91, 46.88]
labels = ['[20.70,30.62)', '[30.63,32.90)', '[32.90,34.91)', '[34.91,46.88)']
male['BMI分组'] = pd.cut(male['bmi'], bins=bins, labels=labels, right=False)

print('>>> 拟合 KM 曲线（事件：Y<4 %）')
best_weeks = []  

plt.figure(figsize=(10, 5))
colors = plt.cm.tab10.colors

for i, grp in enumerate(labels):
    sub = male[male['BMI分组'] == grp]
    sub = sub.dropna(subset=['gestational_week_value', 'event_fail'])
    n = len(sub)
    e = sub['event_fail'].sum()
    print(f'{grp}: 样本={n}, 事件={e}')
    
    if n == 0 or e == 0:  
        best_weeks.append(np.nan)
        continue
    
    kmf = KaplanMeierFitter()
    kmf.fit(sub['gestational_week_value'], event_observed=sub['event_fail'])
    
    try:
        survival_df = kmf.survival_function_
        valid_times = survival_df[survival_df['KM_estimate'] >= 0.95].index
        if len(valid_times) > 0:
            best_week = valid_times.max()
        else:
            best_week = np.nan
    except:
        best_week = np.nan
    
    best_weeks.append(best_week)

    kmf.plot_survival_function(ci_show=False, label=grp, color=colors[i])

plt.axhline(0.95, ls='--', c='k', lw=1, label='95%成功率阈值')
plt.xlabel('孕周')
plt.ylabel('Y浓度≥4%的概率')
plt.title('各 BMI 组 Y浓度≥4% 的概率随孕周变化')
plt.legend()
plt.tight_layout()
plt.savefig('km_curves_single_risk.png', dpi=300)
print('KM 曲线已保存 -> km_curves_single_risk.png')

print('\n===== 各 BMI 组最佳抽血孕周 =====')
for i, grp in enumerate(labels):
    if pd.isna(best_weeks[i]):
        print(f'{grp}: 无法确定最佳孕周')
    else:
        formatted_week = wk_format(best_weeks[i])
        print(f'{grp}: {formatted_week}')

result_df = pd.DataFrame({
    'BMI分组': labels,
    '最佳孕周': [wk_format(w) for w in best_weeks]
})
result_df.to_csv('最佳抽血孕周结果.csv', index=False, encoding='utf-8-sig')
print('\n>>> 结果已保存到: 最佳抽血孕周结果.csv')

print('\n>>> 分析完成！请查看当前目录下的结果文件。')