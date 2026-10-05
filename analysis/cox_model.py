import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def wk_format(weeks):
    if pd.isna(weeks) or np.isinf(weeks):
        return 'NA'
    w, d = int(weeks), round((weeks - int(weeks)) * 7)
    if d >= 7:
        w += 1
        d = 0
    return f'{w}w+{d}'

df = pd.read_excel('数据-筛选-处理-真假-Y浓度.xlsx', sheet_name='Sheet1')
col_map = {
    '孕妇代码': 'subject_id', '年龄': 'age', '身高': 'height', '体重': 'weight',
    '孕妇BMI': 'bmi', '检测孕周': 'gestational_days', 'Y染色体浓度': 'Y_percentage',
    '怀孕次数': 'pregnancy_count', '生产次数': 'production_count',
    '检测抽血次数': 'blood_test_count', '原始读段数': 'total_reads',
    '在参考基因组上比对的比例': 'mapped_ratio', '重复读段的比例': 'duplicate_ratio'
}
df = df.rename(columns=col_map)
df['gestational_week_value'] = df['gestational_days'] / 7


male = df.dropna(subset=['gestational_days', 'Y_percentage', 'bmi', 'age', 'weight']).copy()
male['event'] = (male['Y_percentage'] >= 0.04).astype(int)
male['time'] = male['gestational_week_value']

features = ['bmi', 'age', 'weight', 'blood_test_count', 'pregnancy_count',
            'production_count', 'total_reads', 'mapped_ratio', 'duplicate_ratio']
cox_data = male[['time', 'event'] + features].dropna()

male['bmi_original'] = male['bmi']        

scaler = StandardScaler()
cox_data[features] = scaler.fit_transform(cox_data[features])
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='time', event_col='event')

male = male.dropna(subset=features)
male[features] = scaler.transform(male[features])
male['risk_score'] = cph.predict_partial_hazard(male[features])

risk_q = male['risk_score'].quantile([0, .25, .50, .75, 1])
male['risk_group'] = pd.cut(male['risk_score'], bins=risk_q,
                            labels=['低风险','中低风险','中高风险','高风险'],
                            include_lowest=True)

sorted_bmi = np.sort(male['bmi_original'].values)
n_grp = 4
grp_size = len(sorted_bmi) // n_grp
bmi_intervals = [(sorted_bmi[i*grp_size],
                  sorted_bmi[(i+1)*grp_size-1] if i < n_grp-1 else sorted_bmi[-1])
                 for i in range(n_grp)]

best_weeks, penalty_scores = [], []
plt.figure(figsize=(10, 6))
colors = plt.cm.tab10.colors
for i, grp in enumerate(['低风险','中低风险','中高风险','高风险']):
    sub = male[male['risk_group'] == grp].dropna(subset=['time','event'])
    if sub.empty:
        best_weeks.append(np.nan); penalty_scores.append(np.nan); continue

    kmf = KaplanMeierFitter()
    kmf.fit(sub['time'], sub['event'])
    surv = kmf.survival_function_
    valid = surv[(surv['KM_estimate'] >= 0.95)].index
    valid = valid[(valid >= 10) & (valid <= 25)]
    if len(valid) == 0:
        best_weeks.append(np.nan); penalty_scores.append(np.nan); continue

    penalties = [t + 5*np.log(1/(len(sub[sub['time'] >= t])/len(sub))) for t in valid]
    best_idx = np.argmin(penalties)
    best_weeks.append(valid[best_idx])
    penalty_scores.append(penalties[best_idx])

    kmf.plot_survival_function(ci_show=False, label=grp, color=colors[i])

plt.axhline(0.95, ls='--', c='k', lw=1)
plt.xlabel('孕周'); plt.ylabel('达标概率')
plt.title('各风险组 KM 曲线（Y 浓度≥4%）') 
plt.legend(); plt.tight_layout(); plt.savefig('km_curves_risk_groups.png', dpi=300); plt.show()

print('===== 各风险组最佳孕周 & 原始 BMI 区间 =====')
for i, grp in enumerate(['低风险','中低风险','中高风险','高风险']):
    print(f'{grp}: 最佳孕周={wk_format(best_weeks[i])}, '
          f'BMI 区间=[{bmi_intervals[i][0]:.2f}, {bmi_intervals[i][1]:.2f}]')

plt.figure(figsize=(8, 5))
for grp in ['低风险','中低风险','中高风险','高风险']:
    plt.hist(male[male['risk_group'] == grp]['risk_score'], alpha=0.7, label=grp, bins=20)
plt.xlabel('风险评分'); plt.ylabel('频数')
plt.title('(a)')
plt.legend(); plt.tight_layout(); plt.savefig('risk_score_distribution.png', dpi=300); plt.show()

plt.figure(figsize=(8, 5))
for i, grp in enumerate(['低风险','中低风险','中高风险','高风险']):
    mask = (male['bmi_original'] >= bmi_intervals[i][0]) & (male['bmi_original'] <= bmi_intervals[i][1])
    plt.hist(male[mask]['bmi_original'], alpha=0.7, bins=20,
             label=f'{grp} [{bmi_intervals[i][0]:.2f}, {bmi_intervals[i][1]:.2f}]')
plt.xlabel('BMI'); plt.ylabel('频数')
plt.title('(b)')  
plt.legend(); plt.tight_layout(); plt.savefig('bmi_distribution_by_risk_group.png', dpi=300); plt.show()