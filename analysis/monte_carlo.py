
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.interpolate import interp1d

df = pd.read_excel('数据-筛选-处理.xlsx', sheet_name='Sheet1')
col_map = {'孕妇代码':'subject_id','孕妇BMI':'bmi',
           '检测孕周':'gestational_days','Y染色体浓度':'Y_percentage'}
df = df.rename(columns=col_map)
df['gestational_week_value'] = df['gestational_days'] / 7
df = df.dropna(subset=['gestational_days','Y_percentage'])
male = df[df['Y_percentage'].notna()].copy()
male['event_fail'] = (male['Y_percentage'] < 0.04).astype(int)

bins = [20.70, 30.62, 32.90, 34.91, 46.88]
labels = ['[20.70,30.62)','[30.63,32.90)','[32.90,34.91)','[34.91,46.88)']
male['BMI分组'] = pd.cut(male['bmi'], bins=bins, labels=labels, right=False)

golden_days = {
    '[20.70,30.62)': 13*7 + 1,
    '[30.63,32.90)': 16*7 + 0,
    '[32.90,34.91)': 13*7 + 5,
    '[34.91,46.88)': 12*7 + 6
}

def mc_week(sub, golden_week):

    weeks = sub['gestational_week_value'].values
    fails = sub['event_fail'].values

    mask = np.abs(weeks - golden_week) <= 1.0
    k = fails[mask].sum()
    n = mask.sum()
    if n == 0:
        return np.nan

    p_fail = beta.rvs(k + 1, n - k + 1)          
    p_fail = np.clip(p_fail, 0.001, 0.999)

    delta = np.random.normal(0, 0.05)            
    x0 = golden_week + delta
    slope = np.random.normal(2.5, 0.2)           
    week_grid = np.linspace(10, 20, 200)
    logit_p = slope * (week_grid - x0)
    p_fail_grid = 1 / (1 + np.exp(-logit_p))

    p_succ_grid = 1 - p_fail_grid
    f = interp1d(p_succ_grid, week_grid, bounds_error=False)
    low  = f(0.100)  
    high = f(0.90)   
    if np.isnan(low) or np.isnan(high):
        return np.nan
    return np.random.uniform(low, high)          

N_BOOT = 5_000
np.random.seed(42)
results = []

for grp in labels:
    sub = male[male['BMI分组'] == grp].dropna(subset=['gestational_week_value','event_fail'])
    if len(sub) == 0:
        results.append({'BMI分组':grp,'点估计':np.nan,'下限':np.nan,'上限':np.nan})
        continue

    golden_week = golden_days[grp] / 7
    sim_weeks = [mc_week(sub, golden_week) for _ in range(N_BOOT)]
    low, high = np.nanpercentile(sim_weeks, [2.5, 97.5])
    results.append({
        'BMI分组': grp,
        '点估计': golden_week,
        '下限': low,
        '上限': high
    })

df_out = pd.DataFrame(results)

def wk_format(weeks):
    if pd.isna(weeks):
        return 'NA'
    w = int(weeks)
    d = round((weeks - w) * 7)
    if d >= 7:
        w += 1
        d = 0
    return f'{w}w+{d}'

df_out['点估计'] = df_out['点估计'].apply(wk_format)
df_out['95%CI']  = (df_out['下限'].apply(wk_format) +
                    ' ~ ' +
                    df_out['上限'].apply(wk_format))

df_out[['BMI分组','点估计','95%CI']].to_csv('MonteCarloCI_黄金点.csv',
                                        index=False, encoding='utf-8-sig')
print(df_out[['BMI分组','点估计','95%CI']])