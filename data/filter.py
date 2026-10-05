import pandas as pd
import re
from datetime import datetime, timedelta

file_path = r"数据.xlsx"
df = pd.read_excel(file_path, sheet_name=0)
df.columns = df.columns.str.strip()

def any2date(x):
    if pd.isna(x):
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date()
    if isinstance(x, (int, float)):
        try:
            return (datetime(1899, 12, 30) + timedelta(days=float(x))).date()
        except Exception:
            return None
    if isinstance(x, str):
        x = x.strip()
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%d/%m/%Y']:
            try:
                return datetime.strptime(x, fmt).date()
            except ValueError:
                continue
    return None

df['末次月经日期'] = df['末次月经'].apply(any2date)

df = df[df['检测抽血次数'] != 5]

df['检测日期'] = pd.to_datetime(df['检测日期'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['检测日期'])
df = df[df['末次月经日期'] < df['检测日期'].dt.date]

def parse_gestation_weeks(week_str):
    if pd.isna(week_str):
        return None
    m = re.search(r'(\d+)\s*w\s*\+?\s*(\d*)', str(week_str).strip(), re.I)
    if m:
        return int(m.group(1)) * 7 + (int(m.group(2)) if m.group(2) else 0)
    return None

df['检测孕周'] = df['检测孕周'].apply(parse_gestation_weeks)
df = df[(df['检测孕周'] >= 70) & (df['检测孕周'] <= 175)]
df = df[(df['GC含量'] >= 0.35) & (df['GC含量'] <= 0.65)]

def vote_and_concat_aneu(group: pd.DataFrame) -> pd.DataFrame:
    score = pd.Series(0, index=group.index)
    score.loc[group['原始读段数'].idxmax()] += 1
    score.loc[group['在参考基因组上比对的比例'].idxmax()] += 1
    score.loc[group['重复读段的比例'].idxmin()] += 1
    score.loc[group['唯一比对的读段数'].idxmax()] += 1
    winner_idx = score.idxmax()

    aneu_list = group['染色体的非整倍体'].dropna().astype(str).str.strip()
    aneu_str = '|'.join(dict.fromkeys(aneu_list))
    winner = group.loc[[winner_idx]].copy()
    winner['染色体的非整倍体'] = aneu_str
    return winner

df_final = (df.groupby(['孕妇代码', '检测孕周'], group_keys=False)
              .apply(vote_and_concat_aneu))

out_path = '数据-筛选.xlsx'
df_final.drop(columns=['末次月经日期'], inplace=True, errors='ignore')
df_final.to_excel(out_path, index=False)
print('结果已保存为', out_path)