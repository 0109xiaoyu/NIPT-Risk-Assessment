import pandas as pd
import warnings
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, make_scorer
from decimal import Decimal

warnings.filterwarnings('ignore')

file_path = '数据-筛选-女.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

df['T13_label'] = df['T13'].notna().astype(int)
df['T18_label'] = df['T18'].notna().astype(int)
df['T21_label'] = df['T21'].notna().astype(int)
df['abnormal_label'] = (df['T13_label'] | df['T18_label'] | df['T21_label']).astype(int)
feature_columns = [
    '年龄', '身高', '体重', '检测抽血次数', '检测孕周', '孕妇BMI',
    '原始读段数', '在参考基因组上比对的比例', '重复读段的比例',
    '唯一比对的读段数', 'GC含量', '13号染色体的Z值', '18号染色体的Z值',
    '21号染色体的Z值', 'X染色体的Z值', 'X染色体浓度', '13号染色体的GC含量',
    '18号染色体的GC含量', '21号染色体的GC含量', '被过滤掉读段数的比例',
    '怀孕次数', '生产次数'
]

for col in feature_columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())

def bmi_tag(bmi):
    return '≥32' if pd.notna(bmi) and bmi >= 32 else '＜32'

df['BMI区间'] = df['孕妇BMI'].apply(bmi_tag)
for group_name in ['＜32', '≥32']:
    group_df = df[df['BMI区间'] == group_name].copy()
    if len(group_df) < 10:
        print(f'{group_name} 数据太少，跳过')
        continue

    print(f'\n>>> 正在处理 BMI{group_name} 组，原始 {len(group_df)} 条')

    original_file = f'BMI_{group_name}_原始数据.xlsx'
    group_df.to_excel(original_file, index=False)
    print(f'   已输出：{original_file}')

    X = group_df[feature_columns].values
    y = group_df['abnormal_label'].values

    pipe = Pipeline([
        ('sampler', RandomOverSampler(random_state=42)),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])

    ratio_range = [float(Decimal(i) / 100) for i in range(1, 301)]
    param_grid = {'sampler__sampling_strategy': ratio_range}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, pos_label=1)

    grid = GridSearchCV(pipe,
                        param_grid,
                        cv=cv,
                        scoring=scorer,
                        n_jobs=-1,
                        verbose=0,
                        error_score=0)
    grid.fit(X, y)

    best_ratio = grid.best_params_['sampler__sampling_strategy']
    print(f'   最佳采样比例（少数/多数）: {best_ratio:.2f}，验证 F1: {grid.best_score_:.3f}')

    ros = RandomOverSampler(sampling_strategy=best_ratio, random_state=42)
    X_res, y_res = ros.fit_resample(group_df.reset_index(drop=True), y)

    resampled_df = X_res.copy()
    resampled_df['abnormal_label'] = y_res
    resampled_df['样本类型'] = resampled_df['abnormal_label'].map({0: '正常', 1: '异常'})

    print(f'   重采样后共 {len(resampled_df)} 条')
    expand_file = f'BMI_{group_name}_重采样.xlsx'
    resampled_df.to_excel(expand_file, index=False)
    print(f'   已输出：{expand_file}')

print('\n>>> 全部完成！共 4 个文件已生成。')