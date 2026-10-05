import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig_idx = 0  

def run_task(task_name, y_train, y_test, feat_list, suffix):
    global fig_idx
    if y_train.sum() == 0 or y_test.sum() == 0:
        print(f'{task_name}{suffix} 无异常样本，跳过')
        return

    X_train = df_train[feat_list]
    X_test = df_test[feat_list]

    param_range = np.arange(1, 21)
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        X_train, y_train,
        param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(param_range, train_mean, 'o-', label='训练得分')
    ax.plot(param_range, test_mean, 'o-', label='交叉验证得分')
    ax.set_xlabel('树的最大深度')
    ax.set_ylabel('准确率')
    ax.grid(True)
    ax.legend(loc='lower right',
              bbox_to_anchor=(1.0, 0.0),
              borderaxespad=0.)

    label_char = chr(ord('a') + fig_idx)
    ax.annotate(f'({label_char})',
                xy=(0, 1), xycoords='axes fraction',
                xytext=(-2, 2), textcoords='offset points',
                ha='left', va='bottom',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{task_name}{suffix}_验证曲线.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig_idx += 1

    best_depth = param_range[np.argmax(test_mean)]
    rf = RandomForestClassifier(n_estimators=100, max_depth=best_depth,
                                random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    imp = pd.Series(rf.feature_importances_, index=feat_list).sort_values(ascending=False)

    rules = []
    for feat in imp.index:
        if X_train[feat].std() > 0:
            normal_m = X_train[y_train == 0][feat].median()
            abnormal_m = X_train[y_train == 1][feat].median()
            split = (normal_m + abnormal_m) / 2
            if abnormal_m > normal_m:
                rules.append(f"if {feat} > {split:.3f} → 异常风险高")
            else:
                rules.append(f"if {feat} < {split:.3f} → 异常风险高")
        else:
            rules.append(f"特征 {feat} 无变异，无法生成规则")

    with open(f'{task_name}{suffix}_判定规则.txt', 'w', encoding='utf-8') as f:
        f.write(f'{task_name}{suffix} 判定规则\n')
        f.write(f'训练集：BMI{suffix}_重采样.xlsx\n')
        f.write(f'测试集：BMI{suffix}_原始数据.xlsx\n')
        f.write(f'最优 max_depth: {best_depth}\n')
        f.write(f'测试集准确率: {acc:.4f}\n')
        f.write('全部特征及重要性:\n')
        for feat, v in imp.items():
            f.write(f'  {feat}: {v:.4f}\n')
        f.write('\n简化 if-then 规则:\n')
        for r in rules:
            f.write(f'  {r}\n')

    print(f'{task_name}{suffix} 完成')

fp_feat = ['原始读段数', '重复读段的比例', '唯一比对的读段数',
           '被过滤掉读段数的比例', '在参考基因组上比对的比例']
ane_feat = ['GC含量', '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
            '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量']
for grp in ['＜32', '≥32']:
    print(f'\n>>> 正在处理 BMI{grp} 分组')
    df_train = pd.read_excel(f'BMI_{grp}_重采样.xlsx')
    df_test = pd.read_excel(f'BMI_{grp}_原始数据.xlsx')

    for t in ['T13', 'T18', 'T21']:
        df_train[f'{t}_label'] = (df_train['样本类型'] == '异常') & df_train[t].notna()
        df_test[f'{t}_label'] = (df_test['样本类型'] == '异常') & df_test[t].notna()

    df_train['假阳_label'] = (df_train['样本类型'] == '异常') & (df_train['胎儿是否健康'] == '是')
    df_test['假阳_label'] = (df_test['样本类型'] == '异常') & (df_test['胎儿是否健康'] == '是')
    def fill_med(df, cols):
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c].fillna(df[c].median(), inplace=True)

    fill_med(df_train, fp_feat + ane_feat)
    fill_med(df_test, fp_feat + ane_feat)

    suffix = f'_BMI{grp}'
    for t in ['T13', 'T18', 'T21']:
        run_task(t,
                 df_train[f'{t}_label'].astype(int),
                 df_test[f'{t}_label'].astype(int),
                 ane_feat, suffix)
    run_task('假阳',
             df_train['假阳_label'].astype(int),
             df_test['假阳_label'].astype(int),
             fp_feat, suffix)

print('\n全部完成！共 8 张验证曲线图 + 8 个 txt 规则文件。')