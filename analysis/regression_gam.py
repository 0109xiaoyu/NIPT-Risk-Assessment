import pandas as pd
import numpy as np
import re
import warnings
from pygam import LinearGAM, s, f
warnings.filterwarnings("ignore")
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

data_path = r"数据-筛选-处理-已过滤-标准化.xlsx"
raw = pd.read_excel(data_path, sheet_name="Sheet1")

raw["gest_weeks"] = raw["检测孕周"].apply(
    lambda w: int(re.findall(r'\d+', str(w))[0]) / 7
    if pd.notna(w) and re.findall(r'\d+', str(w))
    else np.nan
)

df = (
    raw.rename(columns={
        "Y染色体浓度": "y_conc",
        "X染色体浓度": "x_conc",
        "检测抽血次数": "draw_num",
        "18号染色体的Z值": "z18",
        "孕妇BMI": "bmi",
    })[["y_conc", "x_conc", "draw_num", "z18", "bmi", "gest_weeks"]]
    .dropna()
)

df["draw_num"] = df["draw_num"].astype(int)
draw_num_unique = sorted(df["draw_num"].unique())
df["draw_num"] = df["draw_num"].map({v: i for i, v in enumerate(draw_num_unique)})
X = df[["x_conc", "draw_num", "z18", "bmi", "gest_weeks"]].values
y = df["y_conc"].values

spline_order, lam = 5, 1e-5

n_splines_list = [
    max(spline_order + 1, min(40, len(np.unique(X[:, i]))))
    for i in range(X.shape[1])
]

gam = LinearGAM(
    s(0, n_splines=n_splines_list[0], spline_order=spline_order, lam=lam) +  
    f(1) +                                                                  
    s(2, n_splines=n_splines_list[2], spline_order=spline_order, lam=lam) +  
    s(3, n_splines=n_splines_list[3], spline_order=spline_order, lam=lam) +  
    s(4, n_splines=n_splines_list[4], spline_order=spline_order, lam=lam),   
    fit_intercept=True,
).fit(X, y)

r2 = gam.statistics_["pseudo_r2"]["explained_deviance"]
print(f"R² = {r2:.4f}\n")
print("========== 模型回归系数表 ==========")
print(gam.summary())

plt.figure(figsize=(8, 6))

plt.scatter(y, gam.predict(X), alpha=0.7, s=60) 
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2)

plt.xlabel("实际值", fontsize=18)
plt.ylabel("预测值", fontsize=18)
plt.title(f"实际值 vs 预测值", fontsize=20, pad=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()

plt.savefig("广义相加图.png", dpi=400, bbox_inches='tight', facecolor='white')
plt.show()