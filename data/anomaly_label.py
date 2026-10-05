
import os
import pandas as pd

input_path = r"数据-筛选-处理.xlsx"
df = pd.read_excel(input_path)

df["假阳"] = ((df["染色体的非整倍体"].notna()) & (df["胎儿是否健康"] == "是")).astype(int)
df["假阴"] = ((df["染色体的非整倍体"].isna()) & (df["胎儿是否健康"] == "否")).astype(int)

output_path = os.path.join(os.path.dirname(input_path), "数据-筛选-处理-真假.xlsx")
df.to_excel(output_path, index=False)

print("处理完成，结果已保存至：", output_path)