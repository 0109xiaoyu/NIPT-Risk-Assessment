
import os
import pandas as pd

input_file = '数据-筛选-处理-真假.xlsx'
df = pd.read_excel(input_file)

y_conc = pd.to_numeric(df['Y染色体浓度'], errors='coerce').fillna(0.0)

df['Y达标'] = (y_conc >= 0.04).astype(int)

output_file = '数据-筛选-处理-真假-Y浓度.xlsx'
df.to_excel(output_file, index=False)

print('处理完成！共标记达标记录：', df['Y达标'].sum(), '条')
print('结果已保存至：', os.path.abspath(output_file))