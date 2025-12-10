import pandas as pd
import itertools
import re

# 1. 读取原始数据
# 直接读取为 DataFrame，不需要先转成 string
input_file_path = 'experiment/four/result_logic_[abcd].csv'
output_file_path = 'experiment/five/result_logic_[abcde]_processed.csv' # 输出文件名

df = pd.read_csv(input_file_path)

# 用于正则匹配带符号的数字，如 -0, +1
pattern = re.compile(r'[+-]\d+')

# 用于收集处理后的新行数据
new_rows_data = []

# 全局集合，用于记录已经生成的组合，防止重复
# 这里存 tuple (key, metric1, metric2...) 方便比对
seen_rows = set()

# 获取列名，方便后续重建 DataFrame
columns = df.columns.tolist()

# 2. 遍历 DataFrame 的每一行
for index, row in df.iterrows():
    # 假设第一列是 K (key)，后面的列是 metrics
    current_key = str(row.iloc[0]) 
    current_metrics = row.iloc[1:].tolist() # 获取除第一列外的所有指标值
    
    # 提取元素
    elements = pattern.findall(current_key)
    
    # 加入 '-0'
    elements.append('-0')
    
    # 生成所有不重复的排列
    unique_permutations = sorted(list(set(itertools.permutations(elements))))
    
    # 3. 处理排列并收集数据
    for p in unique_permutations:
        new_key = "[" + "".join(p) + "]"
        
        # 创建一个用于去重的标识 (Key + 指标值)
        # 将 list 转换为 tuple 才能放入 set 中
        row_signature = (new_key, *current_metrics)
        
        if row_signature not in seen_rows:
            # 这一行没有出现过，加入结果列表
            # 重建这一行的数据字典
            new_row_dict = {columns[0]: new_key}
            for col_name, val in zip(columns[1:], current_metrics):
                new_row_dict[col_name] = val
            
            new_rows_data.append(new_row_dict)
            seen_rows.add(row_signature)

# 4. 生成新的 DataFrame 并保存为 CSV
if new_rows_data:
    df_result = pd.DataFrame(new_rows_data, columns=columns)
    df_result.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"成功！文件已生成: {output_file_path}")
    print(f"原行数: {len(df)}, 新行数: {len(df_result)}")
else:
    print("没有生成任何数据。")