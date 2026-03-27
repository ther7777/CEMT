import json
import sys
import os
import re


json_file = sys.argv[1]
src_output_file = sys.argv[2]
trans_output_file = sys.argv[3]
ref_output_file = sys.argv[4]

# 读取JSON文件
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 处理文本换行问题的函数
def normalize_text(text):
    if not text:
        return ""
    # 替换所有换行符为空格
    text = re.sub(r'\n+', ' ', text)
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 提取并规范化源文本、生成的翻译和参考翻译
aligned_items = []
for item in data:
    source = normalize_text(item.get('source_text', ''))
    reference = normalize_text(item.get('reference_translation', ''))
    generated = normalize_text(item.get('generated_translation', ''))
    
    if source and reference and generated:  # 只处理三者都不为空的条目
        aligned_items.append((source, reference, generated))

# 创建输出文件的目录（如果不存在）
os.makedirs(os.path.dirname(src_output_file), exist_ok=True)
os.makedirs(os.path.dirname(trans_output_file), exist_ok=True)
os.makedirs(os.path.dirname(ref_output_file), exist_ok=True)

# 写入文件 - 每个文档一行，无内部换行
with open(src_output_file, 'w', encoding='utf-8') as f_src, \
     open(ref_output_file, 'w', encoding='utf-8') as f_ref, \
     open(trans_output_file, 'w', encoding='utf-8') as f_trans:
    
    for source, reference, generated in aligned_items:
        f_src.write(source + '\n')
        f_ref.write(reference + '\n')
        f_trans.write(generated + '\n')

print(f"成功处理 {len(aligned_items)} 个对齐样本，已移除文本内换行符")
