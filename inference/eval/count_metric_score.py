import os
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

directory = sys.argv[1]

if not os.path.isdir(directory):
    print("Error: Directory does not exist.")
    sys.exit(1)

# OUTPUT PATH
output_bleu_file = os.path.join(directory, "bleu_results.txt")
output_comet_file = os.path.join(directory, "comet_results.txt")
output_cometkiwi_file = os.path.join(directory, "cometkiwi_results.txt")

# BLEU (逻辑保持不变)
with open(output_bleu_file, "w", encoding="utf-8") as bleu_output:
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".bleu"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    if "version:2.5.1 =" in line:
                        parts = line.split("version:2.5.1 = ")
                        filename_without_ext = '.'.join(filename.split('.')[:-1])
                        score = parts[1].split()[0].strip()
                        bleu_output.write(f"{filename_without_ext}: {score}\n")
                        break


def extract_comet_score_from_text(file_path):
    """从COMET的文本输出中稳健地提取最终平均分。"""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
        if not lines:
            return None


        target_line_index = -1
        if lines[-1].strip().startswith("[NOTICE]"):
            if len(lines) > 1:
                target_line_index = -2
            else:
                return None 
        
        target_line = lines[target_line_index]
        
        try:
            score = target_line.split()[-1]
            float(score)
            return score
        except (IndexError, ValueError):
            return None


# xCOMET
with open(output_comet_file, "w", encoding="utf-8") as xcomet_output:
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".xcomet"):
            file_path = os.path.join(directory, filename)
            score = extract_comet_score_from_text(file_path)
            if score:
                filename_without_ext = '.'.join(filename.split('.')[:-1])
                xcomet_output.write(f"{filename_without_ext}: {score}\n")
            else:
                print(f"Warning: Could not find a valid score in {filename}")


# COMETKIWI
with open(output_cometkiwi_file, "w", encoding="utf-8") as cometkiwi_output:
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".cometkiwi"):
            file_path = os.path.join(directory, filename)
            score = extract_comet_score_from_text(file_path)
            if score:
                filename_without_ext = '.'.join(filename.split('.')[:-1])
                cometkiwi_output.write(f"{filename_without_ext}: {score}\n")
            else:
                print(f"Warning: Could not find a valid score in {filename}")

print(f"汇总完成。请检查目录 {directory} 下的 _results.txt 文件。")