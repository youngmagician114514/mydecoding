def save_first_n_lines_to_jsonl(input_file_path, output_file="sharegpt_llama_example.jsonl", n=10):
    """
    读取输入JSON文件的前n行，另存为指定的JSONL文件
    
    Args:
        input_file_path (str): 源JSON文件的路径
        output_file (str): 输出的JSONL文件名，默认是example.jsonl
        n (int): 要保存的行数，默认10行
    """
    try:
        # 打开源文件读取，打开目标文件写入（覆盖模式，编码为utf-8）
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            line_number = 0
            # 逐行读取源文件
            for line in infile:
                if line_number < n:
                    # 去除行尾多余空白（如换行符）后，添加换行符写入目标文件
                    cleaned_line = line.strip()
                    if cleaned_line:  # 跳过空行（可选，避免写入无效空行）
                        outfile.write(cleaned_line + '\n')
                    line_number += 1
                else:
                    break  # 读取够10行后退出循环
        
        print(f"成功将{input_file_path}的前{n}行保存为{output_file}")
    
    except FileNotFoundError:
        print(f"错误：找不到源文件 '{input_file_path}'，请检查文件路径是否正确")
    except PermissionError:
        print(f"错误：没有权限读取源文件或写入{output_file}，请检查文件权限")
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")

# 调用函数 - 请替换源文件路径为你的实际JSON文件路径
if __name__ == "__main__":
    # 替换为你的源JSON文件路径（相对路径/绝对路径均可）
    source_json_path = "/root/autodl-tmp/Faster_SD-main/data/sharegpt/sharegpt_llama.jsonl"
    # 执行保存操作（默认保存前10行到example.jsonl）
    save_first_n_lines_to_jsonl(source_json_path)