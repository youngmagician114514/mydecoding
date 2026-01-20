def read_json_first_n_lines(file_path, n=20):
    """
    读取JSON文件的前n行内容并输出
    
    Args:
        file_path (str): JSON文件的路径
        n (int): 要读取的行数，默认20行
    """
    try:
        # 以只读模式打开文件，指定编码为utf-8避免中文乱码
        with open(file_path, 'r', encoding='utf-8') as file:
            # 初始化行号计数器
            line_number = 0
            # 逐行读取文件
            for line in file:
                # 只读取前n行
                if line_number < n:
                    # 去除每行末尾的换行符后输出，同时显示行号
                    print(f"第 {line_number + 1} 行: {line.strip()}")
                    line_number += 1
                else:
                    # 读取够n行后退出循环
                    break
                    
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'，请检查文件路径是否正确")
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")

# 调用函数 - 请将这里的路径替换为你的JSON文件实际路径
if __name__ == "__main__":
    # 示例路径，你需要修改为自己的JSON文件路径
    json_file_path = "./data/sharegpt/raw/train.json"
    # 读取前20行并输出
    read_json_first_n_lines(json_file_path, 2)