import json
import pandas as pd
import os
import argparse

def process_excel_to_json(json_file, excel_file):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 获取objHeight值
    obj_height = 67.0  # 默认值
    if 'alg' in data and 'param' in data['alg'] and 'objHeight' in data['alg']['param']:
        obj_height = float(data['alg']['param']['objHeight'])
    
    # 读取Excel文件
    df = pd.read_excel(excel_file, header=None)
    
    # 检查数据维度是否符合要求（18行，每行32个数字）
    if df.shape != (36, 64):
        print(f"警告：数据维度不正确。期望维度：(36, 64)，实际维度：{df.shape}")
        return
    
    # 获取原始的data数组
    numbers = data.get('data', [])
    
    # 将Excel数据转换为一维数组
    excel_data = df.values.flatten().tolist()
    
    # 对每个值进行转换：如果非零，则用objHeight除以该值
    transformed_data = []
    for x in excel_data:
        if x != 0:
            transformed_data.append(obj_height / x)
        else:
            transformed_data.append(0)
    
    # 更新data数组
    data['data'] = transformed_data
    
    # 创建输出目录
    output_dir = "./check-json-fixed/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用输入JSON文件名作为输出文件名
    input_filename = os.path.basename(json_file)
    base_name = os.path.splitext(input_filename)[0]
    output_file = os.path.join(output_dir, f"{base_name}.json")
    
    # 保存修改后的JSON文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"处理完成，已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理Excel文件并更新JSON数据')
    parser.add_argument('--json', required=True, help='输入的JSON文件路径')
    parser.add_argument('--xlsx', required=True, help='输入的Excel文件路径')
    
    args = parser.parse_args()
    
    if os.path.exists(args.json) and os.path.exists(args.xlsx):
        process_excel_to_json(args.json, args.xlsx)
    else:
        print("错误：输入文件不存在")
