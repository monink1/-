import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

def adjust_excel(input_file, output_file, start_prop, end_prop):
    """
    调整Excel文件中的数值，从底部开始按比例减少，直到顶部的0%
    
    Args:
        input_file: 输入的Excel文件路径
        output_file: 输出的Excel文件路径
        prop: 减少比例（例如：30表示30%）
    """
    print(f"开始处理文件: {input_file}")
    
    # 读取Excel文件
    df = pd.read_excel(input_file, header=None)
    
    # 获取数据维度
    ypart, xpart = df.shape
    
    # 创建调整后的数据副本
    df_adjusted = df.copy()
    
    # 计算每行的减少比例
    # 最底部行减少prop%，顶部行减少0%，中间行线性减少
    for col_idx in range(xpart):
        column = df.iloc[:, col_idx]
        
        # 对于每一列，从下往上计算减少比例
        for row_idx in range(ypart):
            # 计算当前行的减少比例
            # 最底部行(ypart-1)减少start_prop%，顶部行(0)减少end_prop%
            # 从底部开始计算，所以需要反转行号
            current_prop = ((1 - (ypart - 1 - row_idx) / (ypart - 1)) * (start_prop - end_prop) + end_prop) / 100
            
            # 获取当前值
            value = column.iloc[row_idx]
            
            # 如果值大于0，则按比例调整
            if value > 0:
                # 正比例表示减少，负比例表示增加
                adjusted_value = value * (1 - current_prop)
                # 确保值不小于0
                adjusted_value = max(0, adjusted_value)
                df_adjusted.iloc[row_idx, col_idx] = adjusted_value
                
                # 显示调整信息
                if current_prop > 0:
                    change_type = "减少"
                else:
                    change_type = "增加"
                print(f"第{row_idx}行第{col_idx}列: 原值{value:.2f} -> 新值{adjusted_value:.2f} ({change_type}{abs(current_prop*100):.1f}%)")
    
    # 保存调整后的Excel文件
    df_adjusted.to_excel(output_file, index=False, header=False)
    print(f"\n调整完成，已保存到: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='调整Excel文件中的数值')
    parser.add_argument('--input', type=str, required=True, help='输入的Excel文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出的Excel文件路径')
    parser.add_argument('--prop', type=str, required=True, help='调整范围，格式为：开始值to结束值。例如：30to0 或 50to20 或 10to-20。对于负值，可以使用等号：--prop=-10to10')
    
    args = parser.parse_args()
    
    input_file = args.input
    # 解析prop参数
    try:
        prop_value = args.prop
        
        # 检查是否包含'to'
        if 'to' not in prop_value:
            raise ValueError("参数格式错误，必须包含'to'")
            
        parts = prop_value.split('to')
        if len(parts) != 2:
            raise ValueError("参数格式错误，必须是两个值用'to'连接")
            
        start_prop = float(parts[0])
        end_prop = float(parts[1])
    except Exception as e:
        print(f"错误：--prop参数格式不正确。正确格式为：开始值to结束值")
        print(f"例如：30to0 或 50to20 或 -10to10 (对于负值，使用等号：--prop=-10to10)")
        print(f"错误详情：{str(e)}")
        exit(1)
    
    # 获取脚本运行目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取输入文件的基本信息
    input_filename = os.path.basename(input_file)
    input_filename_without_ext = os.path.splitext(input_filename)[0]
    
    # 创建excel-fitted文件夹
    excel_fitted_dir = os.path.join(script_dir, 'excel-fitted')
    os.makedirs(excel_fitted_dir, exist_ok=True)
    
    # 生成输出文件名
    output_file = os.path.join(excel_fitted_dir, f"{input_filename_without_ext}_adjusted.xlsx")
    
    if os.path.exists(input_file):
        adjust_excel(input_file, output_file, start_prop, end_prop)
    else:
        print(f"错误：文件 {input_file} 不存在")
