import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from pathlib import Path
import matplotlib.font_manager as fm
from scipy.interpolate import CubicSpline, Rbf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

class KalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=0.1**2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.
        self.estimate_error = 1.
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.estimate = measurement
            self.initialized = True
            return measurement

        # 预测
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # 更新
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate

def linear_interpolate(x, y):
    """使用线性插值 y = ax + b，支持多个点"""
    if len(x) < 2 or len(y) < 2:
        return None, None
    
    # 使用最小二乘法计算线性回归
    A = np.vstack([x, np.ones(len(x))]).T
    try:
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return a, b
    except np.linalg.LinAlgError:
        return None, None

def apply_linear_interpolation(x, a, b):
    """应用线性插值"""
    if a is None or b is None:
        return None
    return a * x + b

def generate_heatmap(df, file_name, output_dir):
    """生成热力图"""
    
    # 添加本地Arial字体
    from matplotlib.font_manager import FontProperties
    font_path = "./Arial.ttf"
    prop = FontProperties(fname=font_path)
    
    # 设置标题和标签的字体大小
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # 获取网格大小
    xpart, ypart = df.shape
    
    # 创建热力图
    plt.figure(figsize=(xpart * 2.5, ypart * 2.5))  # 保持画布大小不变
    
    # 自定义颜色映射：0为白，逐渐变黄、红，最大为黑
    colors_list = [
        (1, 1, 1),    # white (0)
        (1, 1, 0),    # yellow (中等)
        (1, 0, 0),    # red (较高)
        (0, 0, 0)     # black (最大)
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_white_yellow_red_black", colors_list)
    
    # 绘制热力图
    sns.heatmap(df, cmap=cmap, annot=True, fmt='.0f',
               linewidths=1, linecolor='black',
               cbar=False,  # 移除颜色条
               annot_kws={'size': 20, 'fontproperties': prop},  # 将字体大小设置为20
               square=True,  # 保持单元格为正方形
               xticklabels=True,  # 显示x轴标签
               yticklabels=True)  # 显示y轴标签
    
    # 添加英文标题和标签
    plt.title(f'Data Heatmap - {file_name}', fontproperties=prop, fontsize=18)
    plt.xlabel('Columns', fontproperties=prop, fontsize=16)
    plt.ylabel('Rows', fontproperties=prop, fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存热力图
    heatmap_path = os.path.join(output_dir, f'{file_name}_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"热力图已保存到: {heatmap_path}")

def fix_xlsx(input_file, output_file, kalman):
    """修复xlsx文件中的数据异常"""
    print(f"开始处理文件: {input_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制原始文件到输出路径
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    original_output = os.path.join(output_dir, f'{base_name}_original.xlsx')
    import shutil
    shutil.copy2(input_file, original_output)
    print(f"原始文件已复制到: {original_output}")
    
    # 读取xlsx文件，header=None表示第一行就是数据
    df = pd.read_excel(input_file, header=None)
    tensornet = df.values
    ypart, xpart = tensornet.shape
    
    # 保存原始数据的热力图
    generate_heatmap(df, f"{base_name}_original", output_dir)
    
    # 第一轮：处理大于2个非零值的列
    print("\n开始第一轮插值...")
    tensornet_fixed = tensornet.copy()
    columns_to_fix_later = []
    
    for column_idx in range(xpart):
        column = tensornet[:, column_idx]
        non_zero_idx = np.where(column != 0)[0]
        non_zero_count = len(non_zero_idx)
        
        print(f"处理第{column_idx}列，非零值数量：{non_zero_count}")
        
        if non_zero_count > 2:
            # 使用线性插值 y = ax + b
            try:
                a, b = linear_interpolate(non_zero_idx, column[non_zero_idx])
                if a is not None:
                    # 对于这一列中的所有0值进行替换
                    zero_count = 0
                    for i in range(ypart):
                        if column[i] == 0:
                            value = apply_linear_interpolation(i, a, b)
                            if value is not None:
                                tensornet_fixed[i, column_idx] = value
                                zero_count += 1
                                print(f"替换第{column_idx}列的第{i}行，原值0 -> 新值{round(value)}")
                    if zero_count > 0:
                        print(f"第{column_idx}列共替换了{zero_count}个0值")
                else:
                    print(f"列{column_idx}线性插值失败：无法计算a, b")
            except Exception as e:
                print(f"列{column_idx}线性插值失败：{e}")
                columns_to_fix_later.append(column_idx)
        else:
            # 小于等于2个非零值的列留到第二轮处理
            columns_to_fix_later.append(column_idx)

    print("\n第一轮处理后统计：")
    for column_idx in range(xpart):
        column = tensornet_fixed[:, column_idx]
        zero_count = np.count_nonzero(column == 0)
        print(f"第{column_idx}列剩余0值数量：{zero_count}")

    # 第二轮：处理小于等于2个非零值的列
    print("\n开始第二轮插值...")
    for column_idx in columns_to_fix_later:
        column = tensornet_fixed[:, column_idx]
        non_zero_idx = np.where(column != 0)[0]
        non_zero_count = len(non_zero_idx)
        
            
        # 寻找相邻的已修复列
        left_idx = column_idx - 1
        right_idx = column_idx + 1
        reference_column = None
        
        print(f"\n处理第{column_idx}列，非零值数量：{non_zero_count}")
        print("正在寻找参考列...")
        
        while (left_idx >= 0 or right_idx < xpart) and reference_column is None:
            if left_idx >= 0 and left_idx not in columns_to_fix_later:
                ref_col = tensornet_fixed[:, left_idx]
                if np.count_nonzero(ref_col) > 2:  # 只使用已修复的列作为参考
                    reference_column = ref_col
                    print(f"找到左侧参考列：第{left_idx}列")
                    break
            if right_idx < xpart and right_idx not in columns_to_fix_later:
                ref_col = tensornet_fixed[:, right_idx]
                if np.count_nonzero(ref_col) > 2:  # 只使用已修复的列作为参考
                    reference_column = ref_col
                    print(f"找到右侧参考列：第{right_idx}列")
                    break
            left_idx -= 1
            right_idx += 1
        
        if reference_column is not None:
            print(f"使用参考列进行插值...")
            # 使用参考列的形状进行插值
            ref_non_zero = np.where(reference_column != 0)[0]
            
            if len(non_zero_idx) > 0:
                # 使用参考列值的1/3次方作为比例因子
                scale_factor = (column[non_zero_idx[0]] / reference_column[non_zero_idx[0]]) ** (1/3)
            else:
                # 如果当前列没有非零值，直接使用参考列的值
                scale_factor = 1.0
            
            zero_count = 0
            for i in range(ypart):
                if column[i] == 0 and reference_column[i] != 0:
                    tensornet_fixed[i, column_idx] = reference_column[i] * scale_factor
                    zero_count += 1
            if zero_count > 0:
                print(f"第{column_idx}列共替换了{zero_count}个0值")
        else:
            print(f"警告：第{column_idx}列没有找到合适的参考列")

    print("\n最终处理结果统计：")
    for column_idx in range(xpart):
        column = tensornet_fixed[:, column_idx]
        zero_count = np.count_nonzero(column == 0)
        print(f"第{column_idx}列最终剩余0值数量：{zero_count}")

    # 使用卡尔曼滤波器平滑每一列
    if kalman:
        print("\n开始卡尔曼滤波平滑...")
        tensornet_smoothed = tensornet_fixed.copy()
        
        # 减小过程噪声方差，增加测量噪声方差
        process_variance = 1e-5  # 原来是1e-3
        measurement_variance = 0.1**2  # 原来是0.01**2
        
        for column_idx in range(xpart):
            kf = KalmanFilter(process_variance=process_variance, measurement_variance=measurement_variance)
            column = tensornet_smoothed[:, column_idx]
            
            # 从上到下平滑
            for i in range(ypart):
                if column[i] != 0:
                    column[i] = kf.update(column[i])
            
            # 从下到上平滑
            kf = KalmanFilter(process_variance=process_variance, measurement_variance=measurement_variance)
            for i in range(ypart-1, -1, -1):
                if column[i] != 0:
                    column[i] = kf.update(column[i])
            
            tensornet_smoothed[:, column_idx] = column
    else:
        print("\n跳过卡尔曼滤波平滑步骤...")
        tensornet_smoothed = tensornet_fixed.copy()

    # 调整偏差
    print("\n开始调整偏差...")
    import random
    ypart, xpart = tensornet_smoothed.shape
    
    for row_idx in range(ypart):
        row = tensornet_smoothed[row_idx]
        row_median = np.median(row)  # 使用中位数
        adjusted_count = 0
        
        for col_idx in range(xpart):
            value = row[col_idx]
            deviation = abs(value - row_median)
            
            if deviation > row_median * 0.15 or deviation > 18 :
                # 使用中位数±random(0,1)替换
                new_value = row_median + random.uniform(-3, 3)
                tensornet_smoothed[row_idx, col_idx] = new_value
                adjusted_count += 1
                print(f"调整第{row_idx}行第{col_idx}列的值：原值{value} -> 新值{new_value}，偏差{deviation:.4f}")
        
        if adjusted_count > 0:
            print(f"第{row_idx}行共调整了{adjusted_count}个值")

    # 将所有负值替换为0
    print("\n替换所有负值为0...")
    negative_count = 0
    for row_idx in range(ypart):
        for col_idx in range(xpart):
            if tensornet_smoothed[row_idx, col_idx] < 0:
                tensornet_smoothed[row_idx, col_idx] = 0
                negative_count += 1
    if negative_count > 0:
        print(f"共替换{negative_count}个负值为0")

    # 保存结果
    df_fixed = pd.DataFrame(tensornet_smoothed)
    df_fixed.to_excel(output_file, index=False, header=False)
    print(f"\n修复完成，已保存到: {output_file}")
    
    # 生成热力图
    generate_heatmap(df_fixed, base_name, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='修复xlsx文件中的数据异常')
    parser.add_argument('--input', type=str, required=True, help='输入的xlsx文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出的修复后的xlsx文件路径')
    parser.add_argument('--kalman', action='store_true', help='是否启用卡尔曼滤波平滑')
    
    args = parser.parse_args()
    
    input_file = args.input
    
    if args.output is None:
        # 获取脚本运行目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 获取输入文件的基本信息
        input_filename = os.path.basename(input_file)
        input_filename_without_ext = os.path.splitext(input_filename)[0]
        
        # 创建xlsx_processed文件夹
        processed_dir = os.path.join(script_dir, 'xlsx_processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # 确保子文件夹名唯一（如果存在同名文件夹，则添加数字后缀）
        base_output_dir = os.path.join(processed_dir, input_filename_without_ext)
        output_dir = base_output_dir
        counter = 1
        
        while os.path.exists(output_dir):
            output_dir = f"{base_output_dir}_{counter}"
            counter += 1
        
        # 创建最终的输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        output_file = os.path.join(output_dir, f"{input_filename_without_ext}_processed.xlsx")
    else:
        output_file = args.output
    
    if os.path.exists(input_file):
        fix_xlsx(input_file, output_file, args.kalman)
    else:
        print(f"错误：文件 {input_file} 不存在")
