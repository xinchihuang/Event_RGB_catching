#!/usr/bin/env python3
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from evaluation_tools.plot_accuracy import index_to_position

from utils import rotate_matrix_coordinate
from fitting.parabola_fitter import MonoRansac
import matplotlib.pyplot as plt


def index_to_position(point_index,y_bias=-0.05,z_bias=-0.2):
    if point_index>0:
        row=point_index//10
        col=point_index%10
        y=(col)*0.1+y_bias+0.05
        z=(row)*0.1+z_bias+0.05
        return [0,y,z]
    else:
        point_index=-point_index
        row = point_index // 10
        col = point_index % 10
        # print(col,row)
        y = -(col) * 0.1 + y_bias + 0.05
        z = (row) * 0.1 + z_bias + 0.05

        return [0, y, z]

def position_to_index(position, y_bias=0, z_bias=0.2):
    """Convert a 3D position to a grid index."""
    y = position[1] - y_bias - 0.05
    z = position[2] - z_bias - 0.05
    row = int(z // 0.1)
    col = int(y // 0.1)
    print("row:", row, "col:", col, "y:", y, "z:", z)
    if y>=0:
        return -row * 10 + col
    else:
        return row * 10 - col
def process_one_case(path, label_file):
    camera_config = {
        "fx": 987.1678420424932,
        "fy": 986.0683237433913,
        "cy": 739.7418995490075,
        "cx": 513.7733221750243,
        "dist_coeffs": [
            -0.11123899870851763,
            -0.020663002330730863,
            -0.0002997445722290338,
            0.0007964973013426446,
            0.014818755754515196
        ]
    }
    camera_rpy_to = [0, 45, -90]
    camera_rpy_back = [45, 0, 90]
    rotation_matrix_to = rotate_matrix_coordinate(camera_rpy_to[0], camera_rpy_to[1], camera_rpy_to[2])
    rotation_matrix_back = rotate_matrix_coordinate(camera_rpy_back[0], camera_rpy_back[1], camera_rpy_back[2])

    labels = np.loadtxt(label_file, delimiter=",")
    fitter = MonoRansac(camera_params=camera_config, rotation_matrix_to=rotation_matrix_to,
                        rotation_matrix_back=rotation_matrix_back)
    exp_results = []

    for i in range(0, len(labels)):
        # Skip invalid data where label is 0
        if labels[i] == 0:
            print(f"Skipping trajectory {i}: invalid label (0)")
            continue
        try:
            ball_memory=np.loadtxt(os.path.join(path, f"{i}.csv"), delimiter=",")
            if ball_memory.size == 0:
                print(f"Skipping trajectory {i}: empty CSV file")
                continue
            if ball_memory.ndim == 1:
                print(f"Skipping trajectory {i}: invalid data format (1D array)")
                continue
        except Exception as e:
            print(f"Skipping trajectory {i}: error loading data - {e}")
            continue
        observations = ball_memory
        print(observations)
        observations[:, 3] = observations[:, 3] - observations[0, 3]
        observations = observations[:10, :]
        fitter.set_data(observations)
        result = fitter.get_params()

        impact_point_real = index_to_position(labels[i])
        ball = result['points']

        impact_point = fitter.predict()
        dict_to_save = {
                        'impact': impact_point,
                        'label': impact_point_real,
                        }
        exp_results.append(dict_to_save)
    return exp_results



def prepare_data():
    """遍历根目录下的所有子目录并加载数据"""
    root_dir = ("/home/xinchi/Event_RGB_catching_data/old/test")

    normal_path=["/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_event_l",
                 "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_RGB_l",
                 "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_l",
                ]
    clutter_path=["/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_RGB_c",
                 "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_c",]
    dark_path=["/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_event_d",
                 "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_d",]
    dark_clutter_path=[
                 "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_dc",]


    normal_label=os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_tools/label_l.csv")
    clutter_label=os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_tools/label_c.csv")
    dark_label=os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_tools/label_d.csv")
    dark_clutter_label=os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation_tools/label_dc.csv")

    normal_data=[]
    for path in normal_path:
        print("Processing path:", path)
        normal_data.append(process_one_case(path,normal_label))
    clutter_data=[]
    for path in clutter_path:
        print("Processing path:", path)
        clutter_data.append(process_one_case(path,clutter_label))
    dark_data=[]
    for path in dark_path:
        print("Processing path:", path)
        dark_data.append(process_one_case(path,dark_label))
    dark_clutter_data=[]
    for path in dark_clutter_path:
        print("Processing path:", path)
        dark_clutter_data.append(process_one_case(path,dark_clutter_label))
    print(len(normal_data))
    return normal_data,clutter_data,dark_data,dark_clutter_data

def remove_outliers(errors, threshold=None, method='iqr'):
    """去除异常值"""
    if len(errors) == 0:
        return errors, 0, 0.0
    
    errors_array = np.array(errors)
    original_count = len(errors_array)
    
    if method == 'iqr':
        Q1 = np.percentile(errors_array, 25)
        Q3 = np.percentile(errors_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_errors = errors_array[(errors_array >= lower_bound) & (errors_array <= upper_bound)]
    elif method == 'threshold' and threshold is not None:
        filtered_errors = errors_array[errors_array <= threshold]
    else:
        filtered_errors = errors_array
    
    filtered_count = len(filtered_errors)
    removed_count = original_count - filtered_count
    removed_percentage = (removed_count / original_count) * 100 if original_count > 0 else 0.0
    
    return filtered_errors.tolist(), removed_count, removed_percentage

def get_method_from_path(path):
    """从路径中提取方法名称"""
    last_dir = os.path.basename(path)
    if 'event' in last_dir:
        return 'event'
    elif 'RGB' in last_dir:
        return 'rgb'
    elif 'model' in last_dir:
        return 'model'
    return 'unknown'

def calculate_errors(data_groups, data_paths, filter_outliers=True, outlier_threshold=None):
    """计算每组数据中预测impact point和真实impact point之间的距离"""
    all_errors = {}
    all_methods = {}
    
    print("\n过滤统计:")
    
    for group_name, group_data in data_groups.items():
        method_errors = []
        method_names = []
        paths = data_paths[group_name]
        
        for i, method_data in enumerate(group_data):
            single_method_errors = []
            for result in method_data:
                predicted = result['impact']
                real = result['label']
                error = np.linalg.norm(np.array(predicted) - np.array(real))
                if error > 0.1:
                    error = error - 0.1
                single_method_errors.append(error)
            
            method_name = get_method_from_path(paths[i]) if i < len(paths) else 'unknown'
            
            if filter_outliers:
                if outlier_threshold is not None:
                    filtered_errors, removed_count, removed_percentage = remove_outliers(single_method_errors, threshold=outlier_threshold, method='threshold')
                else:
                    filtered_errors, removed_count, removed_percentage = remove_outliers(single_method_errors, method='iqr')
                
                # 打印过滤统计
                label_map = {
                    'normal': 'Normal Illumination',
                    'clutter': 'Background Influence',
                    'dark': 'Low Illumination',
                    'dark_clutter': 'LI+BI'
                }
                display_label = label_map.get(group_name, group_name)
                print(f"{display_label} - {method_name.upper()}: 过滤掉 {removed_count}/{len(single_method_errors)} ({removed_percentage:.1f}%)")
                
                single_method_errors = filtered_errors
            
            method_errors.append(single_method_errors)
            method_names.append(method_name)
        
        all_errors[group_name] = method_errors
        all_methods[group_name] = method_names
    
    return all_errors, all_methods

def plot_boxplots(errors_dict, methods_dict):
    """绘制boxplot，每种data类型一组，组内画多个boxplot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})
    
    data_types = list(errors_dict.keys())
    color_map = {'event': 'lightblue', 'rgb': 'lightgreen', 'model': 'lightcoral'}
    
    # 定义横轴标签映射
    label_map = {
        'normal': 'Normal Illumination',
        'clutter': 'Background Influence',
        'dark': 'Low Illumination',
        'dark_clutter': 'LI+BI'
    }
    
    all_box_data = []
    all_positions = []
    all_colors = []
    
    group_width = 0.8
    group_spacing = 1.2  # 减小组间距离
    
    pos = 1
    for data_type in data_types:
        method_errors = errors_dict[data_type]
        method_names = methods_dict[data_type]
        
        if len(method_errors) > 0:
            box_width = group_width / len(method_errors)
            for i, single_method_errors in enumerate(method_errors):
                if len(single_method_errors) > 0:
                    all_box_data.append(single_method_errors)
                    box_pos = pos + (i - len(method_errors)/2 + 0.5) * box_width
                    all_positions.append(box_pos)
                    method_name = method_names[i]
                    all_colors.append(color_map.get(method_name, 'gray'))
            pos += group_spacing
    
    box_width = 0.2
    if all_box_data:
        bp = ax.boxplot(all_box_data, positions=all_positions, patch_artist=True, widths=box_width)
        
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(all_colors[i])
        
        # 加深中位数线颜色并打印中位数值
        print("\n中位数值:")
        box_index = 0
        for data_type in data_types:
            method_errors = errors_dict[data_type]
            method_names = methods_dict[data_type]
            
            if len(method_errors) > 0:
                for i, single_method_errors in enumerate(method_errors):
                    if len(single_method_errors) > 0:
                        median_val = np.median(single_method_errors)
                        method_name = method_names[i]
                        display_label = label_map.get(data_type, data_type)
                        print(f"{display_label} - {method_name.upper()}: {median_val:.4f}")
                        box_index += 1
        
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        group_positions = []
        group_labels = []
        pos = 1
        for data_type in data_types:
            method_errors = errors_dict[data_type]
            if len(method_errors) > 0 and any(len(me) > 0 for me in method_errors):
                group_positions.append(pos)
                display_label = label_map.get(data_type, data_type)
                group_labels.append(display_label)
                pos += group_spacing
        
        ax.set_xticks(group_positions)
        ax.set_xticklabels(group_labels, fontsize=16)
        ax.set_ylabel('Perception Error(m)', fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 调整y轴刻度标签字体大小
        ax.tick_params(axis='y', labelsize=16)
        
        legend_elements = []
        used_methods = set()
        for method_list in methods_dict.values():
            for method in method_list:
                if method not in used_methods:
                    color = color_map.get(method, 'gray')
                    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, label=method.upper()))
                    used_methods.add(method)
        ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
        
        # 设置x轴范围以减少空白区域
        if group_positions:
            ax.set_xlim(group_positions[0] - 0.8, group_positions[-1] + 0.8)
    
    plt.tight_layout()
    plt.savefig('perception_error.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    normal_data, clutter_data, dark_data, dark_clutter_data = prepare_data()
    
    data_groups = {
        'normal': normal_data,
        'clutter': clutter_data, 
        'dark': dark_data,
        'dark_clutter': dark_clutter_data
    }
    
    data_paths = {
        'normal': ["/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_event_l",
                   "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_RGB_l",
                   "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_l"],
        'clutter': ["/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_RGB_c",
                    "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_c"],
        'dark': ["/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_event_d",
                 "/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_d"],
        'dark_clutter': ["/home/xinchi/Event_RGB_catching_data/old/test/ball_pixel_3D_records_model_dc"]
    }
    
    errors, methods = calculate_errors(data_groups, data_paths, filter_outliers=True, outlier_threshold=0.25)
    plot_boxplots(errors, methods)