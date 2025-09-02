#!/usr/bin/env python3
import cv2
import numpy as np
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

def draw_initial_bbox(image, event_data):
    """演示draw_initial_bbox函数的功能"""
    x, y, w, h = 400, 1100, 60, 60
    
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=rgb_width, 
        sensor_height=rgb_height,
        fps=0.0,
        palette=ColorPalette.Dark
    )
    
    # 创建事件帧
    event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
    event_frame = np.ascontiguousarray(event_frame)
    event_frame_gen.generate_frame(event_data, event_frame, accumulation_time_us=1000)
    
    # 处理事件图像
    event_image_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(event_image_gray, 40, 255, cv2.THRESH_BINARY)
    event_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # 在两个图像上绘制边界框
    rgb_image = image.copy()
    cv2.rectangle(event_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
    cv2.rectangle(rgb_image, (x, y), (x + h, y + w), (0, 255, 0), 2)
    
    # 合并图像
    combined = np.hstack((rgb_image, event_image))
    scale = 0.8
    width = int(combined.shape[1] * scale)
    height = int(combined.shape[0] * scale)
    resized_img = cv2.resize(combined, (width, height), interpolation=cv2.INTER_AREA)
    
    # 保存结果而不是显示（避免GUI超时）
    cv2.imwrite('demo_output.jpg', resized_img)
    print("结果已保存到 demo_output.jpg")
    
    return x, y, w, h

def main():
    """主函数演示如何使用draw_initial_bbox"""
    # 加载图像和事件数据
    try:
        image = cv2.imread("95.jpg")
        event_data = np.load("95.npy")
        
        if image is None:
            print("错误: 无法加载图像文件 '121.jpg'")
            return
        if event_data is None:
            print("错误: 无法加载事件数据文件 '121.npy'")
            return
            
        print(f"加载的图像尺寸: {image.shape}")
        print(f"加载的事件数据形状: {event_data.shape}")
        
        # 调用draw_initial_bbox函数
        bbox = draw_initial_bbox(image, event_data)
        print(f"返回的边界框: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保安装了metavision_sdk_core库")
    except Exception as e:
        print(f"运行时错误: {e}")

if __name__ == "__main__":
    main()