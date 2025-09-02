import cv2
import numpy as np
import time
from typing import Tuple, Optional, List

class BallDetector:
    """
    自动球检测器，用于在AMTTrack初始化时检测球的位置
    """
    
    def __init__(self, 
                 min_radius: int = 10, 
                 max_radius: int = 50,
                 dp: float = 1.0,
                 param1: float = 50,
                 param2: float = 30,
                 min_area: int = 100):
        """
        初始化球检测器
        
        Args:
            min_radius: 最小球半径
            max_radius: 最大球半径  
            dp: 霍夫圆检测的累加器分辨率
            param1: 霍夫圆检测第一个参数（边缘检测阈值）
            param2: 霍夫圆检测第二个参数（累加器阈值）
            min_area: 最小轮廓面积
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.dp = dp
        self.param1 = param1
        self.param2 = param2
        self.min_area = min_area
    
    def detect_by_hough_circles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        使用霍夫圆检测球
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 霍夫圆检测
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=50,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        bboxes = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # 转换圆形为边界框
                bbox_x = max(0, x - r)
                bbox_y = max(0, y - r)
                bbox_w = 2 * r
                bbox_h = 2 * r
                bboxes.append((bbox_x, bbox_y, bbox_w, bbox_h))
        
        return bboxes
    
    def detect_by_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        使用轮廓检测球形物体
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
                
            # 计算轮廓的圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 只保留相对圆形的轮廓
            if circularity > 0.4:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_from_events(self, event_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        从事件图像中检测球
        
        Args:
            event_image: 事件图像 (灰度或BGR)
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if len(event_image.shape) == 3:
            gray = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = event_image
        
        # 对事件图像进行阈值处理
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        
        # 使用DBSCAN聚类检测球形区域
        from sklearn.cluster import DBSCAN
        points = np.column_stack(np.where(binary > 0))
        
        if len(points) == 0:
            return []
        
        dbscan = DBSCAN(eps=9, min_samples=60)
        dbscan.fit(points)
        labels = dbscan.labels_
        
        bboxes = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # 噪声点
                continue
                
            cluster_points = points[labels == label]
            if len(cluster_points) < 100:  # 太小的簇忽略
                continue
                
            y_min, y_max = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
            x_min, x_max = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])
            
            x, y = x_min, y_min
            w, h = x_max - x_min, y_max - y_min
            
            # 过滤太小或形状不合理的检测
            if w > 10 and h > 10 and abs(w - h) / max(w, h) < 0.5:
                bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_ball(self, rgb_image: np.ndarray, event_image: Optional[np.ndarray] = None) -> Optional[Tuple[int, int, int, int]]:
        """
        综合检测球的位置
        
        Args:
            rgb_image: RGB图像
            event_image: 事件图像（可选）
            
        Returns:
            (x, y, w, h) 边界框或None
        """
        all_detections = []
        
        # 方法1: 霍夫圆检测
        hough_detections = self.detect_by_hough_circles(rgb_image)
        all_detections.extend(hough_detections)
        
        # 方法2: 轮廓检测
        contour_detections = self.detect_by_contours(rgb_image)
        all_detections.extend(contour_detections)
        
        # 方法3: 事件图像检测（如果提供）
        if event_image is not None:
            event_detections = self.detect_from_events(event_image)
            all_detections.extend(event_detections)
        
        if not all_detections:
            return None
        
        # 过滤掉明显不合理的检测（太大或太小）
        filtered_detections = []
        for x, y, w, h in all_detections:
            area = w * h
            # 过滤掉太大的检测（可能是整个图像）和太小的检测
            if 100 < area < 10000 and w > 10 and h > 10:
                filtered_detections.append((x, y, w, h))
        
        if not filtered_detections:
            return None
        
        # 选择面积适中的检测结果
        areas = [w * h for x, y, w, h in filtered_detections]
        median_area = np.median(areas)
        
        # 找到面积最接近中位数的检测
        best_idx = np.argmin([abs(area - median_area) for area in areas])
        
        return filtered_detections[best_idx]
    
    def visualize_detections(self, image: np.ndarray, detections: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测到的边界框列表
            
        Returns:
            绘制了边界框的图像
        """
        result = image.copy()
        for i, (x, y, w, h) in enumerate(detections):
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, f'Ball {i+1}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return result

def main():
    """
    测试球检测器的主函数
    """
    import sys
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='测试球检测器')
    parser.add_argument('--image_path', type=str, help='RGB图像路径', default="/home/xinchi/Event_RGB_catching/156.jpg")
    parser.add_argument('--event_path', type=str, help='事件数据路径(.npy文件)',default="/home/xinchi/Event_RGB_catching/157.npy")
    parser.add_argument('--min_radius', type=int, default=5, help='最小球半径')


    parser.add_argument('--max_radius', type=int, default=50, help='最大球半径')
    parser.add_argument('--param2', type=int, default=30, help='霍夫圆检测参数2')
    
    args = parser.parse_args()
    
    # 加载RGB图像
    if not os.path.exists(args.image_path):
        print(f"错误: 图像文件不存在: {args.image_path}")
        return
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"错误: 无法加载图像: {args.image_path}")
        return
    
    print(f"加载图像: {args.image_path}")
    print(f"图像尺寸: {image.shape}")
    
    # 初始化检测器
    detector = BallDetector(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        param2=args.param2
    )
    
    event_image = None
    
    # 如果提供了事件数据，生成事件图像
    if args.event_path:
        if not os.path.exists(args.event_path):
            print(f"警告: 事件文件不存在: {args.event_path}")
        else:
            try:
                from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
                
                event_data = np.load(args.event_path)
                print(f"加载事件数据: {args.event_path}")
                print(f"事件数量: {len(event_data)}")
                
                # 生成事件图像
                rgb_width, rgb_height = image.shape[1], image.shape[0]
                event_frame_gen = PeriodicFrameGenerationAlgorithm(
                    sensor_width=rgb_width, 
                    sensor_height=rgb_height,
                    fps=0.0,
                    palette=ColorPalette.Dark
                )
                
                event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
                event_frame = np.ascontiguousarray(event_frame)
                event_frame_gen.generate_frame(event_data, event_frame, accumulation_time_us=1000)
                event_image = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
                
                # 保存事件图像
                cv2.imwrite("event_image.jpg", event_image)
                cv2.imwrite("event_frame_color.jpg", event_frame)
                print("事件图像生成成功，已保存到 event_image.jpg 和 event_frame_color.jpg")
                
            except Exception as e:
                print(f"生成事件图像失败: {e}")
    
    # 执行检测并计时
    print("开始球检测...")
    
    # 分别测试各种检测方法
    print("\n1. 霍夫圆检测:")
    start_time = time.time()
    hough_detections = detector.detect_by_hough_circles(image)
    hough_time = time.time() - start_time
    print(f"   检测到 {len(hough_detections)} 个球: {hough_detections}")
    print(f"   用时: {hough_time*1000:.2f}ms")
    
    print("\n2. 轮廓检测:")
    start_time = time.time()
    contour_detections = detector.detect_by_contours(image)
    contour_time = time.time() - start_time
    print(f"   检测到 {len(contour_detections)} 个球: {contour_detections}")
    print(f"   用时: {contour_time*1000:.2f}ms")
    
    event_time = 0
    if event_image is not None:
        print("\n3. 事件图像检测:")
        start_time = time.time()
        event_detections = detector.detect_from_events(event_image)
        event_time = time.time() - start_time
        print(f"   检测到 {len(event_detections)} 个球: {event_detections}")
        print(f"   用时: {event_time*1000:.2f}ms")
    
    # 综合检测
    print("\n4. 综合检测:")
    start_time = time.time()
    final_detection = detector.detect_ball(image, event_image)
    total_time = time.time() - start_time
    
    if final_detection:
        print(f"   最终检测结果: x={final_detection[0]}, y={final_detection[1]}, w={final_detection[2]}, h={final_detection[3]}")
    else:
        print("   未检测到球")
    print(f"   综合检测用时: {total_time*1000:.2f}ms")
    
    # 总结时间
    total_all_time = hough_time + contour_time + event_time + total_time
    print(f"\n=== 检测时间总结 ===")
    print(f"霍夫圆检测: {hough_time*1000:.2f}ms")
    print(f"轮廓检测: {contour_time*1000:.2f}ms")
    if event_time > 0:
        print(f"事件检测: {event_time*1000:.2f}ms")
    print(f"综合检测: {total_time*1000:.2f}ms")
    print(f"总用时: {total_all_time*1000:.2f}ms")
    
    # 实时显示检测结果
    print("\n显示检测结果窗口...")
    print("按'q'键退出")
    
    # RGB检测结果
    rgb_result = image.copy()
    if final_detection:
        x, y, w, h = final_detection
        cv2.rectangle(rgb_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(rgb_result, "RGB Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rgb_result, f"Ball: ({x},{y}) {w}x{h}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Event检测结果（如果有）
    event_result = None
    blended = None
    if event_image is not None:
        event_image_color = cv2.cvtColor(event_image, cv2.COLOR_GRAY2BGR)
        event_result = event_image_color.copy()
        
        if final_detection:
            x, y, w, h = final_detection
            cv2.rectangle(event_result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(event_result, "Event Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 创建重合图像
        blended = cv2.addWeighted(rgb_result, 0.6, event_result, 0.4, 0)
        cv2.putText(blended, "RGB + Event Overlay", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # 缩放图像以适应屏幕
    scale = 0.6
    
    while True:
        # 显示RGB检测结果
        rgb_scaled = cv2.resize(rgb_result, (int(rgb_result.shape[1] * scale), int(rgb_result.shape[0] * scale)))
        cv2.imshow('RGB Detection', rgb_scaled)
        
        # 如果有Event数据，显示Event和重合结果
        if event_image is not None and event_result is not None and blended is not None:
            event_scaled = cv2.resize(event_result, (int(event_result.shape[1] * scale), int(event_result.shape[0] * scale)))
            blended_scaled = cv2.resize(blended, (int(blended.shape[1] * scale), int(blended.shape[0] * scale)))
            
            cv2.imshow('Event Detection', event_scaled)
            cv2.imshow('RGB + Event Overlay', blended_scaled)
        
        # 检测键盘输入
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("检测完成")

if __name__ == "__main__":
    main()