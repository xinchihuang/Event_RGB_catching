import cv2
import numpy as np
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
    parser.add_argument('--image_path', type=str, help='RGB图像路径', default="/home/xinchi/Event_RGB_catching/122.jpg")
    parser.add_argument('--event_path', type=str, help='事件数据路径(.npy文件)',default="/home/xinchi/Event_RGB_catching/122.npy")
    parser.add_argument('--min_radius', type=int, default=10, help='最小球半径')


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
    
    # 执行检测
    print("开始球检测...")
    
    # 分别测试各种检测方法
    print("\n1. 霍夫圆检测:")
    hough_detections = detector.detect_by_hough_circles(image)
    print(f"   检测到 {len(hough_detections)} 个球: {hough_detections}")
    
    print("\n2. 轮廓检测:")
    contour_detections = detector.detect_by_contours(image)
    print(f"   检测到 {len(contour_detections)} 个球: {contour_detections}")
    
    if event_image is not None:
        print("\n3. 事件图像检测:")
        event_detections = detector.detect_from_events(event_image)
        print(f"   检测到 {len(event_detections)} 个球: {event_detections}")
    
    # 综合检测
    print("\n4. 综合检测:")
    final_detection = detector.detect_ball(image, event_image)
    if final_detection:
        print(f"   最终检测结果: x={final_detection[0]}, y={final_detection[1]}, w={final_detection[2]}, h={final_detection[3]}")
    else:
        print("   未检测到球")
    
    # 创建重合显示的图像
    if event_image is not None:
        # 将事件图像转换为彩色
        event_image_color = cv2.cvtColor(event_image, cv2.COLOR_GRAY2BGR)
        
        # 创建重合图像：RGB图像作为底色，事件图像叠加
        overlay = image.copy()
        
        # 将事件图像以半透明方式叠加到RGB图像上
        alpha = 0.6  # RGB图像透明度
        beta = 0.4   # 事件图像透明度
        blended = cv2.addWeighted(overlay, alpha, event_image_color, beta, 0)
        
        # 在重合图像上标注检测结果
        if final_detection:
            x, y, w, h = final_detection
            cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(blended, f"Ball: ({x},{y})", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 创建对比图像：左边RGB，右边事件图像，下面重合图像
        rgb_with_detection = image.copy()
        event_with_detection = event_image_color.copy()
        
        if final_detection:
            x, y, w, h = final_detection
            # 在RGB图像上标注
            cv2.rectangle(rgb_with_detection, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(rgb_with_detection, "RGB + Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 在事件图像上标注
            cv2.rectangle(event_with_detection, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(event_with_detection, "Event + Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 在重合图像上添加标签
            cv2.putText(blended, "RGB + Event Overlay", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 水平拼接RGB和事件图像
        top_row = np.hstack((rgb_with_detection, event_with_detection))
        
        # 将重合图像放在下方（需要调整大小匹配宽度）
        blended_resized = cv2.resize(blended, (top_row.shape[1], blended.shape[0]))
        
        # 垂直拼接
        combined_result = np.vstack((top_row, blended_resized))
        
        # 缩放最终图像
        scale = 0.4
        height, width = combined_result.shape[:2]
        new_size = (int(width * scale), int(height * scale))
        final_result = cv2.resize(combined_result, new_size, interpolation=cv2.INTER_AREA)
        
        # 保存重合显示结果
        output_path = "rgb_event_overlay_detection.jpg"
        cv2.imwrite(output_path, final_result)
        print(f"\nRGB+Event重合图像已保存到: {output_path}")
        
        # 也保存单独的重合图像
        blended_scaled = cv2.resize(blended, (int(blended.shape[1] * 0.6), int(blended.shape[0] * 0.6)))
        cv2.imwrite("rgb_event_blended.jpg", blended_scaled)
        print(f"重合图像已保存到: rgb_event_blended.jpg")
        
    elif final_detection:
        # 如果只有RGB检测结果
        result_image = image.copy()
        x, y, w, h = final_detection
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_image, "Detected Ball", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        scale = 0.8
        height, width = result_image.shape[:2]
        new_size = (int(width * scale), int(height * scale))
        resized_img = cv2.resize(result_image, new_size, interpolation=cv2.INTER_AREA)
        
        output_path = "ball_detection_result.jpg"
        cv2.imwrite(output_path, resized_img)
        print(f"\n检测结果已保存到: {output_path}")
    
    print("检测完成")

if __name__ == "__main__":
    main()