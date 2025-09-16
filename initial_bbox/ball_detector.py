import cv2
import numpy as np
import time
from typing import Tuple, Optional, List

class BallDetector:
    """
    自动球检测器，用于在AMTTrack初始化时检测球的位置
    """
    
    def __init__(self, 
                 min_radius: int = 15, 
                 max_radius: int = 30,
                 dp: float = 1.0,
                 param1: float = 50,
                 param2: float = 25,
                 min_area: int = 400,
                 # Event检测参数
                 event_threshold: int = 30,
                 min_component_area: int = 100,
                 max_component_area: int = 3000,
                 min_width: int = 20,
                 max_width: int = 80,
                 min_height: int = 20,
                 max_height: int = 80,
                 min_aspect_ratio: float = 0.6,
                 max_distance_from_center: float = 300.0,
                 center_x: int = 515,
                 center_y: int = 1118):
        """
        初始化球检测器
        
        Args:
            min_radius: 最小球半径
            max_radius: 最大球半径  
            dp: 霍夫圆检测的累加器分辨率
            param1: 霍夫圆检测第一个参数（边缘检测阈值）
            param2: 霍夫圆检测第二个参数（累加器阈值）
            min_area: 最小轮廓面积
            
            # Event检测专用参数
            event_threshold: 事件图像二值化阈值
            min_component_area: 连通组件最小面积
            max_component_area: 连通组件最大面积
            min_width: 候选框最小宽度
            max_width: 候选框最大宽度
            min_height: 候选框最小高度
            max_height: 候选框最大高度
            min_aspect_ratio: 最小长宽比（接近1表示更圆）
            max_distance_from_center: 距离预期中心的最大距离
            center_x: 预期球中心x坐标
            center_y: 预期球中心y坐标
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.dp = dp
        self.param1 = param1
        self.param2 = param2
        self.min_area = min_area
        
        # Event检测参数
        self.event_threshold = event_threshold
        self.min_component_area = min_component_area
        self.max_component_area = max_component_area
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_distance_from_center = max_distance_from_center
        self.center_x = center_x
        self.center_y = center_y
    
    def detect_by_hough_circles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        使用霍夫圆检测球
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        # 增加RGB亮度
        brightened = cv2.convertScaleAbs(image, alpha=5, beta=30)
        gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 霍夫圆检测
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=60,
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
        # 增加RGB亮度和对比度
        brightened = cv2.convertScaleAbs(image, alpha=5, beta=40)
        gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)
        
        # 多种阈值处理方法
        # 方法1: 自适应阈值
        binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 方法2: Otsu阈值
        _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 方法3: 基于均值的阈值
        mean_val = np.mean(gray)
        _, binary3 = cv2.threshold(gray, mean_val - 20, 255, cv2.THRESH_BINARY)
        
        # 组合多种二值化结果
        binary = cv2.bitwise_or(binary1, cv2.bitwise_or(binary2, binary3))
        
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
            
            # 提高圆形度要求
            if circularity > 0.6:
                x, y, w, h = cv2.boundingRect(contour)
                # 检查长宽比，允许稍微椭圆的形状
                aspect_ratio = min(w, h) / max(w, h)
                if aspect_ratio > 0.7:  # 要求更接近圆形
                    bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_from_events_fast(self, event_image: np.ndarray, debug: bool = False) -> List[Tuple[int, int, int, int]]:
        """
        快速事件检测方法 - 使用可调参数
        
        Args:
            event_image: 事件图像 (灰度或BGR)
            debug: 是否打印调试信息
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if len(event_image.shape) == 3:
            gray = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = event_image
        
        # 使用可调的阈值参数
        _, binary = cv2.threshold(gray, self.event_threshold, 255, cv2.THRESH_BINARY)
        
        # 使用连通组件分析替代DBSCAN（更快）
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        candidates = []  # 存储所有候选框信息用于调试
        
        for i in range(1, num_labels):  # 跳过背景（标签0）
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 计算候选框属性
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            center_x, center_y = centroids[i]
            distance = np.sqrt((center_x - self.center_x)**2 + (center_y - self.center_y)**2)
            
            # 收集所有候选框信息（用于调试）
            candidate_info = {
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'center': (center_x, center_y),
                'distance': distance,
                'passes_filters': True
            }
            
            # 应用筛选条件
            if area < self.min_component_area or area > self.max_component_area:
                candidate_info['passes_filters'] = False
                if debug: print(f"候选框{i}: 面积{area}不符合范围[{self.min_component_area}, {self.max_component_area}]")
                
            if w < self.min_width or w > self.max_width:
                candidate_info['passes_filters'] = False
                if debug: print(f"候选框{i}: 宽度{w}不符合范围[{self.min_width}, {self.max_width}]")
                
            if h < self.min_height or h > self.max_height:
                candidate_info['passes_filters'] = False
                if debug: print(f"候选框{i}: 高度{h}不符合范围[{self.min_height}, {self.max_height}]")
                
            if aspect_ratio < self.min_aspect_ratio:
                candidate_info['passes_filters'] = False
                if debug: print(f"候选框{i}: 长宽比{aspect_ratio:.3f}小于{self.min_aspect_ratio}")
                
            if distance > self.max_distance_from_center:
                candidate_info['passes_filters'] = False
                if debug: print(f"候选框{i}: 距离中心{distance:.1f}大于{self.max_distance_from_center}")
            
            candidates.append(candidate_info)
            
            if debug:
                print(f"候选框{i}: 位置=({x},{y}), 尺寸={w}x{h}, 面积={area}, "
                      f"长宽比={aspect_ratio:.3f}, 中心=({center_x:.1f},{center_y:.1f}), "
                      f"距离={distance:.1f}, 通过={candidate_info['passes_filters']}")
        
        # 筛选通过的候选框
        valid_candidates = [c for c in candidates if c['passes_filters']]
        
        if debug:
            print(f"\n总候选框数: {len(candidates)}, 通过筛选: {len(valid_candidates)}")
        
        # 按距离中心的远近排序，选择最接近预期位置的
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x['distance'])
            if debug:
                print("按距离排序的有效候选框:")
                for i, c in enumerate(valid_candidates):
                    print(f"  {i+1}. 距离={c['distance']:.1f}, 位置={c['bbox']}")
            
            return [c['bbox'] for c in valid_candidates]  # 返回所有有效候选框
        
        return []

    def detect_from_events(self, event_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        从事件图像中检测球 - 先尝试快速方法
        
        Args:
            event_image: 事件图像 (灰度或BGR)
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        # 首先尝试快速检测
        fast_result = self.detect_from_events_fast(event_image)
        if fast_result:
            return fast_result
            
        # 如果快速检测失败，使用原有的霍夫圆检测
        if len(event_image.shape) == 3:
            gray = cv2.cvtColor(event_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = event_image
        
        bboxes = []
        
        # 霍夫圆检测作为后备方法
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=2.0,  # 增加dp以减少计算
            minDist=50,
            param1=30,  # 调整参数
            param2=15,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                bbox_x = max(0, x - r)
                bbox_y = max(0, y - r)
                bbox_w = 2 * r
                bbox_h = 2 * r
                bboxes.append((bbox_x, bbox_y, bbox_w, bbox_h))
        
        return bboxes
    
    def detect_ball_fast(self, rgb_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        快速球检测方法 - 仅使用优化的霍夫圆检测
        
        Args:
            rgb_image: RGB图像
            
        Returns:
            (x, y, w, h) 边界框或None
        """
        # 降采样以提高速度
        scale = 0.5 if min(rgb_image.shape[:2]) > 500 else 1.0
        if scale < 1.0:
            small_image = cv2.resize(rgb_image, None, fx=scale, fy=scale)
        else:
            small_image = rgb_image
        
        # 简化的预处理
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        
        # 优化的霍夫圆检测参数
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=2.0,  # 增加dp以减少计算
            minDist=30,
            param1=50,
            param2=20,  # 降低阈值以更容易检测
            minRadius=int(self.min_radius * scale),
            maxRadius=int(self.max_radius * scale)
        )
        
        if circles is None:
            return None
        
        # 取第一个检测到的圆
        circles = np.round(circles[0, :]).astype("int")
        x, y, r = circles[0]  # 只取第一个
        
        # 转换回原始尺寸
        if scale < 1.0:
            x = int(x / scale)
            y = int(y / scale)
            r = int(r / scale)
        
        # 转换为边界框
        bbox_x = max(0, x - r)
        bbox_y = max(0, y - r)
        bbox_w = 2 * r
        bbox_h = 2 * r
        
        return (bbox_x, bbox_y, bbox_w, bbox_h)

    def select_best_candidate(self, candidates: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        从候选框中选择最佳的球检测结果
        
        Args:
            candidates: 候选框列表 [(x, y, w, h), ...]
            
        Returns:
            最佳候选框或None
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # 计算每个候选框的综合得分
        scored_candidates = []
        
        for x, y, w, h in candidates:
            # 计算候选框属性
            area = w * h
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            center_x, center_y = x + w//2, y + h//2
            distance = np.sqrt((center_x - self.center_x)**2 + (center_y - self.center_y)**2)
            
            # 计算各项得分 (0-1范围)
            # 1. 长宽比得分：越接近1(圆形)得分越高
            aspect_score = aspect_ratio
            
            # 2. 距离得分：距离预期中心越近得分越高
            max_reasonable_distance = self.max_distance_from_center
            distance_score = max(0, 1 - distance / max_reasonable_distance)
            
            # 3. 尺寸得分：越接近预期大小得分越高
            ideal_size = (self.min_width + self.max_width) / 2  # 预期尺寸
            size_diff = abs(w - ideal_size) + abs(h - ideal_size)
            size_score = max(0, 1 - size_diff / ideal_size)
            
            # 4. 面积得分：在合理范围内，适中的面积得分更高
            ideal_area = (self.min_component_area + self.max_component_area) / 2
            area_diff = abs(area - ideal_area)
            area_score = max(0, 1 - area_diff / ideal_area)
            
            # 综合得分 (加权平均)
            total_score = (
                aspect_score * 0.3 +      # 长宽比权重30%
                distance_score * 0.4 +    # 距离权重40% 
                size_score * 0.2 +        # 尺寸权重20%
                area_score * 0.1          # 面积权重10%
            )
            
            scored_candidates.append({
                'bbox': (x, y, w, h),
                'score': total_score,
                'aspect_ratio': aspect_ratio,
                'distance': distance,
                'area': area
            })
        
        # 按综合得分降序排序，选择得分最高的
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        return scored_candidates[0]['bbox']
    
    def detect_ball(self, rgb_image: np.ndarray, event_image: Optional[np.ndarray] = None, debug: bool = False) -> Optional[Tuple[int, int, int, int]]:
        """
        综合检测球的位置 - 优先使用event检测并进行候选框筛选
        
        Args:
            rgb_image: RGB图像
            event_image: 事件图像（必需用于准确检测）
            debug: 是否显示候选框选择过程
            
        Returns:
            (x, y, w, h) 边界框或None
        """
        # 优先使用event检测（如果提供）
        if event_image is not None:
            event_detections = self.detect_from_events(event_image)
            if event_detections:
                if debug:
                    print(f"Event检测到{len(event_detections)}个候选框，开始筛选最佳候选框...")
                best_candidate = self.select_best_candidate(event_detections)
                if best_candidate:
                    if debug:
                        print(f"选择最佳候选框: {best_candidate}")
                    return best_candidate
        
        # 如果没有event图像或event检测失败，使用快速RGB检测
        fast_result = self.detect_ball_fast(rgb_image)
        if fast_result is not None:
            return fast_result
        
        # 最后回退到原有的霍夫圆检测，也进行候选框筛选
        hough_detections = self.detect_by_hough_circles(rgb_image)
        if hough_detections:
            best_candidate = self.select_best_candidate(hough_detections)
            return best_candidate
        
        return None
    
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
    parser.add_argument('--image_path', type=str, help='RGB图像路径', default="/home/xinchi/Event_RGB_catching/94.jpg")
    parser.add_argument('--event_path', type=str, help='事件数据路径(.npy文件)',default="/home/xinchi/Event_RGB_catching/94.npy")
    # parser.add_argument('--image_path', type=str, help='RGB图像路径', default="/home/xinchi/Event_RGB_catching/102.jpg")
    # parser.add_argument('--event_path', type=str, help='事件数据路径(.npy文件)',
    #                     default="/home/xinchi/Event_RGB_catching/102.npy")1
    parser.add_argument('--min_radius', type=int, default=1, help='最小球半径')
    parser.add_argument('--max_radius', type=int, default=50, help='最大球半径')
    parser.add_argument('--param2', type=int, default=15, help='霍夫圆检测参数2')
    
    # Event检测参数
    parser.add_argument('--event_threshold', type=int, default=40, help='事件图像二值化阈值')
    parser.add_argument('--min_area', type=int, default=100, help='连通组件最小面积')
    parser.add_argument('--max_area', type=int, default=3000, help='连通组件最大面积')
    parser.add_argument('--min_width', type=int, default=20, help='候选框最小宽度')
    parser.add_argument('--max_width', type=int, default=80, help='候选框最大宽度')
    parser.add_argument('--min_height', type=int, default=20, help='候选框最小高度')
    parser.add_argument('--max_height', type=int, default=80, help='候选框最大高度')
    parser.add_argument('--min_aspect_ratio', type=float, default=0.6, help='最小长宽比')
    parser.add_argument('--max_distance', type=float, default=300.0, help='距离预期中心的最大距离')
    parser.add_argument('--center_x', type=int, default=515, help='预期球中心x坐标')
    parser.add_argument('--center_y', type=int, default=1118, help='预期球中心y坐标')
    parser.add_argument('--debug', action='store_true', help='启用调试输出')
    
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
        param2=args.param2,
        # Event检测参数
        event_threshold=args.event_threshold,
        min_component_area=args.min_area,
        max_component_area=args.max_area,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        min_aspect_ratio=args.min_aspect_ratio,
        max_distance_from_center=args.max_distance,
        center_x=args.center_x,
        center_y=args.center_y
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
                cv2.imwrite("../ROI_generation/event_image.jpg", event_image)
                cv2.imwrite("../ROI_generation/event_frame_color.jpg", event_frame)
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
        # 启用调试模式显示详细信息
        if args.debug:
            print("=== Event检测调试信息 ===")
            event_detections = detector.detect_from_events_fast(event_image, debug=True)
        else:
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
    
    # 使用亮度增强的RGB图像作为基础
    brightened_rgb = cv2.convertScaleAbs(image, alpha=0.3, beta=30)
    rgb_result = brightened_rgb.copy()
    
    # 绘制所有检测方法的结果
    # 霍夫圆检测 - 红色
    for i, (x, y, w, h) in enumerate(hough_detections):
        cv2.rectangle(rgb_result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(rgb_result, f"Hough{i+1}", (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # 轮廓检测 - 蓝色
    for i, (x, y, w, h) in enumerate(contour_detections):
        cv2.rectangle(rgb_result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(rgb_result, f"Contour{i+1}", (x, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    
    # 最终检测结果 - 绿色
    if final_detection:
        x, y, w, h = final_detection
        cv2.rectangle(rgb_result, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(rgb_result, "Final Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rgb_result, f"Ball: ({x},{y}) {w}x{h}", (x, y-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # 添加检测统计信息
    event_count = len(detector.detect_from_events(event_image)) if event_image is not None else 0
    detection_text = f"Total: H={len(hough_detections)}, C={len(contour_detections)}, E={event_count}"
    cv2.putText(rgb_result, detection_text, (10, rgb_result.shape[0]-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Event检测结果（如果有）
    event_result = None
    blended = None
    if event_image is not None:
        event_image_color = cv2.cvtColor(event_image, cv2.COLOR_GRAY2BGR)
        event_result = event_image_color.copy()
        
        # 在事件图像上绘制事件检测结果 - 黄色
        event_detections_only = detector.detect_from_events(event_image)
        for i, (x, y, w, h) in enumerate(event_detections_only):
            cv2.rectangle(event_result, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(event_result, f"Event{i+1}", (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        
        if final_detection:
            x, y, w, h = final_detection
            cv2.rectangle(event_result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(event_result, "Final", (x, y-45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # 创建高清重合图像
        blended = cv2.addWeighted(rgb_result, 0.6, event_result, 0.4, 0)
        
        # 高清字体显示
        cv2.putText(blended, "RGB + Event Overlay (High Quality)", (15, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # 添加检测统计信息（高清）
        stats_text = [
            f"Hough: {len(hough_detections)} detections",
            f"Contour: {len(contour_detections)} detections", 
            f"Event: {len(event_detections_only)} detections"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(blended, text, (15, 120 + i*40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # 最终检测结果信息
        if final_detection:
            x, y, w, h = final_detection
            result_text = f"Final: ({x},{y}) {w}x{h}"
            cv2.putText(blended, result_text, (15, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        
        # 保存高清版本
        cv2.imwrite("../ROI_generation/detection_result_hq.jpg", blended)
        print(f"高清检测结果已保存到: detection_result_hq.jpg")
    
    # 高清显示设置
    display_scale = 0.8  # 提高显示比例
    
    while True:
        # 高清显示RGB检测结果
        rgb_scaled = cv2.resize(rgb_result, (int(rgb_result.shape[1] * display_scale), int(rgb_result.shape[0] * display_scale)))
        cv2.imshow('RGB Detection', rgb_scaled)
        
        # 如果有Event数据，高清显示Event和重合结果
        if event_image is not None and event_result is not None and blended is not None:
            event_scaled = cv2.resize(event_result, (int(event_result.shape[1] * display_scale), int(event_result.shape[0] * display_scale)))
            blended_scaled = cv2.resize(blended, (int(blended.shape[1] * display_scale), int(blended.shape[0] * display_scale)))
            
            cv2.imshow('Event Detection', event_scaled)
            cv2.imshow('RGB + Event Overlay (High Quality)', blended_scaled)
        
        # 检测键盘输入
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print("检测完成")

if __name__ == "__main__":
    main()