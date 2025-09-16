import numpy as np
from typing import Tuple, Optional
from initial_bbox.ball_detector import BallDetector

class RealTimeBallDetector:
    """
    实时球检测器，用于在跟踪过程中动态检测和重新初始化
    """
    
    def __init__(self):
        self.ball_detector = BallDetector(min_radius=15, max_radius=40, param2=25)
        self.confidence_threshold = 0.7
        self.detection_cooldown = 30  # 帧数
        self.last_detection_frame = 0
        self.tracking_lost_threshold = 5  # 连续失败跟踪次数阈值
        self.consecutive_failures = 0
        
    def should_redetect(self, frame_count: int, tracking_confidence: float = 1.0) -> bool:
        """
        判断是否需要重新检测球的位置
        
        Args:
            frame_count: 当前帧数
            tracking_confidence: 跟踪置信度（0-1）
            
        Returns:
            bool: 是否需要重新检测
        """
        # 如果跟踪置信度低
        if tracking_confidence < self.confidence_threshold:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            
        # 检查是否需要重新检测
        frames_since_last = frame_count - self.last_detection_frame
        
        return (self.consecutive_failures >= self.tracking_lost_threshold and 
                frames_since_last >= self.detection_cooldown)
    
    def detect_and_reinitialize(self, rgb_image: np.ndarray, event_image: np.ndarray, 
                               current_bbox: Tuple[int, int, int, int],
                               frame_count: int) -> Optional[Tuple[int, int, int, int]]:
        """
        检测并重新初始化球的位置
        
        Args:
            rgb_image: RGB图像
            event_image: 事件图像
            current_bbox: 当前跟踪的边界框
            frame_count: 当前帧数
            
        Returns:
            新的边界框或None（如果检测失败）
        """
        # 尝试检测新的球位置
        detected_bbox = self.ball_detector.detect_ball(rgb_image, event_image)
        
        if detected_bbox is not None:
            # 验证检测结果是否合理（与当前位置不能相差太远）
            current_x, current_y, current_w, current_h = current_bbox
            detected_x, detected_y, detected_w, detected_h = detected_bbox
            
            # 计算中心点距离
            current_center = (current_x + current_w/2, current_y + current_h/2)
            detected_center = (detected_x + detected_w/2, detected_y + detected_h/2)
            
            distance = np.sqrt((current_center[0] - detected_center[0])**2 + 
                             (current_center[1] - detected_center[1])**2)
            
            # 如果距离太远，可能是误检测
            max_distance = max(current_w, current_h) * 3
            if distance > max_distance:
                print(f"检测位置距离当前位置太远 (距离: {distance:.1f}), 忽略")
                return None
            
            self.last_detection_frame = frame_count
            self.consecutive_failures = 0
            
            print(f"重新检测到球的位置: x={detected_x}, y={detected_y}, w={detected_w}, h={detected_h}")
            return detected_bbox
        
        return None
    
    def get_search_region(self, bbox: Tuple[int, int, int, int], expand_factor: float = 2.0) -> Tuple[int, int, int, int]:
        """
        基于当前边界框获取搜索区域
        
        Args:
            bbox: 当前边界框 (x, y, w, h)
            expand_factor: 扩展因子
            
        Returns:
            扩展后的搜索区域 (x, y, w, h)
        """
        x, y, w, h = bbox
        
        # 计算扩展大小
        expand_w = int(w * expand_factor)
        expand_h = int(h * expand_factor)
        
        # 计算新的位置（确保在图像范围内）
        new_x = max(0, x - (expand_w - w) // 2)
        new_y = max(0, y - (expand_h - h) // 2)
        
        return (new_x, new_y, expand_w, expand_h)