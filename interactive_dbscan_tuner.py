#!/usr/bin/env python3
"""
交互式DBSCAN参数调试器
功能：
- 从.raw文件读取事件数据
- 使用DBSCAN进行聚类
- 在图像上显示聚类结果
- 支持键盘调整DBSCAN参数
- 支持播放前后帧（方向键）
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import sys
import argparse
import time

# Add project path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from metavision_core.event_io import EventsIterator
from config.config_loader import Config, get_config
from data_input.read_from_raw import align_to_rgb


class InteractiveDBSCANTuner:
    """交互式DBSCAN参数调试器"""

    def __init__(self, raw_file, display_scale=0.6):
        """
        初始化

        Args:
            raw_file: .raw文件路径
            display_scale: 显示缩放比例
        """
        self.raw_file = raw_file
        self.display_scale = display_scale

        # 获取配置
        config = get_config()
        self.data_reading = Config.get_data_reading()
        self.tracking = Config.get_tracking()
        self.image_dims = Config.get_image_dimensions()

        # 图像尺寸
        self.width = self.image_dims['rgb_width']
        self.height = self.image_dims['rgb_height']

        # DBSCAN参数
        self.eps = self.tracking.get('dbscan_eps', 12)
        self.min_samples = self.tracking.get('dbscan_min_samples', 70)

        # 读取参数
        self.delta_t = self.data_reading.get('delta_t', 2000)
        self.n_events = self.data_reading.get('n_events', 100000)

        # 加载事件数据
        self.event_chunks = []
        self.current_frame = 0
        self.total_frames = 0

        # 自动播放相关
        self.auto_play = False
        self.fps = 10
        self.last_frame_time = 0

        # 缓存机制
        self.cache = {}  # 缓存聚类结果
        self.last_eps = self.eps
        self.last_min_samples = self.min_samples
        self.last_frame = -1

        print(f"Initialization parameters:")
        print(f"  Image size: {self.width}x{self.height}")
        print(f"  DBSCAN params: eps={self.eps}, min_samples={self.min_samples}")
        print(f"  Read params: delta_t={self.delta_t}us, n_events={self.n_events}")

        self._load_event_data()

    def _load_event_data(self):
        """加载.raw文件中的事件数据"""
        print(f"Loading .raw file: {self.raw_file}")

        # 使用EventsIterator读取事件数据
        iterator = EventsIterator(
            input_path=self.raw_file,
            mode="delta_t",
            delta_t=self.delta_t,
            n_events=self.n_events
        )

        chunk_count = 0
        total_events = 0

        for ev in iterator:
            if len(ev) == 0:
                continue

            # 应用坐标变换 (同read_from_raw.py)
            x = ev['x'].copy()
            y = ev['y'].copy()

            # 步骤1: 交换x,y并取负x
            x_new = y.copy()
            y_new = -x.copy()

            # 步骤2: 应用y坐标偏移
            alignment = Config.get_alignment()
            y_new = y_new + alignment['y_coordinate_offset']

            # 步骤3: 对齐到RGB尺寸
            x_aligned, y_aligned = align_to_rgb({'x': x_new, 'y': y_new})

            # 创建变换后的事件数组
            ev_transformed = np.zeros(ev.shape, dtype=ev.dtype)
            ev_transformed['x'] = x_aligned
            ev_transformed['y'] = y_aligned
            ev_transformed['p'] = ev['p'].copy()
            ev_transformed['t'] = ev['t'].copy()

            self.event_chunks.append(ev_transformed)
            total_events += len(ev_transformed)
            chunk_count += 1

            if chunk_count % 100 == 0:
                print(f"  Loaded {chunk_count} chunks, {total_events} events...")

        self.total_frames = len(self.event_chunks)
        print(f"✓ Loading complete: {self.total_frames} frames, total {total_events} events")

        if self.total_frames > 0:
            first_chunk = self.event_chunks[0]
            last_chunk = self.event_chunks[-1]
            if len(first_chunk) > 0 and len(last_chunk) > 0:
                print(f"  Time range: {first_chunk['t'][0]} to {last_chunk['t'][-1]} us")

    def _merge_events_by_position(self, events):
        """
        按 (x, y) 合并相同位置的 event，不考虑 count，
        同一像素只保留一个事件。
        """
        if len(events) == 0:
            return events

        # N x 2 坐标数组
        coords = np.column_stack([events['x'], events['y']])

        # 找到唯一坐标及代表索引
        unique_coords, unique_indices = np.unique(coords, axis=0, return_index=True)

        # 取代表事件
        merged_events = events[unique_indices].copy()
        # 确保坐标与 unique_coords 对齐
        merged_events['x'] = unique_coords[:, 0]
        merged_events['y'] = unique_coords[:, 1]

        return merged_events

    def _cluster_events(self, events):
        """
        对事件进行DBSCAN聚类（先按像素合并）

        Args:
            events: 事件数组

        Returns:
            tuple: (points, labels, n_clusters, n_noise)
        """
        if len(events) == 0:
            return np.array([]), np.array([]), 0, 0

        # 先按像素合并（相同 (x,y) 的 event 合并为一个）
        merged_events = self._merge_events_by_position(events)

        # 提取坐标点（合并后的）
        points = np.column_stack([merged_events['x'], merged_events['y']])

        # 应用DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(points)

        # 统计聚类结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        return points, labels, n_clusters, n_noise

    def _visualize_clusters(self, points, labels):
        """
        可视化聚类结果 - 所有聚类共享一个总边界框

        Args:
            points: 坐标点数组（合并后的）
            labels: 聚类标签

        Returns:
            result_image: 可视化图像
        """
        # 创建黑色背景图像
        result_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if len(points) == 0 or len(labels) == 0:
            return result_image

        # 分离噪声点和聚类点
        cluster_mask = labels != -1
        cluster_points = points[cluster_mask]
        cluster_labels = labels[cluster_mask]
        noise_points = points[~cluster_mask]

        # 1) 绘制噪声点（红色）- 批量处理
        if len(noise_points) > 0:
            noise_coords = noise_points.astype(int)
            # 确保坐标在图像范围内
            valid_mask = (
                    (noise_coords[:, 0] >= 0) & (noise_coords[:, 0] < self.width) &
                    (noise_coords[:, 1] >= 0) & (noise_coords[:, 1] < self.height)
            )
            valid_noise = noise_coords[valid_mask]
            if len(valid_noise) > 0:
                result_image[valid_noise[:, 1], valid_noise[:, 0]] = [0, 0, 255]

        # 2) 为每个聚类生成颜色，并画点（不再画单独 bbox）
        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                label_mask = cluster_labels == label
                label_points = cluster_points[label_mask]
                if len(label_points) == 0:
                    continue

                # 转换颜色为BGR
                color_bgr = [int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)]

                # 批量设置像素点
                label_coords = label_points.astype(int)
                valid_mask = (
                        (label_coords[:, 0] >= 0) & (label_coords[:, 0] < self.width) &
                        (label_coords[:, 1] >= 0) & (label_coords[:, 1] < self.height)
                )
                valid_coords = label_coords[valid_mask]
                if len(valid_coords) > 0:
                    result_image[valid_coords[:, 1], valid_coords[:, 0]] = color_bgr

        # 3) 画一个“总边界框”：覆盖所有 cluster_points
        if len(cluster_points) > 0:
            x_coords = cluster_points[:, 0]
            y_coords = cluster_points[:, 1]
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())

            # 绿色大框
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 在左上角写一个总标签，比如 "ALL"
            cv2.putText(
                result_image, "ALL", (x_min, max(y_min - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        return result_image

    def _update_display(self):
        """更新显示 - 带缓存优化"""
        if self.current_frame >= self.total_frames:
            self.current_frame = self.total_frames - 1

        if self.current_frame < 0:
            self.current_frame = 0

        # 获取当前帧事件
        events = self.event_chunks[self.current_frame]

        # 检查是否需要重新计算聚类（缓存机制）
        cache_key = (self.current_frame, self.eps, self.min_samples)
        params_changed = (self.eps != self.last_eps or
                          self.min_samples != self.last_min_samples)

        if cache_key not in self.cache or params_changed:
            # 如果参数改变了，清空旧缓存
            if params_changed:
                self.cache.clear()
                self.last_eps = self.eps
                self.last_min_samples = self.min_samples

            # 进行聚类计算并缓存
            points, labels, n_clusters, n_noise = self._cluster_events(events)
            self.cache[cache_key] = (points, labels, n_clusters, n_noise)
        else:
            # 从缓存获取结果
            points, labels, n_clusters, n_noise = self.cache[cache_key]

        self.last_frame = self.current_frame

        # 可视化聚类结果
        result_image = self._visualize_clusters(points, labels)

        # 计算统计信息
        total_points = len(points)  # 合并后用于聚类的点数
        signal_points = total_points - n_noise
        signal_ratio = (signal_points / total_points * 100) if total_points > 0 else 0

        # 添加信息文本
        cache_status = "CACHED" if cache_key in self.cache and not params_changed else "COMPUTED"
        info_text = [
            f"Frame: {self.current_frame + 1}/{self.total_frames} [{cache_status}]",
            f"Events: {len(events)} | unique_pixels: {total_points}",
            f"eps={self.eps} | min_samples={self.min_samples}",
            f"Clusters: {n_clusters} | Signal: {signal_ratio:.1f}% | Noise: {n_noise}",
            f"{'[AUTO]' if self.auto_play else '[MANUAL]'}"
        ]

        y_pos = 30
        for text in info_text:
            cv2.putText(result_image, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30

        # 缩放显示
        display_width = int(self.width * self.display_scale)
        display_height = int(self.height * self.display_scale)
        display_image = cv2.resize(result_image, (display_width, display_height))

        cv2.imshow('Interactive DBSCAN Tuner', display_image)

    def run(self):
        """运行交互式调试器"""
        if self.total_frames == 0:
            print("Error: No event data loaded")
            return

        print("\n" + "=" * 60)
        print("Interactive DBSCAN Parameter Tuner")
        print("=" * 60)
        print("Keyboard Controls:")
        print("  SPACE       - Play/Pause auto-play")
        print("  LEFT/RIGHT  - Previous/Next frame")
        print("  UP/DOWN     - Fast backward/forward (-/+10 frames)")
        print("  h/E         - First/Last frame")
        print("  w/s         - eps +1/-1")
        print("  e/d         - eps +0.5/-0.5")
        print("  a/z         - min_samples +10/-10")
        print("  q/x         - min_samples +5/-5")
        print("  r           - Reset parameters")
        print("  p           - Print current parameters")
        print("  +/-         - Adjust playback speed")
        print("  ESC         - Exit")
        print("=" * 60)
        print()

        try:
            while True:
                current_time = time.time()

                # 自动播放模式
                if self.auto_play and (current_time - self.last_frame_time >= 1.0 / self.fps):
                    self.current_frame += 1
                    if self.current_frame >= self.total_frames:
                        self.current_frame = self.total_frames - 1
                        self.auto_play = False
                        print("→ Reached last frame, auto-play stopped")
                    self.last_frame_time = current_time

                # 更新显示
                self._update_display()

                # 处理按键输入
                key = cv2.waitKey(1)
                key_char = key & 0xFF  # 低8位用于普通键（字母、数字等）

                # --------- 通用退出 / 播放控制 ---------
                if key_char == 27:  # ESC
                    print("\nExiting interactive tuner...")
                    break
                elif key_char == ord(' '):  # 空格键 - 播放/暂停
                    self.auto_play = not self.auto_play
                    status = "Playing" if self.auto_play else "Paused"
                    print(f"→ {status} (Frame: {self.current_frame + 1}/{self.total_frames})")

                # --------- 方向键控制帧 ---------
                # → 右箭头：下一帧
                elif key_char == ord('l'):
                    self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
                    self.auto_play = False
                    print(f"→ Next frame: {self.current_frame + 1}/{self.total_frames}")

                # j：上一帧
                elif key_char == ord('j'):
                    self.current_frame = max(self.current_frame - 1, 0)
                    self.auto_play = False
                    print(f"→ Previous frame: {self.current_frame + 1}/{self.total_frames}")

                # o：快进 +10
                elif key_char == ord('o'):
                    self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
                    self.auto_play = False
                    print(f"→ Fast forward (+10): {self.current_frame + 1}/{self.total_frames}")

                # u：快退 -10
                elif key_char == ord('u'):
                    self.current_frame = max(self.current_frame - 10, 0)
                    self.auto_play = False
                    print(f"→ Fast backward (-10): {self.current_frame + 1}/{self.total_frames}")

                # --------- 其他帧跳转 ---------
                elif key_char == ord('h'):  # 第一帧
                    self.current_frame = 0
                    self.auto_play = False
                    print(f"→ First frame: {self.current_frame + 1}/{self.total_frames}")
                elif key_char == ord('E'):  # 最后一帧
                    self.current_frame = self.total_frames - 1
                    self.auto_play = False
                    print(f"→ Last frame: {self.current_frame + 1}/{self.total_frames}")

                # --------- DBSCAN 参数调整 ---------
                elif key_char == ord('e'):  # eps +1
                    self.eps += 1
                    print(f"→ eps = {self.eps}")
                elif key_char == ord('q'):  # eps -1
                    self.eps = max(0.5, self.eps - 1)
                    print(f"→ eps = {self.eps}")
                elif key_char == ord('d'):  # min_samples +5
                    self.min_samples += 5
                    print(f"→ min_samples = {self.min_samples}")
                elif key_char == ord('a'):  # min_samples -5
                    self.min_samples = max(1, self.min_samples - 5)
                    print(f"→ min_samples = {self.min_samples}")

                # --------- 其他控制 ---------
                elif key_char == ord('r'):  # 重置参数
                    self.eps = self.tracking.get('dbscan_eps', 12)
                    self.min_samples = self.tracking.get('dbscan_min_samples', 70)
                    print(f"→ Reset parameters: eps={self.eps}, min_samples={self.min_samples}")
                elif key_char == ord('p'):  # 打印参数
                    print(f"\nCurrent State:")
                    print(f"  Frame: {self.current_frame + 1}/{self.total_frames}")
                    print(f"  Parameters: eps={self.eps}, min_samples={self.min_samples}")
                    print(f"  Playback: {'Auto' if self.auto_play else 'Manual'}, FPS={self.fps}")
                elif key_char == ord('+') or key_char == ord('='):  # 增加FPS
                    self.fps = min(self.fps + 1, 30)
                    print(f"→ FPS = {self.fps}")
                elif key_char == ord('-'):  # 减少FPS
                    self.fps = max(self.fps - 1, 1)
                    print(f"→ FPS = {self.fps}")

        except KeyboardInterrupt:
            print("\nUser interrupted")
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='交互式DBSCAN参数调试器 - 从.raw文件读取事件数据并进行实时参数调整',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置文件
  python interactive_dbscan_tuner.py path/to/event.raw

  # 指定特定任务配置
  python interactive_dbscan_tuner.py path/to/event.raw --task clutter

  # 自定义显示缩放
  python interactive_dbscan_tuner.py path/to/event.raw --scale 0.8
        """
    )

    parser.add_argument('raw_file', type=str, nargs='?',
                        default='/home/xinchi/Event_RGB_catching_data/saved_data/event/0.raw',
                        )
    parser.add_argument('--task', type=str, default='clutter',
                        choices=['normal', 'dark', 'clutter', 'dark_clutter'],
                        help='任务类型，用于加载对应配置 (默认: clutter)')
    parser.add_argument('--scale', type=float, default=0.6,
                        help='显示缩放比例 (默认: 0.6)')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.raw_file):
        print(f"Error: File not found: {args.raw_file}")
        return 1

    # 加载指定任务的配置
    test_mode_key = os.environ.get("TEST_MODE", "").strip().lower()
    mode_suffix = {"model": "model", "rgb": "rgb", "event": "event"}.get(test_mode_key)
    task_key = (args.task or "").lower()
    task_base = {
        "normal": "config_l",
        "dark": "config_d",
        "clutter": "config_c",
        "dark_clutter": "config_dc"
    }
    legacy_mapping = {
        "normal": "config/config_l.yaml",
        "dark": "config/config_d.yaml",
        "clutter": "config/config_c.yaml",
        "dark_clutter": "config/config_dc.yaml"
    }

    config_file = None
    if mode_suffix:
        base = task_base.get(task_key)
        if base:
            config_file = f"config/{base}_{mode_suffix}.yaml"
    if config_file is None:
        config_file = legacy_mapping.get(task_key)
    if config_file:
        os.environ['ACTIVE_CONFIG_FILE'] = config_file
        config = get_config()
        config.switch_config(config_file)
        print(f"Loading config file: {config_file} (task: {args.task})")

    # 创建并运行调试器
    try:
        tuner = InteractiveDBSCANTuner(args.raw_file, args.scale)
        tuner.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
