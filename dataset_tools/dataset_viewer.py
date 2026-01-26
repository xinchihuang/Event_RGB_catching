#!/usr/bin/env python3
import os
import cv2
import numpy as np
from scipy.io import loadmat

class DatasetViewer:
    def __init__(self, dataset_path="/media/xinchi/data/saved_data/selected_dataset"):
        self.dataset_path = dataset_path
        self.sequences = []
        self.current_sequence = 200
        self.current_frame = 0
        self.paused = False
        self.view_mode = 'combined'  # 'rgb', 'event', 'combined', 'voxel'
        self.show_bbox = True
        self.frame_flags = {}  # {seq_id: {frame_id: flag}}
        self.present_only_mode = True  # 只显示present帧的模式
        self.load_sequences()
        self.load_flags()
        
        # 启动时如果是Present Only模式，找到第一个present帧
        if self.present_only_mode and self.sequences:
            self.current_sequence = 0
            self.current_frame = 0
            self.find_next_present_frame()

    def load_sequences(self):
        """加载所有可用的序列"""
        for seq_name in sorted(os.listdir(self.dataset_path), key=lambda x: int(x) if x.isdigit() else float('inf')):
            seq_path = os.path.join(self.dataset_path, seq_name)
            if os.path.isdir(seq_path) and seq_name.isdigit():
                # 检查必要文件是否存在
                if os.path.exists(os.path.join(seq_path, "absent.txt")) and \
                   os.path.exists(os.path.join(seq_path, "groundtruth.txt")):
                    self.sequences.append(int(seq_name))
        print(f"Found {len(self.sequences)} sequences")

    def load_flags(self):
        """加载已保存的标记"""
        flags_file = os.path.join(self.dataset_path, "frame_flags.txt")
        if os.path.exists(flags_file):
            with open(flags_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        seq_id, frame_id, flag = int(parts[0]), int(parts[1]), int(parts[2])
                        if seq_id not in self.frame_flags:
                            self.frame_flags[seq_id] = {}
                        self.frame_flags[seq_id][frame_id] = bool(flag)

    def save_flags(self):
        """保存标记到文件"""
        flags_file = os.path.join(self.dataset_path, "frame_flags.txt")
        with open(flags_file, 'w') as f:
            for seq_id, frames in self.frame_flags.items():
                for frame_id, flag in frames.items():
                    f.write(f"{seq_id},{frame_id},{int(flag)}\n")
        print(f"Flags saved to {flags_file}")

    def toggle_frame_flag(self):
        """切换当前帧的标记状态"""
        if not self.sequences:
            return
            
        seq_id = self.sequences[self.current_sequence]
        frame_id = self.current_frame
        
        if seq_id not in self.frame_flags:
            self.frame_flags[seq_id] = {}
            
        current_flag = self.frame_flags[seq_id].get(frame_id, False)
        self.frame_flags[seq_id][frame_id] = not current_flag
        
        # 立即保存
        self.save_flags()
        
        flag_status = "FLAGGED" if self.frame_flags[seq_id][frame_id] else "UNFLAGGED"
        print(f"Frame {frame_id} in sequence {seq_id} is now {flag_status}")
        
        # 自动播放下一帧
        self.next_frame()

    def get_frame_flag(self, seq_id, frame_id):
        """获取指定帧的标记状态"""
        return self.frame_flags.get(seq_id, {}).get(frame_id, False)

    def get_sequence_info(self, seq_id):
        """获取序列信息"""
        seq_path = os.path.join(self.dataset_path, str(seq_id))
        
        # 读取absent和groundtruth信息
        absent_file = os.path.join(seq_path, "absent.txt")
        groundtruth_file = os.path.join(seq_path, "groundtruth.txt")
        
        absent_data = []
        groundtruth_data = []
        
        with open(absent_file, 'r') as f:
            absent_data = [line.strip() for line in f.readlines()]
        
        with open(groundtruth_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    groundtruth_data.append([float(x) for x in parts[:4]])
        
        return absent_data, groundtruth_data

    def load_frame(self, seq_id, frame_id, view_mode):
        """加载指定帧的图像"""
        seq_path = os.path.join(self.dataset_path, str(seq_id))
        
        if view_mode == 'voxel':
            # 加载voxel数据（.mat文件）
            voxel_path = os.path.join(seq_path, "voxel", f"{frame_id}.mat")
            if os.path.exists(voxel_path):
                try:
                    mat_data = loadmat(voxel_path)
                    # 假设voxel数据的键名为'voxel'，您可能需要根据实际情况调整
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            voxel_data = mat_data[key]
                            break
                    
                    # 将voxel数据可视化（假设是3D数据，取某个通道或求和）
                    if len(voxel_data.shape) == 3:
                        # 如果是3D，在第三维求和或取最后一层
                        img = np.sum(voxel_data, axis=2)
                    else:
                        img = voxel_data
                    
                    # 归一化到0-255
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    
                    # 转换为3通道
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    return img
                except Exception as e:
                    print(f"Cannot load voxel file {voxel_path}: {e}")
                    return None
        else:
            # 加载图像文件
            img_path = os.path.join(seq_path, view_mode, f"{frame_id}.jpg")
            if os.path.exists(img_path):
                return cv2.imread(img_path)
        
        return None

    def draw_bbox(self, img, bbox):
        """在图像上绘制边界框"""
        if bbox and len(bbox) >= 4:
            x, y, w, h = map(int, bbox[:4])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img

    def show_frame(self):
        """显示当前帧"""
        if not self.sequences:
            print("No sequences found")
            return False

        seq_id = self.sequences[self.current_sequence]
        absent_data, groundtruth_data = self.get_sequence_info(seq_id)
        
        if self.current_frame >= len(absent_data):
            # 自动跳到下一个序列
            if self.current_sequence < len(self.sequences) - 1:
                self.next_sequence()
                return True
            else:
                self.paused = True
                print("Reached end of dataset")
                return True

        # Create overlay for RGB + Event display when in combined mode
        if self.view_mode == 'combined':
            # Load both RGB and event images for overlay
            rgb_img = self.load_frame(seq_id, self.current_frame, 'rgb')
            event_img = self.load_frame(seq_id, self.current_frame, 'event')
            
            if rgb_img is not None and event_img is not None:
                # Create RGB + Event overlay similar to prepare_dataset.py
                rgb_brightened = cv2.convertScaleAbs(rgb_img, alpha=0.7, beta=30)
                img = cv2.addWeighted(rgb_brightened, 0.7, event_img, 0.3, 0)
            elif rgb_img is not None:
                img = rgb_img
            elif event_img is not None:
                img = event_img
            else:
                print(f"Cannot load RGB or Event frame: seq{seq_id}, frame{self.current_frame}")
                self.next_frame()
                return True
        else:
            # 加载单一模式图像
            img = self.load_frame(seq_id, self.current_frame, self.view_mode)
            if img is None:
                print(f"Cannot load frame: seq{seq_id}, frame{self.current_frame}, mode{self.view_mode}")
                # 自动跳到下一帧而不是退出
                self.next_frame()
                return True

        # 绘制边界框（如果启用且目标存在）
        if self.show_bbox and self.current_frame < len(groundtruth_data):
            is_present = absent_data[self.current_frame] != '0'
            if is_present:
                self.draw_bbox(img, groundtruth_data[self.current_frame])

        # 检查当前帧的标记状态
        is_flagged = self.get_frame_flag(seq_id, self.current_frame)
        flag_indicator = "★" if is_flagged else "☆"
        
        # 添加信息文本
        info_text = [
            f"Sequence: {seq_id}/{len(self.sequences)-1} ({self.current_sequence+1}/{len(self.sequences)})",
            f"Frame: {self.current_frame}/{len(absent_data)-1} {flag_indicator}",
            f"Mode: {self.view_mode}",
            f"Target: {'Present' if absent_data[self.current_frame] != '0' else 'Absent'}",
            f"BBox: {'ON' if self.show_bbox else 'OFF'}",
            f"Flagged: {'YES' if is_flagged else 'NO'}",
            f"Present Only: {'ON' if self.present_only_mode else 'OFF'}",
            "",
            "Controls:",
            "SPACE: Play/Pause  A/D: Prev/Next Frame",
            "W/S: Prev/Next Seq  R/E/T/Y: RGB/Event/Combined/Voxel",
            "B: Toggle BBox  F: Flag Frame  P: Present Only  Q: Quit"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_offset += 25

        cv2.imshow('Dataset Viewer', img)
        return True

    def handle_key(self, key):
        """处理键盘输入"""
        if key == ord('q') or key == 27:  # 'q' or Esc
            return False
        elif key == ord(' '):  # Space - 播放/暂停
            self.paused = not self.paused
        elif key == ord('d'):  # 下一帧
            self.next_frame()
        elif key == ord('a'):  # 上一帧
            self.prev_frame()
        elif key == ord('s'):  # 下一序列
            self.next_sequence()
        elif key == ord('w'):  # 上一序列
            self.prev_sequence()
        elif key == ord('r'):  # RGB模式
            self.view_mode = 'rgb'
        elif key == ord('e'):  # Event模式
            self.view_mode = 'event'
        elif key == ord('t'):  # Combined模式
            self.view_mode = 'combined'
        elif key == ord('y'):  # Voxel模式
            self.view_mode = 'voxel'
        elif key == ord('b'):  # 切换边界框显示
            self.show_bbox = not self.show_bbox
        elif key == ord('f'):  # 标记当前帧
            self.toggle_frame_flag()
        elif key == ord('p'):  # 切换present only模式
            self.present_only_mode = not self.present_only_mode
            present_status = "ON" if self.present_only_mode else "OFF"
            print(f"Present Only Mode: {present_status}")
        
        return True

    def next_frame(self):
        """下一帧"""
        if not self.sequences:
            return
        
        seq_id = self.sequences[self.current_sequence]
        absent_data, _ = self.get_sequence_info(seq_id)
        
        # 在Present Only模式下，跳过absent帧
        if self.present_only_mode:
            start_frame = self.current_frame + 1
            found_present = False
            
            # 在当前序列中寻找下一个present帧
            for frame_idx in range(start_frame, len(absent_data)):
                if absent_data[frame_idx] != '0':
                    self.current_frame = frame_idx
                    found_present = True
                    break
            
            if not found_present:
                # 当前序列没有更多present帧，切换到下一序列
                if self.current_sequence < len(self.sequences) - 1:
                    self.next_sequence()
                    # 在新序列中找第一个present帧
                    self.find_next_present_frame()
                else:
                    self.paused = True
                    print("Reached end of dataset")
        else:
            # 正常模式
            if self.current_frame < len(absent_data) - 1:
                self.current_frame += 1
            else:
                # 自动切换到下一个序列的第一帧
                if self.current_sequence < len(self.sequences) - 1:
                    self.next_sequence()
                else:
                    # 如果已经是最后一个序列的最后一帧，停止播放
                    self.paused = True
                    print("Reached end of dataset")

    def prev_frame(self):
        """上一帧"""
        if not self.sequences:
            return
            
        seq_id = self.sequences[self.current_sequence]
        absent_data, _ = self.get_sequence_info(seq_id)
        
        # 在Present Only模式下，跳过absent帧
        if self.present_only_mode:
            found_present = False
            
            # 在当前序列中向前寻找present帧
            for frame_idx in range(self.current_frame - 1, -1, -1):
                if absent_data[frame_idx] != '0':
                    self.current_frame = frame_idx
                    found_present = True
                    break
            
            if not found_present:
                # 当前序列没有更多present帧，切换到上一序列
                if self.current_sequence > 0:
                    self.prev_sequence()
                    # 在新序列中找最后一个present帧
                    self.find_prev_present_frame()
        else:
            # 正常模式
            if self.current_frame > 0:
                self.current_frame -= 1
            else:
                # 切换到上一个序列的最后一帧
                self.prev_sequence()
                if self.sequences:
                    seq_id = self.sequences[self.current_sequence]
                    absent_data, _ = self.get_sequence_info(seq_id)
                    self.current_frame = len(absent_data) - 1

    def next_sequence(self):
        """下一序列"""
        if self.sequences and self.current_sequence < len(self.sequences) - 1:
            self.current_sequence += 1
            self.current_frame = 0
            # 在Present Only模式下，找到第一个present帧
            if self.present_only_mode:
                self.find_next_present_frame()

    def prev_sequence(self):
        """上一序列"""
        if self.current_sequence > 0:
            self.current_sequence -= 1
            self.current_frame = 0
            # 在Present Only模式下，找到第一个present帧
            if self.present_only_mode:
                self.find_next_present_frame()

    def find_next_present_frame(self):
        """在当前序列中找到下一个present帧"""
        if not self.sequences:
            return
            
        seq_id = self.sequences[self.current_sequence]
        absent_data, _ = self.get_sequence_info(seq_id)
        
        for frame_idx in range(self.current_frame, len(absent_data)):
            if absent_data[frame_idx] != '0':
                self.current_frame = frame_idx
                return
        
        # 如果当前序列没有present帧，保持当前帧不变
        
    def find_prev_present_frame(self):
        """在当前序列中找到上一个present帧（从最后往前找）"""
        if not self.sequences:
            return
            
        seq_id = self.sequences[self.current_sequence]
        absent_data, _ = self.get_sequence_info(seq_id)
        
        # 从最后一帧开始向前找
        for frame_idx in range(len(absent_data) - 1, -1, -1):
            if absent_data[frame_idx] != '0':
                self.current_frame = frame_idx
                return
                
        # 如果当前序列没有present帧，设为第一帧
        self.current_frame = 0

    def run(self):
        """运行查看器"""
        print("Dataset Viewer Starting...")
        print("Controls:")
        print("  SPACE: Play/Pause")
        print("  A/D: Previous/Next Frame")
        print("  W/S: Previous/Next Sequence")
        print("  R/E/T/Y: Switch to RGB/Event/Combined/Voxel mode")
        print("  B: Toggle BBox Display")
        print("  F: Flag/Unflag Frame")
        print("  P: Toggle Present Only Mode")
        print("  Q or ESC: Quit")
        print()

        while True:
            if not self.show_frame():
                break

            key = cv2.waitKey(30) & 0xFF
            
            # 自动播放
            if not self.paused and key == 255:  # 没有按键时
                self.next_frame()
                continue
            
            if not self.handle_key(key):
                break

        cv2.destroyAllWindows()
        # 在退出时保存标记
        self.save_flags()

if __name__ == "__main__":
    viewer = DatasetViewer()
    viewer.run()