import cv2
import os
import sys
import numpy as np
import time
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from data_input.read_from_raw import read_from_raw
from data_input.read_from_video import read_from_video

def get_video_info(video_path):
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return frame_count, fps, duration

def get_raw_file_info(raw_file_path):
    """获取raw文件信息（需要实际读取事件数据）"""
    try:
        from metavision_core.event_io import EventsIterator
        mv_iterator = EventsIterator(raw_file_path)
        
        event_count = 0
        start_time = None
        end_time = None
        
        for events in mv_iterator:
            if len(events) > 0:
                if start_time is None:
                    start_time = events['t'][0]
                end_time = events['t'][-1]
                event_count += len(events)
        
        duration_us = end_time - start_time if start_time is not None else 0
        duration_s = duration_us / 1e6
        
        return event_count, duration_s, start_time, end_time
    except Exception as e:
        print(f"Error reading raw file: {e}")
        return None, None, None, None

def debug_timing(record_index, record_dir):
    """调试RGB视频和raw文件的时间同步"""

    rgb_video_path = f"{record_dir}/rgb/{record_index}.mp4"
    raw_file_path = f"{record_dir}/event/{record_index}.raw"
    
    print(f"=== 调试记录 {record_index} ===")
    print(f"RGB视频路径: {rgb_video_path}")
    print(f"Raw文件路径: {raw_file_path}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(rgb_video_path):
        print(f"错误: RGB视频文件不存在 - {rgb_video_path}")
        return
    
    if not os.path.exists(raw_file_path):
        print(f"错误: Raw文件不存在 - {raw_file_path}")
        return
    
    # 获取RGB视频信息
    print("=== RGB视频信息 ===")
    rgb_frames, rgb_fps, rgb_duration = get_video_info(rgb_video_path)
    if rgb_frames is not None:
        print(f"帧数: {rgb_frames}")
        print(f"FPS: {rgb_fps:.2f}")
        print(f"时长: {rgb_duration:.2f} 秒")
        print(f"文件大小: {os.path.getsize(rgb_video_path) / (1024*1024):.2f} MB")
    else:
        print("无法读取RGB视频信息")
        return
    
    print()
    
    # 获取Raw文件信息
    print("=== Raw文件信息 ===")
    raw_events, raw_duration, start_time, end_time = get_raw_file_info(raw_file_path)
    if raw_events is not None:
        print(f"事件数量: {raw_events}")
        print(f"时长: {raw_duration:.2f} 秒")
        print(f"开始时间: {start_time} μs")
        print(f"结束时间: {end_time} μs")
        print(f"文件大小: {os.path.getsize(raw_file_path) / (1024*1024):.2f} MB")
    else:
        print("无法读取Raw文件信息")
        return
    
    print()
    
    # 比较时长
    print("=== 时间同步分析 ===")
    time_diff = abs(rgb_duration - raw_duration)
    print(f"时长差异: {time_diff:.2f} 秒")
    
    if rgb_duration > raw_duration:
        print(f"RGB视频更长 ({rgb_duration:.2f}s vs {raw_duration:.2f}s)")
        print(f"RGB视频多出: {time_diff:.2f} 秒")
    elif raw_duration > rgb_duration:
        print(f"Raw文件更长 ({raw_duration:.2f}s vs {rgb_duration:.2f}s)")
        print(f"Raw文件多出: {time_diff:.2f} 秒")
    else:
        print("两个文件时长相同")
    
    # 计算理论上的同步率
    if rgb_duration > 0 and raw_duration > 0:
        sync_ratio = min(rgb_duration, raw_duration) / max(rgb_duration, raw_duration)
        print(f"同步率: {sync_ratio:.3f} ({sync_ratio*100:.1f}%)")
        
        if sync_ratio < 0.95:
            print("⚠️  警告: 时间差异较大，可能存在同步问题")
        else:
            print("✓ 时间差异在可接受范围内")

def align_to_rgb(event_data, rgb_width=1080, rgb_height=1440, event_width=720, event_height=1280):
    """将事件坐标对齐到RGB坐标系"""
    x_e, y_e = event_data['x'].copy(), event_data['y'].copy()
    rgb_center_x = rgb_width / 2
    rgb_center_y = rgb_height / 2
    event_center_x = event_width / 2
    event_center_y = event_height / 2
    event_pixel = 3.45
    rgb_pixel = 4.86
    
    x = (x_e - event_center_x) * rgb_pixel / event_pixel + rgb_center_x + 30
    y = (y_e - event_center_y) * rgb_pixel / event_pixel + rgb_center_y + 29
    y = np.clip(y, 0, rgb_height - 1)
    x = np.clip(x, 0, rgb_width - 1)
    return x.astype(int), y.astype(int)

def save_adjusted_events(raw_event_buffer, event_frame_offset, record_index, record_dir):
    """保存调整偏移后的纯原始事件数据（无坐标变换）"""
    import shutil
    import tempfile
    
    # 创建输出目录  
    output_dir = f"{record_dir}/adjusted_events"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    raw_path = f"{output_dir}/{record_index}.raw"
    
    try:
        # 如果偏移为0，直接复制原文件
        if event_frame_offset == 0:
            original_raw = f"{record_dir}/event/{record_index}.raw"
            if os.path.exists(original_raw):
                shutil.copy2(original_raw, raw_path)
                print(f"直接复制原始RAW文件到: {raw_path}")
                return
        
        # 否则，提示用户偏移功能暂不支持RAW格式保存
        print(f"注意: 当前不支持带偏移的RAW文件保存")
        print(f"事件帧偏移: {event_frame_offset}")
        print(f"建议: 使用偏移=0或手动调整录制时机")
        
        # 仍然提供复制原文件的选项
        original_raw = f"{record_dir}/event/{record_index}.raw"
        if os.path.exists(original_raw):
            shutil.copy2(original_raw, raw_path)
            print(f"已复制原始RAW文件到: {raw_path} (无偏移调整)")
        
    except Exception as e:
        print(f"保存事件数据时出错: {e}")

def play_overlay_debug(record_index, record_dir, frame_skip=1):
    """播放叠加的RGB和事件帧进行调试"""
    from metavision_core.event_io import EventsIterator
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette

    rgb_video_path = f"{record_dir}/rgb/{record_index}.mp4"
    raw_file_path = f"{record_dir}/event/{record_index}.raw"
    
    # 检查文件是否存在
    if not os.path.exists(rgb_video_path) or not os.path.exists(raw_file_path):
        print("错误: 文件不存在")
        return
    
    # 初始化RGB视频读取
    cap = cv2.VideoCapture(rgb_video_path)
    rgb_fps = cap.get(cv2.CAP_PROP_FPS)
    rgb_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化事件数据读取
    rgb_width, rgb_height = 1080, 1440
    event_frame_gen = PeriodicFrameGenerationAlgorithm(
        sensor_width=rgb_width, 
        sensor_height=rgb_height,
        fps=rgb_fps,
        palette=ColorPalette.Dark
    )
    
    print(f"RGB FPS: {rgb_fps}, 总帧数: {rgb_frame_count}")
    print("按键控制:")
    print("  空格键=暂停/继续, 'q'=退出, 'r'=重置, 'a'/'d'=逐帧")
    print("  '+'/'-'=调整事件帧偏移, 'w'/'s'=调整事件速度比例, '0'=重置, 'e'=保存")
    print("=" * 50)
    
    # 预加载原始事件数据（用于保存）和对齐后数据（用于显示）
    print("正在预加载事件数据...")
    raw_event_buffer = []  # 纯原始事件数据（无坐标变换）
    transformed_event_buffer = []  # 坐标变换后事件数据
    aligned_event_buffer = []  # 对齐后事件数据
    iterator = EventsIterator(input_path=raw_file_path, mode="delta_t", delta_t=2000, n_events=10000)
    
    for ev in iterator:
        if len(ev) > 0:
            # 保存纯原始事件数据（无任何变换）
            raw_event_buffer.append(ev.copy())
            
            # 坐标变换后事件数据
            transformed_ev = ev.copy()
            transformed_ev['x'], transformed_ev['y'] = transformed_ev['y'].copy(), -transformed_ev['x'].copy()
            transformed_ev['y'] += 1279
            transformed_event_buffer.append(transformed_ev)
            
            # 对齐后事件数据（用于显示）
            aligned_ev = transformed_ev.copy()
            aligned_ev['x'], aligned_ev['y'] = align_to_rgb(aligned_ev)
            aligned_event_buffer.append(aligned_ev)
    
    event_buffer = aligned_event_buffer  # 保持原有变量名用于显示
    
    if not event_buffer:
        print("没有找到事件数据")
        return
    
    print(f"加载了 {len(event_buffer)} 个事件块")
    
    # 计算时间映射
    if len(event_buffer[0]) > 0:
        event_start_time = event_buffer[0]['t'][0]
        event_end_time = event_buffer[-1]['t'][-1] if len(event_buffer[-1]) > 0 else event_start_time
    else:
        event_start_time = event_end_time = 0
    
    event_duration = (event_end_time - event_start_time) / 1e6
    rgb_duration = rgb_frame_count / rgb_fps
    
    print(f"RGB时长: {rgb_duration:.2f}s, Event时长: {event_duration:.2f}s")
    
    current_frame = 0
    paused = False
    event_frame_offset = 0  # 事件帧偏移量
    event_speed_ratio = 1.0  # 事件播放速度比例
    
    while True:
        if not paused:
            ret, rgb_frame = cap.read()
            if not ret:
                print("RGB视频播放完毕")
                break
        else:
            # 暂停时重新定位到当前帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, rgb_frame = cap.read()
            if not ret:
                break
        
        # 计算当前事件块索引（基于RGB帧进度和帧偏移）
        rgb_progress = current_frame / max(1, rgb_frame_count - 1)
        
        # 应用帧偏移和速度调整
        adjusted_rgb_frame = (current_frame + event_frame_offset) * event_speed_ratio
        event_progress = adjusted_rgb_frame / max(1, rgb_frame_count - 1)
        event_progress = max(0, min(event_progress, 1))
        
        event_block_index = int(event_progress * (len(event_buffer) - 1))
        event_block_index = max(0, min(event_block_index, len(event_buffer) - 1))
        
        # 创建事件帧
        event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
        
        if event_block_index < len(event_buffer) and len(event_buffer[event_block_index]) > 0:
            events = event_buffer[event_block_index]
            event_frame = np.ascontiguousarray(event_frame)
            event_frame_gen.generate_frame(events, event_frame, accumulation_time_us=2000)
        
        # 对事件帧进行二值化处理
        event_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
        _, event_binary = cv2.threshold(event_gray, 40, 255, cv2.THRESH_BINARY)
        
        # 增加RGB亮度
        rgb_brightened = cv2.convertScaleAbs(rgb_frame, alpha=10, beta=30)
        
        # 创建叠加图像 - 将二值化的事件数据以绿色叠加
        overlay = rgb_brightened.copy()
        event_mask = event_binary > 0
        overlay[event_mask, 1] = 255  # 绿色通道设为最亮
        overlay[event_mask, 0] = overlay[event_mask, 0] // 2  # 蓝色通道减半
        overlay[event_mask, 2] = overlay[event_mask, 2] // 2  # 红色通道减半
        
        # 添加信息显示
        rgb_time = current_frame / rgb_fps
        actual_event_time = event_start_time / 1e6 + (event_block_index / max(1, len(event_buffer) - 1)) * event_duration
        
        info_text = [
            f"RGB Frame: {current_frame}/{rgb_frame_count-1} ({current_frame/rgb_frame_count*100:.1f}%)",
            f"Event Block: {event_block_index}/{len(event_buffer)-1} ({event_block_index/len(event_buffer)*100:.1f}%)",
            f"RGB Time: {rgb_time:.3f}s / {rgb_duration:.3f}s",
            f"Event Time: {actual_event_time:.3f}s / {event_duration:.3f}s",
            f"Frame Offset: {event_frame_offset}, Speed: {event_speed_ratio:.2f}x",
            f"Events in block: {len(event_buffer[event_block_index]) if event_block_index < len(event_buffer) else 0}",
            f"Controls: SPACE=pause, Q=quit, R=reset, A/D=frame, +/-=offset, W/S=speed"
        ]
        
        # 添加半透明背景
        overlay_info = overlay.copy()
        cv2.rectangle(overlay_info, (5, 5), (600, 165), (0, 0, 0), -1)
        overlay = cv2.addWeighted(overlay, 0.3, overlay_info, 0.2, 0)
        
        y_offset = 25
        for i, text in enumerate(info_text):
            cv2.putText(overlay, text, (10, y_offset + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 缩放显示
        scale = 0.8
        width = int(overlay.shape[1] * scale)
        height = int(overlay.shape[0] * scale)
        resized = cv2.resize(overlay, (width, height))
        
        cv2.imshow('RGB + Event Overlay Debug', resized)
        
        # 键盘控制
        if not paused:
            key = cv2.waitKey(max(1, int(1000/rgb_fps))) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF
            
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键暂停/继续
            paused = not paused
            print("暂停" if paused else "继续")
        elif key == ord('r'):  # 重置
            current_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False
            print("重置到开始")
        elif key == ord('d'):  # 'd'键 - 下一帧
            if current_frame < rgb_frame_count - 1:
                current_frame += frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                paused = True
        elif key == ord('a'):  # 'a'键 - 上一帧
            if current_frame > 0:
                current_frame = max(0, current_frame - frame_skip)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                paused = True
        elif key == ord('+') or key == ord('='):  # 增加事件帧偏移
            event_frame_offset += 1
            print(f"事件帧偏移: {event_frame_offset}")
        elif key == ord('-'):  # 减少事件帧偏移
            event_frame_offset -= 1
            print(f"事件帧偏移: {event_frame_offset}")
        elif key == ord('w'):  # 增加事件速度比例
            event_speed_ratio += 0.1
            event_speed_ratio = min(event_speed_ratio, 3.0)
            print(f"事件速度比例: {event_speed_ratio:.2f}x")
        elif key == ord('s'):  # 减少事件速度比例
            event_speed_ratio -= 0.1
            event_speed_ratio = max(event_speed_ratio, 0.1)
            print(f"事件速度比例: {event_speed_ratio:.2f}x")
        elif key == ord('0'):  # 重置调整参数
            event_frame_offset = 0
            event_speed_ratio = 1.0
            print("重置帧调整参数")
        elif key == ord('e'):  # 保存当前调整后的事件数据
            save_adjusted_events(raw_event_buffer, event_frame_offset, record_index, record_dir)
            print(f"已保存调整后的事件数据，偏移量: {event_frame_offset}")
        
        if not paused:
            current_frame += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 自动保存调整后的事件数据
    save_adjusted_events(raw_event_buffer, event_frame_offset, record_index, record_dir)
    print(f"程序结束时自动保存调整后的事件数据，偏移量: {event_frame_offset}")

def batch_debug(record_dir, start_index, end_index):
    """批量调试多个记录"""
    print(f"=== 批量调试记录 {start_index} 到 {end_index} ===")
    print()
    
    results = []
    
    for i in range(start_index, end_index + 1):
        print(f"检查记录 {i}...")
        
        rgb_video_path = f"{record_dir}/rgb/{i}.mp4"
        raw_file_path = f"{record_dir}/event/{i}.raw"
        
        if not os.path.exists(rgb_video_path) or not os.path.exists(raw_file_path):
            print(f"  跳过记录 {i} - 文件不完整")
            continue
        
        # 获取信息
        rgb_frames, rgb_fps, rgb_duration = get_video_info(rgb_video_path)
        raw_events, raw_duration, _, _ = get_raw_file_info(raw_file_path)
        
        if rgb_duration is not None and raw_duration is not None:
            time_diff = abs(rgb_duration - raw_duration)
            longer = "RGB" if rgb_duration > raw_duration else "Raw" if raw_duration > rgb_duration else "Equal"
            
            results.append({
                'index': i,
                'rgb_duration': rgb_duration,
                'raw_duration': raw_duration,
                'time_diff': time_diff,
                'longer': longer
            })
            
            print(f"  RGB: {rgb_duration:.2f}s, Raw: {raw_duration:.2f}s, 差异: {time_diff:.2f}s, 更长: {longer}")
    
    print()
    print("=== 批量结果摘要 ===")
    if results:
        avg_time_diff = np.mean([r['time_diff'] for r in results])
        max_time_diff = max([r['time_diff'] for r in results])
        rgb_longer_count = sum(1 for r in results if r['longer'] == 'RGB')
        raw_longer_count = sum(1 for r in results if r['longer'] == 'Raw')
        
        print(f"检查的记录数: {len(results)}")
        print(f"平均时间差异: {avg_time_diff:.2f} 秒")
        print(f"最大时间差异: {max_time_diff:.2f} 秒")
        print(f"RGB更长的记录: {rgb_longer_count}")
        print(f"Raw更长的记录: {raw_longer_count}")
    else:
        print("没有找到有效的记录")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug RGB video and raw file timing')
    parser.add_argument('--record_index', type=int, default=1,
                       help='Record index to debug (default: 10)')
    parser.add_argument('--record_dir', type=str, 
                       default='/media/xinchi/data/Event_RGB_catching_data/saved_data_d',
                       help='Directory containing recorded data')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch analysis on multiple records')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index for batch analysis (default: 1)')
    parser.add_argument('--end_index', type=int, default=110,
                       help='End index for batch analysis (default: 20)')
    parser.add_argument('--play', action='store_true', default=True,
                       help='Play overlay visualization')
    parser.add_argument('--frame_skip', type=int, default=1,
                       help='Frame skip for seeking (default: 1)')
    
    args = parser.parse_args()
    
    if args.play:
        play_overlay_debug(args.record_index, args.record_dir, args.frame_skip)
    elif args.batch:
        batch_debug(args.record_dir, args.start_index, args.end_index)
    else:
        debug_timing(args.record_index, args.record_dir)