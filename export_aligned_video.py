import os
import sys
import numpy as np
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

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

def export_raw_event_data(record_index, record_dir, output_dir, event_frame_offset=-3, show_preview=False):
    """导出调整偏移后的raw事件数据"""
    from metavision_core.event_io import EventsIterator, DatWriter
    from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
    import cv2
    
    raw_file_path = f"{record_dir}/event/{record_index}.raw"
    rgb_video_path = f"{record_dir}/rgb/{record_index}.mp4"
    
    # 检查文件是否存在
    if not os.path.exists(raw_file_path):
        print("错误: Raw文件不存在")
        return
    
    if not os.path.exists(rgb_video_path):
        print("错误: RGB视频不存在")
        return
    
    # 创建输出目录结构
    output_event_dir = f"{output_dir}/event"
    if not os.path.exists(output_event_dir):
        os.makedirs(output_event_dir)
    
    output_raw_path = f"{output_event_dir}/{record_index}.raw"
    
    print(f"开始导出raw事件数据 {record_index}")
    print(f"事件帧偏移: {event_frame_offset}")
    print(f"输入文件: {raw_file_path}")
    print(f"输出文件: {output_raw_path}")
    
    # 获取RGB视频信息
    cap = cv2.VideoCapture(rgb_video_path)
    rgb_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rgb_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rgb_fps = cap.get(cv2.CAP_PROP_FPS)
    rgb_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 预加载所有事件数据（原始格式，不做变换）
    print("正在加载事件数据...")
    event_buffer = []
    iterator = EventsIterator(input_path=raw_file_path, mode="delta_t", delta_t=2000, n_events=10000)
    
    for ev in iterator:
        if len(ev) > 0:
            event_buffer.append(ev)
    
    if not event_buffer:
        print("没有找到事件数据")
        cap.release()
        return
    
    print(f"加载了 {len(event_buffer)} 个事件块")
    
    # 如果需要显示预览
    if show_preview:
        print("显示叠加预览...")
        print("按 'q' 退出预览，按任意键继续导出")
        
        # 初始化事件帧生成器（用于显示）
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=rgb_width, 
            sensor_height=rgb_height,
            fps=rgb_fps,
            palette=ColorPalette.Dark
        )
        
        current_frame = 0
        while current_frame < min(rgb_frame_count, 100):  # 只显示前100帧
            ret, rgb_frame = cap.read()
            if not ret:
                break
            
            # 应用帧偏移计算事件块索引
            adjusted_frame_index = current_frame + event_frame_offset
            event_block_index = max(0, min(adjusted_frame_index, len(event_buffer) - 1))
            
            # 创建事件帧用于显示（应用RGB对齐）
            event_frame = np.zeros((rgb_height, rgb_width, 3), dtype=np.uint8)
            
            if event_block_index < len(event_buffer) and len(event_buffer[event_block_index]) > 0:
                events = event_buffer[event_block_index].copy()
                # 应用坐标变换（仅用于显示）
                events['x'], events['y'] = events['y'].copy(), -events['x'].copy()
                events['y'] += 1279
                events['x'], events['y'] = align_to_rgb(events, rgb_width, rgb_height)
                
                event_frame = np.ascontiguousarray(event_frame)
                event_frame_gen.generate_frame(events, event_frame, accumulation_time_us=2000)
            
            # 对事件帧进行二值化处理
            event_gray = cv2.cvtColor(event_frame, cv2.COLOR_BGR2GRAY)
            _, event_binary = cv2.threshold(event_gray, 40, 255, cv2.THRESH_BINARY)
            
            # 创建叠加图像
            overlay = rgb_frame.copy()
            event_mask = event_binary > 0
            overlay[event_mask, 1] = 255  # 绿色通道设为最亮
            overlay[event_mask, 0] = overlay[event_mask, 0] // 2  # 蓝色通道减半
            overlay[event_mask, 2] = overlay[event_mask, 2] // 2  # 红色通道减半
            
            # 添加信息显示
            info_text = f"Frame: {current_frame}, Offset: {event_frame_offset}, Event Block: {event_block_index}"
            cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 缩放显示
            scale = 0.6
            width = int(overlay.shape[1] * scale)
            height = int(overlay.shape[0] * scale)
            resized = cv2.resize(overlay, (width, height))
            
            cv2.imshow('Preview - RGB + Event Overlay', resized)
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                return
            
            current_frame += 1
        
        cv2.destroyAllWindows()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频位置
    
    cap.release()
    
    # 重新加载事件数据（原始格式，不做变换）
    print("正在重新加载事件数据用于导出...")
    event_buffer = []
    iterator = EventsIterator(input_path=raw_file_path, mode="delta_t", delta_t=2000, n_events=10000)
    
    for ev in iterator:
        if len(ev) > 0:
            event_buffer.append(ev)
    
    # 应用偏移并写入新的raw文件
    print("正在写入调整后的事件数据...")
    
    try:
        writer = DatWriter(output_raw_path)
        
        total_blocks = len(event_buffer)
        for block_index in range(total_blocks):
            # 应用帧偏移
            adjusted_block_index = block_index + event_frame_offset
            
            # 确保索引在有效范围内
            if 0 <= adjusted_block_index < total_blocks:
                events = event_buffer[adjusted_block_index]
                if len(events) > 0:
                    writer.write(events)
            
            # 显示进度
            if block_index % 100 == 0:
                progress = block_index / total_blocks * 100
                print(f"写入进度: {progress:.1f}% ({block_index}/{total_blocks})")
        
        writer.close()
        print("Raw事件数据导出完成!")
        print(f"输出文件: {output_raw_path}")
        
    except Exception as e:
        print(f"写入Raw文件时出错: {e}")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export raw event data with offset')
    parser.add_argument('--record_index', type=int, default=10,
                       help='Record index to export (default: 10)')
    parser.add_argument('--record_dir', type=str, 
                       default='/media/xinchi/data/Event_RGB_catching_data/saved_data_dc',
                       help='Directory containing recorded data')
    parser.add_argument('--event_frame_offset', type=int, default=0,
                       help='Event frame offset (default: 0)')
    parser.add_argument('--output_dir', type=str, default='/media/xinchi/data/Event_RGB_catching_data/output_videos',
                       help='Output directory for exported data')
    parser.add_argument('--show_preview', action='store_true', default=True,
                       help='Show preview of alignment before export (default: True)')
    parser.add_argument('--no_preview', action='store_true',
                       help='Skip preview and export directly')
    
    args = parser.parse_args()
    
    # 如果指定了no_preview，则不显示预览
    show_preview = not args.no_preview
    
    export_raw_event_data(args.record_index, args.record_dir, args.output_dir, 
                         args.event_frame_offset, show_preview)