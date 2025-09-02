#!/usr/bin/env python3
import os
import sys
import subprocess

def batch_process_events(start_index=0, end_index=110, record_dir='/media/xinchi/data/Event_RGB_catching_data/saved_data_dc'):
    """批量处理事件数据，为每个record_index生成调整后的事件文件"""
    
    for i in range(start_index, end_index + 1):
        print(f"Processing record {i}...")
        
        # 检查文件是否存在
        rgb_path = f"{record_dir}/rgb/{i}.mp4"
        raw_path = f"{record_dir}/event/{i}.raw"
        
        if not os.path.exists(rgb_path) or not os.path.exists(raw_path):
            print(f"  跳过记录 {i} - 文件不存在")
            continue
            
        try:
            # 运行debug_video_timing.py --play模式，会在结束时自动保存
            cmd = ['python', 'debug_video_timing.py', '--play', '--record_index', str(i)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"  ✓ 记录 {i} 处理完成")
            else:
                print(f"  ✗ 记录 {i} 处理失败: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⚠ 记录 {i} 处理超时")
        except Exception as e:
            print(f"  ✗ 记录 {i} 处理出错: {e}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process events for multiple records')
    parser.add_argument('--start_index', type=int, default=1,
                       help='Start index (default: 0)')
    parser.add_argument('--end_index', type=int, default=110,
                       help='End index (default: 110)')
    parser.add_argument('--record_dir', type=str,
                       default='/media/xinchi/data/Event_RGB_catching_data/saved_data_dc',
                       help='Directory containing recorded data')
    
    args = parser.parse_args()
    batch_process_events(args.start_index, args.end_index, args.record_dir)