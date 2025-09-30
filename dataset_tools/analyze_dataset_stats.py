#!/usr/bin/env python3
"""
Dataset Statistics Analysis Tool

This script analyzes the data distribution from dataset_dc and dataset_c,
providing detailed statistics about positive/negative samples and data counts.
"""

import os
import csv

def analyze_single_dataset(dataset_path, dataset_name):
    """分析单个数据集的统计信息"""
    print(f"\n=== 分析 {dataset_name} ===")
    print(f"路径: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return None
    
    stats = {
        'dataset_name': dataset_name,
        'dataset_path': dataset_path,
        'total_sequences': 0,
        'valid_sequences': 0,
        'total_frames': 0,
        'positive_frames': 0,
        'negative_frames': 0,
        'sequences_with_positives': 0,
        'missing_files': 0
    }
    
    # 遍历序列（0-99）
    for i in range(100):
        sequence_path = os.path.join(dataset_path, str(i))
        
        if not os.path.exists(sequence_path):
            continue
            
        stats['total_sequences'] += 1
        
        # 检查必要文件
        absent_path = os.path.join(sequence_path, "absent.txt")
        groundtruth_path = os.path.join(sequence_path, "groundtruth.txt")
        
        if not (os.path.exists(absent_path) and os.path.exists(groundtruth_path)):
            stats['missing_files'] += 1
            continue
            
        stats['valid_sequences'] += 1
        
        # 分析absent.txt
        sequence_positive_frames = 0
        sequence_negative_frames = 0
        
        try:
            with open(absent_path, 'r') as f:
                for line in f:
                    stats['total_frames'] += 1
                    if line.strip() != '0':
                        stats['positive_frames'] += 1
                        sequence_positive_frames += 1
                    else:
                        stats['negative_frames'] += 1
                        sequence_negative_frames += 1
            
            if sequence_positive_frames > 0:
                stats['sequences_with_positives'] += 1
                
        except Exception as e:
            print(f"⚠️  读取序列 {i} 的absent.txt时出错: {e}")
            continue
    
    # 打印统计结果
    print(f"📊 数据集统计:")
    print(f"  总序列数: {stats['total_sequences']}")
    print(f"  有效序列数: {stats['valid_sequences']}")
    print(f"  缺失文件序列: {stats['missing_files']}")
    print(f"  包含正样本的序列: {stats['sequences_with_positives']}")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  正样本帧数: {stats['positive_frames']}")
    print(f"  负样本帧数: {stats['negative_frames']}")
    
    if stats['total_frames'] > 0:
        positive_ratio = stats['positive_frames'] / stats['total_frames'] * 100
        print(f"  正样本比例: {positive_ratio:.2f}%")
        print(f"  负样本比例: {100-positive_ratio:.2f}%")
    
    return stats

def analyze_pick_positive_behavior(source_datasets, negative_sample_probability=0.2):
    """分析pick_positive.py的行为预测"""
    print(f"\n=== 预测 pick_positive.py 行为 ===")
    print(f"负样本包含概率: {negative_sample_probability}")
    
    total_expected_positive = 0
    total_expected_negative = 0
    total_expected_sequences = 0
    
    all_stats = []
    
    for dataset_path in source_datasets:
        dataset_name = os.path.basename(dataset_path)
        stats = analyze_single_dataset(dataset_path, dataset_name)
        
        if stats:
            all_stats.append(stats)
            # 预测选择的样本数
            expected_negative_selected = int(stats['negative_frames'] * negative_sample_probability)
            total_expected_positive += stats['positive_frames']
            total_expected_negative += expected_negative_selected
            total_expected_sequences += stats['sequences_with_positives']
    
    print(f"\n📈 合并后预期结果:")
    print(f"  预期总正样本: {total_expected_positive}")
    print(f"  预期总负样本: {total_expected_negative}")
    print(f"  预期总样本: {total_expected_positive + total_expected_negative}")
    print(f"  预期序列数: {total_expected_sequences}")
    
    if total_expected_positive + total_expected_negative > 0:
        final_positive_ratio = total_expected_positive / (total_expected_positive + total_expected_negative) * 100
        print(f"  最终正样本比例: {final_positive_ratio:.2f}%")
    
    return all_stats

def save_stats_to_csv(all_stats, output_file="dataset_analysis.csv"):
    """保存统计结果到CSV文件"""
    if not all_stats:
        return
        
    fieldnames = all_stats[0].keys()
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"\n💾 统计结果已保存到: {output_file}")

def main():
    # 配置数据集路径（与pick_positive.py相同）
    source_datasets = [
        "/media/xinchi/data/Event_RGB_catching_data/dataset_c",
        "/media/xinchi/data/Event_RGB_catching_data/dataset_dc",
    ]
    
    print("🔍 Event-RGB Catching 数据集统计分析")
    print("=" * 50)
    
    # 分析所有数据集（更新为30%负样本概率，与pick_positive.py一致）
    all_stats = analyze_pick_positive_behavior(source_datasets, negative_sample_probability=0.3)
    
    # 保存结果
    save_stats_to_csv(all_stats, "dataset_analysis.csv")
    
    print(f"\n✅ 分析完成！")

if __name__ == "__main__":
    main()