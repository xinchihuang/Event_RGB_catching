#!/usr/bin/env python3
"""
Analysis script for annotation verification results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
import argparse

def load_verification_results(json_file):
    """Load verification results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['results']

def analyze_results(results):
    """Analyze verification results and generate statistics"""
    df = pd.DataFrame(results)
    
    stats = {
        'total_verified': len(results),
        'by_status': df['verification_status'].value_counts().to_dict(),
        'by_dataset': defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'skipped': 0}),
        'accuracy_by_dataset': {},
        'bbox_analysis': {}
    }
    
    # Analyze by dataset
    for _, row in df.iterrows():
        dataset_idx = row['dataset_idx']
        status = row['verification_status']
        stats['by_dataset'][dataset_idx][status] += 1
    
    # Calculate accuracy per dataset
    for dataset_idx, counts in stats['by_dataset'].items():
        total = counts['correct'] + counts['incorrect']
        if total > 0:
            accuracy = counts['correct'] / total * 100
            stats['accuracy_by_dataset'][dataset_idx] = accuracy
    
    # Analyze bounding box characteristics
    correct_bboxes = []
    incorrect_bboxes = []
    
    for _, row in df.iterrows():
        bbox = row['bbox']
        if bbox != [0, 0, 0, 0]:  # Valid bbox
            x, y, w, h = bbox
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            bbox_info = {
                'area': area,
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio,
                'x': x,
                'y': y
            }
            
            if row['verification_status'] == 'correct':
                correct_bboxes.append(bbox_info)
            elif row['verification_status'] == 'incorrect':
                incorrect_bboxes.append(bbox_info)
    
    stats['bbox_analysis'] = {
        'correct': correct_bboxes,
        'incorrect': incorrect_bboxes
    }
    
    return stats, df

def create_visualizations(stats, df, output_dir):
    """Create visualization plots for verification results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall status distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    status_counts = stats['by_status']
    ax1.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
    ax1.set_title('Overall Verification Status Distribution')
    
    ax2.bar(status_counts.keys(), status_counts.values())
    ax2.set_title('Verification Status Counts')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'status_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy by dataset
    if stats['accuracy_by_dataset']:
        plt.figure(figsize=(12, 6))
        datasets = sorted(stats['accuracy_by_dataset'].keys())
        accuracies = [stats['accuracy_by_dataset'][d] for d in datasets]
        
        plt.bar(datasets, accuracies)
        plt.title('Annotation Accuracy by Dataset')
        plt.xlabel('Dataset Index')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Dataset verification progress
    dataset_stats = []
    for dataset_idx, counts in stats['by_dataset'].items():
        total = sum(counts.values())
        dataset_stats.append({
            'dataset': dataset_idx,
            'total': total,
            'correct': counts['correct'],
            'incorrect': counts['incorrect'],
            'skipped': counts['skipped']
        })
    
    if dataset_stats:
        df_datasets = pd.DataFrame(dataset_stats).sort_values('dataset')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        width = 0.8
        datasets = df_datasets['dataset']
        
        ax.bar(datasets, df_datasets['correct'], width, label='Correct', color='green', alpha=0.7)
        ax.bar(datasets, df_datasets['incorrect'], width, bottom=df_datasets['correct'], 
               label='Incorrect', color='red', alpha=0.7)
        ax.bar(datasets, df_datasets['skipped'], width, 
               bottom=df_datasets['correct'] + df_datasets['incorrect'],
               label='Skipped', color='yellow', alpha=0.7)
        
        ax.set_xlabel('Dataset Index')
        ax.set_ylabel('Number of Annotations')
        ax.set_title('Verification Status by Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'verification_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Bounding box analysis
    correct_bboxes = stats['bbox_analysis']['correct']
    incorrect_bboxes = stats['bbox_analysis']['incorrect']
    
    if correct_bboxes or incorrect_bboxes:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Area comparison
        if correct_bboxes:
            correct_areas = [b['area'] for b in correct_bboxes]
            ax1.hist(correct_areas, alpha=0.7, label='Correct', color='green', bins=20)
        if incorrect_bboxes:
            incorrect_areas = [b['area'] for b in incorrect_bboxes]
            ax1.hist(incorrect_areas, alpha=0.7, label='Incorrect', color='red', bins=20)
        ax1.set_title('Bounding Box Area Distribution')
        ax1.set_xlabel('Area (square pixels)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Aspect ratio comparison
        if correct_bboxes:
            correct_aspects = [b['aspect_ratio'] for b in correct_bboxes if b['aspect_ratio'] > 0]
            ax2.hist(correct_aspects, alpha=0.7, label='Correct', color='green', bins=20)
        if incorrect_bboxes:
            incorrect_aspects = [b['aspect_ratio'] for b in incorrect_bboxes if b['aspect_ratio'] > 0]
            ax2.hist(incorrect_aspects, alpha=0.7, label='Incorrect', color='red', bins=20)
        ax2.set_title('Bounding Box Aspect Ratio Distribution')
        ax2.set_xlabel('Aspect Ratio (Width/Height)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Width vs Height scatter
        if correct_bboxes:
            correct_w = [b['width'] for b in correct_bboxes]
            correct_h = [b['height'] for b in correct_bboxes]
            ax3.scatter(correct_w, correct_h, alpha=0.6, label='Correct', color='green', s=10)
        if incorrect_bboxes:
            incorrect_w = [b['width'] for b in incorrect_bboxes]
            incorrect_h = [b['height'] for b in incorrect_bboxes]
            ax3.scatter(incorrect_w, incorrect_h, alpha=0.6, label='Incorrect', color='red', s=10)
        ax3.set_title('Bounding Box Width vs Height')
        ax3.set_xlabel('Width (pixels)')
        ax3.set_ylabel('Height (pixels)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Position distribution
        if correct_bboxes:
            correct_x = [b['x'] for b in correct_bboxes]
            correct_y = [b['y'] for b in correct_bboxes]
            ax4.scatter(correct_x, correct_y, alpha=0.6, label='Correct', color='green', s=10)
        if incorrect_bboxes:
            incorrect_x = [b['x'] for b in incorrect_bboxes]
            incorrect_y = [b['y'] for b in incorrect_bboxes]
            ax4.scatter(incorrect_x, incorrect_y, alpha=0.6, label='Incorrect', color='red', s=10)
        ax4.set_title('Bounding Box Position Distribution')
        ax4.set_xlabel('X Position (pixels)')
        ax4.set_ylabel('Y Position (pixels)')
        ax4.invert_yaxis()  # Image coordinates
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bbox_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(stats, output_file):
    """Generate a text report of verification results"""
    with open(output_file, 'w') as f:
        f.write("ANNOTATION VERIFICATION ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total annotations verified: {stats['total_verified']}\n")
        
        for status, count in stats['by_status'].items():
            percentage = count / stats['total_verified'] * 100
            f.write(f"{status.capitalize()}: {count} ({percentage:.1f}%)\n")
        
        f.write("\nACCURACY BY DATASET:\n")
        f.write("-" * 30 + "\n")
        f.write(f"{'Dataset':<8} {'Correct':<8} {'Incorrect':<10} {'Total':<8} {'Accuracy':<10}\n")
        f.write("-" * 50 + "\n")
        
        for dataset_idx in sorted(stats['by_dataset'].keys()):
            counts = stats['by_dataset'][dataset_idx]
            total = counts['correct'] + counts['incorrect']
            accuracy = stats['accuracy_by_dataset'].get(dataset_idx, 0)
            f.write(f"{dataset_idx:<8} {counts['correct']:<8} {counts['incorrect']:<10} "
                   f"{total:<8} {accuracy:<9.1f}%\n")
        
        # Overall accuracy
        total_correct = sum(stats['by_status'].get('correct', 0) for _ in [0])
        total_incorrect = sum(stats['by_status'].get('incorrect', 0) for _ in [0])
        total_evaluated = stats['by_status'].get('correct', 0) + stats['by_status'].get('incorrect', 0)
        
        if total_evaluated > 0:
            overall_accuracy = stats['by_status'].get('correct', 0) / total_evaluated * 100
            f.write(f"\nOVERALL ACCURACY: {overall_accuracy:.1f}%\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze annotation verification results")
    parser.add_argument("results_file", help="JSON file containing verification results")
    parser.add_argument("--output_dir", default="verification_analysis", 
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Results file not found: {args.results_file}")
        return
    
    print(f"Loading results from: {args.results_file}")
    results = load_verification_results(args.results_file)
    
    if not results:
        print("No results found in the file")
        return
    
    print(f"Analyzing {len(results)} verification results...")
    stats, df = analyze_results(results)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Creating visualizations...")
    create_visualizations(stats, df, args.output_dir)
    
    # Generate text report
    report_file = os.path.join(args.output_dir, "verification_analysis_report.txt")
    print("Generating report...")
    generate_report(stats, report_file)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print(f"Report: {report_file}")

if __name__ == "__main__":
    main()