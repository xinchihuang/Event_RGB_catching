#!/usr/bin/env python3
"""
Script to update absent.txt files based on manual verification results.
Changes incorrect annotations from 1 to 0 (absent).
"""

import os
import csv
import argparse

def update_absent_files(verification_csv, dataset_root):
    """
    Update absent.txt files based on verification results.
    
    Args:
        verification_csv: Path to the verification CSV file
        dataset_root: Root directory containing dataset folders
    """
    updates_made = 0
    
    # Read verification results
    with open(verification_csv, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if row['verification_status'] == 'incorrect':
                dataset_idx = int(row['dataset_idx'])
                frame_idx = int(row['frame_idx'])
                
                # Path to the absent.txt file
                absent_file = os.path.join(dataset_root, str(dataset_idx), 'absent.txt')
                
                if not os.path.exists(absent_file):
                    print(f"Warning: absent.txt not found for dataset {dataset_idx}")
                    continue
                
                # Read current absent.txt
                with open(absent_file, 'r') as af:
                    lines = af.readlines()
                
                # Check if frame_idx is within bounds
                if frame_idx >= len(lines):
                    print(f"Warning: frame_idx {frame_idx} out of bounds for dataset {dataset_idx} (max: {len(lines)-1})")
                    continue
                
                # Update the line (frame_idx corresponds to line number since we start from frame 0)
                current_value = lines[frame_idx].strip()
                if current_value == '1':
                    lines[frame_idx] = '0\n'
                    
                    # Write back to file
                    with open(absent_file, 'w') as af:
                        af.writelines(lines)
                    
                    print(f"Updated dataset {dataset_idx}, frame {frame_idx}: {current_value} -> 0")
                    updates_made += 1
                elif current_value == '0':
                    print(f"Dataset {dataset_idx}, frame {frame_idx} already marked as absent (0)")
                else:
                    print(f"Warning: unexpected value '{current_value}' in dataset {dataset_idx}, frame {frame_idx}")
    
    print(f"\nTotal updates made: {updates_made}")
    return updates_made

def main():
    parser = argparse.ArgumentParser(description='Update absent.txt files based on verification results')
    parser.add_argument('--verification-csv', 
                        default='/home/xinchi/Event_RGB_catching/dataset_tools/annotation_verification_20250926_213339.csv',
                        help='Path to verification CSV file')
    parser.add_argument('--dataset-root', 
                        default='/media/xinchi/data/Event_RGB_catching_data/dataset_dc',
                        help='Root directory containing dataset folders')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be changed without making actual changes')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.verification_csv):
        print(f"Error: Verification CSV file not found: {args.verification_csv}")
        return 1
    
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root directory not found: {args.dataset_root}")
        return 1
    
    print(f"Verification CSV: {args.verification_csv}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dry run: {args.dry_run}")
    print("-" * 50)
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        # For dry run, we'll simulate the process
        with open(args.verification_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['verification_status'] == 'incorrect':
                    dataset_idx = int(row['dataset_idx'])
                    frame_idx = int(row['frame_idx'])
                    absent_file = os.path.join(args.dataset_root, str(dataset_idx), 'absent.txt')
                    
                    if os.path.exists(absent_file):
                        with open(absent_file, 'r') as af:
                            lines = af.readlines()
                        
                        if frame_idx < len(lines):
                            current_value = lines[frame_idx].strip()
                            print(f"Would update dataset {dataset_idx}, frame {frame_idx}: {current_value} -> 0")
                        else:
                            print(f"Would skip dataset {dataset_idx}, frame {frame_idx}: out of bounds")
                    else:
                        print(f"Would skip dataset {dataset_idx}: absent.txt not found")
    else:
        updates_made = update_absent_files(args.verification_csv, args.dataset_root)
        if updates_made > 0:
            print(f"\nSuccessfully updated {updates_made} entries.")
        else:
            print("\nNo updates needed.")
    
    return 0

if __name__ == '__main__':
    exit(main())