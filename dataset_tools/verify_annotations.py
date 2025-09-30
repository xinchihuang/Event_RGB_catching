#!/usr/bin/env python3
"""
Interactive Annotation Verification Script

This script allows manual verification of bounding box annotations by displaying 
images with their corresponding bounding boxes and recording user feedback.

Controls:
- 'y' or 'Y': Mark as correct annotation
- 'n' or 'N': Mark as incorrect annotation
- 's' or 'S': Skip this annotation
- 'q' or 'Q': Quit and save results
- 'r' or 'R': Restart from beginning
- 'b' or 'B': Go back to previous image
- Space: Next image (same as correct)
- ESC: Quit without saving

Author: Generated for Event-RGB Catching Dataset Verification
"""

import os
import cv2
import csv
import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

class AnnotationVerifier:
    def __init__(self, dataset_root, start_dataset=0, output_file=None):
        self.dataset_root = dataset_root
        self.start_dataset = start_dataset
        self.current_dataset = start_dataset
        self.current_frame = 0
        
        # Results tracking
        self.verification_results = []
        self.current_results = {}
        
        # Setup output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"annotation_verification_{timestamp}.json"
        self.output_file = output_file
        
        # Load datasets
        self.datasets = self._load_datasets()
        self.total_datasets = len(self.datasets)
        
        if not self.datasets:
            raise ValueError("No datasets found in the specified directory")
            
        print(f"Found {self.total_datasets} datasets")
        print(f"Starting from dataset {start_dataset}")
        print(f"Results will be saved to: {output_file}")
        
        # Load existing results if file exists
        self._load_existing_results()
        
    def _load_datasets(self):
        """Load all available datasets"""
        datasets = []
        for item in os.listdir(self.dataset_root):
            item_path = os.path.join(self.dataset_root, item)
            if os.path.isdir(item_path) and item.isdigit():
                dataset_idx = int(item)
                if dataset_idx >= self.start_dataset:
                    datasets.append(dataset_idx)
        return sorted(datasets)
    
    def _load_existing_results(self):
        """Load existing verification results if available"""
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                    self.verification_results = data.get('results', [])
                    self.current_results = {r['image_path']: r for r in self.verification_results}
                print(f"Loaded {len(self.verification_results)} existing verification results")
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
    
    def _save_results(self):
        """Save verification results to file"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'dataset_root': self.dataset_root,
            'total_verified': len(self.verification_results),
            'results': self.verification_results
        }
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Results saved to: {self.output_file}")
            
            # Also save as CSV for easy analysis
            csv_file = self.output_file.replace('.json', '.csv')
            with open(csv_file, 'w', newline='') as f:
                if self.verification_results:
                    writer = csv.DictWriter(f, fieldnames=self.verification_results[0].keys())
                    writer.writeheader()
                    writer.writerows(self.verification_results)
            print(f"Results also saved as CSV: {csv_file}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _load_dataset_data(self, dataset_idx):
        """Load data for a specific dataset"""
        dataset_path = os.path.join(self.dataset_root, str(dataset_idx))
        
        # Load groundtruth
        groundtruth_path = os.path.join(dataset_path, 'groundtruth.txt')
        if not os.path.exists(groundtruth_path):
            return None
            
        bboxes = []
        try:
            with open(groundtruth_path, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 4:
                        x, y, w, h = map(int, row)
                        bboxes.append([x, y, w, h])
        except Exception as e:
            print(f"Error loading groundtruth for dataset {dataset_idx}: {e}")
            return None
        
        # Load absent.txt to determine which frames to show
        absent_path = os.path.join(dataset_path, 'absent.txt')
        absent_status = []
        if os.path.exists(absent_path):
            try:
                with open(absent_path, 'r') as f:
                    for line in f:
                        absent_status.append(int(line.strip()))
            except Exception as e:
                print(f"Error loading absent.txt for dataset {dataset_idx}: {e}")
                # Fallback: use groundtruth bboxes to determine status
                absent_status = [1 if bbox != [0, 0, 0, 0] else 0 for bbox in bboxes]
        else:
            print(f"Warning: absent.txt not found for dataset {dataset_idx}, using groundtruth fallback")
            # Fallback: use groundtruth bboxes to determine status
            absent_status = [1 if bbox != [0, 0, 0, 0] else 0 for bbox in bboxes]
        
        # Check for image directories
        rgb_dir = os.path.join(dataset_path, 'rgb')
        event_dir = os.path.join(dataset_path, 'event')
        combined_dir = os.path.join(dataset_path, 'combined')
        
        # Find available images - prioritize RGB, but also check event directory
        image_files = []
        if os.path.exists(rgb_dir):
            for f in sorted(os.listdir(rgb_dir), key=lambda x: int(x.split('.')[0])):
                if f.endswith('.jpg'):
                    image_files.append(os.path.join(rgb_dir, f))
        elif os.path.exists(event_dir):
            # Fallback to event images if RGB not available
            for f in sorted(os.listdir(event_dir), key=lambda x: int(x.split('.')[0])):
                if f.endswith('.jpg'):
                    image_files.append(os.path.join(event_dir, f))
        
        return {
            'bboxes': bboxes,
            'absent_status': absent_status,
            'image_files': image_files,
            'rgb_dir': rgb_dir,
            'event_dir': event_dir,
            'combined_dir': combined_dir
        }
    
    def _create_event_rgb_overlay(self, rgb_path, event_path, alpha=0.7):
        """Create overlay of RGB and event images"""
        # Load RGB image
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            return None
            
        # Load event image
        event_img = cv2.imread(event_path)
        if event_img is None:
            return rgb_img  # Return RGB only if event not available
            
        # Ensure both images have the same size
        if rgb_img.shape != event_img.shape:
            event_img = cv2.resize(event_img, (rgb_img.shape[1], rgb_img.shape[0]))
        
        # Create overlay: blend RGB and event images
        # RGB as base (alpha), event as overlay (1-alpha)
        overlay = cv2.addWeighted(rgb_img, alpha, event_img, 1-alpha, 0)
        
        return overlay
    
    def _draw_bbox_on_image(self, image, bbox, frame_idx, dataset_idx, is_present=True):
        """Draw bounding box and information on image"""
        x, y, w, h = bbox
        
        # Create a copy to avoid modifying original
        display_image = image.copy()
        
        # Draw bounding box only if present and valid
        if is_present and bbox != [0, 0, 0, 0]:
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Add bbox info
            cv2.putText(display_image, f"BBox: {w}x{h}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif is_present and bbox == [0, 0, 0, 0]:
            # Present but no valid bbox - show warning
            cv2.putText(display_image, "WARNING: Present but no valid bbox", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Add dataset and frame info
        cv2.putText(display_image, f"Dataset: {dataset_idx}", (20, display_image.shape[0] - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display_image, f"Frame: {frame_idx}", (20, display_image.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display_image, f"Progress: {len(self.verification_results)}", (20, display_image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Add controls info
        control_text = "Y=Correct | N=Incorrect | S=Skip | B=Back | Q=Quit"
        text_size = cv2.getTextSize(control_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(display_image, (10, 10), (text_size[0] + 20, 40), (0, 0, 0), -1)
        cv2.putText(display_image, control_text, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display_image
    
    def _record_result(self, dataset_idx, frame_idx, image_path, bbox, verification_status):
        """Record verification result"""
        result = {
            'dataset_idx': dataset_idx,
            'frame_idx': frame_idx,
            'image_path': image_path,
            'bbox': bbox,
            'verification_status': verification_status,  # 'correct', 'incorrect', 'skipped'
            'timestamp': datetime.now().isoformat()
        }
        
        # Remove existing result for this image if it exists
        self.verification_results = [r for r in self.verification_results 
                                   if r['image_path'] != image_path]
        
        # Add new result
        self.verification_results.append(result)
        self.current_results[image_path] = result
    
    def _get_verification_status(self, image_path):
        """Get existing verification status for an image"""
        result = self.current_results.get(image_path, {})
        return result.get('verification_status', '')
    
    def run(self):
        """Run the interactive verification process"""
        print("\n" + "="*60)
        print("ANNOTATION VERIFICATION TOOL")
        print("="*60)
        print("Controls:")
        print("  Y/y - Mark as CORRECT")
        print("  N/n - Mark as INCORRECT") 
        print("  S/s - SKIP")
        print("  B/b - Go BACK")
        print("  Q/q - QUIT and save")
        print("  R/r - RESTART from beginning")
        print("  ESC - Quit without saving")
        print("="*60 + "\n")
        
        navigation_history = []  # For back navigation: (dataset_idx, frame_idx)
        current_dataset_pos = 0  # Position in self.datasets list
        restart_verification = False
        resume_frame_idx = None  # For resuming at a specific frame when going back
        
        while current_dataset_pos < len(self.datasets):
            dataset_idx = self.datasets[current_dataset_pos]
            print(f"\nProcessing dataset {dataset_idx}...")
            
            dataset_data = self._load_dataset_data(dataset_idx)
            if dataset_data is None:
                print(f"Skipping dataset {dataset_idx} - could not load data")
                current_dataset_pos += 1
                continue
            
            bboxes = dataset_data['bboxes']
            absent_status = dataset_data['absent_status']
            image_files = dataset_data['image_files']
            
            if not image_files:
                print(f"No images found for dataset {dataset_idx}")
                current_dataset_pos += 1
                continue
            
            # Count present frames (absent_status == 1)
            present_count = sum(absent_status)
            print(f"Dataset {dataset_idx}: {len(bboxes)} total frames, {present_count} present frames to verify")
            
            # Process each frame - only show frames marked as present (absent_status == 1)
            frame_idx = resume_frame_idx if resume_frame_idx is not None else 0
            resume_frame_idx = None  # Reset after use
            go_back_to_previous_dataset = False
            
            while frame_idx < len(image_files) and not go_back_to_previous_dataset:
                if frame_idx >= len(bboxes) or frame_idx >= len(absent_status):
                    break
                    
                image_path = image_files[frame_idx]
                bbox = bboxes[frame_idx]
                is_present = absent_status[frame_idx]
                
                # Skip frames marked as absent (absent_status == 0)
                if is_present == 0:
                    frame_idx += 1
                    continue
                
                # Load and display image with event overlay
                try:
                    # Check if we should create RGB+Event overlay
                    rgb_path = image_path  # Current path is RGB
                    event_path = image_path.replace('/rgb/', '/event/')
                    is_overlay = False
                    
                    # Create overlay if both RGB and event images exist
                    if os.path.exists(rgb_path) and os.path.exists(event_path):
                        image = self._create_event_rgb_overlay(rgb_path, event_path)
                        if image is None:
                            # Fallback to RGB only
                            image = cv2.imread(rgb_path)
                        else:
                            is_overlay = True
                    else:
                        # Load RGB only if event not available
                        image = cv2.imread(image_path)
                    
                    if image is None:
                        print(f"Could not load image: {image_path}")
                        frame_idx += 1
                        continue
                    
                    # Scale image for display if too large
                    h, w = image.shape[:2]
                    if w > 1200 or h > 900:
                        scale = min(1200/w, 900/h)
                        new_w, new_h = int(w*scale), int(h*scale)
                        image = cv2.resize(image, (new_w, new_h))
                        # Scale bbox accordingly
                        bbox = [int(b*scale) for b in bbox]
                    
                    display_image = self._draw_bbox_on_image(image, bbox, frame_idx, dataset_idx, bool(is_present))
                    
                    # Add overlay indicator
                    if is_overlay:
                        cv2.putText(display_image, "RGB+Event Overlay", (display_image.shape[1] - 200, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check if already verified
                    existing_status = self._get_verification_status(image_path)
                    if existing_status:
                        cv2.putText(display_image, f"Status: {existing_status.upper()}", 
                                   (display_image.shape[1] - 200, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow('Annotation Verification', display_image)
                    
                    # Wait for key press
                    key = cv2.waitKey(0) & 0xFF
                    
                    if key == ord('y') or key == ord('Y') or key == ord(' '):  # Correct
                        self._record_result(dataset_idx, frame_idx, image_path, bbox, 'correct')
                        navigation_history.append((dataset_idx, frame_idx))
                        frame_idx += 1
                    elif key == ord('n') or key == ord('N'):  # Incorrect
                        self._record_result(dataset_idx, frame_idx, image_path, bbox, 'incorrect')
                        navigation_history.append((dataset_idx, frame_idx))
                        frame_idx += 1
                    elif key == ord('s') or key == ord('S'):  # Skip
                        self._record_result(dataset_idx, frame_idx, image_path, bbox, 'skipped')
                        navigation_history.append((dataset_idx, frame_idx))
                        frame_idx += 1
                    elif key == ord('b') or key == ord('B'):  # Back
                        if navigation_history:
                            prev_dataset, prev_frame = navigation_history.pop()
                            if prev_dataset == dataset_idx:
                                # Going back within current dataset
                                if prev_frame > 0:
                                    frame_idx = prev_frame - 1
                                else:
                                    print("Cannot go back further in this dataset")
                            else:
                                # Going back to previous dataset
                                print(f"Going back to dataset {prev_dataset}, frame {prev_frame}")
                                # Find the position of the previous dataset
                                try:
                                    prev_dataset_pos = self.datasets.index(prev_dataset)
                                    current_dataset_pos = prev_dataset_pos
                                    resume_frame_idx = prev_frame  # Resume at the specific frame
                                    go_back_to_previous_dataset = True
                                except ValueError:
                                    print(f"Error: Previous dataset {prev_dataset} not found")
                        else:
                            print("Cannot go back further - no navigation history")
                    elif key == ord('q') or key == ord('Q'):  # Quit and save
                        self._save_results()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('r') or key == ord('R'):  # Restart
                        self.verification_results = []
                        self.current_results = {}
                        navigation_history = []
                        current_dataset_pos = 0
                        restart_verification = True
                        break
                    elif key == 27:  # ESC - quit without saving
                        print("Quitting without saving...")
                        cv2.destroyAllWindows()
                        return
                
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    self._save_results()
                    cv2.destroyAllWindows()
                    return
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    frame_idx += 1
                    continue
            
            # Handle restart case
            if restart_verification:
                restart_verification = False
                continue
                
            # Move to next dataset if we didn't go back
            if not go_back_to_previous_dataset:
                current_dataset_pos += 1
        
        print("Verification complete!")
        self._save_results()
        cv2.destroyAllWindows()
    
    def print_summary(self):
        """Print summary of verification results"""
        if not self.verification_results:
            print("No verification results available")
            return
        
        correct = len([r for r in self.verification_results if r['verification_status'] == 'correct'])
        incorrect = len([r for r in self.verification_results if r['verification_status'] == 'incorrect'])
        skipped = len([r for r in self.verification_results if r['verification_status'] == 'skipped'])
        total = len(self.verification_results)
        
        print(f"\nVerification Summary:")
        print(f"Total verified: {total}")
        print(f"Correct: {correct} ({correct/total*100:.1f}%)")
        print(f"Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
        print(f"Skipped: {skipped} ({skipped/total*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Interactive Annotation Verification Tool")
    parser.add_argument("--dataset_root", default="/media/xinchi/data/Event_RGB_catching_data/dataset_dc",
                       help="Root directory of the dataset")
    parser.add_argument("--start_dataset", type=int, default=0,
                       help="Dataset index to start from")
    parser.add_argument("--output_file", 
                       help="Output file for verification results (JSON format)")
    parser.add_argument("--summary_only", action="store_true",
                       help="Only print summary of existing results")
    
    args = parser.parse_args()
    
    try:
        verifier = AnnotationVerifier(
            dataset_root=args.dataset_root,
            start_dataset=args.start_dataset,
            output_file=args.output_file
        )
        
        if args.summary_only:
            verifier.print_summary()
        else:
            verifier.run()
            verifier.print_summary()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()