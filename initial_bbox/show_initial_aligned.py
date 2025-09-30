#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple RGB and Event image overlay display script
For displaying RGB image and Event data overlay effect only
"""

import cv2
import numpy as np
import os
import argparse


def generate_event_image_simple(event_data: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Simple event image generation method
    
    Args:
        event_data: Event data (.npy format, structured array with 'x', 'y', 'p', 't' fields)
        width: Image width
        height: Image height
        
    Returns:
        Event image (grayscale)
    """
    event_image = np.zeros((height, width), dtype=np.uint8)
    
    if len(event_data) > 0:
        # Extract x, y coordinates from structured array
        if hasattr(event_data, 'dtype') and event_data.dtype.names:
            # Structured array format
            x_coords = event_data['x'].astype(int)
            y_coords = event_data['y'].astype(int)
        else:
            # Regular array format [x, y, t, p]
            x_coords = event_data[:, 0].astype(int)
            y_coords = event_data[:, 1].astype(int)
        
        # Filter valid coordinates
        valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        # Mark event points on event image
        if len(x_coords) > 0:
            event_image[y_coords, x_coords] = 255
    
    return event_image


def show_overlay(rgb_image: np.ndarray, event_data: np.ndarray) -> None:
    """
    Display RGB and Event image overlay effect
    
    Args:
        rgb_image: RGB image
        event_data: Event data
    """
    # Generate event image
    rgb_height, rgb_width = rgb_image.shape[:2]
    event_image = generate_event_image_simple(event_data, rgb_width, rgb_height)
    
    # Convert to color image
    event_image_color = cv2.cvtColor(event_image, cv2.COLOR_GRAY2BGR)
    
    # Create overlay image
    blended = cv2.addWeighted(rgb_image, 0.7, event_image_color, 0.3, 0)
    
    # Add title text
    cv2.putText(blended, "RGB + Event Overlay", (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
    
    # Adjust display size
    display_height, display_width = blended.shape[:2]
    if display_width > 1200 or display_height > 800:
        scale = min(1200 / display_width, 800 / display_height)
        new_width = int(display_width * scale)
        new_height = int(display_height * scale)
        blended = cv2.resize(blended, (new_width, new_height))
    
    print("Displaying RGB+Event overlay image... Press any key to exit.")
    cv2.imshow('RGB + Event Overlay', blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Simple RGB and Event image overlay display')
    parser.add_argument('--rgb_path', type=str, 
                       default="/home/xinchi/Event_RGB_catching/78.jpg",
                       help='RGB image path')
    parser.add_argument('--event_path', type=str,
                       default="/home/xinchi/Event_RGB_catching/78.npy",
                       help='Event data path (.npy file)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.rgb_path):
        print(f"Error: RGB image file does not exist: {args.rgb_path}")
        return
    
    if not os.path.exists(args.event_path):
        print(f"Error: Event data file does not exist: {args.event_path}")
        return
    
    # Load RGB image
    rgb_image = cv2.imread(args.rgb_path)
    if rgb_image is None:
        print(f"Error: Unable to load RGB image: {args.rgb_path}")
        return
    
    # Load Event data
    try:
        event_data = np.load(args.event_path)
        print(f"Successfully loaded:")
        print(f"  RGB image: {args.rgb_path} - Size: {rgb_image.shape}")
        print(f"  Event data: {args.event_path} - Event count: {len(event_data)}")
        
        if len(event_data.shape) != 2 or event_data.shape[1] < 2:
            print("Warning: Event data format may be incorrect, expected [N, >=2] array")
        
    except Exception as e:
        print(f"Error: Unable to load Event data: {e}")
        return
    
    # Display overlay effect
    show_overlay(rgb_image, event_data)
    print("Program ended")


if __name__ == "__main__":
    main()