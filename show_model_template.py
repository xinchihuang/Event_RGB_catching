#!/usr/bin/env python3
"""Display the model template in an isolated OpenCV process."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Show RGB and event model templates")
    parser.add_argument("--config", required=True, help="Model YAML configuration")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    paths = config["paths"]
    image_config = config["image"]
    processing = config["image_processing"]
    bbox = config["roi"]["default_bbox"]

    template_jpg = project_dir / paths["template_jpg"]
    template_npy = project_dir / paths["template_npy"]
    rgb_image = cv2.imread(str(template_jpg))
    if rgb_image is None:
        raise RuntimeError(f"Failed to load RGB template: {template_jpg}")
    events = np.load(template_npy, allow_pickle=True)

    brightness = processing.get("brightness_multiplier", 1.0)
    if brightness != 1.0:
        rgb_image = np.clip(
            rgb_image.astype(np.float32) * brightness, 0, 255
        ).astype(np.uint8)

    height = image_config["rgb_height"]
    width = image_config["rgb_width"]
    event_image = np.zeros((height, width, 3), dtype=np.uint8)
    x_values = events["x"].astype(np.int64)
    y_values = events["y"].astype(np.int64)
    valid = (
        (x_values >= 0)
        & (x_values < width)
        & (y_values >= 0)
        & (y_values < height)
    )
    x_values = x_values[valid]
    y_values = y_values[valid]
    polarities = events["p"][valid]
    event_image[y_values[polarities == 0], x_values[polarities == 0]] = (255, 0, 0)
    event_image[y_values[polarities != 0], x_values[polarities != 0]] = (0, 0, 255)

    x, y, box_width, box_height = bbox
    cv2.rectangle(
        rgb_image, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2
    )
    cv2.rectangle(
        event_image, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2
    )
    cv2.putText(
        rgb_image,
        "Model Template",
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        event_image,
        "Event Template",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    combined = np.hstack((rgb_image, event_image))
    scale = processing.get("display_scale", 0.8)
    preview = cv2.resize(
        combined,
        (int(combined.shape[1] * scale), int(combined.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )

    window_name = "Ball Detection (RGB + Event) - Press any key to continue"
    print(f"Showing model template preview: {template_jpg}", flush=True)
    print("Preview: creating OpenCV window", flush=True)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_scale = float(os.environ.get("GUI_WINDOW_SCALE", "3"))
    cv2.resizeWindow(window_name, int(400 * window_scale), int(300 * window_scale))
    print("Preview: displaying image", flush=True)
    cv2.imshow(window_name, preview)
    print("Preview: waiting up to 2 seconds", flush=True)
    cv2.waitKey(2000)
    print("Preview: closing window", flush=True)
    cv2.destroyAllWindows()
    print("Model template preview closed", flush=True)


if __name__ == "__main__":
    main()
