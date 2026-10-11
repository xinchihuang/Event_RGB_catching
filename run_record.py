#!/usr/bin/env python3
"""
Auto-restart wrapper script for record.py
Automatically restarts record.py with incremented record_index after completion
Supports batch execution of multiple parameter sets, saved in one new directory
per batch invocation.
"""

import subprocess
import argparse
import time
import sys
import os
import signal
import json
import csv
from datetime import datetime
from pathlib import Path
from replay_options import positive_ratio
from project_paths import resolve_path

PROJECT = Path(__file__).resolve().parent


def load_indices_from_csv(csv_path):
    if not csv_path:
        return []
    path = csv_path
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        print(f"Error: index CSV {path} does not exist.")
        return []
    indices = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                value = row[0].strip()
                if not value:
                    continue
                if value.lower() in {"trajectory_id", "index", "record_index"}:
                    continue
                try:
                    idx = int(float(value))
                except ValueError:
                    print(f"Warning: invalid index '{value}' in {path}, skipping")
                    continue
                indices.append(idx)
    except Exception as e:
        print(f"Error reading index CSV {path}: {e}")
        return []
    return indices

class RecordRunner:
    def __init__(self, start_index=40, max_runs=None, restart_delay=5, timeout=180,
                 test_mode="Event", task="normal", index_list=None, evaluation_root=None, ratio=10, record_dir=None):
        self.record_dir = str(resolve_path(record_dir)) if record_dir is not None else None
        self.ratio = positive_ratio(ratio)
        self.start_index = start_index
        self.current_index = start_index
        self.max_runs = max_runs
        self.restart_delay = restart_delay
        self.timeout = timeout
        self.test_mode = test_mode
        self.task = task
        self.evaluation_root = evaluation_root
        self.index_list = list(index_list) if index_list else None
        self.index_pos = 0
        self.run_count = 0
        self.should_stop = False
        self.last_index = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {sig}, stopping after current run...")
        self.should_stop = True
    
    def run_record_process(self):
        """Run record.py with the current record_index"""
        cmd = [sys.executable, "record.py", "--record_index", str(self.current_index)]
        
        # Add test_mode and task parameters
        cmd.extend(["--test_mode", self.test_mode])
        cmd.extend(["--task", self.task])
        cmd.extend(["--ratio", str(self.ratio)])
        if self.record_dir is not None:
            cmd.extend(["--record-dir", self.record_dir])
        
        print(f"=== Starting record.py with record_index={self.current_index}, test_mode={self.test_mode}, task={self.task} ===")
        print(f"Command: {' '.join(cmd)}")
        print(f"Run #{self.run_count + 1}" + (f" of {self.max_runs}" if self.max_runs else ""))
        print("-" * 60)
        
        try:
            env = os.environ.copy()
            if self.evaluation_root is not None:
                # All samples in this batch share its reserved results directory.
                env['EVALUATION_RESULTS_DIR'] = str(self.evaluation_root)
            # Run record.py with a timeout to prevent hanging
            result = subprocess.run(cmd, 
                                  cwd=os.path.dirname(os.path.abspath(__file__)), 
                                  timeout=self.timeout, env=env)
            return result.returncode
        except subprocess.TimeoutExpired:
            print(f"Warning: record.py timed out after {self.timeout} seconds, terminating...")
            return 124  # Standard timeout exit code
        except FileNotFoundError:
            print("Error: record.py not found in current directory")
            return 1
        except Exception as e:
            print(f"Error running record.py: {e}")
            return 1
    
    def run_loop(self):

        """Main execution loop with auto-restart"""
        print(f"Starting record runner with initial index: {self.start_index}")
        if self.index_list:
            total_planned = len(self.index_list)
            if self.max_runs:
                total_planned = min(self.max_runs, total_planned)
            print(f"Index list mode: {total_planned} runs from CSV list")
        elif self.max_runs:
            print(f"Maximum runs: {self.max_runs}")
        print(f"Restart delay: {self.restart_delay} seconds")
        print("=" * 60)
        
        while not self.should_stop:
            # Check if we've reached max runs
            if self.max_runs and self.run_count >= self.max_runs:
                print(f"Reached maximum runs ({self.max_runs}), stopping...")
                break

            if self.index_list:
                if self.index_pos >= len(self.index_list):
                    print("Reached end of index list, stopping...")
                    break
                self.current_index = self.index_list[self.index_pos]
            
            # Run record.py
            start_time = time.time()
            returncode = self.run_record_process()
            end_time = time.time()
            duration = end_time - start_time
            
            self.run_count += 1
            
            print("-" * 60)
            print(f"Run #{self.run_count} completed with return code: {returncode}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Record index used: {self.current_index}")
            self.last_index = self.current_index
            
            # Check return code
            if returncode != 0:
                print(f"Warning: record.py exited with non-zero code ({returncode})")
            
            # Increment index for next run
            if self.index_list:
                self.index_pos += 1
            else:
                self.current_index += 1
            
            # Check if we should continue
            if self.should_stop:
                print("Stop requested, not restarting...")
                break
            
            if self.max_runs and self.run_count >= self.max_runs:
                print(f"Completed all {self.max_runs} runs")
                break
            
            # Close all OpenCV windows before restart
            print("Closing all OpenCV windows...")
            try:
                import cv2
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Process any pending GUI events
            except ImportError:
                print("OpenCV not available, skipping window cleanup")
            except Exception as e:
                print(f"Error closing OpenCV windows: {e}")
            
            # Wait before restart
            print(f"Waiting {self.restart_delay} seconds before restart...")
            for i in range(self.restart_delay, 0, -1):
                if self.should_stop:
                    print("Stop requested during delay, exiting...")
                    return
                print(f"Restarting in {i} seconds...", end='\r')
                time.sleep(1)
            print()
        
        print("=" * 60)
        print(f"Record runner finished. Total runs: {self.run_count}")
        if self.last_index is not None:
            print(f"Final record index: {self.last_index}")
        else:
            print("Final record index: n/a")

def run_batch_config(batch_configs, output_root=None, ratio=None):
    """Reserve one new run directory for all configurations and sample IDs."""
    if isinstance(batch_configs, dict):
        batch_configs = [batch_configs]
    elif not isinstance(batch_configs, list):
        raise ValueError("batch_config must be a JSON list or object")
    if not batch_configs or not all(isinstance(config, dict) for config in batch_configs):
        raise ValueError("batch_config must contain at least one configuration object")

    # Validate all ratios before creating output; CLI overrides per-entry JSON.
    batch_configs = [dict(config, ratio=positive_ratio(
        ratio if ratio is not None else config.get('ratio', 10)))
        for config in batch_configs]

    output = (resolve_path(output_root) if output_root is not None else
              PROJECT / 'evaluation_tools' / 'batch_runs' /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    # Never reuse an existing batch, including a user-supplied output directory.
    output.mkdir(parents=True, exist_ok=False)
    results = output / 'results'
    results.mkdir()
    (output / 'batch_config.json').write_text(
        json.dumps(batch_configs, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    total_configs = len(batch_configs)
    print(f"Starting batch execution with {total_configs} configurations")
    print(f"Batch output: {output}", flush=True)
    print(f"Results root: {results}", flush=True)
    print("=" * 80)
    
    for config_idx, config in enumerate(batch_configs, 1):
        print(f"\n### BATCH CONFIG {config_idx}/{total_configs} ###")
        print(f"Configuration: {config}")
        print("=" * 60)
        
        # Create runner for this configuration
        index_list = config.get("index_list")
        index_csv = config.get("index_csv")
        if index_list and not isinstance(index_list, list):
            print("Error: index_list must be a JSON list of integers")
            continue
        if index_csv:
            index_list = load_indices_from_csv(index_csv)
            if not index_list:
                print(f"Error: no valid indices loaded from {index_csv}, skipping config")
                continue

        runner = RecordRunner(
            start_index=config.get('start_index', 40),
            max_runs=config.get('max_runs'),
            restart_delay=config.get('restart_delay', 5),
            timeout=config.get('timeout', 180),
            test_mode=config.get('test_mode', "Event"),
            task=config.get('task', "normal"),
            index_list=index_list,
            evaluation_root=results,
            ratio=config['ratio'],
            record_dir=config.get('record_dir'),
        )
        
        try:
            runner.run_loop()
            print(f"\n### BATCH CONFIG {config_idx}/{total_configs} COMPLETED ###")
        except KeyboardInterrupt:
            print(f"\n### BATCH CONFIG {config_idx}/{total_configs} INTERRUPTED ###")
            print("Stopping batch execution...")
            break
        except Exception as e:
            print(f"\n### BATCH CONFIG {config_idx}/{total_configs} FAILED ###")
            print(f"Error: {e}")
            # Continue with next configuration instead of stopping
            continue
    
    print("\n" + "=" * 80)
    print("Batch execution completed")
    print(f"Batch output: {output}", flush=True)
    return output

def parse_batch_config(batch_config_arg):
    """Parse batch configuration from file or JSON string"""
    if not batch_config_arg:
        return None
    
    # Try to parse as file path first
    if os.path.exists(batch_config_arg):
        try:
            with open(batch_config_arg, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {batch_config_arg}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file {batch_config_arg}: {e}")
            sys.exit(1)
    else:
        path_hint = os.path.splitext(batch_config_arg)[1].lower() == ".json"
        path_hint = path_hint or (os.sep in batch_config_arg)
        if os.altsep:
            path_hint = path_hint or (os.altsep in batch_config_arg)
        if path_hint:
            print(f"Error: batch config file not found: {batch_config_arg}")
            sys.exit(1)
        # Try to parse as JSON string
        try:
            return json.loads(batch_config_arg)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON string: {e}")
            print("Batch config should be either a valid JSON file path or JSON string")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Auto-restart wrapper for record.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Example batch_config.json content:
  [
    {"test_mode": "Event", "task": "normal", "start_index": 0, "max_runs": 106},
    {"test_mode": "Event", "task": "dark", "start_index": 0, "max_runs": 106},
    {"test_mode": "RGB", "task": "clutter", "start_index": 0, "max_runs": 50}
  ]
        """
    )

    parser.add_argument('--ratio', type=positive_ratio, default=None,
                        help='Override every JSON entry ratio; otherwise use its ratio or 10.')
    parser.add_argument('--batch_config', type=str,default='batch_config.json',
                       help='JSON file or JSON string containing batch parameter configurations')
    parser.add_argument('--output-root', type=Path,
                        help='New batch directory; default: evaluation_tools/batch_runs/<timestamp>. '
                             'An existing directory is never reused.')
    
    args = parser.parse_args()


    # Check if batch configuration is provided
    batch_configs = parse_batch_config(args.batch_config)

    try:
        run_batch_config(batch_configs, output_root=args.output_root, ratio=args.ratio)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"Unexpected error in batch mode: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
