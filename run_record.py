#!/usr/bin/env python3
"""
Auto-restart wrapper script for record.py
Automatically restarts record.py with incremented record_index after completion
"""

import subprocess
import argparse
import time
import sys
import os
import signal

class RecordRunner:
    def __init__(self, start_index=40, max_runs=None, restart_delay=5):
        self.start_index = start_index
        self.current_index = start_index
        self.max_runs = max_runs
        self.restart_delay = restart_delay
        self.run_count = 0
        self.should_stop = False
        
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
        
        print(f"=== Starting record.py with record_index={self.current_index} ===")
        print(f"Command: {' '.join(cmd)}")
        print(f"Run #{self.run_count + 1}" + (f" of {self.max_runs}" if self.max_runs else ""))
        print("-" * 60)
        
        try:
            # Run record.py and wait for completion
            result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            return result.returncode
        except FileNotFoundError:
            print("Error: record.py not found in current directory")
            return 1
        except Exception as e:
            print(f"Error running record.py: {e}")
            return 1
    
    def run_loop(self):

        """Main execution loop with auto-restart"""
        print(f"Starting record runner with initial index: {self.start_index}")
        if self.max_runs:
            print(f"Maximum runs: {self.max_runs}")
        print(f"Restart delay: {self.restart_delay} seconds")
        print("=" * 60)
        
        while not self.should_stop:
            # Check if we've reached max runs
            if self.max_runs and self.run_count >= self.max_runs:
                print(f"Reached maximum runs ({self.max_runs}), stopping...")
                break
            
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
            
            # Check return code
            if returncode != 0:
                print(f"Warning: record.py exited with non-zero code ({returncode})")
            
            # Increment index for next run
            self.current_index += 1
            
            # Check if we should continue
            if self.should_stop:
                print("Stop requested, not restarting...")
                break
            
            if self.max_runs and self.run_count >= self.max_runs:
                print(f"Completed all {self.max_runs} runs")
                break
            
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
        print(f"Final record index: {self.current_index - 1}")

def main():
    parser = argparse.ArgumentParser(
        description='Auto-restart wrapper for record.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_record.py                              # Start from index 40, run indefinitely
  python run_record.py --start_index 1             # Start from index 1
  python run_record.py --max_runs 5                # Run exactly 5 times
  python run_record.py --start_index 10 --max_runs 3  # Start from 10, run 3 times
  python run_record.py --restart_delay 10          # Wait 10 seconds between restarts
        """
    )
    
    parser.add_argument('--start_index', type=int, default=69,
                       help='Initial record index (default: 0)')
    parser.add_argument('--max_runs', type=int, default=110,
                       help='Maximum number of runs (default: unlimited)')
    parser.add_argument('--restart_delay', type=int, default=2,
                       help='Delay in seconds before restart (default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_index < 0:
        print("Error: start_index must be non-negative")
        sys.exit(1)
    
    if args.max_runs is not None and args.max_runs <= 0:
        print("Error: max_runs must be positive")
        sys.exit(1)
    
    if args.restart_delay < 0:
        print("Error: restart_delay must be non-negative")
        sys.exit(1)
    
    # Create and run the runner
    runner = RecordRunner(
        start_index=args.start_index,
        max_runs=args.max_runs,
        restart_delay=args.restart_delay
    )
    
    try:
        runner.run_loop()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()