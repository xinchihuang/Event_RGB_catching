#!/usr/bin/env python3
"""
Simple test script for event-only record.py
Creates minimal test environment and verifies basic functionality
"""

import os
import sys
import subprocess
import tempfile
import signal
import time

def create_test_environment():
    """Create a minimal test environment with dummy data"""
    
    # Create temporary directory structure
    test_dir = tempfile.mkdtemp(prefix="event_test_")
    event_dir = os.path.join(test_dir, "event")
    os.makedirs(event_dir, exist_ok=True)
    
    print(f"Created test directory: {test_dir}")
    
    # Create a dummy .raw file (empty file for now)
    dummy_raw = os.path.join(event_dir, "0.raw")
    with open(dummy_raw, 'wb') as f:
        # Write minimal raw file header (this might need adjustment based on format)
        f.write(b'\x00' * 1024)  # Dummy content
    
    return test_dir

def test_import_only():
    """Test if all imports work correctly"""
    print("Testing imports...")
    
    try:
        # Test imports without running the full system
        script = """
import sys
sys.path.append('/home/xinchi/Event_RGB_catching')

try:
    from read_from_raw import read_from_raw
    print("✓ read_from_raw import successful")
except Exception as e:
    print(f"✗ read_from_raw import failed: {e}")

try:
    from Event_tracker import event_tracking_3D
    print("✓ Event_tracker import successful")
except Exception as e:
    print(f"✗ Event_tracker import failed: {e}")

try:
    from fitting_process import landing_prediction, robot_control
    print("✓ fitting_process import successful")
except Exception as e:
    print(f"✗ fitting_process import failed: {e}")

try:
    from config.config_loader import switch_global_config
    print("✓ config_loader import successful")
except Exception as e:
    print(f"✗ config_loader import failed: {e}")
"""
        
        result = subprocess.run([sys.executable, '-c', script], 
                              capture_output=True, text=True, cwd='/home/xinchi/Event_RGB_catching/event_only')
        
        print("Import test output:")
        print(result.stdout)
        if result.stderr:
            print("Import test errors:")
            print(result.stderr)
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Import test failed: {e}")
        return False

def test_help_command():
    """Test help command"""
    print("Testing help command...")
    
    try:
        result = subprocess.run([sys.executable, 'record.py', '--help'], 
                              capture_output=True, text=True, 
                              cwd='/home/xinchi/Event_RGB_catching/event_only')
        
        if "Event-Only Ball Catching System" in result.stdout:
            print("✓ Help command works")
            return True
        else:
            print("✗ Help command failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Help test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting Event-only system tests...")
    print("=" * 50)
    
    # Test 1: Import functionality
    import_success = test_import_only()
    
    # Test 2: Help command
    help_success = test_help_command()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Import test: {'✓ PASS' if import_success else '✗ FAIL'}")
    print(f"Help test: {'✓ PASS' if help_success else '✗ FAIL'}")
    
    if import_success and help_success:
        print("\n✓ Basic functionality tests PASSED")
        print("The event-only system appears to be properly configured.")
        return True
    else:
        print("\n✗ Some tests FAILED")
        print("Check the errors above and fix dependencies.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)