# Keyboard Function Update Guide

## Overview

Updated the keyboard function in Event_RGB_catching project with enhanced features:
- State change detection and voice notifications
- Added 'w' key support for setting waiting state
- Converted all Chinese comments and outputs to English

## Modified Files

1. `/home/xinchi/Event_RGB_catching/live_real.py:48-101`
2. `/home/xinchi/Event_RGB_catching/record.py:46-99` 
3. `/home/xinchi/Event_RGB_catching/event_only/record.py:69-132`

## Key Changes

### 1. New Key Bindings

| Key | Action | Previous | Current |
|-----|--------|----------|---------|
| 't' | Set tracking state | ✓ | ✓ |
| 'w' | Set waiting state | ✗ | ✓ |
| Other keys | Set exit state | ✓ | ✓ |

### 2. Voice Notifications

**English Voice Messages:**
- live_real.py: "System started tracking"
- record.py: "System started tracking"  
- event_only/record.py: "Event system started tracking"

### 3. Comment Language Changes

**Before (Chinese):**
```python
def play_voice_notification(message):
    """播放语音提示"""
    engine.setProperty('rate', 150)  # 语速
    engine.setProperty('volume', 0.9)  # 音量
    print(f"语音提示: {message} (pyttsx3 not available)")
    print(f"语音播放失败: {e}, 使用文本提示: {message}")

def state_monitor():
    """监控状态变化并发出语音提示"""
    print(f"状态变化检测: {previous_state} -> {current_state}")
    # 在新线程中播放语音，避免阻塞
    time.sleep(0.1)  # 检查间隔100ms

# 启动状态监控线程
```

**After (English):**
```python
def play_voice_notification(message):
    """Play voice notification"""
    engine.setProperty('rate', 150)  # Speech rate
    engine.setProperty('volume', 0.9)  # Volume level
    print(f"Voice notification: {message} (pyttsx3 not available)")
    print(f"Voice playback failed: {e}, using text notification: {message}")

def state_monitor():
    """Monitor state changes and trigger voice notifications"""
    print(f"State change detected: {previous_state} -> {current_state}")
    # Play voice in new thread to avoid blocking
    time.sleep(0.1)  # Check interval 100ms

# Start state monitoring thread
```

## Updated Function Structure

### Live Real & Record Files
```python
def keyboard(shared_value,system_state):
    from pynput import keyboard
    import threading
    import time
    
    def play_voice_notification(message):
        """Play voice notification"""
        # English implementation
    
    def state_monitor():
        """Monitor state changes and trigger voice notifications"""
        # English implementation
    
    # Start state monitoring thread
    monitor_thread = threading.Thread(target=state_monitor, daemon=True)
    monitor_thread.start()
    
    def on_press(key):
        try:
            print(f"Key {key} pressed")
            if key.char=='t':
                system_state.value='t'
            elif key.char=='w':  # NEW
                system_state.value='w'
        except AttributeError:
            system_state.value='e'

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
```

### Event Only File
```python
def keyboard(shared_value, system_state):
    """
    Handle keyboard input for system control.
    """
    try:
        from pynput import keyboard
        import threading
        import time
        
        def play_voice_notification(message):
            """Play voice notification"""
            # English implementation with EVENT SYSTEM prefix
        
        def state_monitor():
            """Monitor state changes and trigger voice notifications"""
            # English implementation with EVENT SYSTEM prefix
        
        # Start state monitoring thread
        monitor_thread = threading.Thread(target=state_monitor, daemon=True)
        monitor_thread.start()
        
    except ImportError:
        print("EVENT SYSTEM: Keyboard control not available (pynput not installed)")
        return
        
    def on_press(key):
        try:
            print(f"EVENT SYSTEM: Key {key} pressed")
            if key.char == 't':
                system_state.value = 't'
            elif key.char == 'w':  # NEW
                system_state.value = 'w'
        except AttributeError:
            system_state.value = 'e'

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
```

## Usage Examples

### Setting Waiting State
```bash
# Start any entry point
python live_real.py --test_mode Event
python record.py --test_mode Show
cd event_only && python record.py

# Press 'w' key to set waiting state
# Press 't' key to set tracking state (with voice notification)
```

### State Transitions with New 'w' Key

```
Initial: 'w' (waiting)
Press 'w' → 'w' (waiting) 
Press 't' → 't' (tracking) + Voice: "System started tracking"
Press 'w' → 'w' (waiting)
Press other → 'e' (exit)
```

## Benefits of Changes

### 1. Enhanced Control
- Manual control over waiting state
- More flexible state management
- Better debugging capabilities

### 2. Internationalization 
- English-only codebase for better collaboration
- Consistent language across project
- Professional documentation standards

### 3. State Monitoring
- Real-time feedback via voice
- Visual confirmation through console
- Non-blocking operation

## Testing

1. **Verify 'w' key functionality:**
   ```bash
   python live_real.py --test_mode Event
   # Press 'w' → Should see: system_state.value = 'w'
   ```

2. **Test voice notifications:**
   ```bash
   python record.py --test_mode Show
   # Press 't' → Should hear: "System started tracking"
   ```

3. **Check console output:**
   ```
   Key 'w' pressed
   Key 't' pressed  
   State change detected: w -> t
   Voice notification: System started tracking
   ```

## Backwards Compatibility

- All existing functionality preserved
- Original key bindings ('t', other keys) unchanged  
- Voice notifications are additive (can be disabled if pyttsx3 not installed)
- New 'w' key is optional (doesn't break existing workflows)

## Configuration

Voice settings can be customized in each file:
```python
engine.setProperty('rate', 150)     # Speech rate (50-300)
engine.setProperty('volume', 0.9)   # Volume (0.0-1.0)
time.sleep(0.1)                     # Monitor interval (seconds)
```