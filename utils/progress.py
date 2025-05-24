import sys
import time
from datetime import timedelta

def show_progress(current, total, start_time=None, description=None):
    """
    Display a professional progress bar with percentage, elapsed time, and ETA.
    
    Args:
        current: Current progress value
        total: Total items to process
        start_time: Time when processing started (optional)
        description: Short text description of what's being processed (optional)
    """
    bar_length = 30
    progress = min(1.0, current / total) if total > 0 else 1.0
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percentage = progress * 100
    
    # Add description if provided
    desc_text = f"{description}: " if description else ""
    
    # Time information if start_time is provided
    time_info = ""
    if start_time is not None:
        elapsed = time.time() - start_time
        if progress > 0:
            eta = elapsed * (1 - progress) / progress
            remaining = timedelta(seconds=int(eta))
            elapsed_formatted = timedelta(seconds=int(elapsed))
            time_info = f" | {elapsed_formatted} elapsed | ETA: {remaining}"
    
    sys.stdout.write(f'\r{desc_text}[{bar}] {current}/{total} ({percentage:.1f}%){time_info}')
    sys.stdout.flush()
    
    # Return true when complete
    return current == total

def clear_progress():
    """
    Clear the current progress bar line
    """
    sys.stdout.write('\r' + ' ' * 100 + '\r')  # Clear line
    sys.stdout.flush()