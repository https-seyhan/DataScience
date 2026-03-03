import smtplib
import traceback
import sys
from email.mime.text import MIMEText
from datetime import datetime
import os
import functools

# ==================== CONFIGURATION ====================
# Set these variables with your email credentials
SMTP_SERVER = 'smtp.gmail.com'  # Change for other providers
SMTP_PORT = 587
SENDER_EMAIL = 'your-email@gmail.com'
SENDER_PASSWORD = 'your-app-specific-password'  # Use app password for Gmail
RECIPIENT_EMAIL = 'recipient@example.com'
# =======================================================

def send_failure_email(program_name, error_message, traceback_str):
    """
    Send an email notification about program failure.
    
    Args:
        program_name: Name of the program that failed
        error_message: The error message
        traceback_str: The full traceback
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create email content
    subject = f"⚠️ PROGRAM FAILURE: {program_name} at {timestamp}"
    
    body = f"""
PROGRAM FAILURE NOTIFICATION
============================
Time: {timestamp}
Program: {program_name}

Error Message:
-------------
{error_message}

Traceback:
---------
{traceback_str}

This is an automated notification from your monitoring system.
    """
    
    # Create message
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    
    try:
        # Connect to SMTP server and send
        print(f"Attempting to send failure notification for {program_name}...")
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✅ Failure notification sent successfully to {RECIPIENT_EMAIL}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email notification: {e}")
        return False

# ==================== USAGE METHODS ====================

# Method 1: Direct try-except (simplest)
def method1_direct_try_except():
    """Example of direct try-except with email notification."""
    print("\n--- Method 1: Direct try-except ---")
    
    try:
        # Your code that might fail
        print("Running risky operation...")
        x = 1 / 0  # This will cause ZeroDivisionError
    except Exception as e:
        error_message = str(e)
        traceback_str = traceback.format_exc()
        program_name = "method1_direct_try_except"
        
        # Send email notification
        send_failure_email(program_name, error_message, traceback_str)
        
        # Option 1: Re-raise the exception if you want the program to fail
        # raise
        
        # Option 2: Handle the error and continue
        print(f"Error caught: {error_message}. Continuing execution...")

# Method 2: Decorator for automatic notifications
def notify_on_failure(func):
    """Decorator that sends email if the decorated function fails."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            program_name = func.__name__
            error_message = str(e)
            traceback_str = traceback.format_exc()
            
            send_failure_email(program_name, error_message, traceback_str)
            
            # Re-raise the exception
            raise
    return wrapper

# Example using the decorator
@notify_on_failure
def risky_database_operation():
    """Example function that might fail."""
    print("Connecting to database...")
    # Simulate database error
    raise ConnectionError("Database connection timeout after 30 seconds")

# Method 3: Context manager using function
def failure_notification_context(program_name):
    """
    Context manager for failure notification.
    Use with 'with' statement.
    """
    class FailureContext:
        def __init__(self, name):
            self.program_name = name
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                traceback_str = ''.join(traceback.format_exception(exc_type, exc_val, exc_tb))
                send_failure_email(self.program_name, str(exc_val), traceback_str)
            return False  # Don't suppress the exception
    
    return FailureContext(program_name)

# Method 4: Simple wrapper function
def run_with_notification(func, *args, **kwargs):
    """
    Run a function and send email notification if it fails.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        program_name = func.__name__
        error_message = str(e)
        traceback_str = traceback.format_exc()
        
        send_failure_email(program_name, error_message, traceback_str)
        raise  # Re-raise the exception

# ==================== EXAMPLE FUNCTIONS ====================

def calculate_average(numbers):
    """Function that might fail with empty list."""
    return sum(numbers) / len(numbers)

def read_config_file():
    """Function that might fail if file doesn't exist."""
    with open('config.json', 'r') as f:
        return f.read()

def api_call():
    """Function that might fail with network error."""
    import time
    time.sleep(1)
    raise TimeoutError("API request timed out")

# ==================== MAIN DEMO ====================

def main():
    print("=" * 50)
    print("EMAIL NOTIFICATION ON PROGRAM FAILURE")
    print("=" * 50)
    
    # Method 1: Direct try-except
    method1_direct_try_except()
    
    # Method 2: Using decorator
    print("\n--- Method 2: Using decorator ---")
    try:
        risky_database_operation()
    except Exception as e:
        print(f"Caught exception from decorator example: {e}")
    
    # Method 3: Using context manager
    print("\n--- Method 3: Using context manager ---")
    try:
        with failure_notification_context("file_operation"):
            # Code that might fail
            print("Opening file...")
            with open('nonexistent_file.txt', 'r') as f:
                content = f.read()
    except Exception as e:
        print(f"Caught exception from context manager: {e}")
    
    # Method 4: Using wrapper function
    print("\n--- Method 4: Using wrapper function ---")
    try:
        run_with_notification(calculate_average, [])  # Empty list will cause ZeroDivisionError
    except Exception as e:
        print(f"Caught exception from wrapper: {e}")

# ==================== SIMPLE PRACTICAL EXAMPLE ====================

def monitor_data_processing():
    """Practical example: Monitor a data processing task."""
    print("\n" + "=" * 50)
    print("PRACTICAL EXAMPLE: Data Processing Monitor")
    print("=" * 50)
    
    data_files = ['data1.csv', 'data2.csv', 'data3.csv']
    
    for file in data_files:
        try:
            print(f"\nProcessing {file}...")
            
            # Simulate processing
            if file == 'data2.csv':
                raise ValueError(f"Corrupted data in {file}")
            
            print(f"✅ Successfully processed {file}")
            
        except Exception as e:
            error_msg = str(e)
            trace = traceback.format_exc()
            
            # Send email notification
            send_failure_email(
                program_name=f"data_processing - {file}",
                error_message=error_msg,
                traceback_str=trace
            )
            
            print(f"❌ Failed to process {file}: {error_msg}")
            
            # Option: Continue with next file instead of stopping
            continue

# ==================== CONFIGURATION HELPER ====================

def setup_email_config():
    """Helper to set up email configuration from environment variables."""
    global SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL
    
    # Use environment variables if available (more secure)
    SENDER_EMAIL = os.environ.get('EMAIL_SENDER', SENDER_EMAIL)
    SENDER_PASSWORD = os.environ.get('EMAIL_PASSWORD', SENDER_PASSWORD)
    RECIPIENT_EMAIL = os.environ.get('EMAIL_RECIPIENT', RECIPIENT_EMAIL)
    
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        print("⚠️  Warning: Email configuration incomplete!")
        print(f"Current settings:")
        print(f"  Sender: {SENDER_EMAIL}")
        print(f"  Recipient: {RECIPIENT_EMAIL}")
        print(f"  Password: {'*' * len(SENDER_PASSWORD) if SENDER_PASSWORD else 'NOT SET'}")
        return False
    return True

# ==================== QUICK ONE-LINER VERSION ====================

def quick_fail_notify(error=None):
    """
    Ultra-simple one-liner style notification.
    Call this in your except block.
    """
    if error is None:
        error = sys.exc_info()[1]
    
    trace = traceback.format_exc()
    
    send_failure_email(
        program_name=os.path.basename(sys.argv[0]),
        error_message=str(error),
        traceback_str=trace
    )

# ==================== RUN THE EXAMPLES ====================

if __name__ == "__main__":
    # Check configuration first
    if not setup_email_config():
        print("\nPlease set up your email configuration before running.")
        print("You can either:")
        print("1. Edit the configuration variables at the top of the file")
        print("2. Set environment variables:")
        print("   export EMAIL_SENDER='your-email@gmail.com'")
        print("   export EMAIL_PASSWORD='your-app-password'")
        print("   export EMAIL_RECIPIENT='recipient@example.com'")
        sys.exit(1)
    
    # Run the examples
    main()
    monitor_data_processing()
    
    # Example of quick one-liner usage
    print("\n" + "=" * 50)
    print("QUICK ONE-LINER EXAMPLE")
    print("=" * 50)
    
    try:
        # Some code that might fail
        numbers = [1, 2, 3]
        result = numbers[10]  # IndexError
    except:
        quick_fail_notify()
        print("Notification sent using quick one-liner")