import smtplib
import traceback
import functools
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── Email Configuration ──────────────────────────────────────────────────────
EMAIL_CONFIG = {
    "sender":    "your_email@gmail.com",
    "password":  "your_app_password",       # Use an App Password, not your real password
    "recipient": "alert_recipient@gmail.com",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
}


def send_failure_email(subject: str, error: Exception, context: str = ""):
    """Sends an HTML email with full traceback when an error occurs."""
    tb = traceback.format_exc()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_body = f"""
    <html><body style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="color: #c0392b;">🚨 Program Failure Alert</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr><td style="padding: 6px; font-weight: bold;">Time</td>
                <td style="padding: 6px;">{timestamp}</td></tr>
            <tr style="background:#f9f9f9;"><td style="padding: 6px; font-weight: bold;">Context</td>
                <td style="padding: 6px;">{context or "N/A"}</td></tr>
            <tr><td style="padding: 6px; font-weight: bold;">Error Type</td>
                <td style="padding: 6px; color: #c0392b;">{type(error).__name__}</td></tr>
            <tr style="background:#f9f9f9;"><td style="padding: 6px; font-weight: bold;">Message</td>
                <td style="padding: 6px;">{error}</td></tr>
        </table>
        <h3>Full Traceback</h3>
        <pre style="background:#f4f4f4; padding:12px; border-left:4px solid #c0392b;
                    font-size:13px; overflow-x:auto;">{tb}</pre>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_CONFIG["sender"]
    msg["To"]      = EMAIL_CONFIG["recipient"]
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(EMAIL_CONFIG["smtp_host"], EMAIL_CONFIG["smtp_port"]) as server:
        server.starttls()
        server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
        server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["recipient"], msg.as_string())

    print(f"[{timestamp}] Failure email sent to {EMAIL_CONFIG['recipient']}")


# ── Option 1: Decorator ───────────────────────────────────────────────────────
def notify_on_failure(subject="Program Failure", context=""):
    """Decorator — wraps a function and emails if it raises an exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                send_failure_email(
                    subject=f"{subject}: {type(e).__name__}",
                    error=e,
                    context=context or func.__name__,
                )
                raise   # Re-raise so the program still exits with an error
        return wrapper
    return decorator


# ── Option 2: Context Manager ─────────────────────────────────────────────────
class NotifyOnFailure:
    """Context manager — emails if anything inside the `with` block fails."""
    def __init__(self, subject="Program Failure", context=""):
        self.subject = subject
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            send_failure_email(
                subject=f"{self.subject}: {exc_type.__name__}",
                error=exc_val,
                context=self.context,
            )
        return False    # False = don't suppress the exception


# ── Examples ──────────────────────────────────────────────────────────────────
# --- Decorator usage ---
@notify_on_failure(subject="Data Pipeline Failed", context="ETL Job")
def run_etl():
    print("Running ETL...")
    raise ValueError("Database connection timed out")   # Simulated failure


# --- Context manager usage ---
def run_report():
    with NotifyOnFailure(subject="Report Generation Failed", context="Monthly Report"):
        print("Generating report...")
        raise FileNotFoundError("data/sales.csv not found")   # Simulated failure


if __name__ == "__main__":
    # Uncomment the example you want to test:

    # run_etl()       # Decorator example
    # run_report()    # Context manager example

    print("Uncomment one of the example calls above to test.")
    print("Remember to update EMAIL_CONFIG with your credentials first.")
