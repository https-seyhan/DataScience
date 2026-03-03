import smtplib
import ssl
import traceback
from email.message import EmailMessage
import os
import sys


# ==============================
# CONFIGURATION (Use ENV VARS)
# ==============================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465  # SSL
SENDER_EMAIL = os.getenv("ALERT_SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("ALERT_SENDER_PASSWORD")
RECIPIENT_EMAIL = "recipient@example.com"


def send_failure_email(error_message: str):
    """Send failure notification email with stack trace."""

    msg = EmailMessage()
    msg["Subject"] = "🚨 Program Failure Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL

    msg.set_content(f"""
Program has failed.

Error Details:
{error_message}
""")

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)


# ==============================
# MAIN PROGRAM
# ==============================
def main():
    # ---- Your actual program logic here ----
    print("Running program...")
    
    # Example failure
    x = 10 / 0   # Intentional error


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_trace = traceback.format_exc()
        
        try:
            send_failure_email(error_trace)
        except Exception as email_error:
            print("Failed to send alert email:", email_error)
        
        print("Program failed. Email alert sent.")
        sys.exit(1)