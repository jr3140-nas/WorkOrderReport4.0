
import os, smtplib, ssl
from typing import Optional
from email.message import EmailMessage

def send_email_with_attachment(
    to_address: str,
    subject: str,
    body: str,
    attachment_bytes: bytes,
    attachment_filename: str,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_username: Optional[str] = None,
    smtp_password: Optional[str] = None,
    smtp_sender: Optional[str] = None,
    use_tls: bool = True
) -> str:
    smtp_host = smtp_host or os.getenv("SMTP_HOST")
    smtp_port = int(smtp_port or os.getenv("SMTP_PORT", "587"))
    smtp_username = smtp_username or os.getenv("SMTP_USERNAME")
    smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
    smtp_sender = smtp_sender or os.getenv("SMTP_SENDER") or smtp_username

    if not (smtp_host and smtp_username and smtp_password and smtp_sender):
        raise RuntimeError("SMTP settings are missing. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and SMTP_SENDER.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_sender
    msg["To"] = to_address
    msg.set_content(body)

    if attachment_bytes:
        msg.add_attachment(attachment_bytes, maintype="application", subtype="pdf", filename=attachment_filename)

    context = ssl.create_default_context()
    if use_tls:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
    else:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
    return "Email sent"
