from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import os
import base64

def send_report(to_email, pdf_path):
    message = Mail(
        from_email=os.getenv("FROM_EMAIL"),
        to_emails=to_email,
        subject="Your Energy Consumption Report",
        html_content="<p>Your energy report is attached.</p>"
    )

    with open(pdf_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    message.add_attachment(
        encoded,
        "application/pdf",
        "energy_report.pdf",
        "attachment"
    )

    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
    sg.send(message)
