import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'noreply@yourdomain.com')
SENDER_NAME = os.getenv('SENDER_NAME', 'Smart Energy AI')

def send_report(recipient_email, pdf_path):
    """Send PDF report via SendGrid"""
    
    if not SENDGRID_API_KEY:
        print("⚠️ SendGrid API key not configured")
        return False
    
    try:
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        # Encode to base64
        encoded_file = base64.b64encode(pdf_data).decode()
        
        # Create message
        message = Mail(
            from_email=(SENDER_EMAIL, SENDER_NAME),
            to_emails=recipient_email,
            subject='Your Smart Energy AI - Prediction Report',
            html_content=f'''
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                        <h2 style="color: #00f0ff;">Smart Energy AI - Prediction Report</h2>
                        
                        <p>Hello,</p>
                        
                        <p>Thank you for using Smart Energy AI Platform!</p>
                        
                        <p>Please find attached your comprehensive energy prediction report. 
                        This report includes:</p>
                        
                        <ul>
                            <li>Summary statistics of your energy consumption</li>
                            <li>Recent prediction history</li>
                            <li>Usage patterns and efficiency metrics</li>
                        </ul>
                        
                        <p>Keep monitoring your energy consumption to optimize efficiency!</p>
                        
                        <p style="margin-top: 30px;">
                            <strong>Best regards,</strong><br/>
                            Smart Energy AI Team
                        </p>
                        
                        <hr style="margin-top: 30px; border: none; border-top: 1px solid #ddd;">
                        
                        <p style="font-size: 12px; color: #666;">
                            This is an automated message. Please do not reply to this email.
                        </p>
                    </div>
                </body>
            </html>
            '''
        )
        
        # Attach PDF
        attachedFile = Attachment(
            FileContent(encoded_file),
            FileName('energy_report.pdf'),
            FileType('application/pdf'),
            Disposition('attachment')
        )
        message.attachment = attachedFile
        
        # Send email
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        print(f"✅ Email sent successfully to {recipient_email}")
        print(f"   Status code: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False


def send_simple_email(recipient_email, subject, html_content):
    """Send a simple email via SendGrid"""
    
    if not SENDGRID_API_KEY:
        print("⚠️ SendGrid API key not configured")
        return False
    
    try:
        message = Mail(
            from_email=(SENDER_EMAIL, SENDER_NAME),
            to_emails=recipient_email,
            subject=subject,
            html_content=html_content
        )
        
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        print(f"✅ Email sent successfully to {recipient_email}")
        return True
        
    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False
