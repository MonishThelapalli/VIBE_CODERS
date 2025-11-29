# templates_db.py
from typing import Dict
from models import CustomerCategory

# Simulated "LLM templates DB"
# We use format placeholders that will be filled with customer data.
TEMPLATES: Dict[CustomerCategory, Dict[str, str]] = {
    "critical": {
        "subject": "Thank you for being a valued customer – your latest bill",
        "body": (
            "Dear Customer {customer_id},\n\n"
            "Thank you for staying with us for {tenure} month(s). "
            "Your current monthly charge is ₹{monthly_charges:.2f}, and your "
            "total spend with us so far is ₹{total_charges:.2f}.\n\n"
            "This is a friendly update about your account. No action is required "
            "if your payments are up to date. If you ever need any help with your "
            "plan or billing, our support team is happy to assist you.\n\n"
            "Warm regards,\nCustomer Care Team"
        ),
    },
    "occasional_defaulter": {
        "subject": "Reminder – avoid late fees on your telecom bill",
        "body": (
            "Dear Customer {customer_id},\n\n"
            "We appreciate you as our customer for the last {tenure} month(s). "
            "We’ve noticed a few delays in recent payments. Your usual monthly "
            "charge is around ₹{monthly_charges:.2f}, and your total spend so far "
            "is ₹{total_charges:.2f}.\n\n"
            "To avoid late fees or service interruptions, please ensure upcoming "
            "bills are paid on time. You can use our app or website for quick and "
            "secure payments.\n\n"
            "If you’re facing any difficulty, please reach out – we’re here to help.\n\n"
            "Regards,\nBilling Team"
        ),
    },
    "habitual_defaulter": {
        "subject": "Urgent: please clear your outstanding telecom dues",
        "body": (
            "Dear Customer {customer_id},\n\n"
            "Our records indicate repeated delays or missed payments on your account. "
            "You have been with us for {tenure} month(s), with a typical monthly "
            "charge of ₹{monthly_charges:.2f}, and a total billed amount of "
            "₹{total_charges:.2f} so far.\n\n"
            "Please clear your outstanding dues at the earliest to avoid service "
            "suspension and additional late charges as per policy.\n\n"
            "If you are facing financial difficulty, contact our support team to "
            "discuss possible payment options.\n\n"
            "Sincerely,\nCollections Team"
        ),
    },
}


def get_template(category: CustomerCategory) -> Dict[str, str]:
    return TEMPLATES[category]
