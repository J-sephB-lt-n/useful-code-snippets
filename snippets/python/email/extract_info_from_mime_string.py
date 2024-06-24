"""
TAGS: attachment|attachments|email|extract|mime|parse
DESCRIPTION: An example of how to extract the content from an email stored in MIME format 
NOTES: I am not confident that this code will work for all cases - I don't understand the MIME format well.
"""

import email
from email.message import EmailMessage

rawtext: str = "email content MIME here"
email_message = email.message_from_string(rawtext, _class=EmailMessage)

print(
    f"""
   FROM: {email_message.get("From")}
     TO: {email_message.get("To")}
SUBJECT: {email_message.get("Subject")}
"""
)

for attachment in email_message.iter_attachments():
    filename = attachment.get_filename()
    content_type = attachment.get_content_type()
    content_bytes = attachment.get_payload(decode=True)
    print(filename, content_type)
    # save attachments as local files #
    with open(filename, "wb") as file:
        file.write(content_bytes)
