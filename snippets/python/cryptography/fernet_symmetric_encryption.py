"""
TAGS: crypt|cryptography|decrypt|encrypt|fernet|password|secret
DESCRIPTION: Securely encrypt a string with a password using Fernet symmetric encryption
"""

import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def encrypt(message: str, password: str, salt: bytes) -> bytes:
    """Convert string message into encrypted bytes using password and salt"""
    bytes_message: bytes = message.encode("utf-8")
    bytes_password: bytes = password.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(bytes_password))
    fernet = Fernet(key)
    encrypted_message = fernet.encrypt(bytes_message)
    return encrypted_message


def decrypt(bytes_message: bytes, password: str, salt: bytes) -> str:
    """Convert encrypted bytes string to original message string using password and salt"""
    bytes_password: bytes = password.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(bytes_password))
    fernet = Fernet(key)
    decrypted_message_bytes: bytes = fernet.decrypt(bytes_message)
    return decrypted_message_bytes.decode(encoding="utf-8")


if __name__ == "__main__":
    salt: bytes = os.urandom(16)
    original_message: str = input("Please provide a message to encrypt: ")
    encrypted_message: bytes = encrypt(
        message=original_message,
        password="Extremely$ecureP@ssword69420!",
        salt=salt,
    )
    print("encrypted message:", encrypted_message)
    decrypted_message: str = decrypt(
        bytes_message=encrypted_message,
        password="Extremely$ecureP@ssword69420!",
        salt=salt,
    )
    print("decrypted message:", decrypted_message)
