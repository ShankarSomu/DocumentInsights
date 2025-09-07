import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

def generate_key_from_password(password: str, salt: bytes = None) -> tuple:
    if salt is None:
        salt = os.urandom(16)
    
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return Fernet(key), salt

def encrypt_env_file(password: str):
    with open('.env', 'r') as f:
        content = f.read()
    
    fernet, salt = generate_key_from_password(password)
    encrypted_content = fernet.encrypt(content.encode())
    
    with open('.env.encrypted', 'wb') as f:
        f.write(salt + encrypted_content)
    
    print("Environment file encrypted successfully!")
    print("You can now delete the .env file and commit .env.encrypted")

def decrypt_env_content(password: str) -> dict:
    try:
        with open('.env.encrypted', 'rb') as f:
            data = f.read()
        
        salt = data[:16]
        encrypted_content = data[16:]
        
        fernet, _ = generate_key_from_password(password, salt)
        decrypted_content = fernet.decrypt(encrypted_content).decode()
        
        env_vars = {}
        for line in decrypted_content.strip().split('\n'):
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"')
        
        return env_vars
    except Exception as e:
        raise Exception(f"Failed to decrypt environment file: {e}")

if __name__ == "__main__":
    password = input("Enter encryption password: ")
    encrypt_env_file(password)