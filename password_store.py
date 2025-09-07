import os
import keyring
import getpass

def store_password():
    """Store encryption password securely in system keyring"""
    password = getpass.getpass("Enter encryption password to store: ")
    keyring.set_password("DocumentInsights", "env_password", password)
    print("✅ Password stored securely in system keyring")

def get_stored_password():
    """Get stored password from system keyring"""
    try:
        return keyring.get_password("DocumentInsights", "env_password")
    except:
        return None

def clear_stored_password():
    """Clear stored password from system keyring"""
    try:
        keyring.delete_password("DocumentInsights", "env_password")
        print("✅ Password cleared from keyring")
    except:
        pass

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_stored_password()
    else:
        store_password()