import sqlite3
import os
import socket
from cryptography.fernet import Fernet
import base64
import hashlib

class SecureConfig:
    def __init__(self):
        self.db_path = "local_data/config.db"
        os.makedirs("local_data", exist_ok=True)
        self._init_db()
        self._check_hostname_change()
    
    def _init_db(self):
        """Initialize config database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def _get_encryption_key(self):
        """Generate encryption key from hostname"""
        hostname = socket.gethostname()
        key_material = hashlib.sha256(hostname.encode()).digest()
        return base64.urlsafe_b64encode(key_material)
    
    def _check_hostname_change(self):
        """Check if hostname changed and clear old keys"""
        current_hostname = socket.gethostname()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM config WHERE key = 'hostname'")
        result = cursor.fetchone()
        
        if result and result[0] != current_hostname:
            # Hostname changed, clear all encrypted data
            cursor.execute("DELETE FROM config WHERE key IN ('groq_api_key', 'pinecone_api_key')")
            print(f"Hostname changed from {result[0]} to {current_hostname}. Cleared old encrypted keys.")
        
        # Update stored hostname
        cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", 
                      ('hostname', current_hostname))
        
        conn.commit()
        conn.close()
    
    def store_api_keys(self, groq_key, pinecone_key=None):
        """Store encrypted API keys"""
        fernet = Fernet(self._get_encryption_key())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Encrypt and store GROQ key
        encrypted_groq = fernet.encrypt(groq_key.encode()).decode()
        cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                      ('groq_api_key', encrypted_groq))
        
        # Encrypt and store Pinecone key if provided
        if pinecone_key:
            encrypted_pinecone = fernet.encrypt(pinecone_key.encode()).decode()
            cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                          ('pinecone_api_key', encrypted_pinecone))
        
        conn.commit()
        conn.close()
    
    def get_api_keys(self):
        """Get decrypted API keys"""
        try:
            fernet = Fernet(self._get_encryption_key())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            config = {}
            
            # Get GROQ key
            cursor.execute("SELECT value FROM config WHERE key = 'groq_api_key'")
            result = cursor.fetchone()
            if result:
                config['GROQ_API_KEY'] = fernet.decrypt(result[0].encode()).decode()
            
            # Get Pinecone key
            cursor.execute("SELECT value FROM config WHERE key = 'pinecone_api_key'")
            result = cursor.fetchone()
            if result:
                config['PINECONE_API_KEY'] = fernet.decrypt(result[0].encode()).decode()
            
            conn.close()
            return config
            
        except Exception as e:
            raise Exception(f"Failed to decrypt API keys: {e}")
    
    def has_keys(self):
        """Check if API keys are stored"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM config WHERE key IN ('groq_api_key', 'pinecone_api_key')")
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0