import os
from env_crypto import decrypt_env_content

def load_config():
    """Load configuration from encrypted environment file or environment variables"""
    config = {}
    
    # Try to load from encrypted file first
    if os.path.exists('.env.encrypted'):
        password = os.getenv('ENV_PASSWORD')
        if not password:
            raise Exception("ENV_PASSWORD environment variable required for encrypted config")
        
        try:
            config = decrypt_env_content(password)
        except Exception as e:
            print(f"Warning: Could not decrypt .env.encrypted: {e}")
            config = {}
    
    # Fallback to environment variables
    config['GROQ_API_KEY'] = config.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
    config['PINECONE_API_KEY'] = config.get('PINECONE_API_KEY') or os.getenv('PINECONE_API_KEY')
    
    if not config['GROQ_API_KEY']:
        raise Exception("GROQ_API_KEY not found in encrypted config or environment variables")
    
    return config