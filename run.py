import uvicorn
import os
from secure_config import SecureConfig

def setup_environment():
    """Setup secure configuration from .env file"""
    
    secure_config = SecureConfig()
    
    # Check if keys are already stored
    if secure_config.has_keys():
        print("✅ Configuration loaded from secure storage")
        return
    
    # Check if .env exists to import keys
    if os.path.exists('.env'):
        print("Importing API keys to secure storage...")
        
        # Read .env file
        env_vars = {}
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value.strip('"')
        
        # Store in secure config
        secure_config.store_api_keys(
            groq_key=env_vars.get('GROQ_API_KEY'),
            pinecone_key=env_vars.get('PINECONE_API_KEY')
        )
        
        # Delete .env file
        os.remove('.env')
        
        # Create .env.example
        with open('.env.example', 'w') as f:
            f.write('# API Keys Configuration\n')
            f.write('# Copy this file to .env and add your actual API keys\n\n')
            f.write('GROQ_API_KEY=your_groq_api_key_here\n')
            f.write('PINECONE_API_KEY=your_pinecone_api_key_here\n')
        
        print("✅ API keys stored securely in database")
        print("🗑️ .env file deleted automatically")
        print("📝 .env.example created with instructions")
        
    else:
        print("❌ No .env file found. Please create one with your API keys.")
        exit(1)

if __name__ == "__main__":
    setup_environment()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)